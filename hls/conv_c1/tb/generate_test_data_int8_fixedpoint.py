"""
Generates conv_c1_int8_fixedpoint_test_data.h: same real int8-quantized C1
weights + real int8-quantized MNIST test image as
generate_test_data_int8.py, but the reference output is now computed with
GENUINE FIXED-POINT integer arithmetic for the requantization step
(int32 accumulator -> int8 output), not float32 multiply/divide.

This follows the TFLite/gemmlowp "quantization multiplier" technique (see
e.g. tensorflow/lite/kernels/internal/quantization_util.cc's
QuantizeMultiplier()): a real-valued multiplier `combined_scale/out_scale`
is decomposed, OFFLINE and ONCE (never on the hardware's per-pixel
critical path), into
  - a Q31 fixed-point mantissa `M` (an int32 representation of a
    normalized [0.5, 1.0) significand -- always fits comfortably inside a
    signed int32, per quantize_multiplier() below), and
  - an integer right-shift `S`,
such that `(acc * M) >> S`, rounded, approximates `acc * (combined_scale
/ out_scale)`. `M`/`S` are computed once per layer and burned into the
generated header as plain integer constants -- conv_c1_int8_fixedpoint.cpp
never sees a float x_scale/w_scale/out_scale at all.

Note: this reproduces gemmlowp's NORMALIZED-SIGNIFICAND decomposition
(same math), but combines gemmlowp's two separate steps -- a fixed 31-bit
"SaturatingRoundingDoublingHighMul" descale plus a variable
"RoundingDivideByPOT" for the exponent -- into ONE wider (64-bit) rounding
shift. Those primitives are split in gemmlowp for ARM NEON SIMD
performance reasons, not because the split is numerically required; a
single 64-bit intermediate is simpler to implement and verify correctly
here, with the same underlying multiplier+shift technique and equivalent
rounding behavior.

Reuses the project's proven references exactly like
generate_test_data_int8.py does: quantize_int_real() (weights),
quantize_activation() (activations), conv2d_int() (int8x int8->int32 MAC
+ dequantize). Since conv2d_int() only returns the DEQUANTIZED float
output (by design -- see its own docstring), the raw int32 accumulator is
recovered from it via the exact inverse of conv2d_int()'s own
`acc.astype(np.float32) * combined_scale` step, and the recovery is
VERIFIED exact (not assumed) before being trusted as the fixed-point
reference's starting point.

Run from repo root: python hls/conv_c1/tb/generate_test_data_int8_fixedpoint.py
"""
import os
import sys
import math
import zipfile
import tempfile

import numpy as np
import h5py
import torchvision

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
from manual_layers import relu
from integer_layers import conv2d_int, quantize_activation
from quantization import quantize_int_real

N_BITS = 8

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
MODEL_PATH = os.path.join(REPO_ROOT, "models", "lenet5_relu.keras")
OUT_PATH = os.path.join(os.path.dirname(__file__), "conv_c1_int8_fixedpoint_test_data.h")


def quantize_multiplier(real_multiplier):
    """
    (M, S) such that (value * M) >> S approximates value * real_multiplier,
    via the TFLite/gemmlowp normalized-significand technique: decompose
    real_multiplier = significand * 2**exponent with significand in
    [0.5, 1.0) (math.frexp does exactly this), quantize the significand to
    Q31 fixed point (M, always in [2**30, 2**31 - 1] -- comfortably inside
    signed int32), and fold BOTH the Q31 descale (31 bits) and the
    significand's own exponent into one combined shift S.
    """
    assert real_multiplier > 0, "requantization multiplier must be positive"
    significand, exponent = math.frexp(real_multiplier)
    M = round(significand * (1 << 31))
    if M == (1 << 31):  # rounded up to exactly 1.0 -- renormalize
        M //= 2
        exponent += 1
    S = 31 - exponent
    assert 0 < M < (1 << 31), f"M={M} does not fit comfortably in signed int32"
    assert S >= 0, f"S={S} is negative -- unexpected for this network's scale ratios"
    return M, S


def requantize_fixed(acc_nonneg, M, S):
    """
    (acc * M) rounded right-shifted by S bits, matching
    conv_c1_int8_fixedpoint.cpp's int64 multiply + rounding-shift exactly.
    acc_nonneg must already be >= 0 (ReLU applied) and M > 0, so the
    product is always non-negative -- a plain "add half, then shift"
    round-to-nearest (ties round up) is unambiguous, no negative-number
    shift semantics to worry about.
    """
    acc64 = acc_nonneg.astype(np.int64)
    prod = acc64 * np.int64(M)
    half = np.int64(1) << (S - 1) if S > 0 else np.int64(0)
    return (prod + half) >> S


# ---- 1. Load trained C1 weights straight out of the .keras archive ----
with tempfile.TemporaryDirectory() as tmpdir:
    with zipfile.ZipFile(MODEL_PATH) as z:
        z.extractall(tmpdir)
    with h5py.File(os.path.join(tmpdir, "model.weights.h5"), "r") as f:
        arrays = {}
        for group_name in f["layers"]:
            group = f["layers"][group_name]
            if "vars" not in group or len(group["vars"]) == 0:
                continue
            v = group["vars"]
            if len(v) >= 2:
                arrays[group_name] = (v["0"][()], v["1"][()])

c1_w = c1_b = None
for name, (w, b) in arrays.items():
    if w.shape == (5, 5, 1, 6):
        c1_w, c1_b = w.astype(np.float32), b.astype(np.float32)

assert c1_w is not None, "Could not find C1 weights (shape (5,5,1,6)) in the model archive"
print("Loaded C1 weights:", c1_w.shape, "bias:", c1_b.shape)

# ---- 2. Load one real MNIST test image (test[0]) ----
ds = torchvision.datasets.MNIST(root=os.path.join(tempfile.gettempdir(), "mnist_data"),
                                 train=False, download=True)
img_pil, true_label = ds[0]
image = (np.array(img_pil, dtype=np.float32) / 255.0).reshape(28, 28, 1)
print(f"Test image: MNIST test[0], true label = {true_label}")

# ---- 3. Quantize weights and input activation, same proven quantizers ----
w_int, w_scale = quantize_int_real(c1_w, N_BITS)
x_int, x_scale = quantize_activation(image, N_BITS)
w_scale = np.float32(w_scale)
x_scale = np.float32(x_scale)
combined_scale = np.float32(x_scale * w_scale)
print(f"x_scale = {x_scale}, w_scale = {w_scale}, combined_scale = {combined_scale}")

# ---- 4. Get the proven MAC+dequantize reference, then recover the raw
#         int32 accumulator that conv2d_int() computed internally but
#         doesn't return (see its own docstring) -- via the EXACT inverse
#         of its own `acc.astype(np.float32) * combined_scale` step,
#         VERIFIED, not assumed, to round-trip losslessly. ----
float_out = conv2d_int(x_int, w_int, c1_b, x_scale, w_scale, padding="same")

acc_recovered = np.round(float_out.astype(np.float64) / float(combined_scale)).astype(np.int64)
reconstructed = (acc_recovered.astype(np.float32) * combined_scale)
assert np.array_equal(reconstructed, float_out), (
    "Could not exactly recover conv2d_int()'s internal int32 accumulator "
    "from its float output -- fixed-point reference would be built on an "
    "unverified foundation."
)
assert acc_recovered.min() >= np.iinfo(np.int32).min and acc_recovered.max() <= np.iinfo(np.int32).max, \
    "Recovered accumulator does not fit in int32"
print(f"Accumulator recovery verified exact. acc range: [{acc_recovered.min()}, {acc_recovered.max()}]")

bias_int = np.round(c1_b / combined_scale).astype(np.int32)

# ---- 5. ReLU directly on the integer accumulator (sign(acc) == sign of
#         the dequantized value since combined_scale > 0 -- exact, no
#         float dequant needed at all in this design). ----
acc_relu = np.maximum(acc_recovered, 0)

# ---- 6. Derive this layer's OUTPUT activation scale the usual way
#         (data-driven, from the float ReLU'd output) -- this is the one
#         place float statistics are still used, and only OFFLINE, at
#         generation time, same as real quantization-calibration workflows. ----
relu_out_float = relu(float_out).astype(np.float32)
_, out_scale = quantize_activation(relu_out_float, N_BITS)
out_scale = np.float32(out_scale)

real_multiplier = float(combined_scale) / float(out_scale)
M, S = quantize_multiplier(real_multiplier)
print(f"out_scale = {out_scale}")
print(f"real_multiplier = combined_scale/out_scale = {real_multiplier}")
print(f"quantize_multiplier -> M = {M}, S = {S}  (M/2**31 * 2**(31-S) = {M / (1 << 31) * 2 ** (31 - S)})")

# ---- 7. The fixed-point reference output itself: GENUINE integer
#         multiply + rounding right-shift, matching
#         conv_c1_int8_fixedpoint.cpp's arithmetic exactly (not the float
#         division/reciprocal-multiply used in the earlier two versions). ----
requantized = requantize_fixed(acc_relu, M, S)
expected_output = np.clip(requantized, -128, 127).astype(np.int8)

print(f"expected_output range: [{expected_output.min()}, {expected_output.max()}]")
print(f"Nonzero fraction: {np.mean(expected_output != 0):.3f}")


def emit_int_array(f, name, dims, arr, ctype):
    flat = arr.flatten()
    dim_str = "".join(f"[{d}]" for d in dims)
    f.write(f"static const {ctype} {name}{dim_str} = {{\n")
    for i in range(0, len(flat), 16):
        chunk = flat[i:i + 16]
        f.write("    " + ", ".join(str(int(v)) for v in chunk) + ",\n")
    f.write("};\n\n")


with open(OUT_PATH, "w") as f:
    f.write("// AUTO-GENERATED by generate_test_data_int8_fixedpoint.py -- do not hand-edit.\n")
    f.write("// Real int8-quantized Models/lenet5_relu.keras C1 weights + a real\n")
    f.write("// int8-quantized MNIST test[0] image, plus a GENUINE fixed-point\n")
    f.write("// (int64 multiply + rounding right-shift) reference output, for\n")
    f.write("// conv_c1_int8_fixedpoint_tb.cpp to check the fully-integer INT8 HLS\n")
    f.write("// kernel against. See quantize_multiplier()/requantize_fixed() in the\n")
    f.write("// generator script for the exact arithmetic this must match.\n")
    f.write("#ifndef CONV_C1_INT8_FIXEDPOINT_TEST_DATA_H\n#define CONV_C1_INT8_FIXEDPOINT_TEST_DATA_H\n\n")

    f.write(f"#define TRUE_LABEL {int(true_label)}\n")
    f.write(f"#define REQUANT_MULT {M}\n")
    f.write(f"#define REQUANT_SHIFT {S}\n\n")

    emit_int_array(f, "input_data", [28, 28, 1], x_int, "signed char")
    emit_int_array(f, "weights_data", [5, 5, 1, 6], w_int, "signed char")
    emit_int_array(f, "bias_int_data", [6], bias_int, "int")
    emit_int_array(f, "expected_output", [28, 28, 6], expected_output, "signed char")

    f.write("#endif\n")

print(f"\nWrote {OUT_PATH}")
