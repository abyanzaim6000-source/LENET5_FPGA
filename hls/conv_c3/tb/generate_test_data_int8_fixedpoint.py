"""
Generates conv_c3_int8_fixedpoint_test_data.h: real int8-quantized C3
weights + a real int8-quantized S2-pooled activation map (via the C1->S2
int8 pipeline) as conv_c3_int8_fixedpoint's input, plus the reference
output computed with GENUINE fixed-point integer arithmetic -- same
technique as hls/conv_c1/tb/generate_test_data_int8_fixedpoint.py, but
this script derives C3's OWN Q31 multiplier/shift (M3, S3) and verifies
C3's OWN accumulator recovery independently -- nothing here is reused
from C1's already-computed M/S; C1's pipeline is only run first because
C3's real input (S2's pooled output) doesn't exist without it.

Pipeline (all real data, no synthetic values):
  1. C1 conv (int8x int8->int32 MAC via conv2d_int(), ReLU on the
     accumulator, fixed-point requantize) -> real 28x28x6 int8
     activations. This duplicates C1's own already-proven steps (see
     hls/conv_c1/tb/generate_test_data_int8_fixedpoint.py) only because
     C3 needs a real input to convolve over -- C1's own M/S are derived
     fresh here too, not imported, keeping this script self-contained.
  2. S2 max pool (int8, no rescale) -> 14x14x6, C3's real input. Pooling
     doesn't change the scale factor, so this array shares C1's exact
     out_scale.
  3. C3 conv (int8x int8->int32 MAC via conv2d_int(), padding="valid",
     C1's out_scale as C3's INPUT scale, C3's own weights quantized
     fresh) -> the raw int32 accumulator is recovered from conv2d_int()'s
     float output via its exact inverse and the round-trip is VERIFIED
     exact -- C3's OWN verification, independent of C1's.
  4. ReLU directly on C3's accumulator, then C3's own
     quantize_multiplier()-derived (M3, S3) fixed-point requantize to
     int8 -- this IS the bit-exact-in-Python reference
     conv_c3_int8_fixedpoint.cpp must match, checked here before any C++
     is written.

Run from repo root: python hls/conv_c3/tb/generate_test_data_int8_fixedpoint.py
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
from integer_layers import conv2d_int, quantize_activation
from quantization import quantize_int_real
from manual_layers import relu

N_BITS = 8
POOL_SIZE = 2
STRIDE = 2

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
MODEL_PATH = os.path.join(REPO_ROOT, "models", "lenet5_relu.keras")
OUT_PATH = os.path.join(os.path.dirname(__file__), "conv_c3_int8_fixedpoint_test_data.h")


def quantize_multiplier(real_multiplier):
    """TFLite/gemmlowp-style decomposition -- see conv_c1's generator for
    the full explanation. Duplicated here (not imported) so this script
    stays self-contained, matching the project's per-stage convention."""
    assert real_multiplier > 0
    significand, exponent = math.frexp(real_multiplier)
    M = round(significand * (1 << 31))
    if M == (1 << 31):
        M //= 2
        exponent += 1
    S = 31 - exponent
    assert 0 < M < (1 << 31), f"M={M} does not fit comfortably in signed int32"
    assert S >= 0, f"S={S} is negative -- unexpected for this network's scale ratios"
    return M, S


def requantize_fixed(acc_nonneg, M, S):
    acc64 = acc_nonneg.astype(np.int64)
    prod = acc64 * np.int64(M)
    half = np.int64(1) << (S - 1) if S > 0 else np.int64(0)
    return (prod + half) >> S


def maxpool_int8(x_int8, pool_size, stride):
    H, W, C = x_int8.shape
    out_H = (H - pool_size) // stride + 1
    out_W = (W - pool_size) // stride + 1
    y = np.zeros((out_H, out_W, C), dtype=np.int8)
    for ch in range(C):
        for r in range(out_H):
            for c in range(out_W):
                r0, c0 = r * stride, c * stride
                window = x_int8[r0:r0 + pool_size, c0:c0 + pool_size, ch]
                y[r, c, ch] = np.max(window)
    return y


# ---- 1. Load trained C1 + C3 weights, and a real MNIST image ----
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

c1_w = c1_b = c3_w = c3_b = None
for name, (w, b) in arrays.items():
    if w.shape == (5, 5, 1, 6):
        c1_w, c1_b = w.astype(np.float32), b.astype(np.float32)
    elif w.shape == (5, 5, 6, 16):
        c3_w, c3_b = w.astype(np.float32), b.astype(np.float32)
assert c1_w is not None, "Could not find C1 weights (shape (5,5,1,6))"
assert c3_w is not None, "Could not find C3 weights (shape (5,5,6,16))"
print("Loaded C3 weights:", c3_w.shape, "bias:", c3_b.shape)

ds = torchvision.datasets.MNIST(root=os.path.join(tempfile.gettempdir(), "mnist_data"),
                                 train=False, download=True)
img_pil, true_label = ds[0]
image = (np.array(img_pil, dtype=np.float32) / 255.0).reshape(28, 28, 1)
print(f"Test image: MNIST test[0], true label = {true_label}")

# ---- 2. C1 conv (int8, fixed-point) -- duplicated from C1's own
#         generator purely to produce a real input for C3 to convolve
#         over; C1's own M/S are re-derived fresh here, not imported. ----
c1_w_int, c1_w_scale = quantize_int_real(c1_w, N_BITS)
x_int, x_scale = quantize_activation(image, N_BITS)
x_scale = np.float32(x_scale)
c1_w_scale = np.float32(c1_w_scale)
c1_combined_scale = np.float32(x_scale * c1_w_scale)

c1_float_out = conv2d_int(x_int, c1_w_int, c1_b, x_scale, c1_w_scale, padding="same")
c1_acc = np.round(c1_float_out.astype(np.float64) / float(c1_combined_scale)).astype(np.int64)
assert np.array_equal(c1_acc.astype(np.float32) * c1_combined_scale, c1_float_out), \
    "C1 accumulator recovery failed (needed only to produce C3's real input)"
c1_acc_relu = np.maximum(c1_acc, 0)
c1_relu_float = relu(c1_float_out).astype(np.float32)
_, c1_out_scale = quantize_activation(c1_relu_float, N_BITS)
c1_out_scale = np.float32(c1_out_scale)
c1_M, c1_S = quantize_multiplier(float(c1_combined_scale) / float(c1_out_scale))
c1_out_int8 = np.clip(requantize_fixed(c1_acc_relu, c1_M, c1_S), -128, 127).astype(np.int8)
print(f"C1 int8 output: shape {c1_out_int8.shape}, range [{c1_out_int8.min()}, {c1_out_int8.max()}]")

# ---- 3. S2 max pool (int8, no rescale) -- C3's real input, sharing
#         C1's exact out_scale untouched ----
s2_out_int8 = maxpool_int8(c1_out_int8, POOL_SIZE, STRIDE)
print(f"S2 int8 output: shape {s2_out_int8.shape}")

# ---- 4. C3's OWN quantization: weights, MAC, and the accumulator
#         recovery -- verified independently, not reused from C1. ----
c3_w_int, c3_w_scale = quantize_int_real(c3_w, N_BITS)
c3_w_scale = np.float32(c3_w_scale)
c3_combined_scale = np.float32(c1_out_scale * c3_w_scale)
print(f"C3: x_scale (=C1 out_scale) = {c1_out_scale}, w_scale = {c3_w_scale}, "
      f"combined_scale = {c3_combined_scale}")

c3_float_out = conv2d_int(s2_out_int8, c3_w_int, c3_b, c1_out_scale, c3_w_scale, padding="valid")
assert c3_float_out.shape == (10, 10, 16), f"Expected C3 output (10,10,16), got {c3_float_out.shape}"

c3_acc_recovered = np.round(c3_float_out.astype(np.float64) / float(c3_combined_scale)).astype(np.int64)
c3_reconstructed = c3_acc_recovered.astype(np.float32) * c3_combined_scale
assert np.array_equal(c3_reconstructed, c3_float_out), (
    "Could not exactly recover conv2d_int()'s internal C3 accumulator from "
    "its float output -- C3's fixed-point reference would be built on an "
    "unverified foundation."
)
assert c3_acc_recovered.min() >= np.iinfo(np.int32).min and c3_acc_recovered.max() <= np.iinfo(np.int32).max, \
    "Recovered C3 accumulator does not fit in int32"
print(f"C3 accumulator recovery verified exact. acc range: "
      f"[{c3_acc_recovered.min()}, {c3_acc_recovered.max()}]")

c3_bias_int = np.round(c3_b / c3_combined_scale).astype(np.int32)

# ---- 5. ReLU directly on C3's accumulator, then C3's OWN
#         quantize_multiplier()-derived fixed-point requantize. ----
c3_acc_relu = np.maximum(c3_acc_recovered, 0)

c3_relu_float = relu(c3_float_out).astype(np.float32)
_, c3_out_scale = quantize_activation(c3_relu_float, N_BITS)
c3_out_scale = np.float32(c3_out_scale)

c3_real_multiplier = float(c3_combined_scale) / float(c3_out_scale)
c3_M, c3_S = quantize_multiplier(c3_real_multiplier)
print(f"C3 out_scale = {c3_out_scale}")
print(f"C3 real_multiplier = combined_scale/out_scale = {c3_real_multiplier}")
print(f"C3 quantize_multiplier -> M = {c3_M}, S = {c3_S}  "
      f"(M/2**31 * 2**(31-S) = {c3_M / (1 << 31) * 2 ** (31 - c3_S)})")

c3_requantized = requantize_fixed(c3_acc_relu, c3_M, c3_S)
expected_output = np.clip(c3_requantized, -128, 127).astype(np.int8)

print(f"C3 expected_output range: [{expected_output.min()}, {expected_output.max()}], "
      f"nonzero fraction: {np.mean(expected_output != 0):.3f}")

# ---- Bit-exactness sanity check, BEFORE writing any C++: recompute the
#      SAME quantity a second, genuinely independent way -- true float
#      division of the already-dequantized/relu'd value by out_scale
#      (c3_relu_float IS acc_relu*combined_scale already; dividing it by
#      out_scale directly, with NO M/S involved at all, is independent of
#      quantize_multiplier()/requantize_fixed() entirely) -- and confirm
#      it lands on the same int8 bucket everywhere. This catches a
#      mismatched M/S derivation or an off-by-one in requantize_fixed()
#      itself before any C++ is written, the same role the
#      reciprocal-multiply-vs-division cross-check played for C1.
check_float_path = np.round(c3_relu_float.astype(np.float64) / float(c3_out_scale))
check_output = np.clip(check_float_path, -128, 127).astype(np.int8)
mismatch_vs_float_check = np.sum(check_output != expected_output)
max_abs_diff_vs_float_check = np.max(np.abs(check_float_path - c3_requantized.astype(np.float64)))
print(f"Cross-check vs. independent float division (no M/S involved): "
      f"{mismatch_vs_float_check}/{expected_output.size} int8 buckets differ "
      f"(expected: 0, or very few at exact half-integer rounding boundaries); "
      f"max abs diff in the pre-clip requantized value: {max_abs_diff_vs_float_check}")


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
    f.write("// Real int8-quantized Models/lenet5_relu.keras C3 weights + a real\n")
    f.write("// int8-quantized S2-pooled activation map (via the C1->S2 int8 pipeline),\n")
    f.write("// plus a GENUINE fixed-point (int64 multiply + rounding right-shift)\n")
    f.write("// reference output -- C3's OWN M/S, independently derived and verified,\n")
    f.write("// not reused from C1. For conv_c3_int8_fixedpoint_tb.cpp.\n")
    f.write("#ifndef CONV_C3_INT8_FIXEDPOINT_TEST_DATA_H\n#define CONV_C3_INT8_FIXEDPOINT_TEST_DATA_H\n\n")

    f.write(f"#define TRUE_LABEL {int(true_label)}\n")
    f.write(f"#define REQUANT_MULT {c3_M}\n")
    f.write(f"#define REQUANT_SHIFT {c3_S}\n\n")

    emit_int_array(f, "input_data", [14, 14, 6], s2_out_int8, "signed char")
    emit_int_array(f, "weights_data", [5, 5, 6, 16], c3_w_int, "signed char")
    emit_int_array(f, "bias_int_data", [16], c3_bias_int, "int")
    emit_int_array(f, "expected_output", [10, 10, 16], expected_output, "signed char")

    f.write("#endif\n")

print(f"\nWrote {OUT_PATH}")
