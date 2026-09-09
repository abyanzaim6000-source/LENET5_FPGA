"""
Generates dense_f6_int8_fixedpoint_test_data.h: real int8-quantized F6
weights + a real int8-quantized C5 activation vector (via the full
C1->S2->C3->S4->flatten->C5 int8 pipeline) as dense_f6_int8_fixedpoint's
input, plus the reference output computed with GENUINE fixed-point
integer arithmetic -- same technique as dense_c5's generator, but this
script derives F6's OWN Q31 multiplier/shift (M6, S6) and verifies F6's
OWN accumulator recovery independently -- nothing here is reused from
any earlier layer's already-computed M/S; every prior layer's pipeline
is only run first because F6's real input (C5's output) doesn't exist
without it.

Pipeline (all real data, no synthetic values), duplicated fresh here (not
imported) same convention as every prior generator script in this project:
  1. C1 conv (int8, fixed-point) -> 28x28x6.
  2. S2 max pool (int8, no rescale) -> 14x14x6.
  3. C3 conv (int8, fixed-point, "valid") -> 10x10x16.
  4. S4 max pool (int8, no rescale) -> 5x5x16.
  5. Flatten (NumPy default C-order) -> 400 int8 values.
  6. C5 dense (int8x int8->int32 MAC via dense_int(), fixed-point
     requantize) -> 120 int8 values, F6's real input.
  7. F6 dense (int8x int8->int32 MAC via dense_int()) -- the raw int32
     accumulator is recovered from dense_int()'s float output via its
     exact inverse and the round-trip is VERIFIED exact -- F6's OWN
     verification, independent of every earlier layer's.
  8. ReLU directly on F6's accumulator, then F6's own
     quantize_multiplier()-derived (M6, S6) fixed-point requantize to
     int8 -- this IS the bit-exact-in-Python reference
     dense_f6_int8_fixedpoint.cpp must match, checked here before any C++
     is written.

Run from repo root: python hls/dense_f6/tb/generate_test_data_int8_fixedpoint.py
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
from integer_layers import conv2d_int, dense_int, quantize_activation
from quantization import quantize_int_real
from manual_layers import relu

N_BITS = 8
POOL_SIZE = 2
STRIDE = 2

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
MODEL_PATH = os.path.join(REPO_ROOT, "models", "lenet5_relu.keras")
OUT_PATH = os.path.join(os.path.dirname(__file__), "dense_f6_int8_fixedpoint_test_data.h")


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


def conv_int8_fixedpoint(x_int, b, w_int, x_scale, w_scale, padding):
    """One int8 conv layer end-to-end: MAC (via the proven conv2d_int()),
    verified accumulator recovery, ReLU on the accumulator, and
    fixed-point requantize to int8. Returns (output_int8, out_scale)."""
    combined_scale = np.float32(x_scale * w_scale)
    float_out = conv2d_int(x_int, w_int, b, x_scale, w_scale, padding=padding)

    acc_recovered = np.round(float_out.astype(np.float64) / float(combined_scale)).astype(np.int64)
    assert np.array_equal(acc_recovered.astype(np.float32) * combined_scale, float_out), \
        "Could not exactly recover conv2d_int()'s internal accumulator"
    acc_relu = np.maximum(acc_recovered, 0)

    relu_out_float = relu(float_out).astype(np.float32)
    _, out_scale = quantize_activation(relu_out_float, N_BITS)
    out_scale = np.float32(out_scale)
    real_multiplier = float(combined_scale) / float(out_scale)
    M, S = quantize_multiplier(real_multiplier)

    output_int8 = np.clip(requantize_fixed(acc_relu, M, S), -128, 127).astype(np.int8)
    return output_int8, out_scale


def dense_int8_fixedpoint(x_int, b, w_int, x_scale, w_scale):
    """One int8 dense (ReLU) layer end-to-end: MAC (via the proven
    dense_int()), verified accumulator recovery, ReLU on the accumulator,
    and fixed-point requantize to int8. Returns (output_int8, out_scale)."""
    combined_scale = np.float32(x_scale * w_scale)
    float_out = dense_int(x_int, w_int, b, x_scale, w_scale, activation=None)

    acc_recovered = np.round(float_out.astype(np.float64) / float(combined_scale)).astype(np.int64)
    assert np.array_equal(acc_recovered.astype(np.float32) * combined_scale, float_out), \
        "Could not exactly recover dense_int()'s internal accumulator"
    acc_relu = np.maximum(acc_recovered, 0)

    relu_out_float = relu(float_out).astype(np.float32)
    _, out_scale = quantize_activation(relu_out_float, N_BITS)
    out_scale = np.float32(out_scale)
    real_multiplier = float(combined_scale) / float(out_scale)
    M, S = quantize_multiplier(real_multiplier)

    output_int8 = np.clip(requantize_fixed(acc_relu, M, S), -128, 127).astype(np.int8)
    return output_int8, out_scale, acc_recovered, M, S


# ---- 1. Load trained C1 + C3 + C5 + F6 weights, and a real MNIST image ----
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

c1_w = c1_b = c3_w = c3_b = c5_w = c5_b = f6_w = f6_b = None
for name, (w, b) in arrays.items():
    if w.shape == (5, 5, 1, 6):
        c1_w, c1_b = w.astype(np.float32), b.astype(np.float32)
    elif w.shape == (5, 5, 6, 16):
        c3_w, c3_b = w.astype(np.float32), b.astype(np.float32)
    elif w.shape == (400, 120):
        c5_w, c5_b = w.astype(np.float32), b.astype(np.float32)
    elif w.shape == (120, 84):
        f6_w, f6_b = w.astype(np.float32), b.astype(np.float32)
assert c1_w is not None, "Could not find C1 weights (shape (5,5,1,6))"
assert c3_w is not None, "Could not find C3 weights (shape (5,5,6,16))"
assert c5_w is not None, "Could not find C5 weights (shape (400,120))"
assert f6_w is not None, "Could not find F6 weights (shape (120,84))"
print("Loaded F6 weights:", f6_w.shape, "bias:", f6_b.shape)

ds = torchvision.datasets.MNIST(root=os.path.join(tempfile.gettempdir(), "mnist_data"),
                                 train=False, download=True)
img_pil, true_label = ds[0]
image = (np.array(img_pil, dtype=np.float32) / 255.0).reshape(28, 28, 1)
print(f"Test image: MNIST test[0], true label = {true_label}")

# ---- 2. C1 conv (int8, fixed-point) ----
c1_w_int, c1_w_scale = quantize_int_real(c1_w, N_BITS)
x_int, x_scale = quantize_activation(image, N_BITS)
x_scale = np.float32(x_scale)
c1_w_scale = np.float32(c1_w_scale)
c1_out_int8, c1_out_scale = conv_int8_fixedpoint(
    x_int, c1_b, c1_w_int, x_scale, c1_w_scale, padding="same")
print(f"C1 int8 output: shape {c1_out_int8.shape}, range [{c1_out_int8.min()}, {c1_out_int8.max()}]")

# ---- 3. S2 max pool ----
s2_out_int8 = maxpool_int8(c1_out_int8, POOL_SIZE, STRIDE)
print(f"S2 int8 output: shape {s2_out_int8.shape}")

# ---- 4. C3 conv (int8, fixed-point, "valid") ----
c3_w_int, c3_w_scale = quantize_int_real(c3_w, N_BITS)
c3_w_scale = np.float32(c3_w_scale)
c3_out_int8, c3_out_scale = conv_int8_fixedpoint(
    s2_out_int8, c3_b, c3_w_int, c1_out_scale, c3_w_scale, padding="valid")
print(f"C3 int8 output: shape {c3_out_int8.shape}, range [{c3_out_int8.min()}, {c3_out_int8.max()}]")
assert c3_out_int8.shape == (10, 10, 16), f"Expected C3 output (10,10,16), got {c3_out_int8.shape}"

# ---- 5. S4 max pool, then flatten ----
s4_out_int8 = maxpool_int8(c3_out_int8, POOL_SIZE, STRIDE)
c5_input_int8 = s4_out_int8.flatten()
assert c5_input_int8.shape == (400,), f"Expected flattened C5 input (400,), got {c5_input_int8.shape}"
print(f"S4 int8 output: shape {s4_out_int8.shape}, flattened: {c5_input_int8.shape}")

# ---- 6. C5 dense (int8, fixed-point) -> F6's real input ----
c5_w_int, c5_w_scale = quantize_int_real(c5_w, N_BITS)
c5_w_scale = np.float32(c5_w_scale)
c5_out_int8, c5_out_scale, _, _, _ = dense_int8_fixedpoint(
    c5_input_int8, c5_b, c5_w_int, c3_out_scale, c5_w_scale)
assert c5_out_int8.shape == (120,), f"Expected C5 output (120,), got {c5_out_int8.shape}"
print(f"C5 int8 output: shape {c5_out_int8.shape}, range [{c5_out_int8.min()}, {c5_out_int8.max()}], "
      f"nonzero fraction: {np.mean(c5_out_int8 != 0):.3f}")

# ---- 7. F6's OWN quantization: weights, MAC (via dense_int()), and the
#         accumulator recovery -- verified independently, not reused from
#         any earlier layer's. ----
f6_w_int, f6_w_scale = quantize_int_real(f6_w, N_BITS)
f6_w_scale = np.float32(f6_w_scale)
f6_combined_scale = np.float32(c5_out_scale * f6_w_scale)
print(f"F6: x_scale (=C5 out_scale) = {c5_out_scale}, w_scale = {f6_w_scale}, "
      f"combined_scale = {f6_combined_scale}")

f6_float_out = dense_int(c5_out_int8, f6_w_int, f6_b, c5_out_scale, f6_w_scale, activation=None)
assert f6_float_out.shape == (84,), f"Expected F6 output (84,), got {f6_float_out.shape}"

f6_acc_recovered = np.round(f6_float_out.astype(np.float64) / float(f6_combined_scale)).astype(np.int64)
f6_reconstructed = f6_acc_recovered.astype(np.float32) * f6_combined_scale
assert np.array_equal(f6_reconstructed, f6_float_out), (
    "Could not exactly recover dense_int()'s internal F6 accumulator from "
    "its float output -- F6's fixed-point reference would be built on an "
    "unverified foundation."
)
assert f6_acc_recovered.min() >= np.iinfo(np.int32).min and f6_acc_recovered.max() <= np.iinfo(np.int32).max, \
    "Recovered F6 accumulator does not fit in int32"
print(f"F6 accumulator recovery verified exact. acc range: "
      f"[{f6_acc_recovered.min()}, {f6_acc_recovered.max()}]")

f6_bias_int = np.round(f6_b / f6_combined_scale).astype(np.int32)

# ---- 8. ReLU directly on F6's accumulator, then F6's OWN
#         quantize_multiplier()-derived fixed-point requantize. ----
f6_acc_relu = np.maximum(f6_acc_recovered, 0)

f6_relu_float = relu(f6_float_out).astype(np.float32)
_, f6_out_scale = quantize_activation(f6_relu_float, N_BITS)
f6_out_scale = np.float32(f6_out_scale)

f6_real_multiplier = float(f6_combined_scale) / float(f6_out_scale)
f6_M, f6_S = quantize_multiplier(f6_real_multiplier)
print(f"F6 out_scale = {f6_out_scale}")
print(f"F6 real_multiplier = combined_scale/out_scale = {f6_real_multiplier}")
print(f"F6 quantize_multiplier -> M = {f6_M}, S = {f6_S}  "
      f"(M/2**31 * 2**(31-S) = {f6_M / (1 << 31) * 2 ** (31 - f6_S)})")

f6_requantized = requantize_fixed(f6_acc_relu, f6_M, f6_S)
expected_output = np.clip(f6_requantized, -128, 127).astype(np.int8)

print(f"F6 expected_output range: [{expected_output.min()}, {expected_output.max()}], "
      f"nonzero fraction: {np.mean(expected_output != 0):.3f}")

# ---- Bit-exactness sanity check, BEFORE writing any C++ ----
check_float_path = np.round(f6_relu_float.astype(np.float64) / float(f6_out_scale))
check_output = np.clip(check_float_path, -128, 127).astype(np.int8)
mismatch_vs_float_check = np.sum(check_output != expected_output)
max_abs_diff_vs_float_check = np.max(np.abs(check_float_path - f6_requantized.astype(np.float64)))
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
    f.write("// Real int8-quantized Models/lenet5_relu.keras F6 weights + a real\n")
    f.write("// int8-quantized C5 activation vector (via the full\n")
    f.write("// C1->S2->C3->S4->flatten->C5 int8 pipeline), plus a GENUINE\n")
    f.write("// fixed-point (int64 multiply + rounding right-shift) reference output --\n")
    f.write("// F6's OWN M/S, independently derived and verified, not reused from any\n")
    f.write("// earlier layer's. For dense_f6_int8_fixedpoint_tb.cpp.\n")
    f.write("#ifndef DENSE_F6_INT8_FIXEDPOINT_TEST_DATA_H\n#define DENSE_F6_INT8_FIXEDPOINT_TEST_DATA_H\n\n")

    f.write(f"#define TRUE_LABEL {int(true_label)}\n")
    f.write(f"#define REQUANT_MULT {f6_M}\n")
    f.write(f"#define REQUANT_SHIFT {f6_S}\n\n")

    emit_int_array(f, "input_data", [120], c5_out_int8, "signed char")
    emit_int_array(f, "weights_data", [120, 84], f6_w_int, "signed char")
    emit_int_array(f, "bias_int_data", [84], f6_bias_int, "int")
    emit_int_array(f, "expected_output", [84], expected_output, "signed char")

    f.write("#endif\n")

print(f"\nWrote {OUT_PATH}")
