"""
Generates pool_s2_int8_test_data.h: a real int8-quantized C1 activation
map (computed with the same proven int8 pipeline as
hls/conv_c1/tb/generate_test_data_int8_fixedpoint.py -- real trained
weights, real MNIST image, int32 MAC, ReLU on the accumulator, then a
genuine fixed-point requantize) as pool_s2_int8's input, plus the max-pool
reference output, for pool_s2_int8_tb.cpp.

MaxPool needs NO rescaling at all, unlike conv_c1: every value in a 2x2
pooling window shares the exact same scale factor (they're all the same
C1-output activation tensor), and since that scale factor is always
positive, comparing the raw int8 values directly gives EXACTLY the same
ordering -- and so the same argmax/max -- as comparing the real
(dequantized) values would. So the reference here is simply
np.max() over int8 values directly; no multiplier, no shift, no
quantize_multiplier() needed, unlike every one of conv_c1's INT8 stages.

Reuses conv_c1's exact int8 pipeline (quantize_int_real, quantize_activation,
conv2d_int, the accumulator-recovery-and-verify step, and the same
fixed-point quantize_multiplier()/requantize_fixed() helpers) purely to
produce REALISTIC quantized C1-output test data for pool_s2 -- this
script does not re-derive or re-verify conv_c1's own correctness (that's
already proven in hls/conv_c1/tb/generate_test_data_int8_fixedpoint.py),
it just needs a real int8 activation map to pool over.

Run from repo root: python hls/pool/tb/generate_test_data_int8.py
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
OUT_PATH = os.path.join(os.path.dirname(__file__), "pool_s2_int8_test_data.h")


def quantize_multiplier(real_multiplier):
    """Same TFLite/gemmlowp-style decomposition as conv_c1's generator --
    see hls/conv_c1/tb/generate_test_data_int8_fixedpoint.py for the full
    explanation. Duplicated here (not imported) so this script stays
    self-contained, matching the project's per-stage generator convention."""
    assert real_multiplier > 0
    significand, exponent = math.frexp(real_multiplier)
    M = round(significand * (1 << 31))
    if M == (1 << 31):
        M //= 2
        exponent += 1
    S = 31 - exponent
    assert 0 < M < (1 << 31)
    assert S >= 0
    return M, S


def requantize_fixed(acc_nonneg, M, S):
    acc64 = acc_nonneg.astype(np.int64)
    prod = acc64 * np.int64(M)
    half = np.int64(1) << (S - 1) if S > 0 else np.int64(0)
    return (prod + half) >> S


def maxpool_int8(x_int8, pool_size, stride):
    """MaxPool directly on int8 values -- no scale factor needed at all
    (see module docstring): max of raw int8 values == max of their
    dequantized values, since every value in the window shares the same
    positive scale."""
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


# ---- 1. Load trained C1 weights + a real MNIST image (same as conv_c1's generator) ----
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

ds = torchvision.datasets.MNIST(root=os.path.join(tempfile.gettempdir(), "mnist_data"),
                                 train=False, download=True)
img_pil, true_label = ds[0]
image = (np.array(img_pil, dtype=np.float32) / 255.0).reshape(28, 28, 1)
print(f"Test image: MNIST test[0], true label = {true_label}")

# ---- 2. Run C1's proven int8 pipeline to get a REAL int8 activation map
#         (28x28x6) to pool over -- same steps as
#         generate_test_data_int8_fixedpoint.py, condensed. ----
w_int, w_scale = quantize_int_real(c1_w, N_BITS)
x_int, x_scale = quantize_activation(image, N_BITS)
w_scale = np.float32(w_scale)
x_scale = np.float32(x_scale)
combined_scale = np.float32(x_scale * w_scale)

float_out = conv2d_int(x_int, w_int, c1_b, x_scale, w_scale, padding="same")
acc_recovered = np.round(float_out.astype(np.float64) / float(combined_scale)).astype(np.int64)
assert np.array_equal(acc_recovered.astype(np.float32) * combined_scale, float_out), \
    "Could not exactly recover conv2d_int()'s internal accumulator"
acc_relu = np.maximum(acc_recovered, 0)

relu_out_float = relu(float_out).astype(np.float32)
_, out_scale = quantize_activation(relu_out_float, N_BITS)
out_scale = np.float32(out_scale)
real_multiplier = float(combined_scale) / float(out_scale)
M, S = quantize_multiplier(real_multiplier)

c1_output_int8 = np.clip(requantize_fixed(acc_relu, M, S), -128, 127).astype(np.int8)
print(f"C1 int8 output range: [{c1_output_int8.min()}, {c1_output_int8.max()}], "
      f"nonzero fraction: {np.mean(c1_output_int8 != 0):.3f}")

# ---- 3. Pool_s2 reference: plain max over int8 values, no scale involved ----
expected_output = maxpool_int8(c1_output_int8, POOL_SIZE, STRIDE)
print(f"pool_s2 int8 output shape: {expected_output.shape}, "
      f"range: [{expected_output.min()}, {expected_output.max()}]")


def emit_int_array(f, name, dims, arr, ctype):
    flat = arr.flatten()
    dim_str = "".join(f"[{d}]" for d in dims)
    f.write(f"static const {ctype} {name}{dim_str} = {{\n")
    for i in range(0, len(flat), 16):
        chunk = flat[i:i + 16]
        f.write("    " + ", ".join(str(int(v)) for v in chunk) + ",\n")
    f.write("};\n\n")


with open(OUT_PATH, "w") as f:
    f.write("// AUTO-GENERATED by generate_test_data_int8.py -- do not hand-edit.\n")
    f.write("// A real int8-quantized C1 activation map (via conv_c1's own proven\n")
    f.write("// int8 pipeline -- real Models/lenet5_relu.keras weights + real MNIST\n")
    f.write("// test[0]) as pool_s2_int8's input, plus the max-pool reference output\n")
    f.write("// (plain np.max over int8 values -- no scale factor needed, see the\n")
    f.write("// generator script's docstring), for pool_s2_int8_tb.cpp.\n")
    f.write("#ifndef POOL_S2_INT8_TEST_DATA_H\n#define POOL_S2_INT8_TEST_DATA_H\n\n")

    f.write(f"#define TRUE_LABEL {int(true_label)}\n\n")

    emit_int_array(f, "input_data", [28, 28, 6], c1_output_int8, "signed char")
    emit_int_array(f, "expected_output", [14, 14, 6], expected_output, "signed char")

    f.write("#endif\n")

print(f"\nWrote {OUT_PATH}")
