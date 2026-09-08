"""
Generates pool_s4_int8_test_data.h: a real int8-quantized C3 activation
map as pool_s4_int8's input, plus the max-pool reference output.

Unlike pool_s2 (which only needed to chain through C1), S4 sits after
BOTH C1 and C3, so producing a real (non-synthetic) 10x10x16 int8 input
means running the full C1 -> S2 -> C3 int8 pipeline in Python first:
  1. C1 conv (int8x int8->int32 MAC, ReLU on the accumulator, fixed-point
     requantize) -- same steps as conv_c1's own generator scripts.
  2. S2 max pool -- int8 in, int8 out, no rescale (max commutes with any
     positive affine rescale, so pooling never changes the scale factor
     at all -- s2's output shares C1's exact out_scale).
  3. C3 conv (padding="valid", 5x5x6->16 channels) -- same MAC ->
     accumulator-recovery-and-verify -> ReLU -> fixed-point-requantize
     pattern as C1, just with C1's own out_scale as C3's INPUT scale
     (since S2 didn't change it) and C3's own weights/bias.
  4. That C3 int8 output (10x10x16) is pool_s4_int8's real input.

No HLS C3 IP exists yet (only C1 has been converted to INT8 hardware so
far) -- this script only needs C3's correct INT8 *arithmetic* in Python
to produce realistic test data for S4, not a synthesizable C3 kernel.

Like pool_s2's generator, S4's own reference output needs NO rescaling
at all: max() over int8 values directly, since every value in a 2x2
window shares the exact same scale.

Run from repo root: python hls/pool_s4/tb/generate_test_data_int8.py
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
OUT_PATH = os.path.join(os.path.dirname(__file__), "pool_s4_int8_test_data.h")


def quantize_multiplier(real_multiplier):
    """Same TFLite/gemmlowp-style decomposition as conv_c1's generators --
    see hls/conv_c1/tb/generate_test_data_int8_fixedpoint.py for the full
    explanation. Duplicated here so this script stays self-contained,
    matching the project's per-stage generator convention."""
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
    """MaxPool directly on int8 values -- no scale factor needed (max
    commutes with any positive affine rescale)."""
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
    fixed-point requantize to int8. Returns (output_int8, out_scale) --
    out_scale is needed by the NEXT conv layer as its input scale (pooling
    layers in between don't change it)."""
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

# ---- 3. S2 max pool (int8, no rescale -- output shares c1_out_scale) ----
s2_out_int8 = maxpool_int8(c1_out_int8, POOL_SIZE, STRIDE)
print(f"S2 int8 output: shape {s2_out_int8.shape}")

# ---- 4. C3 conv (int8, fixed-point, "valid" padding) -- s2_out_int8 is
#         the input, with c1_out_scale as ITS scale (pooling didn't
#         change it), c3 weights quantized fresh ----
c3_w_int, c3_w_scale = quantize_int_real(c3_w, N_BITS)
c3_w_scale = np.float32(c3_w_scale)
c3_out_int8, c3_out_scale = conv_int8_fixedpoint(
    s2_out_int8, c3_b, c3_w_int, c1_out_scale, c3_w_scale, padding="valid")
print(f"C3 int8 output: shape {c3_out_int8.shape}, range [{c3_out_int8.min()}, {c3_out_int8.max()}], "
      f"nonzero fraction: {np.mean(c3_out_int8 != 0):.3f}")
assert c3_out_int8.shape == (10, 10, 16), f"Expected C3 output (10,10,16), got {c3_out_int8.shape}"

# ---- 5. S4 max pool reference: plain max over int8 values, no scale involved ----
expected_output = maxpool_int8(c3_out_int8, POOL_SIZE, STRIDE)
print(f"S4 int8 output: shape {expected_output.shape}, range [{expected_output.min()}, {expected_output.max()}]")


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
    f.write("// A real int8-quantized C3 activation map -- produced by chaining the\n")
    f.write("// real int8 C1 -> S2 -> C3 pipeline in Python (real Models/lenet5_relu.keras\n")
    f.write("// weights, real MNIST test[0] image; C3 has no HLS IP yet, so this is\n")
    f.write("// pure-Python int8 arithmetic, same MAC/ReLU/fixed-point-requantize\n")
    f.write("// pattern as conv_c1's own generators) -- as pool_s4_int8's input, plus\n")
    f.write("// the max-pool reference output (plain np.max over int8 values -- no\n")
    f.write("// scale factor needed), for pool_s4_int8_tb.cpp.\n")
    f.write("#ifndef POOL_S4_INT8_TEST_DATA_H\n#define POOL_S4_INT8_TEST_DATA_H\n\n")

    f.write(f"#define TRUE_LABEL {int(true_label)}\n\n")

    emit_int_array(f, "input_data", [10, 10, 16], c3_out_int8, "signed char")
    emit_int_array(f, "expected_output", [5, 5, 16], expected_output, "signed char")

    f.write("#endif\n")

print(f"\nWrote {OUT_PATH}")
