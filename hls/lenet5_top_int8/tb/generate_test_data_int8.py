"""
Generates lenet5_top_int8_test_data.h: a real MNIST test[0] image + the
real trained Models/lenet5_relu.keras weights, quantized to INT8 at every
layer via the SAME fixed-point (TFLite/gemmlowp quantize_multiplier)
pipeline every individual INT8 layer's own generator script already used
(duplicated here, not imported, matching the project's per-stage
convention) -- for lenet5_top_int8_tb.cpp to check both:

  1. every intermediate stage's output (C1, S2, C3, S4, C5, F6), BIT-EXACT,
     against this same Python integer arithmetic (each conv/dense stage's
     accumulator is verified exactly recoverable from its float output,
     same assertion every earlier per-layer generator script already
     used);
  2. the final softmax probabilities, to a tight float tolerance (same
     1e-5 standard as dense_output_int8_tb.cpp's own check) -- softmax is
     the one stage that is deliberately NOT fixed-point (see
     dense_output_int8.h's own docstring for why: it's the network's
     final activation, read as a probability distribution, not fed
     forward to another quantized layer).

Pipeline (all real data, no synthetic values):
  1. C1 conv (int8, fixed-point, "same") -> 28x28x6.
  2. S2 max pool (int8, no rescale) -> 14x14x6.
  3. C3 conv (int8, fixed-point, "valid", LUT-decode architecture) -> 10x10x16.
  4. S4 max pool (int8, no rescale) -> 5x5x16.
  5. Flatten (NumPy default C-order) -> 400 int8 values.
  6. C5 dense (int8, fixed-point) -> 120 int8 values.
  7. F6 dense (int8, fixed-point) -> 84 int8 values.
  8. Output dense (int8x int8->int32 MAC via dense_int()) -- the raw
     int32 accumulator is recovered from dense_int()'s float output via
     its exact inverse, and the round-trip is VERIFIED exact.
  9. Softmax: dequantize the verified accumulator with
     combined_scale = x_scale*w_scale, then max-subtract/exp/sum/divide
     in float.

Every conv/dense layer's requant_mult (M) / requant_shift (S) is derived
INDEPENDENTLY from that layer's own (x_scale, w_scale, out_scale) triple
-- never reused from a different layer, exactly as this project's
per-layer generator scripts already established piece by piece (C1, C3,
C5, F6 each computed their own M/S in their own generate_test_data
scripts; this script just chains all of that together instead of
starting from Python-only stand-in data at each stage).

Run from repo root: python hls/lenet5_top_int8/tb/generate_test_data_int8.py
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
MODEL_PATH = os.path.join(REPO_ROOT, "Models", "lenet5_relu.keras")
OUT_PATH = os.path.join(os.path.dirname(__file__), "lenet5_top_int8_test_data.h")


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
    """One int8 conv layer end-to-end (ReLU activation): MAC (via the
    proven conv2d_int()), verified accumulator recovery, ReLU on the
    accumulator, and fixed-point requantize to int8. Returns
    (output_int8, out_scale, M, S, bias_int) -- bias_int and (M, S) are
    exactly what the HLS kernel's own runtime interface needs."""
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
    bias_int = np.round(b / combined_scale).astype(np.int32)
    return output_int8, out_scale, M, S, bias_int


def dense_int8_fixedpoint(x_int, b, w_int, x_scale, w_scale):
    """One int8 dense (ReLU) layer end-to-end -- same structure as
    conv_int8_fixedpoint() above, via the proven dense_int()."""
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
    bias_int = np.round(b / combined_scale).astype(np.int32)
    return output_int8, out_scale, M, S, bias_int


# ---- 1. Load trained C1 + C3 + C5 + F6 + Output weights, and a real
#         MNIST image ----
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

c1_w = c1_b = c3_w = c3_b = c5_w = c5_b = f6_w = f6_b = out_w = out_b = None
for name, (w, b) in arrays.items():
    if w.shape == (5, 5, 1, 6):
        c1_w, c1_b = w.astype(np.float32), b.astype(np.float32)
    elif w.shape == (5, 5, 6, 16):
        c3_w, c3_b = w.astype(np.float32), b.astype(np.float32)
    elif w.shape == (400, 120):
        c5_w, c5_b = w.astype(np.float32), b.astype(np.float32)
    elif w.shape == (120, 84):
        f6_w, f6_b = w.astype(np.float32), b.astype(np.float32)
    elif w.shape == (84, 10):
        out_w, out_b = w.astype(np.float32), b.astype(np.float32)
assert c1_w is not None, "Could not find C1 weights (shape (5,5,1,6))"
assert c3_w is not None, "Could not find C3 weights (shape (5,5,6,16))"
assert c5_w is not None, "Could not find C5 weights (shape (400,120))"
assert f6_w is not None, "Could not find F6 weights (shape (120,84))"
assert out_w is not None, "Could not find Output weights (shape (84,10))"
print("Loaded weight shapes:")
print("  C1:", c1_w.shape, c1_b.shape)
print("  C3:", c3_w.shape, c3_b.shape)
print("  C5:", c5_w.shape, c5_b.shape)
print("  F6:", f6_w.shape, f6_b.shape)
print("  Output:", out_w.shape, out_b.shape)

ds = torchvision.datasets.MNIST(root=os.path.join(tempfile.gettempdir(), "mnist_data"),
                                 train=False, download=True)
img_pil, true_label = ds[0]
image = (np.array(img_pil, dtype=np.float32) / 255.0).reshape(28, 28, 1)
print(f"Test image: MNIST test[0], true label = {true_label}")

# ---- 2. C1 conv (int8, fixed-point, "same") ----
c1_w_int, c1_w_scale = quantize_int_real(c1_w, N_BITS)
c1_w_scale = np.float32(c1_w_scale)
x_int, x_scale = quantize_activation(image, N_BITS)
x_scale = np.float32(x_scale)
c1_out_int8, c1_out_scale, c1_M, c1_S, c1_bias_int = conv_int8_fixedpoint(
    x_int, c1_b, c1_w_int, x_scale, c1_w_scale, padding="same")
print(f"C1: M={c1_M}, S={c1_S}, out_scale={c1_out_scale}, "
      f"range [{c1_out_int8.min()}, {c1_out_int8.max()}]")

# ---- 3. S2 max pool ----
s2_out_int8 = maxpool_int8(c1_out_int8, POOL_SIZE, STRIDE)

# ---- 4. C3 conv (int8, fixed-point, "valid") ----
c3_w_int, c3_w_scale = quantize_int_real(c3_w, N_BITS)
c3_w_scale = np.float32(c3_w_scale)
c3_out_int8, c3_out_scale, c3_M, c3_S, c3_bias_int = conv_int8_fixedpoint(
    s2_out_int8, c3_b, c3_w_int, c1_out_scale, c3_w_scale, padding="valid")
assert c3_out_int8.shape == (10, 10, 16), f"Expected C3 output (10,10,16), got {c3_out_int8.shape}"
print(f"C3: M={c3_M}, S={c3_S}, out_scale={c3_out_scale}, "
      f"range [{c3_out_int8.min()}, {c3_out_int8.max()}]")

# ---- 5. S4 max pool, then flatten ----
s4_out_int8 = maxpool_int8(c3_out_int8, POOL_SIZE, STRIDE)
c5_input_int8 = s4_out_int8.flatten()
assert c5_input_int8.shape == (400,), f"Expected flattened C5 input (400,), got {c5_input_int8.shape}"

# ---- 6. C5 dense (int8, fixed-point) ----
c5_w_int, c5_w_scale = quantize_int_real(c5_w, N_BITS)
c5_w_scale = np.float32(c5_w_scale)
c5_out_int8, c5_out_scale, c5_M, c5_S, c5_bias_int = dense_int8_fixedpoint(
    c5_input_int8, c5_b, c5_w_int, c3_out_scale, c5_w_scale)
assert c5_out_int8.shape == (120,), f"Expected C5 output (120,), got {c5_out_int8.shape}"
print(f"C5: M={c5_M}, S={c5_S}, out_scale={c5_out_scale}, "
      f"range [{c5_out_int8.min()}, {c5_out_int8.max()}]")

# ---- 7. F6 dense (int8, fixed-point) ----
f6_w_int, f6_w_scale = quantize_int_real(f6_w, N_BITS)
f6_w_scale = np.float32(f6_w_scale)
f6_out_int8, f6_out_scale, f6_M, f6_S, f6_bias_int = dense_int8_fixedpoint(
    c5_out_int8, f6_b, f6_w_int, c5_out_scale, f6_w_scale)
assert f6_out_int8.shape == (84,), f"Expected F6 output (84,), got {f6_out_int8.shape}"
print(f"F6: M={f6_M}, S={f6_S}, out_scale={f6_out_scale}, "
      f"range [{f6_out_int8.min()}, {f6_out_int8.max()}]")

# ---- 8. Output's OWN quantization: weights and the MAC accumulator --
#         verified independently, not reused from any earlier layer's. ----
out_w_int, out_w_scale = quantize_int_real(out_w, N_BITS)
out_w_scale = np.float32(out_w_scale)
out_combined_scale = np.float32(f6_out_scale * out_w_scale)
print(f"Output: x_scale (=F6 out_scale) = {f6_out_scale}, w_scale = {out_w_scale}, "
      f"combined_scale = {out_combined_scale}")

out_float_logits = dense_int(f6_out_int8, out_w_int, out_b, f6_out_scale, out_w_scale, activation=None)
assert out_float_logits.shape == (10,), f"Expected Output logits (10,), got {out_float_logits.shape}"

out_acc_recovered = np.round(out_float_logits.astype(np.float64) / float(out_combined_scale)).astype(np.int64)
out_reconstructed = out_acc_recovered.astype(np.float32) * out_combined_scale
assert np.array_equal(out_reconstructed, out_float_logits), (
    "Could not exactly recover dense_int()'s internal Output accumulator from "
    "its float output -- Output's reference would be built on an unverified "
    "foundation."
)
assert out_acc_recovered.min() >= np.iinfo(np.int32).min and out_acc_recovered.max() <= np.iinfo(np.int32).max, \
    "Recovered Output accumulator does not fit in int32"
print(f"Output accumulator recovery verified exact. acc (logits) range: "
      f"[{out_acc_recovered.min()}, {out_acc_recovered.max()}]")

out_bias_int = np.round(out_b / out_combined_scale).astype(np.int32)

# ---- 9. Softmax: dequantize the verified int32 accumulator with
#         combined_scale, then max-subtract/exp/sum/divide in float -- no
#         requantization to int8 (this is the network's final output). ----
dequantized_logits = out_acc_recovered.astype(np.float32) * out_combined_scale
assert np.array_equal(dequantized_logits, out_float_logits), \
    "Dequantized logits (from the verified accumulator) must equal dense_int()'s own float output exactly"

shifted = dequantized_logits - np.max(dequantized_logits)
exp_vals = np.exp(shifted)
expected_output = (exp_vals / np.sum(exp_vals)).astype(np.float32)

predicted_label = int(np.argmax(expected_output))
print(f"Output softmax reference: {expected_output}")
print(f"Sum: {np.sum(expected_output):.8f}, predicted label = {predicted_label} "
      f"(true label = {true_label})")


def emit_int_array(f, name, dims, arr, ctype):
    flat = arr.flatten()
    dim_str = "".join(f"[{d}]" for d in dims)
    f.write(f"static const {ctype} {name}{dim_str} = {{\n")
    for i in range(0, len(flat), 16):
        chunk = flat[i:i + 16]
        f.write("    " + ", ".join(str(int(v)) for v in chunk) + ",\n")
    f.write("};\n\n")


def emit_float_array(f, name, dims, arr):
    flat = arr.astype(np.float32).flatten()
    dim_str = "".join(f"[{d}]" for d in dims)
    f.write(f"static const float {name}{dim_str} = {{\n")
    for i in range(0, len(flat), 8):
        chunk = flat[i:i + 8]
        f.write("    " + ", ".join(f"{v:.9e}f" for v in chunk) + ",\n")
    f.write("};\n\n")


with open(OUT_PATH, "w") as f:
    f.write("// AUTO-GENERATED by generate_test_data_int8.py -- do not hand-edit.\n")
    f.write("// Real MNIST test[0] image + real Models/lenet5_relu.keras weights,\n")
    f.write("// quantized end-to-end through the full C1->S2->C3->S4->flatten->\n")
    f.write("// C5->F6->Output int8 pipeline. Each conv/dense layer's own\n")
    f.write("// requant_mult (M)/requant_shift (S) is independently derived from\n")
    f.write("// that layer's own (x_scale, w_scale, out_scale) triple. Includes\n")
    f.write("// every intermediate stage's expected output (bit-exact reference)\n")
    f.write("// plus the final softmax probability distribution (tight-tolerance\n")
    f.write("// reference), for lenet5_top_int8_tb.cpp.\n")
    f.write("#ifndef LENET5_TOP_INT8_TEST_DATA_H\n#define LENET5_TOP_INT8_TEST_DATA_H\n\n")

    f.write(f"#define TRUE_LABEL {int(true_label)}\n")
    f.write(f"#define EXPECTED_CLASS {predicted_label}\n\n")

    f.write(f"#define C1_REQUANT_MULT {c1_M}\n#define C1_REQUANT_SHIFT {c1_S}\n")
    f.write(f"#define C3_REQUANT_MULT {c3_M}\n#define C3_REQUANT_SHIFT {c3_S}\n")
    f.write(f"#define C5_REQUANT_MULT {c5_M}\n#define C5_REQUANT_SHIFT {c5_S}\n")
    f.write(f"#define F6_REQUANT_MULT {f6_M}\n#define F6_REQUANT_SHIFT {f6_S}\n")
    f.write(f"#define OUTPUT_X_SCALE {float(f6_out_scale):.9e}f\n")
    f.write(f"#define OUTPUT_W_SCALE {float(out_w_scale):.9e}f\n\n")

    emit_int_array(f, "image_data", [28, 28, 1], x_int, "signed char")

    emit_int_array(f, "c1_weights_data", [5, 5, 1, 6], c1_w_int, "signed char")
    emit_int_array(f, "c1_bias_data", [6], c1_bias_int, "int")
    emit_int_array(f, "c1_expected", [28, 28, 6], c1_out_int8, "signed char")
    emit_int_array(f, "s2_expected", [14, 14, 6], s2_out_int8, "signed char")

    emit_int_array(f, "c3_weights_data", [5, 5, 6, 16], c3_w_int, "signed char")
    emit_int_array(f, "c3_bias_data", [16], c3_bias_int, "int")
    emit_int_array(f, "c3_expected", [10, 10, 16], c3_out_int8, "signed char")
    emit_int_array(f, "s4_expected", [5, 5, 16], s4_out_int8, "signed char")

    emit_int_array(f, "c5_weights_data", [400, 120], c5_w_int, "signed char")
    emit_int_array(f, "c5_bias_data", [120], c5_bias_int, "int")
    emit_int_array(f, "c5_expected", [120], c5_out_int8, "signed char")

    emit_int_array(f, "f6_weights_data", [120, 84], f6_w_int, "signed char")
    emit_int_array(f, "f6_bias_data", [84], f6_bias_int, "int")
    emit_int_array(f, "f6_expected", [84], f6_out_int8, "signed char")

    emit_int_array(f, "output_weights_data", [84, 10], out_w_int, "signed char")
    emit_int_array(f, "output_bias_data", [10], out_bias_int, "int")
    emit_float_array(f, "expected_output", [10], expected_output)

    f.write("#endif\n")

print(f"\nWrote {OUT_PATH}")
