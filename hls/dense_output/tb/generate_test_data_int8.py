"""
Generates dense_output_int8_test_data.h: real int8-quantized Output-layer
weights + a real int8-quantized F6 activation vector (via the full
C1->S2->C3->S4->flatten->C5->F6 int8 pipeline) as dense_output_int8's
input, plus the reference softmax probability distribution.

Unlike every earlier INT8 layer (C1, C3, C5, F6), this is NOT a
"_fixedpoint" reference: the Output layer's softmax is the network's
FINAL activation, with no downstream quantized layer to feed, so there
is no reason to requantize its result back to int8 or to do exp/sum/divide
in fixed-point at all. Stage 1 (the MAC accumulation) is still genuine
int8x int8->int32 integer arithmetic, verified bit-exact via accumulator
recovery exactly like every earlier layer. Stage 2 (softmax) dequantizes
that int32 accumulator to float using the same combined_scale = x_scale *
w_scale every conv/dense layer already uses, then reuses dense_output.cpp's
own proven float max-subtract/exp/sum/divide -- matching
src/integer_layers.py's dense_int(..., activation="softmax"), which does
exactly this (integer MAC, then float dequantize+softmax).

Pipeline (all real data, no synthetic values), duplicated fresh here (not
imported), same convention as every prior generator script in this project:
  1. C1 conv (int8, fixed-point) -> 28x28x6.
  2. S2 max pool (int8, no rescale) -> 14x14x6.
  3. C3 conv (int8, fixed-point, "valid") -> 10x10x16.
  4. S4 max pool (int8, no rescale) -> 5x5x16.
  5. Flatten (NumPy default C-order) -> 400 int8 values.
  6. C5 dense (int8, fixed-point) -> 120 int8 values.
  7. F6 dense (int8, fixed-point) -> 84 int8 values, Output's real input.
  8. Output dense (int8x int8->int32 MAC via dense_int()) -- the raw
     int32 accumulator is recovered from dense_int()'s float output via
     its exact inverse and the round-trip is VERIFIED exact -- Output's
     OWN verification, independent of every earlier layer's.
  9. Softmax: dequantize the verified accumulator with
     combined_scale = x_scale*w_scale, then max-subtract/exp/sum/divide
     in float -- this IS the reference dense_output_int8.cpp's Stage 2
     must match (to a tight float tolerance, not bit-exact -- softmax
     involves exp()/summation-order effects that differ slightly between
     NumPy and C++'s std::exp, exactly like this project's own
     lenet5_top float32 reference, which matched to 6.5e-11, not exact
     bit-identity).

Run from repo root: python hls/dense_output/tb/generate_test_data_int8.py
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
OUT_PATH = os.path.join(os.path.dirname(__file__), "dense_output_int8_test_data.h")


def quantize_multiplier(real_multiplier):
    """TFLite/gemmlowp-style decomposition -- see conv_c1's generator for
    the full explanation. Duplicated here (not imported) so this script
    stays self-contained, matching the project's per-stage convention.
    Only used by the ReLU layers in this script's own chain (C1/C3/C5/F6);
    the Output layer itself never calls this -- see the module docstring."""
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
    (output_int8, out_scale)."""
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
    return output_int8, out_scale


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
print("Loaded Output weights:", out_w.shape, "bias:", out_b.shape)

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

# ---- 4. C3 conv (int8, fixed-point, "valid") ----
c3_w_int, c3_w_scale = quantize_int_real(c3_w, N_BITS)
c3_w_scale = np.float32(c3_w_scale)
c3_out_int8, c3_out_scale = conv_int8_fixedpoint(
    s2_out_int8, c3_b, c3_w_int, c1_out_scale, c3_w_scale, padding="valid")
assert c3_out_int8.shape == (10, 10, 16), f"Expected C3 output (10,10,16), got {c3_out_int8.shape}"

# ---- 5. S4 max pool, then flatten ----
s4_out_int8 = maxpool_int8(c3_out_int8, POOL_SIZE, STRIDE)
c5_input_int8 = s4_out_int8.flatten()
assert c5_input_int8.shape == (400,), f"Expected flattened C5 input (400,), got {c5_input_int8.shape}"

# ---- 6. C5 dense (int8, fixed-point) ----
c5_w_int, c5_w_scale = quantize_int_real(c5_w, N_BITS)
c5_w_scale = np.float32(c5_w_scale)
c5_out_int8, c5_out_scale = dense_int8_fixedpoint(
    c5_input_int8, c5_b, c5_w_int, c3_out_scale, c5_w_scale)
assert c5_out_int8.shape == (120,), f"Expected C5 output (120,), got {c5_out_int8.shape}"

# ---- 7. F6 dense (int8, fixed-point) -> Output's real input ----
f6_w_int, f6_w_scale = quantize_int_real(f6_w, N_BITS)
f6_w_scale = np.float32(f6_w_scale)
f6_out_int8, f6_out_scale = dense_int8_fixedpoint(
    c5_out_int8, f6_b, f6_w_int, c5_out_scale, f6_w_scale)
assert f6_out_int8.shape == (84,), f"Expected F6 output (84,), got {f6_out_int8.shape}"
print(f"F6 int8 output: shape {f6_out_int8.shape}, range [{f6_out_int8.min()}, {f6_out_int8.max()}], "
      f"nonzero fraction: {np.mean(f6_out_int8 != 0):.3f}")

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
#         combined_scale, then the SAME max-subtract/exp/sum/divide as
#         the proven float32 dense_output.cpp -- no requantization to
#         int8 (this is the network's final output, read as a
#         probability distribution, not fed to another quantized layer). ----
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
    f.write("// Real int8-quantized Models/lenet5_relu.keras Output-layer weights + a\n")
    f.write("// real int8-quantized F6 activation vector (via the full\n")
    f.write("// C1->S2->C3->S4->flatten->C5->F6 int8 pipeline), plus the reference\n")
    f.write("// softmax probability distribution (int8x int8->int32 MAC, verified\n")
    f.write("// exact accumulator recovery, then float dequantize + max-subtract/\n")
    f.write("// exp/sum/divide -- NOT fixed-point, see the generator script's\n")
    f.write("// docstring for why). For dense_output_int8_tb.cpp.\n")
    f.write("#ifndef DENSE_OUTPUT_INT8_TEST_DATA_H\n#define DENSE_OUTPUT_INT8_TEST_DATA_H\n\n")

    f.write(f"#define TRUE_LABEL {int(true_label)}\n")
    f.write(f"#define X_SCALE {float(f6_out_scale):.9e}f\n")
    f.write(f"#define W_SCALE {float(out_w_scale):.9e}f\n\n")

    emit_int_array(f, "input_data", [84], f6_out_int8, "signed char")
    emit_int_array(f, "weights_data", [84, 10], out_w_int, "signed char")
    emit_int_array(f, "bias_int_data", [10], out_bias_int, "int")
    emit_float_array(f, "expected_output", [10], expected_output)

    f.write("#endif\n")

print(f"\nWrote {OUT_PATH}")
