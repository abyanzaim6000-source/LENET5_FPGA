"""
Generates conv_c1_int8_test_data.h: real int8-quantized C1 weights/bias +
a real int8-quantized MNIST test image, plus the reference INT8 output,
all embedded as C arrays for conv_c1_int8_tb.cpp.

Quantization uses the project's own proven functions -- not reimplemented
here:
  - weights: quantize_int_real() from src/quantization.py (as asked for
    explicitly -- true data-driven INT8 quantization, real integers).
  - input activation: quantize_activation() from src/integer_layers.py
    (the same activation quantizer integer_inference.py's real int8
    forward pass uses).
  - MAC + dequantize: conv2d_int() from src/integer_layers.py -- the
    proven int8 x int8 -> int32 reference this HLS kernel must match.

conv2d_int() returns a dequantized float (pre-ReLU) output, matching what
the paper's "acc * combined_scale" step produces; it deliberately leaves
activation and re-quantization to the caller (see its own docstring).
This script applies relu() (src/manual_layers.py, same one
integer_inference.py's pipeline uses) and then requantizes down to int8
via a single round(x/out_scale) + saturate step -- exactly what
conv_c1_int8.cpp's hardware datapath does. out_scale itself is produced by
quantize_activation() applied to the ReLU'd float output, matching how
integer_inference.py determines every layer's next-stage input scale.

The one deliberate departure from quantize_activation()'s own rounding is
round-half-AWAY-from-zero (matching conv_c1_int8.cpp's `x >= 0 ? +0.5 :
-0.5` hardware rounding) instead of numpy's round-half-to-even. The two
only disagree exactly at x.5000... boundaries, which do not occur with
real trained-model floating-point data -- so this does not change which
reference is being matched, only makes the rounding rule explicit and
bit-reproducible in C++.

Loads the .keras archive directly via h5py instead of tf.keras.models,
and one real MNIST test image via torchvision -- same reason and same
approach as hls/lenet5_top/tb/generate_test_data.py (TensorFlow isn't
installed in this environment; both loaders serve the same canonical
MNIST test set in the same order).

Run from repo root: python hls/conv_c1/tb/generate_test_data_int8.py
"""
import os
import sys
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
OUT_PATH = os.path.join(os.path.dirname(__file__), "conv_c1_int8_test_data.h")

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

# ---- 3. Quantize weights and input activation using the project's own,
#         already-proven quantizers ----
w_int, w_scale = quantize_int_real(c1_w, N_BITS)          # src/quantization.py
x_int, x_scale = quantize_activation(image, N_BITS)        # src/integer_layers.py
w_scale = np.float32(w_scale)
x_scale = np.float32(x_scale)
print(f"x_scale = {x_scale}, w_scale = {w_scale}")

# ---- 4. Reference MAC + dequantize, via the proven conv2d_int() ----
float_out = conv2d_int(x_int, w_int, c1_b, x_scale, w_scale, padding="same")
relu_out = relu(float_out).astype(np.float32)

# bias_int, exactly as conv2d_int() computes it internally (so the value
# embedded for the HLS kernel's pre-quantized bias input matches what the
# Python reference used to produce relu_out above).
combined_scale = np.float32(x_scale * w_scale)
bias_int = np.round(c1_b / combined_scale).astype(np.int32)

# ---- 5. Requantize to int8: this layer's output activation scale, then
#         round-half-away-from-zero + saturate (matches the HLS hardware) ----
_, out_scale = quantize_activation(relu_out, N_BITS)
out_scale = np.float32(out_scale)

# Multiply by the precomputed reciprocal, NOT relu_out/out_scale directly --
# this must match conv_c1_int8.cpp's `dequant * inv_out_scale` bit-for-bit,
# since a/b is not always bit-identical to a*(1/b) in IEEE754 float32, even
# though they're mathematically equivalent.
inv_out_scale = np.float32(1.0) / out_scale
requant = relu_out * inv_out_scale
rounded = np.where(requant >= 0,
                    np.floor(requant + np.float32(0.5)),
                    np.ceil(requant - np.float32(0.5)))
expected_output = np.clip(rounded, -128, 127).astype(np.int8)

print(f"out_scale = {out_scale}")
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
    f.write("// AUTO-GENERATED by generate_test_data_int8.py -- do not hand-edit.\n")
    f.write("// Real int8-quantized Models/lenet5_relu.keras C1 weights + a real\n")
    f.write("// int8-quantized MNIST test[0] image, plus the conv2d_int()+relu()+\n")
    f.write("// requantize reference output, for conv_c1_int8_tb.cpp to check the\n")
    f.write("// INT8 HLS kernel against.\n")
    f.write("#ifndef CONV_C1_INT8_TEST_DATA_H\n#define CONV_C1_INT8_TEST_DATA_H\n\n")

    f.write(f"#define TRUE_LABEL {int(true_label)}\n")
    f.write(f"#define X_SCALE {x_scale:.9e}f\n")
    f.write(f"#define W_SCALE {w_scale:.9e}f\n")
    f.write(f"#define OUT_SCALE {out_scale:.9e}f\n\n")

    emit_int_array(f, "input_data", [28, 28, 1], x_int, "signed char")
    emit_int_array(f, "weights_data", [5, 5, 1, 6], w_int, "signed char")
    emit_int_array(f, "bias_int_data", [6], bias_int, "int")
    emit_int_array(f, "expected_output", [28, 28, 6], expected_output, "signed char")

    f.write("#endif\n")

print(f"\nWrote {OUT_PATH}")
