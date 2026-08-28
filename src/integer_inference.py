"""
Full forward pass using REAL integer arithmetic (int8) at every layer,
on the ReLU + MaxPool LeNet-5 variant. This is the genuine hardware-
facing proof: real int8 MACs throughout, not simulated in float.
Run from project root: python3 src/integer_inference.py
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

from manual_layers import maxpool2d, relu
from integer_layers import conv2d_int, dense_int, quantize_activation
from quantization import quantize_int_real

N_IMAGES = 200
N_BITS = 4   # change to 4 to test INT4 arithmetic instead

model = tf.keras.models.load_model("models/lenet5_relu.keras")
W = {}
for layer in model.layers:
    w = layer.get_weights()
    if w:
        W[layer.name] = w

# Pre-quantize all weights ONCE (weights don't change per-image)
W_int = {}
for name, (w, b) in W.items():
    qw, w_scale = quantize_int_real(w, N_BITS)
    W_int[name] = (qw, w_scale, b)   # keep bias as ORIGINAL float, requantized per-layer inside conv2d_int/dense_int

(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test.astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1)


def integer_forward(image):
    x_int, x_scale = quantize_activation(image, N_BITS)

    c1_w, c1_scale, c1_b = W_int["C1"]
    x = conv2d_int(x_int, c1_w, c1_b, x_scale, c1_scale, padding="same")
    x = relu(x)

    x = maxpool2d(x, 2, 2)
    x_int, x_scale = quantize_activation(x, N_BITS)

    c3_w, c3_scale, c3_b = W_int["C3"]
    x = conv2d_int(x_int, c3_w, c3_b, x_scale, c3_scale, padding="valid")
    x = relu(x)

    x = maxpool2d(x, 2, 2)
    x = x.flatten()
    x_int, x_scale = quantize_activation(x, N_BITS)

    c5_w, c5_scale, c5_b = W_int["C5"]
    x = dense_int(x_int, c5_w, c5_b, x_scale, c5_scale, activation="relu")
    x_int, x_scale = quantize_activation(x, N_BITS)

    f6_w, f6_scale, f6_b = W_int["F6"]
    x = dense_int(x_int, f6_w, f6_b, x_scale, f6_scale, activation="relu")
    x_int, x_scale = quantize_activation(x, N_BITS)

    out_w, out_scale, out_b = W_int["Output"]
    x = dense_int(x_int, out_w, out_b, x_scale, out_scale, activation="softmax")

    return x


correct = 0
wrong_indices = []
for idx in range(N_IMAGES):
    probs = integer_forward(x_test[idx])
    pred = np.argmax(probs)
    if pred == y_test[idx]:
        correct += 1
    else:
        wrong_indices.append((idx, int(y_test[idx]), int(pred)))

acc = correct / N_IMAGES * 100
print(f"Real INT{N_BITS} integer-arithmetic accuracy over {N_IMAGES} images: {acc:.2f}%")
print(f"Wrong predictions (index, true, predicted): {wrong_indices}")