"""
Verifies that manual NumPy inference (ReLU + MaxPool variant) is
numerically equivalent to Keras, layer by layer, across multiple images.
Run from project root: python3 src/verify_relu.py
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

from manual_layers import conv2d, maxpool2d, dense, relu

N_IMAGES = 10

model = tf.keras.models.load_model("models/lenet5_relu.keras")

W = {}
for layer in model.layers:
    w = layer.get_weights()
    if w:
        W[layer.name] = w

layer_names = ["C1", "S2", "C3", "S4", "C5", "F6", "Output"]

(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test.astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1)


def manual_forward(image):
    stages = {}

    c1_w, c1_b = W["C1"]
    x = relu(conv2d(image, c1_w, c1_b, padding="same"))
    stages["C1"] = x

    x = maxpool2d(x, 2, 2)
    stages["S2"] = x

    c3_w, c3_b = W["C3"]
    x = relu(conv2d(x, c3_w, c3_b, padding="valid"))
    stages["C3"] = x

    x = maxpool2d(x, 2, 2)
    stages["S4"] = x

    x = x.flatten()

    c5_w, c5_b = W["C5"]
    x = relu(dense(x, c5_w, c5_b, activation=None))
    stages["C5"] = x

    f6_w, f6_b = W["F6"]
    x = relu(dense(x, f6_w, f6_b, activation=None))
    stages["F6"] = x

    out_w, out_b = W["Output"]
    x = dense(x, out_w, out_b, activation="softmax")
    stages["Output"] = x

    return stages


max_err = {name: 0.0 for name in layer_names}
sq_err_sum = {name: 0.0 for name in layer_names}
count = {name: 0 for name in layer_names}

for idx in range(N_IMAGES):
    image = x_test[idx]

    x = tf.convert_to_tensor(image[np.newaxis, ...])
    keras_stages = {}
    for layer in model.layers:
        x = layer(x)
        if layer.name in layer_names:
            keras_stages[layer.name] = x[0].numpy()

    manual_stages = manual_forward(image)

    for name in layer_names:
        diff = np.abs(keras_stages[name] - manual_stages[name])
        max_err[name] = max(max_err[name], diff.max())
        sq_err_sum[name] += np.sum(diff ** 2)
        count[name] += diff.size

print(f"Verification (ReLU + MaxPool) over {N_IMAGES} images")
print("=" * 55)
print(f"{'Layer':10s} {'Max Abs Error':>15s} {'RMSE':>15s}")
print("-" * 55)
for name in layer_names:
    rmse = np.sqrt(sq_err_sum[name] / count[name])
    print(f"{name:10s} {max_err[name]:>15.8f} {rmse:>15.8f}")