"""
Verifies that manual NumPy inference is numerically equivalent to Keras.
Compares Keras vs. manual output at EVERY intermediate layer, across
multiple test images, using max absolute error and RMSE.
Run from project root: python3 src/verify.py
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import Model

from manual_layers import conv2d, avgpool2d, dense

N_IMAGES = 10  # how many test images to check

model = tf.keras.models.load_model("models/lenet5.keras")

W = {}
for layer in model.layers:
    w = layer.get_weights()
    if w:
        W[layer.name] = w

# Rebuild the computation graph explicitly using the SAME trained layer
# objects (weights are untouched), so we can expose every intermediate
# output. This avoids relying on model.input, which Keras 3's Sequential
# models don't reliably expose.
layer_names = ["C1", "S2", "C3", "S4", "C5", "F6", "Output"]

inp = tf.keras.Input(shape=(28, 28, 1))
x = inp
tapped_outputs = []
for layer in model.layers:
    x = layer(x)
    if layer.name in layer_names:
        tapped_outputs.append(x)

keras_probe = Model(inputs=inp, outputs=tapped_outputs)

(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test.astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1)


def manual_forward(image):
    """Returns a dict of {layer_name: output} for one image, same stages as keras_probe."""
    stages = {}

    c1_w, c1_b = W["C1"]
    x = np.tanh(conv2d(image, c1_w, c1_b, padding="same"))
    stages["C1"] = x

    x = avgpool2d(x, 2, 2)
    stages["S2"] = x

    c3_w, c3_b = W["C3"]
    x = np.tanh(conv2d(x, c3_w, c3_b, padding="valid"))
    stages["C3"] = x

    x = avgpool2d(x, 2, 2)
    stages["S4"] = x

    x = x.flatten()

    c5_w, c5_b = W["C5"]
    x = dense(x, c5_w, c5_b, activation="tanh")
    stages["C5"] = x

    f6_w, f6_b = W["F6"]
    x = dense(x, f6_w, f6_b, activation="tanh")
    stages["F6"] = x

    out_w, out_b = W["Output"]
    x = dense(x, out_w, out_b, activation="softmax")
    stages["Output"] = x

    return stages


# Accumulate errors per layer across all images
max_err = {name: 0.0 for name in layer_names}
sq_err_sum = {name: 0.0 for name in layer_names}
count = {name: 0 for name in layer_names}

for idx in range(N_IMAGES):
    image = x_test[idx]

    keras_outputs = keras_probe.predict(image[np.newaxis, ...], verbose=0)
    keras_stages = dict(zip(layer_names, [o[0] for o in keras_outputs]))

    manual_stages = manual_forward(image)

    for name in layer_names:
        diff = np.abs(keras_stages[name] - manual_stages[name])
        max_err[name] = max(max_err[name], diff.max())
        sq_err_sum[name] += np.sum(diff ** 2)
        count[name] += diff.size

print(f"Verification over {N_IMAGES} images")
print("=" * 55)
print(f"{'Layer':10s} {'Max Abs Error':>15s} {'RMSE':>15s}")
print("-" * 55)
for name in layer_names:
    rmse = np.sqrt(sq_err_sum[name] / count[name])
    print(f"{name:10s} {max_err[name]:>15.8f} {rmse:>15.8f}")