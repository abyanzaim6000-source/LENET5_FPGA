"""
Full manual (NumPy-only) forward pass through the trained ReLU+MaxPool
LeNet-5 variant. Same structure as manual_inference.py, but uses
relu/maxpool2d instead of tanh/avgpool2d, and loads lenet5_relu.keras.
Run from project root: python3 src/manual_inference_relu.py
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

from manual_layers import conv2d, maxpool2d, dense, relu

# ---- 1. Load trained weights ----
model = tf.keras.models.load_model("models/lenet5_relu.keras")

W = {}
for layer in model.layers:
    w = layer.get_weights()
    if w:
        W[layer.name] = w

# ---- 2. Load one MNIST test image ----
(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test.astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1)

image = x_test[0]
true_label = y_test[0]

# ---- 3. Manual forward pass ----
c1_w, c1_b = W["C1"]
x = relu(conv2d(image, c1_w, c1_b, padding="same"))
print("After C1:", x.shape)

x = maxpool2d(x, pool_size=2, stride=2)
print("After S2:", x.shape)

c3_w, c3_b = W["C3"]
x = relu(conv2d(x, c3_w, c3_b, padding="valid"))
print("After C3:", x.shape)

x = maxpool2d(x, pool_size=2, stride=2)
print("After S4:", x.shape)

x = x.flatten()
print("After Flatten:", x.shape)

c5_w, c5_b = W["C5"]
x = dense(x, c5_w, c5_b, activation=None)
x = relu(x)
print("After C5:", x.shape)

f6_w, f6_b = W["F6"]
x = dense(x, f6_w, f6_b, activation=None)
x = relu(x)
print("After F6:", x.shape)

out_w, out_b = W["Output"]
x = dense(x, out_w, out_b, activation="softmax")
print("After Output:", x.shape)

predicted_label = np.argmax(x)

print()
print("=" * 40)
print(f"True label      : {true_label}")
print(f"Predicted label : {predicted_label}")
print(f"Confidence      : {x[predicted_label]*100:.2f}%")
print("=" * 40)