"""
LeNet5_FPGA Project
Training the 3x3-kernel custom CNN
"""

import tensorflow as tf
import numpy as np

from tensorflow.keras.datasets import mnist
from cnn3x3_relu import build_cnn3x3_relu

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

model = build_cnn3x3_relu()
model.summary()

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1
)

test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

print("\n" + "=" * 60)
print("TEST RESULTS (3x3 kernel CNN, ReLU + MaxPool)")
print("=" * 60)
print(f"Test Accuracy : {test_accuracy*100:.2f}%")
print(f"Test Loss     : {test_loss:.4f}")

model.save("models/cnn3x3_relu.keras")
print("\nModel saved to models/cnn3x3_relu.keras")