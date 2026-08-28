"""
LeNet-5 FPGA Project
Step 2 : Exploring the MNIST Dataset
"""

# ======================================
# Import Libraries
# ======================================

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from lenet5_relu import build_lenet5_relu

# ======================================
# Load Dataset
# ======================================

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# ======================================
# Dataset Information
# ======================================

print("=" * 60)
print("MNIST DATASET")
print("=" * 60)

print("Training Images :", x_train.shape)
print("Training Labels :", y_train.shape)

print("Testing Images  :", x_test.shape)
print("Testing Labels  :", y_test.shape)

print("\n")

# ======================================
# Display First Image
# ======================================

plt.figure(figsize=(5,5))
plt.imshow(x_train[0], cmap="gray")
plt.colorbar()
plt.title(f"Label = {y_train[0]}")
plt.axis("off")
plt.show()

# ======================================
# Print Pixel Values
# ======================================

print("Pixel Matrix of First Image:\n")

print(x_train[0])
# ======================================
# Normalize Images
# ======================================

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

print("=" * 60)
print("AFTER NORMALIZATION")
print("=" * 60)

print("Minimum Pixel Value :", x_train.min())
print("Maximum Pixel Value :", x_train.max())

print()
print("First 5 pixel values of first row:")
print(x_train[0][0][:5])

# ======================================
# Reshape Images
# ======================================

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

print()
print("=" * 60)
print("AFTER RESHAPING")
print("=" * 60)

print("Training Shape :", x_train.shape)
print("Testing Shape  :", x_test.shape)
# ======================================
# Build LeNet-5 Model
# ======================================

model = build_lenet5_relu()

print()
print("=" * 60)
print("LENET-5 MODEL SUMMARY(ReLU + MaxPool)")
print("=" * 60)

model.summary()
# ======================================
# Compile Model
# ======================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel compiled successfully!")
# ======================================
# Train Model
# ======================================

print("\nTraining LeNet-5...\n")

history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1
)
# ======================================
# Test Model
# ======================================

test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print("\n" + "=" * 60)
print("TEST RESULTS(ReLU + MaxPool)")
print("=" * 60)

print(f"Test Accuracy : {test_accuracy*100:.2f}%")
print(f"Test Loss     : {test_loss:.4f}")
# ======================================
# Save Model
# ======================================

model.save("models/lenet5_relu.keras")

print("\nModel saved successfully!")
print("Location : models/lenet5_relu.keras")