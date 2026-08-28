"""
Interactive single-image test: pick a test image, run it through the
model, and show both the image and the prediction visually.
Run from project root: python3 src/test_single_image.py
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# ---- Choose which model to test ----
MODEL_PATH = "models/lenet5.keras"          # baseline (tanh + AvgPool)
# MODEL_PATH = "models/lenet5_relu.keras"   # uncomment to test the ReLU variant instead

model = tf.keras.models.load_model(MODEL_PATH)

(_, _), (x_test, y_test) = mnist.load_data()
x_test_norm = x_test.astype("float32") / 255.0
x_test_norm = x_test_norm.reshape(-1, 28, 28, 1)

# ---- Pick an image: change this index to test different digits ----
INDEX = 19   # try any number from 0 to 9999

image = x_test_norm[INDEX]
true_label = y_test[INDEX]

# ---- Run prediction ----
probs = model.predict(image[np.newaxis, ...], verbose=0)[0]
predicted_label = np.argmax(probs)
confidence = probs[predicted_label] * 100

# ---- Show result ----
print(f"Testing image index {INDEX}")
print(f"True label      : {true_label}")
print(f"Predicted label : {predicted_label}")
print(f"Confidence      : {confidence:.2f}%")
print(f"Correct?        : {'YES' if predicted_label == true_label else 'NO'}")

print("\nFull probability breakdown:")
for digit in range(10):
    bar = "#" * int(probs[digit] * 50)
    print(f"  {digit}: {probs[digit]*100:5.2f}%  {bar}")

# ---- Visual display ----
plt.figure(figsize=(4, 4))
plt.imshow(x_test[INDEX], cmap="gray")
plt.title(f"True: {true_label}  |  Predicted: {predicted_label} ({confidence:.1f}%)",
          color="green" if predicted_label == true_label else "red")
plt.axis("off")
plt.show()