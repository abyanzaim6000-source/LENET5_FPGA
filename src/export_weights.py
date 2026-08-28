"""
Exports trained LeNet-5 weights/biases to disk in two formats:
  - .npy  : for reuse in our own Python scripts (quantization, etc.)
  - .dat  : plain text, human-readable, matches the paper's convention
            for sharing weights with hardware/HLS drivers.
Run from project root: python3 src/export_weights.py
"""

import os
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("models/lenet5.keras")

OUT_DIR = "weights/float32"
os.makedirs(OUT_DIR, exist_ok=True)

for layer in model.layers:
    w = layer.get_weights()
    if not w:
        continue  # pooling/flatten layers have no weights

    weights, bias = w
    name = layer.name

    # .npy — exact binary float32, easiest for us to reload in Python
    np.save(f"{OUT_DIR}/{name}_weights.npy", weights)
    np.save(f"{OUT_DIR}/{name}_bias.npy", bias)

    # .dat — flat, plain-text, one value per line (paper's convention)
    np.savetxt(f"{OUT_DIR}/{name}_weights.dat", weights.flatten(), fmt="%.8f")
    np.savetxt(f"{OUT_DIR}/{name}_bias.dat", bias.flatten(), fmt="%.8f")

    print(f"{name:10s}  weights shape={weights.shape}  bias shape={bias.shape}")

print("\nAll weights exported to:", OUT_DIR)