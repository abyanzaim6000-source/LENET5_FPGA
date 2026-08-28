"""
Runs manual inference on the ReLU + MaxPool variant using quantized
(fixed-point-simulated or INTn) weights and activations, comparing
against its own float32 baseline. Mirrors quantized_inference.py.
Run from project root: python3 src/quantized_inference_relu.py
"""

import os
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

from manual_layers import conv2d, maxpool2d, dense, relu
from quantization import (
    quantize_fixed,
    quantize_int_dynamic,
    quantize_layer_weights,
    quantize_layer_weights_int,
)

N_IMAGES = 500

# ---- Load trained weights ----
model = tf.keras.models.load_model("models/lenet5_relu.keras")
W = {}
for layer in model.layers:
    w = layer.get_weights()
    if w:
        W[layer.name] = w

# ---- Load test data ----
(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test.astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1)

rng = np.random.default_rng(seed=42)
test_indices = rng.choice(len(x_test), size=N_IMAGES, replace=False)


def forward_pass(image, weights, int_bits=None, frac_bits=None, n_bits=None):
    def q(x):
        if n_bits is not None:
            out, _, _, _ = quantize_int_dynamic(x, n_bits)
            return out
        elif int_bits is not None:
            out, _, _, _ = quantize_fixed(x, int_bits, frac_bits)
            return out
        else:
            return x

    c1_w, c1_b = weights["C1"]
    x = q(relu(conv2d(image, c1_w, c1_b, padding="same")))

    x = q(maxpool2d(x, 2, 2))

    c3_w, c3_b = weights["C3"]
    x = q(relu(conv2d(x, c3_w, c3_b, padding="valid")))

    x = q(maxpool2d(x, 2, 2))

    x = x.flatten()

    c5_w, c5_b = weights["C5"]
    x = q(relu(dense(x, c5_w, c5_b, activation=None)))

    f6_w, f6_b = weights["F6"]
    x = q(relu(dense(x, f6_w, f6_b, activation=None)))

    out_w, out_b = weights["Output"]
    x = dense(x, out_w, out_b, activation="softmax")

    return x


def evaluate(weights, int_bits=None, frac_bits=None, n_bits=None, label=""):
    correct = 0
    for idx in test_indices:
        probs = forward_pass(x_test[idx], weights, int_bits, frac_bits, n_bits)
        pred = np.argmax(probs)
        if pred == y_test[idx]:
            correct += 1
    acc = correct / len(test_indices) * 100
    print(f"{label:25s} accuracy over {len(test_indices)} images: {acc:.2f}%")
    return acc


def compute_memory_bytes(weights_dict, bits_per_value):
    total_values = 0
    for name, (w, b) in weights_dict.items():
        total_values += w.size + b.size
    return total_values * bits_per_value / 8


def compute_weight_error(original_weights, quantized_weights):
    all_orig, all_quant = [], []
    for name in original_weights:
        ow, ob = original_weights[name]
        qw, qb = quantized_weights[name]
        all_orig.append(ow.flatten())
        all_orig.append(ob.flatten())
        all_quant.append(qw.flatten())
        all_quant.append(qb.flatten())
    orig = np.concatenate(all_orig)
    quant = np.concatenate(all_quant)
    diff = np.abs(orig - quant)
    return diff.max(), diff.mean(), np.sqrt(np.mean(diff ** 2))


# ---- Run comparison ----
results = []

acc = evaluate(W, label="Float32 (manual)")
results.append({
    "format": "Float32", "source": "baseline",
    "accuracy": acc, "max_err": 0.0, "mean_err": 0.0, "rmse": 0.0,
    "memory_bytes": compute_memory_bytes(W, 32),
})

configs = [
    ("Fixed16 <13,3>", "paper",     quantize_layer_weights(W, int_bits=12, frac_bits=3), dict(int_bits=12, frac_bits=3), 16),
    ("Ultra-low (1,2)", "our test", quantize_layer_weights(W, int_bits=1, frac_bits=2),   dict(int_bits=1, frac_bits=2), 4),
    ("INT8",            "our ext",  quantize_layer_weights_int(W, n_bits=8),               dict(n_bits=8),                8),
    ("INT4",            "our ext",  quantize_layer_weights_int(W, n_bits=4),               dict(n_bits=4),                4),
]

for label, source, W_q, kwargs, bits in configs:
    print(f"\nRunning {label}...")
    acc = evaluate(W_q, label=label, **kwargs)
    max_err, mean_err, rmse = compute_weight_error(W, W_q)
    results.append({
        "format": label, "source": source,
        "accuracy": acc, "max_err": max_err, "mean_err": mean_err, "rmse": rmse,
        "memory_bytes": compute_memory_bytes(W_q, bits),
    })

os.makedirs("results", exist_ok=True)
with open("results/quantization_comparison_relu.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["format", "source", "accuracy", "max_err", "mean_err", "rmse", "memory_bytes"])
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print("\n" + "=" * 90)
print(f"{'Format':18s}{'Source':10s}{'Accuracy':>10s}{'MaxErr':>10s}{'MeanErr':>10s}{'RMSE':>10s}{'Mem(KB)':>10s}")
print("-" * 90)
for r in results:
    print(f"{r['format']:18s}{r['source']:10s}{r['accuracy']:>9.2f}%{r['max_err']:>10.5f}{r['mean_err']:>10.5f}{r['rmse']:>10.5f}{r['memory_bytes']/1024:>9.2f}K")