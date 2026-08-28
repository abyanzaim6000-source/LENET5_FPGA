"""
Combines quantization_comparison.csv (tanh+AvgPool) and
quantization_comparison_relu.csv (ReLU+MaxPool) into one table and plot.
Run from project root: python3 src/compare_variants.py
"""

import csv
import matplotlib.pyplot as plt

def load_csv(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["accuracy"] = float(row["accuracy"])
            row["max_err"] = float(row["max_err"])
            row["mean_err"] = float(row["mean_err"])
            row["rmse"] = float(row["rmse"])
            row["memory_bytes"] = float(row["memory_bytes"])
            rows.append(row)
    return rows

tanh_results = load_csv("results/quantization_comparison.csv")
relu_results = load_csv("results/quantization_comparison_relu.csv")

# ---- Combined text table ----
print("=" * 100)
print(f"{'Format':18s}{'tanh+AvgPool':>15s}{'ReLU+MaxPool':>15s}{'Difference':>15s}")
print("-" * 100)
for t, r in zip(tanh_results, relu_results):
    assert t["format"] == r["format"], "CSV row order mismatch!"
    diff = r["accuracy"] - t["accuracy"]
    print(f"{t['format']:18s}{t['accuracy']:>14.2f}%{r['accuracy']:>14.2f}%{diff:>+14.2f}%")

# ---- Save combined table to CSV too ----
with open("results/combined_comparison.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["format", "source", "tanh_accuracy", "relu_accuracy", "difference",
                      "tanh_rmse", "relu_rmse", "memory_kb"])
    for t, r in zip(tanh_results, relu_results):
        writer.writerow([
            t["format"], t["source"],
            f"{t['accuracy']:.2f}", f"{r['accuracy']:.2f}", f"{r['accuracy']-t['accuracy']:+.2f}",
            f"{t['rmse']:.5f}", f"{r['rmse']:.5f}", f"{t['memory_bytes']/1024:.2f}",
        ])

# ---- Plot: grouped bar chart ----
formats = [r["format"] for r in tanh_results]
tanh_acc = [r["accuracy"] for r in tanh_results]
relu_acc = [r["accuracy"] for r in relu_results]

x = range(len(formats))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar([i - width/2 for i in x], tanh_acc, width, label="tanh + AvgPool")
ax.bar([i + width/2 for i in x], relu_acc, width, label="ReLU + MaxPool")

ax.set_ylabel("Accuracy (%)")
ax.set_title("Accuracy vs Quantization Format: tanh+AvgPool vs ReLU+MaxPool")
ax.set_xticks(list(x))
ax.set_xticklabels(formats, rotation=15)
ax.set_ylim(min(min(tanh_acc), min(relu_acc)) - 5, 100)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("results/accuracy_comparison_plot.png", dpi=150)
print("\nPlot saved to results/accuracy_comparison_plot.png")
plt.show()