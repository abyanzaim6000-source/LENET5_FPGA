"""
Renders the *_python_fmaps.txt / *_hls_fmaps.txt pairs written by
dump_feature_maps.py into one comparison PNG per layer, for slides/reports
that need a visual (not just a numeric table) proof that the real Vitis HLS
C-simulation matches the Python int8 reference.

This is a pure visualization pass over files that already exist -- it does
not re-run C-simulation or regenerate any reference data. Run
dump_feature_maps.py first if hls/lenet5_top_int8/feature_maps/ is empty or
stale.

Layout:
  * spatial layers (C1, S2, C3, S4): one figure, 3 rows x C columns --
    Python heatmap / HLS heatmap / (Python - HLS) diff heatmap, one column
    per channel, shared colormap+scale across the Python/HLS rows so they
    are directly comparable by eye.
  * dense layers (C5, F6, Output): one figure, 2 stacked subplots -- values
    overlaid (Python vs HLS) on top, per-element difference on the bottom
    with the y-axis scaled to actually show the difference even when it is
    tiny (e.g. Output's ~1e-7 softmax rounding).

Usage (after dump_feature_maps.py has populated hls/lenet5_top_int8/feature_maps/):
    python3 hls/lenet5_top_int8/tb/visualize_feature_maps.py
    python3 hls/lenet5_top_int8/tb/visualize_feature_maps.py --fmaps-dir <dir> --out-dir <dir>
"""
import argparse
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TB_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FMAPS_DIR = os.path.join(TB_DIR, "..", "feature_maps")
DEFAULT_OUT_DIR = os.path.join(DEFAULT_FMAPS_DIR, "visuals")

# (layer name, shape, tolerance, kind) -- kind picks the spatial-grid layout
# vs. the stacked-line-plot layout below. Kept independent of
# compare_hls_vs_python.LAYERS so this script has no import-time dependency
# on the C-simulation/header machinery, only on the *_fmaps.txt files.
LAYERS = [
    ("C1",     (28, 28, 6),  0.0,   "spatial"),
    ("S2",     (14, 14, 6),  0.0,   "spatial"),
    ("C3",     (10, 10, 16), 0.0,   "spatial"),
    ("S4",     (5, 5, 16),   0.0,   "spatial"),
    ("C5",     (120,),       0.0,   "dense"),
    ("F6",     (84,),        0.0,   "dense"),
    ("Output", (10,),        1e-5,  "dense"),
]


def load_spatial_fmap(path, shape):
    """Parse a 'Feature Map <c>:' block-formatted file back into an
    (H, W, C) array."""
    H, W, C = shape
    grid = np.zeros((H, W, C), dtype=np.float64)
    ch, row = None, 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.match(r"Feature Map (\d+):$", line)
            if m:
                ch, row = int(m.group(1)), 0
                continue
            vals = [float(x) for x in line.split()]
            if ch is None or len(vals) != W:
                sys.exit(f"ERROR: unexpected line while parsing {path}: {line!r}")
            grid[row, :, ch] = vals
            row += 1
    return grid


def load_dense_fmap(path, shape):
    """Parse a '<Layer> output (N values):' header + one-value-per-line
    file back into a flat (N,) array."""
    (N,) = shape
    with open(path) as f:
        lines = f.read().splitlines()
    values = [float(line.strip()) for line in lines[1:] if line.strip()]
    arr = np.array(values, dtype=np.float64)
    if arr.size != N:
        sys.exit(f"ERROR: {path} has {arr.size} values, expected {N}")
    return arr


def plot_spatial_layer(name, shape, py, hls, out_path):
    H, W, C = shape
    diff = py - hls
    max_abs_diff = float(np.max(np.abs(diff))) if diff.size else 0.0

    vmin = min(py.min(), hls.min())
    vmax = max(py.max(), hls.max())
    dmax = max_abs_diff if max_abs_diff > 0 else 1e-9

    fig, axes = plt.subplots(3, C, figsize=(2.1 * C, 7.2), dpi=150)

    row_labels = ["Python", "HLS", "Diff"]
    im_py = im_hls = im_diff = None
    for ch in range(C):
        im_py = axes[0, ch].imshow(py[:, :, ch], cmap="viridis", vmin=vmin, vmax=vmax)
        axes[0, ch].set_title(f"ch {ch}", fontsize=9)

        im_hls = axes[1, ch].imshow(hls[:, :, ch], cmap="viridis", vmin=vmin, vmax=vmax)

        ch_diff = diff[:, :, ch]
        ch_max = float(np.max(np.abs(ch_diff)))
        im_diff = axes[2, ch].imshow(ch_diff, cmap="RdBu_r", vmin=-dmax, vmax=dmax)
        axes[2, ch].set_title(f"max|Δ|={ch_max:g}", fontsize=8)

        for r in range(3):
            axes[r, ch].set_xticks([])
            axes[r, ch].set_yticks([])

    for r, label in enumerate(row_labels):
        axes[r, 0].set_ylabel(label, fontsize=12, fontweight="bold")

    fig.colorbar(im_py, ax=axes[0, :].tolist(), fraction=0.015, pad=0.01)
    fig.colorbar(im_hls, ax=axes[1, :].tolist(), fraction=0.015, pad=0.01)
    fig.colorbar(im_diff, ax=axes[2, :].tolist(), fraction=0.015, pad=0.01)

    exact_str = "bit-exact" if max_abs_diff == 0 else f"max abs diff: {max_abs_diff:g}"
    fig.suptitle(f"{name} — {C} channels, {exact_str}", fontsize=14, fontweight="bold")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return max_abs_diff


def plot_dense_layer(name, shape, py, hls, tol, out_path):
    (N,) = shape
    diff = py - hls
    max_abs_diff = float(np.max(np.abs(diff))) if diff.size else 0.0
    x = np.arange(N)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(max(10, N * 0.14), 8), dpi=150,
        gridspec_kw={"height_ratios": [2, 1]})

    ax1.plot(x, py, "o-", label="Python", alpha=0.7, markersize=4, linewidth=1)
    ax1.plot(x, hls, "x--", label="HLS", alpha=0.7, markersize=6, linewidth=1)
    ax1.set_xlabel("index")
    ax1.set_ylabel("value")
    ax1.set_title(f"{name} values — Python vs HLS ({N} values)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.bar(x, diff, color="tab:red", width=0.8)
    ax2.set_xlabel("index")
    ax2.set_ylabel("Python − HLS")
    ax2.grid(alpha=0.3)

    if max_abs_diff > 0:
        margin = max_abs_diff * 1.5
        ax2.set_ylim(-margin, margin)
        ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax2.set_title(f"per-element difference (max abs diff: {max_abs_diff:.3e})")
    else:
        ax2.set_ylim(-1e-9, 1e-9)
        ax2.text(0.5, 0.5, "all differences exactly 0 (bit-exact)",
                  transform=ax2.transAxes, ha="center", va="center", fontsize=10)
        ax2.set_title("per-element difference (max abs diff: 0)")

    tol_str = f" (within {tol:.0e} tolerance)" if tol > 0 else " (bit-exact)"
    fig.suptitle(f"{name} — {N} values, max abs diff: {max_abs_diff:.3e}{tol_str}",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return max_abs_diff


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--fmaps-dir", default=DEFAULT_FMAPS_DIR,
                    help="directory holding the *_python_fmaps.txt / *_hls_fmaps.txt "
                         "files written by dump_feature_maps.py")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                    help="where to write the *_comparison.png files "
                         "(default: hls/lenet5_top_int8/feature_maps/visuals)")
    args = ap.parse_args()

    fmaps_dir = os.path.normpath(args.fmaps_dir)
    out_dir = os.path.normpath(args.out_dir)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    print()
    print("Rendering feature-map comparison PNGs for lenet5_top_int8")
    print(f"Feature maps dir: {fmaps_dir}")
    print(f"Output directory: {out_dir}")
    print()
    print("=" * 78)
    print(f"{'Layer':8s}{'Shape':14s}{'Max abs diff':>16s}{'PNG':>28s}")
    print("-" * 78)

    missing = []
    for name, shape, tol, kind in LAYERS:
        py_path = os.path.join(fmaps_dir, f"{name.lower()}_python_fmaps.txt")
        hls_path = os.path.join(fmaps_dir, f"{name.lower()}_hls_fmaps.txt")
        if not os.path.exists(py_path) or not os.path.exists(hls_path):
            missing.append(f"{name}: missing {py_path if not os.path.exists(py_path) else hls_path}")
            continue

        out_name = f"{name.lower()}_comparison.png"
        out_path = os.path.join(out_dir, out_name)

        if kind == "spatial":
            py = load_spatial_fmap(py_path, shape)
            hls = load_spatial_fmap(hls_path, shape)
            max_abs_diff = plot_spatial_layer(name, shape, py, hls, out_path)
        else:
            py = load_dense_fmap(py_path, shape)
            hls = load_dense_fmap(hls_path, shape)
            max_abs_diff = plot_dense_layer(name, shape, py, hls, tol, out_path)

        shape_str = "x".join(str(d) for d in shape)
        print(f"{name:8s}{shape_str:14s}{max_abs_diff:>16.3e}{out_name:>28s}")

    print("=" * 78)
    print()

    if missing:
        print("Some layers were skipped -- run dump_feature_maps.py first:")
        for m in missing:
            print(f"  {m}")
        return 1

    print(f"All {len(LAYERS)} comparison PNGs written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
