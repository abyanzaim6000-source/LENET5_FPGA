"""
Writes each layer's activations out as readable feature-map matrices, for
BOTH sides of the lenet5_top_int8 verification -- two files per layer, so
the Python reference and the HLS C-simulation can be opened side by side
and compared by eye rather than only through a numeric summary.

This is a pure reshaping/formatting pass over data that already exists:

  * the HLS side comes from the flat per-layer dumps written by
    lenet5_top_int8_tb.cpp during C-simulation (hls_c1_out.txt, ...);
  * the Python side comes from the chained Python int8 reference
    activations (c1_expected ... expected_output) in
    lenet5_top_int8_test_data.h.

Both are stored flat in row-major (H, W, C) order, which is what makes
reshaping them to the same (H, W, C) grid well-defined -- the element-wise
agreement itself is what compare_hls_vs_python.py checks.

Note the Python side is the INT8 reference chain, not the float32
activations from src/manual_inference_relu.py: the HLS kernels compute in
int8, so the int8 chain is the only reference on the same numeric scale.
Putting float32 activations beside int8 ones would show a difference at
every single value and tell you nothing.

Output format, per layer:
  * conv/pool layers (H, W, C) -- one labeled block per output channel,
    "Feature Map <c>:" followed by that channel's full H x W grid of
    numbers, then a blank line before the next block;
  * dense layers (N,) -- no spatial shape, so a single header followed by
    the plain list of values, one per line.

This is an ADDITIONAL output alongside compare_hls_vs_python.py's summary
table, not a replacement -- that table remains the quick pass/fail check.

Usage (after running the C-simulation):
    python3 hls/lenet5_top_int8/tb/dump_feature_maps.py
    python3 hls/lenet5_top_int8/tb/dump_feature_maps.py --dump-dir <dir> --out-dir <dir>
"""
import argparse
import os
import sys

import numpy as np

TB_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TB_DIR)

# Reuse the reference-header parsing, dump loading and layer table that
# compare_hls_vs_python.py already defines, so the two scripts can never
# drift apart on shapes, filenames or where the reference comes from.
import compare_hls_vs_python as cmp

DEFAULT_OUT_DIR = os.path.join(TB_DIR, "..", "feature_maps")


def format_value(v, is_float):
    """int8 values are padded to a fixed width so grid columns line up
    when two files are read side by side; softmax probabilities are
    written at full float32 precision instead."""
    if is_float:
        return f"{v:.9e}"
    return f"{int(round(v)):4d}"


def write_feature_maps(path, flat, shape, is_float, layer):
    lines = []
    if len(shape) == 3:
        H, W, C = shape
        grid = flat.reshape(H, W, C)
        for ch in range(C):
            lines.append(f"Feature Map {ch}:")
            for r in range(H):
                lines.append(" ".join(format_value(grid[r, c, ch], is_float)
                                      for c in range(W)))
            lines.append("")
    else:
        # Dense layer: no spatial shape, so a single header and the plain
        # list of values.
        (N,) = shape
        lines.append(f"{layer} output ({N} values):")
        for i in range(N):
            lines.append(format_value(flat[i], is_float).strip())
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    return len(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--dump-dir", default=None,
                    help="directory holding the testbench's hls_*_out.txt dumps "
                         "(default: the csim build dir, else the current directory)")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                    help="where to write the *_fmaps.txt files "
                         "(default: hls/lenet5_top_int8/feature_maps)")
    args = ap.parse_args()

    header = cmp.read_header(cmp.HEADER_PATH)

    # The Python reference side needs no C-simulation, so a missing dump
    # directory is reported but does not stop that half being written.
    try:
        dump_dir = cmp.find_dump_dir(args.dump_dir)
    except SystemExit:
        dump_dir = None

    out_dir = os.path.normpath(args.out_dir)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    print()
    print("Writing feature-map matrices for lenet5_top_int8 (Python int8 reference vs. HLS csim)")
    print(f"Python reference: {os.path.relpath(cmp.HEADER_PATH, os.getcwd())}")
    print(f"HLS dumps:        {dump_dir if dump_dir else 'NOT FOUND -- writing the Python side only'}")
    print(f"Output directory: {out_dir}")
    print()

    print("=" * 88)
    print(f"{'Layer':8s}{'Shape':12s}{'Blocks':>14s}  {'Python file':26s}{'HLS file':24s}")
    print("-" * 88)

    missing = []
    for name, filename, ref_name, shape, tol in cmp.LAYERS:
        is_float = tol > 0.0
        blocks = shape[2] if len(shape) == 3 else 1

        ref = cmp.parse_array(header, ref_name)
        py_name = f"{name.lower()}_python_fmaps.txt"
        write_feature_maps(os.path.join(out_dir, py_name), ref, shape, is_float, name)

        hls_name = f"{name.lower()}_hls_fmaps.txt"
        hls_written = "-"
        if dump_dir:
            path = os.path.join(dump_dir, filename)
            if os.path.exists(path):
                hls, _ = cmp.load_dump(path)
                if hls.size == int(np.prod(shape)):
                    write_feature_maps(os.path.join(out_dir, hls_name), hls,
                                       shape, is_float, name)
                    hls_written = hls_name
                else:
                    missing.append(f"{name}: dump has {hls.size} values, "
                                   f"expected {int(np.prod(shape))}")
            else:
                missing.append(f"{name}: {filename} not found")
        else:
            missing.append(f"{name}: no dump directory")

        shape_str = "x".join(str(d) for d in shape)
        blocks_str = (f"{blocks} channels" if len(shape) == 3
                      else "plain list")
        print(f"{name:8s}{shape_str:12s}{blocks_str:>14s}  "
              f"{py_name:26s}{hls_written:24s}")

    print("=" * 88)
    print()

    if missing:
        print("HLS side incomplete -- the Python side was still written:")
        for m in missing:
            print(f"  {m}")
        print("\nRun the C-simulation to produce the HLS dumps:")
        print("  cd hls/lenet5_top_int8 && vitis_hls -f run_hls_int8_pynqz2.tcl")
        return 1

    print("Both sides written. Open a matching pair side by side, e.g.:")
    print(f"  {os.path.join(out_dir, 'c1_python_fmaps.txt')}")
    print(f"  {os.path.join(out_dir, 'c1_hls_fmaps.txt')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
