"""
Layer-by-layer comparison of the lenet5_top_int8 C-simulation against the
Python int8 reference, for one real MNIST image.

Two independent computations of the same forward pass are tabulated
against each other:

  * the HLS side -- the ACTUAL values computed by the INT8 HLS C++ kernels
    during C-simulation, dumped one plain text file per layer by
    lenet5_top_int8_tb.cpp (hls_c1_out.txt ... hls_output_out.txt);
  * the Python side -- the chained Python int8 reference activations
    (c1_expected ... f6_expected, expected_output) computed by
    generate_test_data_int8.py from the real MNIST test[0] image and the
    real Models/lenet5_relu.keras weights, and emitted into
    lenet5_top_int8_test_data.h.

Reading the reference straight out of the generated header (rather than
re-running the Keras/torchvision pipeline) is deliberate: it needs nothing
but NumPy, so it runs on the HLS machine right after csim, and it compares
against exactly the reference the testbench itself was built against.

Expected result: C1/S2/C3/S4/C5/F6 bit-exact (max difference 0 -- they are
pure integer arithmetic, so there is no rounding left to disagree about),
and the final softmax equal to within float32 rounding (~1e-7, checked
against the same 1e-5 tolerance the testbench and dense_output_int8_tb.cpp
already use, softmax being the one deliberately non-fixed-point stage).

Usage (after running the C-simulation):
    python3 hls/lenet5_top_int8/tb/compare_hls_vs_python.py
    python3 hls/lenet5_top_int8/tb/compare_hls_vs_python.py --dump-dir <dir>

Exits 0 if every layer matches, 1 otherwise.
"""
import argparse
import os
import re
import sys

import numpy as np

TB_DIR = os.path.dirname(os.path.abspath(__file__))
HEADER_PATH = os.path.join(TB_DIR, "lenet5_top_int8_test_data.h")

# Softmax is the one stage that is deliberately NOT fixed-point, so it is
# held to a float tolerance instead of bit-exactness -- the same 1e-5
# standard lenet5_top_int8_tb.cpp itself uses.
SOFTMAX_TOL = 1e-5

# (layer name, dumped filename, reference array in the header, shape, tolerance)
LAYERS = [
    ("C1",     "hls_c1_out.txt",     "c1_expected",     (28, 28, 6),  0.0),
    ("S2",     "hls_s2_out.txt",     "s2_expected",     (14, 14, 6),  0.0),
    ("C3",     "hls_c3_out.txt",     "c3_expected",     (10, 10, 16), 0.0),
    ("S4",     "hls_s4_out.txt",     "s4_expected",     (5, 5, 16),   0.0),
    ("C5",     "hls_c5_out.txt",     "c5_expected",     (120,),       0.0),
    ("F6",     "hls_f6_out.txt",     "f6_expected",     (84,),        0.0),
    ("Output", "hls_output_out.txt", "expected_output", (10,),        SOFTMAX_TOL),
]


def read_header(path):
    if not os.path.exists(path):
        sys.exit(f"ERROR: reference header not found: {path}\n"
                 f"       Regenerate it with tb/generate_test_data_int8.py.")
    with open(path) as f:
        return f.read()


def parse_define(text, name):
    m = re.search(r"#define\s+" + name + r"\s+(-?\d+)", text)
    return int(m.group(1)) if m else None


def parse_array(text, name):
    """Pull one `static const <type> <name>[dims] = { ... };` array out of the
    generated header and return it flat, in the order it was emitted (which
    is NumPy .flatten() order -- see the emit helpers in
    generate_test_data_int8.py)."""
    m = re.search(
        r"static\s+const\s+(?:signed\s+char|unsigned\s+char|char|int|float)\s+"
        + name + r"\s*((?:\[\d+\])+)\s*=\s*\{(.*?)\};",
        text, re.S)
    if not m:
        sys.exit(f"ERROR: could not find reference array '{name}' in "
                 f"{os.path.basename(HEADER_PATH)}")
    dims = tuple(int(d) for d in re.findall(r"\[(\d+)\]", m.group(1)))
    values = [v.strip().rstrip("f") for v in m.group(2).split(",")]
    values = [v for v in values if v]
    arr = np.array([float(v) for v in values], dtype=np.float64)
    expected = int(np.prod(dims))
    if arr.size != expected:
        sys.exit(f"ERROR: reference array '{name}' has {arr.size} values, "
                 f"expected {expected} for shape {dims}")
    return arr


def load_dump(path):
    """Load one layer dump written by lenet5_top_int8_tb.cpp: '#' comment
    lines carrying the layer/shape metadata, then one value per line."""
    values, meta = [], {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                if ":" in line:
                    k, v = line[1:].split(":", 1)
                    meta[k.strip()] = v.strip()
                continue
            values.append(float(line))
    return np.array(values, dtype=np.float64), meta


def find_dump_dir(explicit):
    """Locate the directory holding the C-simulation's dumps. Vitis HLS runs
    csim from <proj>/solution1/csim/build, which is where the testbench's
    plain relative filenames land."""
    if explicit:
        candidates = [explicit]
    else:
        candidates = [
            os.path.join(TB_DIR, "..", "lenet5_top_int8_proj",
                         "solution1", "csim", "build"),
            os.getcwd(),
        ]
    for d in candidates:
        if os.path.exists(os.path.join(d, LAYERS[0][1])):
            return os.path.normpath(d)
    searched = "\n".join(f"  {os.path.normpath(d)}" for d in candidates)
    sys.exit("ERROR: no C-simulation dumps found. Looked in:\n" + searched +
             "\n       Run the C-simulation first:\n"
             "         cd hls/lenet5_top_int8 && vitis_hls -f run_hls_int8_pynqz2.tcl\n"
             "       or point this script at the dumps with --dump-dir.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--dump-dir", default=None,
                    help="directory holding the testbench's hls_*_out.txt dumps "
                         "(default: the csim build dir, else the current directory)")
    args = ap.parse_args()

    dump_dir = find_dump_dir(args.dump_dir)
    header = read_header(HEADER_PATH)

    true_label = parse_define(header, "TRUE_LABEL")
    expected_class = parse_define(header, "EXPECTED_CLASS")

    print()
    print("Layer-by-layer verification: lenet5_top_int8 C-simulation vs. Python int8 reference")
    print(f"Image:            MNIST test[0], true label = {true_label}")
    print(f"HLS dumps:        {dump_dir}")
    print(f"Python reference: {os.path.relpath(HEADER_PATH, os.getcwd())}")
    print()

    print("=" * 84)
    print(f"{'Layer':10s}{'Shape':12s}{'Values':>8s}{'Max difference':>18s}"
          f"{'Tolerance':>13s}{'Result':>12s}")
    print("-" * 84)

    all_match = True
    hls_output = None

    for name, filename, ref_name, shape, tol in LAYERS:
        path = os.path.join(dump_dir, filename)
        if not os.path.exists(path):
            print(f"{name:10s}{'x'.join(str(d) for d in shape):12s}"
                  f"{'-':>8s}{'dump missing':>18s}{'-':>13s}{'NO MATCH':>12s}")
            all_match = False
            continue

        hls, _ = load_dump(path)
        ref = parse_array(header, ref_name)

        if hls.size != ref.size:
            print(f"{name:10s}{'x'.join(str(d) for d in shape):12s}"
                  f"{hls.size:>8d}{'size mismatch':>18s}{'-':>13s}{'NO MATCH':>12s}")
            all_match = False
            continue

        if name == "Output":
            hls_output = hls

        max_diff = float(np.max(np.abs(hls - ref))) if hls.size else 0.0
        ok = max_diff <= tol
        all_match = all_match and ok

        diff_str = f"{int(max_diff)}" if tol == 0.0 else f"{max_diff:.3e}"
        tol_str = "exact" if tol == 0.0 else f"{tol:.0e}"

        print(f"{name:10s}{'x'.join(str(d) for d in shape):12s}{hls.size:>8d}"
              f"{diff_str:>18s}{tol_str:>13s}{'MATCH' if ok else 'NO MATCH':>12s}")

    print("=" * 84)
    print()

    if hls_output is not None:
        hls_class = int(np.argmax(hls_output))
        print(f"Predicted class -- HLS: {hls_class}, Python int8 reference: "
              f"{expected_class}, MNIST true label: {true_label}")
        if hls_class != expected_class:
            all_match = False

    if all_match:
        print("RESULT: every layer matches the Python int8 reference "
              "(C1-F6 bit-exact, softmax within tolerance).")
        return 0

    print("RESULT: at least one layer did NOT match -- see the table above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
