# Layer-by-Layer Verification — INT8 LeNet-5 Top (`lenet5_top_int8`)

**Status: Confirmed via real Vitis HLS C-simulation (2026-09-16)**

Layer-by-layer comparison of the `lenet5_top_int8` C-simulation (Vitis HLS
2022.2, part `xc7z020clg400-1`) against the chained Python INT8 reference,
for MNIST test image `test[0]` (true label 7).

Two independent computations of the same forward pass were compared:

- **HLS side** — the actual values computed by the INT8 HLS C++ kernels
  during `csim_design`, dumped one plain-text file per layer by
  `lenet5_top_int8_tb.cpp`.
- **Python side** — the chained Python INT8 reference activations computed
  by `generate_test_data_int8.py` from the real MNIST image and the real
  `Models/lenet5_relu.keras` weights, emitted into
  `lenet5_top_int8_test_data.h`.

Reproduced with:

```bash
cd hls/lenet5_top_int8
vitis_hls -f run_hls_int8_pynqz2.tcl
python3 tb/compare_hls_vs_python.py
```

## Result

```
====================================================================================
Layer     Shape         Values    Max difference    Tolerance      Result
------------------------------------------------------------------------------------
C1        28x28x6         4704                 0        exact       MATCH
S2        14x14x6         1176                 0        exact       MATCH
C3        10x10x16        1600                 0        exact       MATCH
S4        5x5x16           400                 0        exact       MATCH
C5        120              120                 0        exact       MATCH
F6        84                84                 0        exact       MATCH
Output    10                10         1.193e-07        1e-05       MATCH
====================================================================================

Predicted class -- HLS: 7, Python int8 reference: 7, MNIST true label: 7
RESULT: every layer matches the Python int8 reference (C1-F6 bit-exact, softmax within tolerance).
```

C1 through F6 are pure integer arithmetic and match **bit-exact** (max
difference 0). The final softmax stage (`Output`) is the one deliberately
non-fixed-point stage and matches the Python reference to within float32
rounding (1.193e-07), well inside the 1e-5 tolerance.

The C-simulation itself also reported:

```
TEST PASSED -- every intermediate INT8 stage matches the Python reference chain bit-exact, final softmax matches within tolerance
```

C-synthesis (`csynth_design`) completed successfully for the same run,
estimated Fmax 136.99 MHz.

Per-layer feature maps for both sides are written to
`hls/lenet5_top_int8/feature_maps/` by
[`tb/dump_feature_maps.py`](../hls/lenet5_top_int8/tb/dump_feature_maps.py)
for manual inspection.
