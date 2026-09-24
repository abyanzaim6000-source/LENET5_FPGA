# IP internal structure — C1 / C3 (INT8 fixed-point)

| File | Content |
|---|---|
| `c1_internal_structure.png` | `conv_c1_int8_fixedpoint` (`conv_c1_int8_fixedpoint_proj/solution1`): 8-PE MAC array |
| `c3_internal_structure.png` | `conv_c3_int8_fixedpoint` (`conv_c3_int8_fixedpoint_proj/solution1`): 6-PE partial-sum split with LUT index decode |
| `IP_Internal_Structure.pdf` | Both diagrams with captions, one per page |
| `make_diagrams.py` | Regenerates all three (`python3 Docs/ip_internal_structure/make_diagrams.py`, needs matplotlib) |

## Where the numbers come from

- **Structure** (PE_COUNT, loop bounds, pragmas, buffer shapes, LUT tables, interfaces) comes from
  `hls/conv_c1/src/conv_c1_int8_fixedpoint.{h,cpp}` and `hls/conv_c3/src/conv_c3_int8_fixedpoint.{h,cpp}`.
- **Resources and timing** (DSP/BRAM/FF/LUT, latency, slack/Fmax, Bind Op core names and DSP split,
  C3's achieved II) come from the csynth / Bind Op Report figures for those two solutions, as
  recorded in `Results/hls_results.md` (sections "Follow-up — genuine fixed-point requantization" and
  "C3 Convolution — INT8 Quantization").
- The raw `solution1/syn/report/csynth.rpt` files are **not in the repo**: `.gitignore` excludes every
  `*proj*/` directory. To check the diagrams against the raw reports, re-run
  `run_hls_int8_fixedpoint_pynqz2.tcl` in `hls/conv_c1/` or `hls/conv_c3/`.

| | C1 | C3 |
|---|---|---|
| PEs | 8 × `mac_muladd_8s_8s_32s_32_4_1` (1 DSP each) | 6 × `mac_muladd_8s_8s_20s` (1 DSP each) |
| Rescale multiplier | 1 × `mul_31ns_32s_63_2_1` (3 DSP) | 1 × `mul_31ns_32s_63` (3 DSP) |
| DSP / BRAM_18K / FF / LUT | 11 / 5 / 9,889 / 16,124 | 9 / 0 / 1,523 / 4,919 |
| Latency (cycles) | 124,991 | 150,401 |
| MAC loop | `PIPELINE II=1` pragma (achieved II for this solution is not recorded) | `VITIS_LOOP_61_5`: trip count 25, II 3 achieved vs. 1 targeted |
| Slack @ 10 ns / est. Fmax | 0.00 ns / 136.99 MHz | 0.00 ns / 136.99 MHz |
