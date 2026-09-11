# lenet5_int8 bitstream — PYNQ-Z2 build, 2026-09-11

Sibling of `Lenet5_fpga_top_pynqz2/` (the float32 PYNQ-Z2 build), targeting
the same board/part (`xc7z020clg400-1`, board file
`tul.com.tw:pynq-z2:part0:1.0`) but wrapping `hls/lenet5_top_int8`'s
combined INT8 IP instead of the float32 `lenet5_top`. A separate project
directory, not a mutation of the float32 one — same "old build stays
untouched" convention this whole project follows.

## Project/BD names shortened — Windows MAX_PATH

The project was first attempted as `Lenet5_fpga_top_int8_pynqz2` with BD
design name `lenet5_int8_system` (the direct INT8 analogue of the
float32 build's naming). `generate_target` failed there with
`ERROR: [IP_Flow 19-167] Failed to deliver one or more file(s)` on every
HLS-exported RAM `.dat` file — a genuine Windows 260-char `MAX_PATH`
ceiling, not a script bug: measured directly, the failing file's full
path was 272 characters, versus 237 for the float32 build's equivalent
path. The extra length comes from `lenet5_top_int8` being 5 characters
longer than `lenet5_top` and appearing TWICE in every HLS-generated RTL
filename, plus the longer project/BD names each appearing twice in the
nested `.gen/.../ip/...` path. Renaming to the current short project name
(`L5_int8_pynqz2`) and BD design name (`lenet5_int8`) brought the same
path down to 232 characters, and `generate_target` succeeded on the very
next attempt with no other changes. The stray, now-empty
`Lenet5_fpga_top_int8_pynqz2/` directory from the first attempt was left
in place (locked by a stray OS file handle at the time, harmless, empty).

## Board file / HLS IP reuse

Same one-time PYNQ-Z2 board file install as the float32 build (see that
project's own `reports/README.md`), and no HLS re-export needed for the
same reason (`hls/lenet5_top_int8`'s `component.xml` declares
`supportedFamilies={zynq}`, family-level not part/package-specific).

## Build scripts

Run from this directory, in order:
1. `vivado -mode batch -source create_project_pynqz2.tcl`
2. `vivado -mode batch -source build_lenet5_bd_pynqz2.tcl` — builds `lenet5_int8`
   (PS7 + `lenet5_top_int8_0`, PS7 brought up via `apply_board_preset`, S_AXI_HP0
   enabled explicitly, all 7 of the IP's `m_axi` masters — `gmem_img`, `gmem_c1`,
   `gmem_c3`, `gmem_c5`, `gmem_f6`, `gmem_out`, `gmem_res` — fanned manually into
   one shared `axi_interconnect` since the `axi4` automation rule doesn't handle
   an IP with this many master interfaces, same as the float32 build).
3. `vivado -mode batch -source prepare_synth_pynqz2.tcl` — generates the wrapper
   and runs `generate_target`. Succeeded on the first attempt this time (no retry
   needed for the `::ipgen_iptclns` interpreter crash the float32 build hit
   intermittently) — smaller design, shorter run.
4. `vivado -mode batch -source build_bitstream_pynqz2.tcl` — synthesis +
   implementation through `write_bitstream`, single-job (`-jobs 1`) for the same
   RAM-constraint reason as the float32 build.

**Result: SUCCESS**, and with noticeably MORE timing margin than the float32
PYNQ-Z2 build: **WNS +0.121060 ns, WHS +0.015439 ns** (float32: +0.033169 ns /
+0.020045 ns) — consistent with the INT8 HLS core's own higher estimated Fmax
(136.99 MHz vs 103.49 MHz, see `Results/hls_results.md`'s combined top-level
INT8 log). All user-specified timing constraints met, 0 failing endpoints on
setup, hold, or pulse-width.

| Resource | float32 (`Lenet5_fpga_top_pynqz2`) | int8 (this build) | Change |
|---|---|---|---|
| Slice LUTs | 29,420 / 53,200 (55.3%) | **16,927 / 53,200 (31.8%)** | **−42.5%** |
| Slice Registers | 41,107 / 106,400 (38.6%) | **22,562 / 106,400 (21.2%)** | **−45.1%** |
| Block RAM Tile | 119.5 / 140 (85.4%) | **36 / 140 (25.7%)** | **−69.9%** |
| DSP48E1 | 24 / 220 (10.9%) | **46 / 220 (20.9%)** | **+91.7% (worse)** |

Fits comfortably on every metric. LUT/FF/BRAM all drop sharply at the full
system level too (not just in isolated HLS csynth estimates), same
mechanism documented throughout `Results/hls_results.md`'s INT8 conversion
log: narrower integer datapaths and buffers throughout. DSP goes up, not
down, same already-documented reason as the HLS-level combined-top finding
— `dense_output_int8`'s float softmax finish plus every conv/dense layer's
own fixed-point rescale multiply cost more DSP than the float32 layers'
free `ReLU` compare ever needed; 46/220 (20.9%) still leaves ample headroom.

`drc.rpt`: 0 errors, 78 violations, all `Warning`/`Advisory` severity — same
generic HLS-datapath DSP-pipelining/LUT-equation categories as the float32
build's own 93 (fewer here, not more, despite this build using more DSPs —
these warnings track DSP48 internal-register usage patterns, not DSP count).

The bitstream (`lenet5_int8_wrapper.bit`) is written to
`L5_int8_pynqz2.runs/impl_1/lenet5_int8_wrapper.bit`, excluded from git by
the repo's generic `*.runs/` ignore rule. Re-run steps 3–4 above to
recreate it.
