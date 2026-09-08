# lenet5_system bitstream — PYNQ-Z2 build, 2026-09-07/08

Sibling of `Lenet5_fpga_top/`, rebuilt to target the actual PYNQ-Z2 board
(part `xc7z020clg400-1`, board file `tul.com.tw:pynq-z2:part0:1.0`) instead
of the `xc7z020iclg484-1L` part the original project targets. Different
package (`clg400` vs `clg484`) and different grade — not interchangeable
hardware.

## Board file

Vivado 2022.2 had no PYNQ-Z2 board file installed by default, and it isn't
in Digilent's own `vivado-boards` repo (PYNQ-Z2 is a TUL Corporation board,
not Digilent). Installed from
[xupsh/pynq-supported-board-file](https://github.com/xupsh/pynq-supported-board-file)
(`pynq-z2/A.0/`) into Vivado's standard board_files location:
`C:/Xilinx/Vivado/2022.2/data/boards/board_files/pynq-z2/A.0/`. This is a
one-time, machine-local install — it isn't part of this repo and needs
redoing on any other Vivado install building this project.

## HLS IP — no re-export needed

`hls/lenet5_top`'s exported `component.xml` declares
`<xilinx:supportedFamilies><xilinx:family>zynq</xilinx:family>` — family-level,
not part/package-specific. The same export the `Lenet5_fpga_top` project
uses is reused here unmodified (via the same `ip_repo_paths` entry).

## Build scripts

Run from this directory, in order:
1. `vivado -mode batch -source create_project_pynqz2.tcl` — one-time project creation.
2. `vivado -mode batch -source build_lenet5_bd_pynqz2.tcl` — builds `lenet5_system`
   (same topology as `Lenet5_fpga_top/build_lenet5_bd.tcl`), with the PS7 brought up
   via `apply_board_preset` so DDR3 timing (board's MT41J256M16) and clocks come from
   the board file instead of generic Zynq-7020 defaults.
3. `vivado -mode batch -source prepare_synth_pynqz2.tcl` — generates the wrapper and
   runs `generate_target`. **Retry this step (fresh process each time) if it fails**
   with `ERROR: [Common 17-232] Could not create slave interpreter '::ipgen_iptclns'`
   — a Windows Tcl interpreter/TLS-slot exhaustion that hit this board_part-enabled
   project intermittently (reproduced twice), never the original non-board project.
   It's cheap to retry (~1 min) since it's kept separate from the actual synth+impl run.
4. `vivado -mode batch -source build_bitstream_pynqz2.tcl` — synthesis + implementation
   through `write_bitstream`, once step 3 has succeeded.

(Skipping the explicit `generate_target` call entirely — the first workaround tried for
the interpreter crash — is NOT viable: it leaves the wrapper unrecognized as a synthesis
source, `synth_1` fails immediately with `ERROR: [Synth 8-439] module
'lenet5_system_wrapper' not found`.)

**Result: SUCCESS.** Timing met, but with far less margin than the
`xc7z020iclg484-1L` build (WNS +3.659 ns there): **WNS +0.033169 ns, WHS
+0.020045 ns**. All resources fit the xc7z020clg400-1 budget (same fabric,
near-identical utilization to the other part as expected):

| Resource | Used | Available | Utilization |
|---|---|---|---|
| Slice LUTs | 29,420 | 53,200 | 55.3% |
| Slice Registers | 41,107 | 106,400 | 38.6% |
| DSP48E1 | 24 | 220 | 10.9% |
| Block RAM Tile | 119.5 | 140 | 85.4% |

`drc.rpt`: 0 errors, 65 warnings, 28 advisories — same generic HLS-datapath pipelining/DSP
advisories as the other build, unrelated to the part change.

The bitstream (`lenet5_system_wrapper.bit`) is written to
`Lenet5_fpga_top_pynqz2.runs/impl_1/lenet5_system_wrapper.bit`, excluded from git by the
repo's generic `*.runs/` ignore rule. Re-run steps 3–4 above to recreate it.

## Timing margin — worth watching

+0.033 ns WNS is a hair's-breadth pass, not a comfortable one (two orders of magnitude
tighter than the other part's build). Re-running implementation with a different seed/
directive, or any future change to `lenet5_top`'s II/pipelining, could tip this negative.
If that happens, the fix is standard: try `-directive Explore` on `place_design`/
`route_design`, or revisit clock frequency assumptions from the board's DDR3 preset.

## Pre-existing part mismatch (unrelated to this build, noted for awareness)

The *committed* `Lenet5_fpga_top` project targets `xc7z020iclg484-1L` (industrial temp,
low-power grade), while `hls/lenet5_top/run_hls.tcl` exports against `xc7z020clg484-1`
(commercial grade) — a pre-existing mismatch between the HLS export and the Vivado
project part that predates and is unrelated to this PYNQ-Z2 work. Harmless in practice
since the HLS IP is family-level only (see above), but worth reconciling at some point.
