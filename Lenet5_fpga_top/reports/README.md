# lenet5_system bitstream — 2026-09-07 build

`build_bitstream.tcl` (run via `vivado -mode batch -source build_bitstream.tcl` from this
directory) generates the `lenet5_system_wrapper` HDL wrapper, then runs synthesis and
implementation through `write_bitstream`.

**Result: SUCCESS.** Timing met (WNS +3.659 ns, WHS +0.008 ns), all resources fit the
xc7z020iclg484-1L budget:

| Resource | Used | Available | Utilization |
|---|---|---|---|
| Slice LUTs | 29,403 | 53,200 | 55.3% |
| Slice Registers | 41,096 | 106,400 | 38.6% |
| DSP48E1 | 24 | 220 | 10.9% |
| Block RAM Tile | 119.5 | 140 | 85.4% |

Full detail in `timing_summary.rpt` and `utilization.rpt` in this directory. `build_bitstream.tcl`
also produces a `drc.rpt` (0 errors, 65 warnings, 28 advisories) alongside them, not committed here
but reproducible the same way.

The bitstream itself (`lenet5_system_wrapper.bit`, ~3.9 MB) is written to
`Lenet5_fpga_top.runs/impl_1/lenet5_system_wrapper.bit`, which is excluded from git by
`*.runs/` in `.gitignore` — it's a build artifact, not source, and is cheap to regenerate from
the command above against the committed `.xci`/`.bda` sources. Re-run `build_bitstream.tcl` to
recreate it; the reports here can be regenerated the same way (or, if the routed checkpoint
still exists under `Lenet5_fpga_top.runs/impl_1/`, by `open_run impl_1 -name impl_1_routed`
followed by the three `report_*` commands, without re-running synthesis/implementation).
