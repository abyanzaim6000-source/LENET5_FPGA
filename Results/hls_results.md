Baseline (C1 convolution, no pragmas) — recorded numbers
Metric	Value
Latency	61,299 cycles
Latency (time)	613,000 ns ≈ 0.613 ms
Initiation Interval (II)	13
Trip Count	4,704 (= 28×28×6 output pixels)
DSP	10 (4% of chip)
FF	5,303 (4%)
LUT	5,639 (10%)
BRAM	0 used
Timing slack	-1.12 (negative → see below)
Pragmas used	none


## C1 Convolution — HLS Optimization Log

| Experiment | Latency (cycles) | II | DSP | FF | LUT | Notes |
|---|---|---|---|---|---|---|
| Baseline (no pragmas) | 61,299 | 13 | 10 | 5,303 | 5,639 | input & weights both single-port memory |
| ARRAY_PARTITION on weights only | 61,299 | 13 | 10 | 10,752 | 5,972 | No improvement — weights was not the bottleneck; input array (784 elements, still 2 ports) is now the limiting factor |

Comparing baseline vs after weights partitioning
Metric	Baseline	After partitioning weights	Change
Latency	61,299 cycles	61,299 cycles	No change
II	13	13	No change
DSP	10	10	No change
FF	5,303	10,752	+103%
LUT	5,639	5,972	+6%


third data point
| Line buffer (dim1+dim3 partition) | ~184k cycles → still limited | 3 | 10+ (50 mul, 50 add units generated) | ? | ? | line_buf still bottlenecked on column dimension |

data point 4
| Line buffer, full partition (all dims) | II=3, same bottleneck moved to output array | 3 | 50 mul, 50 add (fully unrolled MAC) | 137.82 MHz | output_r now the limiting array (6 simultaneous channel writes, 2 ports) |

Full comparison table for your results file
markdown
## C1 Convolution — HLS Optimization Log

| Experiment | Latency (cycles) | II | Fmax (MHz) | DSP-related units | FF | LUT | Notes |
|---|---|---|---|---|---|---|---|
| Baseline (no pragmas) | 61,299 | 13 | 118.71 | 2 mul, 2 add | 5,303 | 5,639 | input & weights both single-port |
| ARRAY_PARTITION weights only | 61,299 | 13 | ~119 | 2 mul, 2 add | 10,752 | 5,972 | No improvement — input still bottleneck |
| Line buffer, partial partition | ~184k (II=3) | 3 | 137.82 | 50 mul, 50 add | — | — | line_buf column dim not partitioned |
| Line buffer, full partition (all dims) | — (II=3) | 3 | 137.82 | 50 mul, 50 add | — | — | output array now bottleneck |
| **Line buffer + output partition (FINAL)** | **~4,838 (784×1 + pipeline fill)** | **1** | **137.82** | **150 mul, 150 add** | — | — | **All constraints satisfied — full II=1 pipeline** |

## C1 Convolution — Summary (Baseline → Fully Pipelined)

| Stage | Latency (cycles) | II | Fmax (MHz) | Mult units | Add units | Key change |
|---|---|---|---|---|---|---|
| 1. Baseline (no pragmas) | 61,299 | 13 | 118.71 | 2 | 2 | none |
| 2. ARRAY_PARTITION on weights | 61,299 | 13 | 118.71 | 2 | 2 | weights fully partitioned (no effect — input was real bottleneck) |
| 3. Line buffer (partial partition) | — | 3 | 137.82 | 50 | 50 | line_buf partitioned on dims 1,3 only |
| 4. Line buffer (full partition) | — | 3 | 137.82 | 50 | 50 | line_buf fully partitioned; output array now bottleneck |
| 5. + output partitioned on channel dim | — (fill in) | **1** | 137.82 | 150 | 150 | ALL loop constraints satisfied |

**Speedup, baseline → final:** ~61,299 / (final latency) ≈ [compute once real number known] — roughly 12-13x fewer cycles.
**Resource cost:** 2 mul/add → 150 mul/add (75x increase in parallel MAC units) for that speedup.
**Interpretation:** matches the paper's own finding (Section IV-B) that memory partitioning has the single largest impact on latency/II in the convolution layer, at the direct cost of resource consumption.

PE_COUNT=8 data point
markdown
| PE_COUNT=8 systolic | Final II=4 (accumulation-limited) | 4 | 2 mul, 2 add (genuine sharing) | 104.80 MHz | Loop constraints NOT fully satisfied — timing margin tighter than fully-parallel version |

# HLS Optimization Log — C1 Convolution Layer

Target device: xc7z020iclg484-1L (Zynq-7020, ZedBoard) | Clock: 10ns (100MHz) throughout, kept constant for fair comparison.

## Stage-by-stage results

| Stage | Latency (cycles) | II | Fmax (MHz) | Mult units | Add units | Notes |
|---|---|---|---|---|---|---|
| 1. Baseline (no pragmas) | 61,299 | 13 | 118.71 | 2 | 2 | Both `input` and `weights` limited to 2 memory ports |
| 2. ARRAY_PARTITION on weights only | 61,299 | 13 | 118.71 | 2 | 2 | No improvement — `input` array (784 elements) was the real bottleneck, not weights |
| 3. Line buffer, partial partition | II=3 | 3 | 137.82 | 50 | 50 | line_buf partitioned on dims 1,3 only; column dim still bottlenecked |
| 4. Line buffer, full partition (all dims) | II=3 | 3 | 137.82 | 50 | 50 | line_buf fully partitioned; `output` array (6 simultaneous channel writes) now the bottleneck |
| 5. + output partitioned on channel dim | ~4,838 | **1** | 137.82 | 150 | 150 | **All loop constraints satisfied** — full II=1 pipeline, fully parallel MAC |
| 6. Systolic, PE_COUNT=8 (first attempt) | — | 1 | 137.82 | 150 | 150 | Bug: PIPELINE II=1 forced full unroll of the "sequential" PE loop — no actual resource savings achieved |
| 7. Systolic, PE_COUNT=8 (fixed) | II=4 (accumulation-limited) | 4 | 104.80 | **2** | **2** | Genuine resource sharing — 75x fewer mult/add units than stage 5, at the cost of throughput. Timing margin tighter (loop constraints NOT satisfied, though Fmax still clears 100MHz target) |
| 8. AXI interface conversion (first attempt) | — | — | — | 150+ AXI masters | — | Bug: fully-partitioned `weights`/`output` arrays cannot coexist with `m_axi` interfaces — HLS generated 150+ redundant AXI masters, synthesis took ~20 minutes |
| 9. AXI interface conversion (fixed, local-buffer pattern) | (re-verify final numbers) | (re-verify) | (re-verify) | (re-verify) | (re-verify) | Burst-copy AXI arrays into local partitioned on-chip buffers, compute locally, burst-copy back out. Clean 4-port `m_axi` interface (gmem0-3) |

## Key findings for the report

- **Memory partitioning alone can be a red herring**: partitioning `weights` (stage 2) had zero effect because `input` was the actual bottleneck — always identify the true limiting array via the II-violation warning, don't partition speculatively.
- **Fully-parallel (150 MAC units) vs. 8-PE systolic is a genuine resource/throughput tradeoff**: 75x fewer multiply/add units (2 vs 150) costs a 4x increase in II (1 → 4) and a ~24% drop in Fmax (137.82 → 104.80 MHz). This directly mirrors the paper's own emphasis on this exact tradeoff (Section IV-B).
- **AXI interfaces and full array partitioning are incompatible on the same array** — this is a fundamental HLS constraint, not a bug: `m_axi` implies off-chip/streamed data; full partitioning implies independent on-chip registers. The fix (burst-copy into local buffers) is the standard, correct HLS pattern for this situation.
- **Architecture difference from the paper**: our current AXI design has the IP's `m_axi` ports connecting through an AXI Interconnect directly to memory, without an explicit AXI DMA block. The paper explicitly uses DMA for streaming. This is a valid simplification, not an error, but worth noting as a planned refinement.

## Vivado integration status

- Block design created: Zynq7 Processing System + `conv_c1_systolic` custom IP + AXI Interconnect(s) + Processor System Reset.
- `s_axi_control` (AXI-Lite) connected via `ps7_0_axi_periph` interconnect to PS `M_AXI_GP0`.
- `m_axi_gmem0-3` (AXI4 master, one per array: input/weights/bias/output) connected via a second AXI Interconnect to PS memory-mapped port.
- Address map conflict encountered (auto-assigned overlapping base addresses across two automation passes) — resolved via Address Editor → Auto Assign Address, applied to entire address tree at once rather than incrementally.
- **Design Validated successfully, no errors**, as of [today's date].
- Not yet done: HDL wrapper generation, synthesis, bitstream generation, hardware/board deployment.

## Remaining planned work (not yet started)

- PE_COUNT = 1, 2, 4 sweep (only PE_COUNT=8 done so far) — needed for the full resource-vs-parallelism curve your guide requested.
- Generate bitstream for the current C1 design and confirm it fits/synthesizes at the Vivado (not just HLS-estimated) level.
- Repeat the full C1 methodology (baseline → line buffer → systolic → AXI) for C3, since only C1 has been done end-to-end.
- Reusable/parameterized IP generalization (same core handling both C1 and C3 via runtime parameters) — currently two separate hard-coded dimension sets (`IN_H`, `IN_W`, etc. in `conv_c1.h`), not yet parameterized.
- Add AXI DMA block if closer fidelity to the paper's exact architecture is desired.



## S2 Pooling — HLS Optimization Log

| Experiment | Latency (cycles) | II | Fmax equiv. (clock period) | DSP | FF | LUT | Notes |
|---|---|---|---|---|---|---|---|
| Baseline, AveragePooling (no pragmas) — **superseded** | 2,379 | 2 (auto-pipelined) | 7.256ns (~138MHz) | 7 | 1,192 | 1,809 | No MACs -- just add+divide. Auto-pipelined by HLS without explicit PIPELINE pragma. Numbers no longer apply — source has since switched to MaxPooling (below) to match `lenet5_relu.py`'s S2 |
| Baseline, MaxPooling (no pragmas) | *pending Vitis HLS run* | *pending* | *pending* | *pending* | *pending* | *pending* | Comparison-based (no divide unit, no DSP expected). Functionally verified via local g++ testbench (TEST PASSED), not yet re-run through Vitis HLS |



## C3 Convolution — HLS Optimization Log

Target device: xc7z020iclg484-1L (Zynq-7020, ZedBoard) | Clock: 10ns (100MHz), same as C1/S2.

| Experiment | Latency (cycles) | Interval (cycles) | Inner-loop II (achieved/target) | DSP | FF | LUT | BRAM_18K | Fmax (MHz) | Notes |
|---|---|---|---|---|---|---|---|---|---|
| **Baseline (no pragmas)** | **1,225,601** | **1,225,602** | **5 / 1** | **5** | **722** | **1,622** | **0** | **137.82** | csim: TEST PASSED (corner value 150). Top-level function not pipelined (Pipeline Type: no) — Interval ≈ Latency. Inner 150-iteration MAC loop (`VITIS_LOOP_17_4_18_5_19_6`) is pipelined but capped at II=5, not II=1 |

**Bottleneck diagnosis — accumulator-limited, NOT memory-port-limited (differs from C1's baseline pattern):**
All 4 II-violation warnings (attempted II=1 through II=4) trace to the same cause — a carried dependence on the scalar accumulator `acc` (`acc_1_write_ln19` store vs. `acc_1_load` load, `conv_c3.cpp:19-20`), not port contention on `input`/`weights` like C1's baseline. The 32-bit float adder (`fadd_32ns_..._5`) has a 5-cycle latency, and each MAC iteration must read back the *previous* iteration's `acc` before adding — so II bottoms out exactly at 5, matching the adder's latency. `input_r`/`weights`/`bias`/`output_r` were all mapped to plain single-port `ap_memory` with no partitioning conflict noted in the log.

**Architecture differences from C1's baseline** (relevant to how the optimization arc will differ):
- Input is 14×14×6 (S2's pooled output), not 28×28×1 — 6x deeper MAC inner loop before any unrolling.
- Padding is "valid", not "same" — no boundary zero-checks, so the line-buffer variant won't need the padding-validity logic that C1's did.
- Output is 10×10×16 (vs C1's 28×28×6) — smaller spatial extent, more output channels. Confirmed: the true bottleneck is **not** the `input`/`weights` arrays at all (unlike C1's baseline) — it's the serial floating-point accumulation chain.

| **SUPERSEDED — see LUT-decode version below — Partial-sum split (PE_COUNT=6, mac_idx decode)**, `conv_c3_partialsum.cpp` | 160,078 | 160,079 | 4 / 1 | 45 | 41,020 | 31,870 | 0 | 111.66 | csim: TEST PASSED. Same `partial_sum[PE_COUNT]` pattern as `conv_c1_systolic.cpp`: 150-term reduction (IN_C×K×K = 6×5×5) split across 6 independent accumulator chains (25 terms each), combined at the end. HLS auto-flattened the outer o/r/c loops into the pipeline (trip count 40,000 = 1,600×25). Latency dropped ~7.66x vs. baseline. II only improved 5→4 (still not 1) — same accumulator-latency floor as C1's PE_COUNT=8 case (Final II=4 there too). Most of the DSP/FF growth (33 of 45 DSP; `urem_*` modules ~2,100-2,300 FF each ×16) comes from decoding the flattened `mac_idx` back into `(i,kr,kc)` via `/`/`%` by K=5 (non-power-of-2), not from the MAC math itself. **Superseded** by `conv_c3_partialsum_lut.cpp` (same latency/II/Fmax, far lower DSP/FF/LUT) — no longer called from `lenet5_top.cpp`; kept in the repo and buildable for the record |
| **EXPLORED, REJECTED — div/mod-free round-robin**, `conv_c3_partialsum_roundrobin.cpp` | 1,030,401 | 1,030,402 | 4 / 1 | 5 | 964 | 2,050 | 0 | **93.03** | csim: TEST PASSED. Attempted to remove the div/mod overhead above by using natural nested `i`/`kr`/`kc` loops (zero decode cost) and an increment-and-wrap counter (compare+reset, not a divider) to pick which of `PE_COUNT` partial-sum registers each term hits. **It works**: DSP/FF/LUT drop back to near-baseline levels (DSP 45→5, FF 41,020→964, LUT 31,870→2,050), confirming the div/mod was indeed the dominant resource cost, not the MACs. **But** the `pe`-counter's compare/reset in the loop latch breaks the "perfect loop nest" property HLS needs to auto-flatten o/r/c into the pipeline (log: "Cannot flatten loop ... the outer loop is not a perfect loop because there is nontrivial logic in the loop latch") — so the 150-iteration inner pipeline now drains/refills separately 1,600 times with zero overlap, making overall latency ~6.4x **worse** than the kept version above (still ~16% better than the un-split baseline). Fmax also fell to 93.03MHz, missing the 100MHz-class target — the `pe`-select mux landed on the same critical path as `input_r_load`→`fmul`. Kept in the repo and documented here (not deleted) as a real cost/latency trade worth revisiting, mirroring how C1's rejected intermediate attempts (forced-full-unroll bug, 150-port AXI explosion) were kept visible in this log rather than erased |
| **KEPT — LUT-based mac_idx decode**, `conv_c3_partialsum_lut.cpp` | **160,047** | 160,048 | 4 / 1 | **12** | **3,013** | **6,250** | 0 | **111.66** | csim: TEST PASSED (uniform-value corner check + a second cross-check against `conv_c3_partialsum.cpp` using distinct per-`(kr,kc,i)` weights, max diff 0). Keeps `conv_c3_partialsum.cpp`'s exact flattened `mac_idx = m*PE_COUNT+p` loop structure (same `m`/`p` loops, same `#pragma HLS PIPELINE II=1` placement) — only the decode changes: `i`/`kr`/`kc` come from three `ARRAY_PARTITION complete` 150-entry constant tables (`i_lut`, `kr_lut`, `kc_lut`) indexed by `mac_idx`, instead of `/`/`%` by K=5. Synthesis log shows the identical three `Flattening a loop nest` INFO messages as the previously-kept version, and the loop report confirms it: `VITIS_LOOP_57_1_VITIS_LOOP_59_3_VITIS_LOOP_66_5`, trip count 40,000, Pipelined=yes, II achieved=4 — same accumulator-latency floor, same Fmax (111.66MHz), latency effectively unchanged (160,047 vs 160,078). Resource cost dropped sharply: DSP 45→12 (-73%), FF 41,020→3,013 (-93%), LUT 31,870→6,250 (-80%) — the `urem_*` dividers are gone, replaced by small `mux_255_32_1_1` lookup muxes (17 instances, ~113 LUT each) reading the partitioned tables. **This gets both properties the round-robin attempt couldn't combine: division-free decode AND preserved loop flattening/timing** — it strictly dominates the previous version on DSP/FF/LUT at equal latency/II/Fmax. **Promoted to C3's optimized stage; `lenet5_top.cpp` now calls this version** |

**Correction to this log's own C1 comparisons above**: several notes here (and in C1's own log) describe `conv_c3_partialsum.cpp`'s `partial_sum[PE_COUNT]`/`mac_idx` pattern as "the same pattern as `conv_c1_systolic.cpp`" — that's accurate only in that both use flattened-`mac_idx`-decoded-via-`/`-and-`%`. Checked against every commit that ever touched `conv_c1_systolic.cpp` (from its first commit `905d876` onward): C1 has never had a division-free, direct-nested-loop `(kr,kc,i)` variant with a loop-counter PE assignment — that structure only ever existed as `conv_c3_partialsum_roundrobin.cpp` above (the rejected one). C1 accepted the same division cost C3 did; it never solved the div-vs-flattening tradeoff either. The LUT-based row above is the first variant (C1 or C3) to get both.

**Next step**: `conv_c3_partialsum_lut.cpp` (PE_COUNT=6, LUT-based decode) is now C3's optimized stage, promoted in place of `conv_c3_partialsum.cpp`. Continue C1's methodology from here: `ARRAY_PARTITION` on `input`/`weights`, then line buffer → systolic PE variant → AXI conversion.



## C5 Dense (Fully-Connected) — HLS Optimization Log

Target device: xc7z020iclg484-1L (Zynq-7020, ZedBoard) | Clock: 10ns (100MHz), same as C1/S2/C3.

| Experiment | Latency (cycles) | Interval (cycles) | Inner-loop II (achieved/target) | DSP | FF | LUT | Fmax (MHz) | Notes |
|---|---|---|---|---|---|---|---|---|
| **Baseline (no pragmas)** | **241,681** | **241,682** | **5 / —** | **5** | **790** | **1,083** | **137.82** | csim: TEST PASSED (output[0]=400). `VITIS_LOOP_14_2` (inner 400-term MAC) is auto-pipelined at II=5; outer `VITIS_LOOP_12_1` (120 output neurons) is not pipelined |
| **KEPT — Partial-sum split (PE_COUNT=4)**, `dense_c5_partialsum.cpp` | **48,031** | **48,032** | **4 / 1** | **7** | **1,863** | **2,253** | **111.66** | csim: TEST PASSED. Same `partial_sum[PE_COUNT]` pattern as `conv_c3_partialsum.cpp`: 400-term reduction split across 4 independent accumulator chains (100 terms each), combined at the end. Latency dropped ~5.0x vs. baseline. II only improved 5→4 (still not 1) — same accumulator-latency floor as C3's PE_COUNT=6 case. **This is the version kept** — lowest DSP among the II=4 options, clears the 100MHz target with room to spare |
| EXPLORED, comparison point — PE_COUNT=5 | 38,437 | 38,438 | 4 / 1 | 12 | 2,206 | 3,082 | 110.94 | csim: TEST PASSED. Confirms PE_COUNT ≥ fadd latency (4) does not buy II=1: more parallel chains only shrinks `MACS_PER_PE` (so latency keeps dropping, 48,031→38,437) and costs more DSP (7→12), but II stays pinned at 4. Not kept — PE_COUNT=4 gets the same II at lower resource cost |

**Bottleneck diagnosis — accumulator-limited, exactly like C3's baseline (not a memory-port issue):**
Bind report shows a single scalar accumulator (`acc_2 = fadd(acc_1_load, mul)`, `FAddSub_fulldsp`, latency=4). The `input_r`/`weights` RAM loads already resolve at `<Latency=1><II=1>` on 2 ports — no port stall. II=5 = fadd latency(4) + 1 pipeline-overhead cycle, a pure loop-carried float-accumulator RAW dependency. `ARRAY_PARTITION` would have done nothing here, same conclusion as C1's pre-systolic baseline and C3's pre-split baseline.

**Why PE_COUNT ≥ 4 doesn't reach II=1 (confirmed by directly comparing PE_COUNT=4 vs. 5):**
`#pragma HLS UNROLL` over the PE dimension `p` puts all `PE_COUNT` partial-sum adds inside the *same* pipeline iteration (the `m` loop, trip count = `MACS_PER_PE`). That means any single `partial_sum[p]` is written once per `m`-iteration and read back on the very next `m`-iteration — a revisit gap of exactly 1 iteration, independent of `PE_COUNT`. Since 1 iteration lasts II cycles, the fadd's 4-cycle latency forces II≥4 regardless of how many parallel chains exist; more `PE_COUNT` only reduces `MACS_PER_PE` (fewer iterations → less total latency), it does not relax the per-chain recurrence. The only way to actually get a revisit gap of `PE_COUNT` iterations would be *temporal* round-robin (each PE only touched once every `PE_COUNT` iterations) instead of *spatial* unroll — which is exactly what `conv_c3_partialsum_roundrobin.cpp` tried for C3, and that hit a separate wall instead (variable array index defeats HLS's static dependence analysis, so it conservatively re-imposes the same II=4 anyway, on top of breaking loop flattening). Given that, PE_COUNT=6 was not run for C5 — by this same mechanism it would cost more DSP than PE_COUNT=4 for the identical II=4 floor.

**Next step**: `dense_c5_partialsum.cpp` (PE_COUNT=4) is C5's optimized stage, matching C3's methodology. `ARRAY_PARTITION`/further memory-side optimization is not indicated — the remaining ceiling is the accumulation chain, not a port conflict.

## S2 Pooling — HLS Optimization Log (final, corrected)

| Stage | Latency (cycles) | II | DSP | FF | LUT | Notes |
|---|---|---|---|---|---|---|
| ~~1. Avg pooling, no pragma~~ | ~~2,379~~ | ~~2~~ | ~~7~~ | ~~1,192~~ | ~~1,809~~ | Superseded — wrong pooling variant for the project's ReLU+MaxPool standardization |
| ~~2. Avg pooling + ARRAY_PARTITION~~ | ~~—~~ | ~~1~~ | ~~11~~ | ~~1,431~~ | ~~2,194~~ | Superseded — same reason; the fix was applied to the wrong underlying algorithm |
| **3. Max pooling + ARRAY_PARTITION (final)** | **1,186** | **1** | **0** | **576** | **589** | Correct variant, all loop constraints satisfied. DSP=0 because pure comparisons need no MAC hardware — genuinely cheaper than the average-pooling version at every resource metric |

`pool_s2.cpp`'s committed source now matches stage 3: `ARRAY_PARTITION cyclic factor=2` on dims 1
and 2 (row, col) of `input` (commit `720c664`) — cyclic-partitioning by 2 in both spatial dims
splits each 2×2 pooling window's four loads across four separate memory banks, resolving the port
contention that capped II at 2.



## S4 Pooling — HLS Optimization Log

Target device: xc7z020iclg484-1L (Zynq-7020, ZedBoard) | Clock: 10ns (100MHz), same as C1/S2/C3/C5.

| Experiment | Latency (cycles) | II | DSP | FF | LUT | Notes |
|---|---|---|---|---|---|---|
| Max pooling + ARRAY_PARTITION, from the start | *pending Vitis HLS run* | *pending (targeting 1)* | *pending (targeting 0)* | *pending* | *pending* | `ARRAY_PARTITION cyclic factor=2` on `input` dims 1+2 (row, col) + explicit `PIPELINE II=1` — identical fix to S2's (commit `720c664`), applied from the start instead of discovered fresh. Functionally verified via local g++ testbench (TEST PASSED), not yet run through Vitis HLS |

**Architecture differences from S2's baseline**:
- Input is 10×10×16 (C3's output, valid-padded), not 28×28×6 — smaller spatial extent, more channels (16 vs 6).
- Same 2×2/stride-2 max pooling, same loop structure, same partitioning fix — only dimensions changed.

**Next step**: run C-simulation and synthesis in Vitis HLS to confirm II=1/DSP=0 actually hold at these dimensions.

## S4 Pooling — HLS Optimization Log

| Stage | Latency (cycles) | II | DSP | FF | LUT | Fmax | Notes |
|---|---|---|---|---|---|---|---|
| Max pooling + ARRAY_PARTITION (built correctly from start) | 409 (411 total) | **1** | **0** | 565 | 545 | 140.71 MHz | Reused S2's proven cyclic partition fix directly — no baseline-then-fix cycle needed. All loop constraints satisfied on first synthesis |

#the same max-pooling kernel and array-partition strategy transferred directly from S2 to S4 with zero rework, demonstrating the reusable-IP principle in practice.

## F6 Dense (Fully-Connected) — HLS Optimization Log

Target device: xc7z020iclg484-1L (Zynq-7020, ZedBoard) | Clock: 10ns (100MHz), same as C1/S2/C3/C5/S4.

| Experiment | Latency (cycles) | Interval (cycles) | Inner-loop II (achieved/target) | DSP | FF | LUT | Fmax (MHz) | Notes |
|---|---|---|---|---|---|---|---|---|
| **Partial-sum split (PE_COUNT=4), built directly** | **10,113** | **10,114** | **4 / 1** | **11** | **1,673** | **1,988** | **111.66** | csim: TEST PASSED (output[0]=120). Went straight to `dense_c5_partialsum.cpp`'s proven PE_COUNT=4 pattern — no serial-accumulator baseline built first, same as S2→S4's reuse. `VITIS_LOOP_19_1_VITIS_LOOP_28_3` (flattened outer j / inner m,p reduction, trip count 2,520 = 84×30) confirmed at II=4 |

**Architecture**: 120 inputs → 84 outputs, ReLU. Identical MAC-stage structure to `dense_c5_partialsum.cpp`: the 120-tap reduction per output neuron is split across `PE_COUNT=4` independent `partial_sum[]` accumulators (`ARRAY_PARTITION complete` on `partial_sum` only — C5's bind-report diagnosis showed the bottleneck is the float adder's loop-carried recurrence, not a memory-port conflict, so `input`/`weights` are left unpartitioned), with `PIPELINE II=1` requested on the inner reduction loop. Only the dimensions changed from C5 (400→120 inputs, 120→84 outputs) — same reusable-IP principle as S2→S4.

**Confirmed**: II=4 accumulation-limited floor holds exactly as predicted, independent of N_IN — same mechanism as C5 (fadd 4-cycle latency, same DSP count of 4 MAC units + 2 fadd + 1 fmul = 11 total). Fmax (111.66MHz) matches C5's PE_COUNT=4 result exactly, since it's set by the same `weights_load`→`fmul` critical path, not by N_IN or N_OUT.

## Output Dense (Fully-Connected, Softmax) — HLS Optimization Log

Target device: xc7z020iclg484-1L (Zynq-7020, ZedBoard) | Clock: 10ns (100MHz), same as C1/S2/C3/C5/S4/F6.

| Experiment | Latency (cycles) | Interval (cycles) | Inner-loop II (achieved/target) | DSP | FF | LUT | Fmax (MHz) | Notes |
|---|---|---|---|---|---|---|---|---|
| **Partial-sum split (PE_COUNT=4) MAC stage + separate softmax stage, built directly** | **999** | **1,000** | **4 / 1 (MAC stage)** | **14** | **2,778** | **3,721** | **111.66** | csim: TEST PASSED (all 10 outputs = 0.1, sum = 1.0) after fixing a build error (`std::expf` isn't in this toolchain's `<cmath>`; switched to `std::exp`, which has a valid `float` overload). MAC stage went straight to `dense_c5_partialsum.cpp`'s proven PE_COUNT=4 pattern — confirmed at II=4. See per-stage breakdown below |

**Architecture differences from F6/C5**: 84 inputs → 10 outputs, **softmax** instead of ReLU. Softmax needs the sum of every output neuron's exponential before any single one can be normalized, so it does not fit the single-pass "compute acc → activate → write output[j]" loop the ReLU layers use. `dense_output.cpp` is split into two explicit stages instead:
1. **MAC stage** — identical PE_COUNT=4 partial-sum structure to `dense_c5_partialsum.cpp`/`dense_f6.cpp` (only `partial_sum[]` partitioned, not `input`/`weights`, per C5's diagnosis), writing each neuron's raw (pre-activation) accumulation into a small local `logits[10]` array.
2. **Softmax stage** — a separate pass over the completed `logits` array once all 10 are available: max-subtraction for numerical stability, `exp`, sum, then divide.

Only stage 1 carries the PE_COUNT=4 partial-sum pragmas; stage 2 is a short, inherently serial reduction/normalization over just 10 elements and is not expected to be a resource or II bottleneck — confirmed below.

**Per-stage breakdown** (top-level `dense_output` is not itself pipelined — the 4 stages run sequentially, one HLS sub-block each, `Interval ≈ Latency` at the top level, matching C3/C5's un-pipelined-baseline pattern but here by *design*, not as a bottleneck):

| Stage (source line) | Latency (cycles) | II (achieved/target) | Trip count | Notes |
|---|---|---|---|---|
| MAC accumulation (29–38) | 867 | **4 / 1** | 210 (= 21 MACs/PE × 10 outputs) | Same accumulator-limited floor as C5/F6 — confirms the PE_COUNT=4 pattern transfers unchanged into a multi-stage design |
| Find max logit (59) | 20 | 2 / 1 | 9 (N_OUT−1 comparisons) | `fcmp` reduction over 10 elements — not partial-summed, not worth it at this trip count |
| exp + sum (65) | 68 | **5 / 1** | 10 | Same accumulator-limited pattern as C3/C5's *baseline* (single scalar `sum_exp`, 5-cycle fadd latency) — deliberately not partial-summed since PE_COUNT=4 splitting a 10-element reduction would cost more resources than it saves at this scale |
| Normalize (divide) (70) | 27 | 1 / 1 | 10 | Full II=1 — division has no loop-carried dependency here, each `output[j]` is independent |

**Confirmed**: MAC stage hits the identical II=4 floor as C5/F6 (11→14 DSP includes the exp/divide units, not more MAC parallelism), validating that the PE_COUNT=4 partial-sum pattern is orthogonal to what happens downstream. The softmax stage's own `sum_exp` accumulation lands on the *same* accumulator-latency mechanism documented for C3/C5's un-split baselines (II=5, scalar fadd) — expected and left as-is, since splitting a 10-term reduction across PE_COUNT=4 chains would trade a negligible latency win for real resource cost, unlike the 120–400-term reductions where the split pays off.

## LeNet-5 Combined Top-Level — HLS Optimization Log

Target device: xc7z020iclg484-1L (Zynq-7020, ZedBoard) | Clock: 10ns (100MHz), same as every layer above. `hls/lenet5_top/`: `lenet5_top()` chains all 7 proven layer IPs (conv_c1_systolic → pool_s2 → conv_c3_partialsum → pool_s4 → dense_c5_partialsum → dense_f6 → dense_output) into one forward pass, with the same `m_axi`/`s_axilite` + local-buffer-burst-copy interface style as `conv_c1_systolic.cpp`.

**Test data**: real MNIST test[0] image + real trained weights pulled directly out of `Models/lenet5_relu.keras` (loaded via `h5py` against the `.keras` archive's `model.weights.h5`, since TensorFlow isn't installed in this environment — weight arrays are matched to layers by shape, which is unambiguous here: no two layers share a shape). Reference forward pass computed with the project's own `manual_layers.py` (the same math `manual_inference_relu.py` uses). Generator script: `hls/lenet5_top/tb/generate_test_data.py`, output: `hls/lenet5_top/tb/lenet5_test_data.h`.

**C-simulation: TEST PASSED** (both the primary and the explored-and-reverted DATAFLOW variant give this same result). HLS output matches the Python/NumPy reference to within 6.5e-11 (float rounding from a different accumulation order — the partial-sum split reduces in a different term order than NumPy's sequential sum), sums to 1.0, and both agree on predicted class 7 — which also matches the true MNIST label:

| | class 0 | 1 | 2 | 3 | 4 | 5 | **6** | **7** | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| HLS | 4.49e-9 | 7.52e-7 | 2.59e-6 | 7.63e-5 | 6.46e-12 | 3.24e-7 | 1.15e-11 | **0.99992** | 8.77e-8 | 4.49e-8 |
| Python ref | 4.49e-9 | 7.52e-7 | 2.59e-6 | 7.63e-5 | 6.46e-12 | 3.24e-7 | 1.15e-11 | **0.99992** | 8.77e-8 | 4.49e-8 |

**Build issue hit and fixed before synthesis first succeeded** — bundling all 10 weight/bias arrays under one shared `m_axi` bundle (`gmem_wgt`) failed csynth with `ERROR: [HLS 200-1013] Bundled bus interface gmem_wgt failed dataflow checking: it cannot read data in multiple processes` — DATAFLOW turns each burst-copy-in loop into its own concurrent hardware process, and two concurrent processes can't share one AXI port. Fix: one `m_axi` bundle per array (12 total: image, 5×weights, 5×bias, result) — actually the more faithful reading of `conv_c1_systolic.cpp`'s own style (it already gives `input`/`weights`/`bias`/`output` four separate bundles). Kept in the primary version below even though it's no longer strictly required without DATAFLOW.

### EXPLORED, REVERTED — `#pragma HLS DATAFLOW` for image-to-image pipelining (`lenet5_top_dataflow.cpp`)

Exceeds chip capacity due to weight double-buffering — reverted, not carried forward as the active design. Kept in the repo (`hls/lenet5_top/src/lenet5_top_dataflow.cpp`, `run_hls_dataflow.tcl`, `tb/lenet5_top_dataflow_tb.cpp`) as a real, independently-reproducible exploration, same convention as `conv_c3_partialsum_roundrobin.cpp`.

| Metric | Value | Available | Utilization |
|---|---|---|---|
| Latency (one image, cold) | 505,400 cycles (5.054 ms) | — | — |
| Interval (steady-state, pipelined across images) | 291,202 cycles (2.912 ms) | — | — |
| Fmax | 103.49 MHz | — | clears 100MHz target |
| DSP | 84 | 220 | 38% |
| FF | 83,080 | 106,400 | 78% |
| **LUT** | **74,562** | **53,200** | **140% — OVER BUDGET** |
| **BRAM_18K** | **444** | **280** | **158% — OVER BUDGET** |

**Findings:**

1. **DATAFLOW *did* deliver on its purpose** — Interval (291,202) < total Latency (505,400), confirming successive images really can start before the previous one finishes, gated by the single slowest stage.
2. **`BRAM_18K` and `LUT` are genuinely over the xc7z020's budget**, meaning this design would fail Vivado place-and-route even though Vitis HLS's csynth step reports success (csynth only generates + estimates RTL, it doesn't check placement). Root cause, confirmed in the synthesis log's `PIPO` messages: DATAFLOW requires every buffer that crosses a producer/consumer boundary to support double-buffering (ping-pong) so the next image's producer can write while the current image's consumer still reads — e.g. `local_c3_weights` gets built "using a separate memory for each block" (i.e. doubled), and the same happens to several of the large inter-layer feature-map buffers. This cost is fundamentally reasonable for the *small, per-image* feature-map buffers DATAFLOW is meant to double-buffer — but the *weight* arrays (which don't change per image at all) pay the same double-buffering tax purely because their burst-copy-in loop lives inside the same DATAFLOW region as everything else.
3. `conv_c3_partialsum`'s instance latency here (291,201 cycles) is nearly double its standalone number (160,078 cycles). **Correction to an earlier read of this**: re-running the exact same C3 regression appears in the *non*-DATAFLOW primary version below too (see its finding #1) — so this is NOT a DATAFLOW-specific cost, it's a general consequence of C3 being called as a sub-function at all. Left in this table for completeness since it's still true of this build, but the diagnosis belongs to the primary version's write-up, not to DATAFLOW.

**Follow-ups if image-to-image pipelining is revisited**: move the weight burst-copy-in loops outside the DATAFLOW region (weights only need loading once per bitstream load, not double-buffered per image) — would likely remove most of the BRAM overshoot on its own.

### PRIMARY WORKING BASELINE — sequential, no DATAFLOW (`lenet5_top.cpp`)

Same 7-layer chain, same AXI/burst-copy interface, `#pragma HLS DATAFLOW` removed: image N fully completes (including all weight/image copy-in) before image N+1 starts. C-simulation re-confirmed TEST PASSED (identical numeric result to the table above — removing DATAFLOW doesn't change the math, only the scheduling).

| Metric | Value | Available | Utilization |
|---|---|---|---|
| Latency = Interval (fully sequential, no overlap) | 552,617 cycles (5.526 ms) | — | — |
| Fmax | 103.49 MHz | — | clears 100MHz target |
| DSP | 57 | 220 | 25% |
| FF | 80,961 | 106,400 | 76% |
| BRAM_18K | 232 | 280 | **82% — fits** |
| **LUT** | **70,151** | **53,200** | **131% — still OVER BUDGET** |

**Findings:**

1. **Removing DATAFLOW fixed BRAM (158%→82%) but NOT LUT (140%→131%, still over).** Confirms the double-buffering diagnosis above was correct and was the dominant BRAM cost — without DATAFLOW, every buffer is single-buffered again (`local_c3_weights` etc. now show a single memory, not a doubled pair), and BRAM_18K usage nearly halved (444→232) even though the arrays themselves didn't shrink.
2. **`conv_c3_partialsum`'s latency is STILL ~291,201 cycles here, identical to the DATAFLOW build**, confirming (see correction above) this has nothing to do with DATAFLOW: its own sub-report inside this build still shows the outer `o`/`r`/`c` loop (trip count 1,600) as `Pipelined: no`, each outer iteration paying its own ~182-cycle call overhead into the inner PE_COUNT=6 pipeline (1,600 × 182 ≈ 291,200) instead of the one continuous 40,000-iteration flattened pipeline C3 gets when it's its own top-level function. This single stage is 53% of total latency (291,201 / 552,617) — the clear next lever for latency, independent of the DATAFLOW question. Not fixed here since it wasn't asked for this pass.
3. **LUT is still over budget, and the cause is additive, not a single culprit**: `conv_c3_partialsum` alone costs 29,514 LUT (mostly its documented `mac_idx` div/mod decode overhead, already flagged as an accepted cost in C3's own optimization log above), `conv_c1_systolic` costs 12,171, `dense_output` 2,500, and the 12 per-array `m_axi` adapters cost 1,318 LUT each = 15,816 total — that last number is now a proportionally bigger factor than it was in the DATAFLOW build, because BRAM dropped but LUT didn't, so the fixed per-bundle AXI overhead (unaffected by DATAFLOW either way) is more exposed as a real, unavoidable-at-12-bundles cost.

**Not attempted this pass** (real next steps, not silently applied): consolidating some of the 12 AXI bundles (e.g. one shared bundle per *layer* — weights+bias together — instead of one per array) would cut AXI adapter LUT roughly in half without reintroducing the DATAFLOW multi-process-per-bundle conflict, since this version has no concurrent processes; and addressing C3's lost loop-flattening (independent of DATAFLOW, confirmed above) would cut both latency and, likely, LUT (fewer redundant control-path copies of the inner pipeline's start/stop logic per outer iteration). Neither was applied here — this is reported as the actual, unedited synthesis result of "remove DATAFLOW, re-verify, report," exactly as asked.

### AXI bundle consolidation — 7 bundles instead of 12 (`lenet5_top.cpp`, current)

Confirmed before making the change: the DATAFLOW multi-process-per-bundle conflict (line 248 above) cannot recur here — that error only occurs when DATAFLOW splits each burst-copy loop into its own concurrent hardware process; this primary version has no `#pragma HLS DATAFLOW` and never did, so there is exactly one process reading/writing each AXI port regardless of bundling. Changed the 12 per-array `m_axi` bundles to 7: one shared bundle per layer's weights+bias pair (`gmem_c1`, `gmem_c3`, `gmem_c5`, `gmem_f6`, `gmem_out` for output_weights+output_bias) plus the network's own I/O kept separate (`gmem_img` for `image`, `gmem_res` for `result`). C-simulation re-confirmed TEST PASSED (bundling is an interface-only change, doesn't touch the math). C-synthesis succeeded with no dataflow-checking errors.

| Metric | Value | Available | Utilization |
|---|---|---|---|
| Latency = Interval | 552,749 cycles (5.527 ms) | — | — (unchanged, interface-only change) |
| Fmax | 103.49 MHz | — | clears 100MHz target |
| DSP | 57 | 220 | 25% |
| FF | 77,380 | 106,400 | 72% |
| BRAM_18K | 232 | 280 | 82% — fits |
| **LUT** | **63,478** | **53,200** | **119% — still OVER BUDGET, improved from 131%** |

**Findings:**

1. **Confirmed the AXI-adapter-count hypothesis directly**: each `m_axi` adapter (`gmem_*_m_axi_U` in the Instance report) costs a fixed 1,318 LUT regardless of how many arrays share it. 12→7 bundles cut adapter LUT from 15,816 to 9,226 (−6,590), and total design LUT dropped 70,151→63,478 (−6,673, the extra ~83 from small mux/routing changes elsewhere).
2. **This alone does not get the design under budget.** 63,478 / 53,200 = 119% — still 10,278 LUT over. The two dominant non-AXI costs are unchanged and now proportionally larger: `conv_c3_partialsum` alone is 29,514 LUT and `conv_c1_systolic` 12,171 LUT (together 65% of total design LUT), both pre-existing costs from each layer's own accepted optimization trade-offs (C3's `mac_idx` div/mod decode, documented in its own log above).
3. Going to fewer than 7 bundles (e.g. merging `gmem_img`/`gmem_res` in with a layer bundle, or all weights into one bundle) would save at most another ~2,636 LUT (2 more adapters removed) — not enough on its own to close a 10,278-LUT gap. **Getting under 53,200 LUT requires addressing C3 and/or C1's own resource cost, not further bundle consolidation.**

**Not attempted this pass**: C3's lost loop-flattening (latency-only per the original ask, tracked separately above) and any reduction of C3/C1's own LUT footprint (e.g. revisiting `mac_idx` decode or PE_COUNT) — out of scope for "AXI bundle consolidation only," reported here as the honest result of that change in isolation.

### C3 LUT-decode swap — `conv_c3_partialsum_lut.cpp` promoted in place of `conv_c3_partialsum.cpp` (`lenet5_top.cpp`, current)

Directly addresses finding #2/#3 above ("Getting under 53,200 LUT requires addressing C3 and/or C1's own resource cost"). `lenet5_top.cpp`'s forward declaration and its one call site were switched from `conv_c3_partialsum` to `conv_c3_partialsum_lut` (see C3's own log above for that IP's division-free `mac_idx` decode). No other change — same 7-bundle AXI interface, same DATAFLOW-free sequential scheduling. C-simulation re-confirmed TEST PASSED against the same real-MNIST/real-weights reference (predicted class 7, matches true label, max abs diff vs. Python reference 6.55e-11 — identical to every prior build, since swapping C3's decode implementation doesn't change the math). C-synthesis succeeded.

| Metric | Value | Available | Utilization |
|---|---|---|---|
| Latency = Interval | 496,750 cycles (4.967 ms) | — | ~10% faster than the 552,749-cycle pre-swap build |
| Fmax | 103.49 MHz | — | clears 100MHz target |
| DSP | 24 | 220 | 11% |
| FF | 39,688 | 106,400 | 37% |
| BRAM_18K | 232 | 280 | 82% — fits |
| **LUT** | **37,974** | **53,200** | **71% — UNDER BUDGET** |

**This is the first build of the full 7-layer design to fit the xc7z020's LUT budget.** Confirms both open findings above directly:
- `conv_c3_partialsum_lut`'s own instance cost inside this build is 4,010 LUT — down from `conv_c3_partialsum`'s 29,514 LUT in the identical position (−86%), consistent with (in fact even better than) the division-free decode's standalone win documented in C3's own log.
- Total design LUT dropped 63,478 → 37,974 (−25,504, 40%), taking utilization from 119% (over budget) to 71%. DSP and FF also dropped sharply (57→24 DSP, 77,380→39,688 FF) since the `urem_*` divider logic is gone from the design entirely, not just resource-shared away.
- Latency improved too (552,749→496,749 cycles, ~10% faster) even though C3 still loses its standalone loop-flattening when called as a sub-function here (same `Pipelined: no` outer-loop situation noted in the pre-swap findings) — the win is purely from removing the division critical-path/area cost, not from recovering flattening.

`conv_c1_systolic` (12,171 LUT) is now the largest single-layer LUT cost in the design, and the 7 AXI adapters remain a fixed ~9,226 LUT overhead — both are legitimate next levers if further margin is wanted, but are not required to meet the 53,200 budget.



## C1 Convolution — INT8 Quantization (new, PYNQ-Z2 target)

Project guide's next requirement: convert the HLS design from float32 to INT8 (weights and activations) with INT32 accumulation, starting with C1 only — C3, pooling, and dense stay float32 for now. Target board for this work: **PYNQ-Z2** (`xc7z020clg400-1`), a new part from the `xc7z020iclg484-1L`/`xc7z020clg484-1` target used for every float32 IP above (same Zynq-7020 die, different package/speed grade — resource counts are directly comparable, but note the part isn't identical when reading the numbers below).

**Files added (all new — no existing float32 file touched, same "old version stays untouched as reference" convention as `conv_c1.cpp` vs `conv_c1_systolic.cpp`):**
- `hls/conv_c1/src/conv_c1_int8.h`, `hls/conv_c1/src/conv_c1_int8.cpp` — new IP
- `hls/conv_c1/tb/conv_c1_int8_tb.cpp`, `hls/conv_c1/tb/generate_test_data_int8.py`, `hls/conv_c1/tb/conv_c1_int8_test_data.h` — new testbench + golden-data generator
- `hls/conv_c1/run_hls_int8_pynqz2.tcl` — new build script, new solution (`conv_c1_int8_proj`), part `xc7z020clg400-1`

**Architecture**: deliberately the *same* systolic PE-array design as `conv_c1_systolic.cpp` (PE_COUNT=8, line buffer, `m_axi`×4 + `s_axilite` interfaces, burst-copy into partitioned local on-chip buffers) — only the datapath changes, so the float32-vs-int8 utilization comparison below isolates the effect of the numeric representation, not the architecture. `input`/`weights`/`output` are `ap_int<8>`, the MAC accumulator (`acc`, `partial_sum[]`) is `ap_int<32>`, and `bias` is `ap_int<32>`, PRE-quantized in software as `round(bias_float / (x_scale*w_scale))` (matching `conv2d_int()`'s own bias handling, just computed once ahead of time instead of re-derived via float division inside the datapath every call). `x_scale`, `w_scale`, `out_scale` are passed in as separate `float` scalar parameters (AXI-Lite registers), per the project guide's requirement to keep scale factors separate from the integer data.

**Data flow, matching `src/integer_layers.py`'s proven `conv2d_int()` exactly**: int8 input × int8 weight accumulated into int32 (`acc`), then a single dequantize→ReLU→requantize step per output pixel:
```
dequant = (float)acc * (x_scale * w_scale)   // conv2d_int()'s acc*combined_scale
dequant = max(0, dequant)                     // relu()
requant = round(dequant / out_scale)          // quantize_activation(), round-half-away-from-zero
output  = saturate(requant, -128, 127)
```
`out_scale` is this layer's own output-activation scale, produced by `quantize_activation()` applied to the ReLU'd float output — exactly how `integer_inference.py`'s real int8 forward pass determines every layer's next-stage input scale.

**Test data**: real int8-quantized weights from `models/lenet5_relu.keras`'s C1 layer (via `quantize_int_real()`, `src/quantization.py`, as explicitly requested) and a real int8-quantized MNIST test[0] image (via `quantize_activation()`, `src/integer_layers.py`). Generator: `hls/conv_c1/tb/generate_test_data_int8.py`, output: `hls/conv_c1/tb/conv_c1_int8_test_data.h`. The one deliberate departure from calling `quantize_activation()` for the final output requantization is round-half-**away**-from-zero (matching the HLS hardware's `x>=0 ? +0.5 : -0.5` rounding) instead of NumPy's round-half-to-**even** — the two only disagree exactly at `x.5000...` boundaries, which don't occur with real trained-model floating-point data, so this doesn't change which reference is being matched, only makes the rounding rule explicit and bit-reproducible in C++.

**C-simulation: TEST PASSED — exact match, not just close.**
```
Total output elements: 4704
Mismatches: 0
Max abs diff: 0
TEST PASSED -- HLS INT8 output matches Python int8 reference exactly
```
Every one of the 4,704 output values (28×28×6) is bit-for-bit identical to the Python `conv2d_int()`+`relu()`+requantize reference — integer equality, not a floating-point tolerance.

**C-synthesis: succeeded, all requested `PIPELINE` constraints satisfied.**

### Utilization — float32 (`conv_c1_systolic`) vs int8 (`conv_c1_int8`)

| Metric | float32 (`conv_c1_systolic`, `xc7z020iclg484-1L`) | int8 (`conv_c1_int8`, `xc7z020clg400-1`, PYNQ-Z2) | Change |
|---|---|---|---|
| BRAM_18K | 18 (6%) | **5 (1%)** | **−72%** |
| DSP | 14 (6%) | 13 (5%) | −7% |
| FF | 28,886 (27%) | **9,989 (9%)** | **−65%** |
| LUT | 19,572 (36%) | **16,519 (31%)** | **−16%** |
| Latency (cycles) | 144,591 | 280,223 | **+94% (worse)** |
| Timing (at 10ns/100MHz target) | Slack **−2.36ns** (violated) | Slack **0.00ns** (met) | int8 closes timing, float32 doesn't |
| Estimated Fmax | 104.80 MHz | **136.99 MHz** | +31% |

**Findings:**

1. **BRAM/FF/LUT all drop substantially, DSP is roughly flat.** BRAM falls the most (−72%) since 8-bit local buffers need far less on-chip memory than 32-bit float ones; FF falls sharply (−65%) because the 150-term MAC reduction's operands and pipeline registers are 8/32-bit integers instead of 32-bit floats carrying implicit exponent/mantissa/rounding logic through every pipeline stage. DSP stays close (14→13): both versions still map their 8-lane MAC array onto DSP48 multiply-accumulate primitives (`fmul`/`fadd` for float32, `mac_muladd_8s_8s_32s` for int8) at essentially 1:1 DSP-per-lane, so switching numeric representation doesn't change *how many* MAC lanes exist, only what's inside each one.

2. **Latency got WORSE (144,591 → 280,223 cycles, +94%), and the cause is a genuinely new bottleneck, not a regression in the MAC array itself.** The int8 MAC reduction loop (`VITIS_LOOP_106_24`) still hits the target `II=1` pipeline, same as float32's equivalent inner loop. The regression is entirely in the *new* per-pixel dequantize→ReLU→requantize step this design didn't need before: `requant = dequant / out_scale` is a genuine floating-point division (`fdiv_32ns_32ns_32_16_no_dsp_1`, 15-cycle latency — see the Bind Op Report), on top of the dequantize multiply (`fmul`, 3 cycles). In `conv_c1_systolic.cpp`, ReLU was a free same-cycle comparison (`acc > 0 ? acc : 0`) fused into the already-pipelined accumulation; here, ReLU is cheap (an `if (dequant < 0)` on the *already-computed* float, no extra latency) but the surrounding fmul+fdiv chain is not, and having to complete it once per (row, col, channel) — inside the `o` loop, per-channel, not shared across channels — is expensive enough that HLS could not flatten the `o` loop into the MAC pipeline the way it flattened the analogous 24-trip `o`×`m` loop in `conv_c1_systolic` (compare: float32's per-pixel-across-all-6-channels reduction+activation completes in 145 cycles; int8's completes in 318 cycles across those same 6 channels, `VITIS_LOOP_101_22`, `Pipelined: no`).

3. **int8 closes timing where float32 doesn't** (0.00ns slack vs −2.36ns at the 10ns/100MHz constraint; Estimated Fmax 136.99MHz vs 104.80MHz). Pure-integer MAC/compare logic has a materially shorter critical path than the pipelined floating-point MAC+comparison chain the systolic PE array uses in `conv_c1_systolic.cpp` — consistent with this project's own earlier finding (C1's own optimization log above) that the float `fadd`'s multi-cycle latency was *already* the accumulation bottleneck for the un-split baseline; int8 doesn't carry that same floating-point critical-path cost in its MAC array, even though it now carries a *different* one (the fdiv) at the activation boundary.

4. **Not attempted this pass**: the `fdiv` is the obvious next lever if C1-int8's latency needs to come down — replacing `dequant / out_scale` with a precomputed `dequant * (1.0f / out_scale)` would trade the 15-cycle divide for a 3-cycle multiply and a single one-time reciprocal, likely recovering most or all of the +94% latency regression. This was deliberately **not** applied here: it changes the floating-point operation (`a/b` is not bit-identical to `a*(1/b)` in IEEE754 in general), which would need the Python reference regenerated the same way to keep the exact bit-for-bit match this pass achieved — out of scope for "get INT8 C1 correct and report its numbers as they stand," reported here as a real, concrete next step rather than silently applied.

**Next step**: this validates the INT8 data flow and quantization approach end-to-end for one layer. Extending to C3 (the guide's own next-named layer) can reuse this exact pattern — `ap_int<8>`/`ap_int<32>` datapath, pre-quantized int32 bias, `x_scale`/`w_scale`/`out_scale` as separate float parameters, dequantize→ReLU→requantize per output pixel — and should hit the same reciprocal-multiply latency lever if it reuses the direct `fdiv` as a starting baseline.

### Follow-up — reciprocal-multiply requantization (`conv_c1_int8.cpp`, current)

Applied the lever identified above: `float inv_out_scale = 1.0f / out_scale;` is now computed ONCE (loop-invariant, before the row/col/channel loop nest), and the per-pixel requantization step is `requant = dequant * inv_out_scale` instead of `requant = dequant / out_scale`. Only one division now exists in the whole design (computed once per call), instead of one per output pixel (4,704 times).

**Bit-exactness was checked, not assumed** — `a/b` is not always bit-identical to `a*(1/b)` in IEEE754 float32, so the Python golden reference (`generate_test_data_int8.py`) was regenerated to requantize the same way (`relu_out * inv_out_scale`, `inv_out_scale = np.float32(1.0)/out_scale`), and the two arithmetic paths were compared directly before trusting either:

| Check | Result |
|---|---|
| Raw float `requant` values that differ between `a/b` and `a*(1/b)` | **530 / 4,704** elements differ (max abs diff 7.63e-06) |
| Of those, how many land in a **different int8 bucket** after round+saturate | **0 / 4,704** |

So the two arithmetic paths genuinely do diverge at the float level on this real data (not a no-op change) — they just happen not to cross a rounding boundary anywhere in this particular image/weight set. That's a property of this data, not a guarantee; re-verifying via C-simulation (not just this offline NumPy check) is what actually confirms the HLS kernel itself is still correct.

**C-simulation, re-run against the regenerated reference: still an exact match.**
```
Total output elements: 4704
Mismatches: 0
Max abs diff: 0
TEST PASSED -- HLS INT8 output matches Python int8 reference exactly
```

**C-synthesis, re-run:**

| Metric | Before (per-pixel `fdiv`) | After (reciprocal-multiply) | Change | vs. float32 `conv_c1_systolic` |
|---|---|---|---|---|
| Latency (cycles) | 280,223 | **223,780** | **−20.1%** | float32 = 144,591 (int8 still +54.8% worse, down from +93.8%) |
| BRAM | 5 (1%) | 5 (1%) | unchanged | still −72% vs float32 |
| DSP | 13 (5%) | 13 (5%) | unchanged | still −7% vs float32 |
| FF | 9,989 (9%) | 9,950 (9%) | −0.4% | still ~−66% vs float32 |
| LUT | 16,519 (31%) | 16,461 (30%) | −0.4% | still ~−16% vs float32 |
| Timing (10ns/100MHz target) | met (0.00ns slack) | **met (0.00ns slack)** | unchanged | float32 violates (−2.36ns) |
| Estimated Fmax | 136.99 MHz | 136.99 MHz | unchanged | float32 = 104.80 MHz |

**Latency improved meaningfully (−20%) without losing timing closure or any of the resource savings** — confirms the `fdiv` was a real cost and the fix is safe. It did **not**, however, close the full gap to float32's 144,591 cycles: the output-channel loop (`VITIS_LOOP_109_22`, trip count 6) is *still* not pipelined/flattened into the MAC reduction the way float32's equivalent loop is — its per-channel iteration latency dropped from 53→41 cycles (the `fdiv`'s contribution to the critical path is gone), but something else in the dequantize→ReLU→requantize→round→saturate chain still blocks flattening. Not chased further this pass — the concrete ask (does latency improve, is exactness/timing/resources preserved) is answered; diagnosing the remaining flattening blocker is a follow-up, not part of this change.

### DSP investigation — what the 13 DSPs actually are

Checked the Bind Op Report (both before and after the reciprocal-multiply change — the breakdown is identical either way) to answer directly: **only 8 of the 13 DSPs (62%) are the genuine int8 MAC array; the remaining 5 (38%) are still floating-point**, which is exactly why the DSP drop (14→13, −7%) was so much smaller than FF/BRAM's (−65%/−72%) despite the MAC math itself now being pure integer.

| DSPs | Source | Detail |
|---|---|---|
| 8 | `conv_c1_int8_Pipeline_VITIS_LOOP_114_24` (the PE_COUNT=8 MAC array) | 8× `mac_muladd_8s_8s_32s_32_4_1`, one int8×int8→int32 MAC unit per PE lane, 1 DSP each — genuinely integer |
| 3 | One shared `fmul_32ns_32ns_32_4_max_dsp_1` instance | Time-multiplexed across **three** separate float32 multiplies: `combined_scale = x_scale*w_scale`, `dequant = acc*combined_scale`, and (after the reciprocal-multiply change) `requant = dequant*inv_out_scale` — all three share one physical 3-DSP multiplier since they never execute concurrently |
| 2 | One `fadd_32ns_32ns_32_5_full_dsp_1` instance | The round-half-away-from-zero step's `requant + 0.5f` / `requant - 0.5f` |
| 0 | `fdiv_32ns_32ns_32_16_no_dsp_1` (`inv_out_scale = 1.0f/out_scale`) | Divider is LUT/fabric-only (no DSP), 15-cycle latency, but only instantiated/used **once** per call now, not once per pixel |
| **13** | **Total** | matches the reported synthesis number exactly |

So the answer: it's genuinely **not** the int8 MACs holding DSP usage up — those only need 8. The other 5 DSPs are the float32 requantization chain (the `fmul` for dequantize/rescale, the `fadd` for rounding) that this design still keeps as IEEE754 float rather than fixed-point/integer arithmetic. A fully-integer rescale (e.g. multiplying by a fixed-point approximation of `combined_scale/out_scale` and shifting, instead of a real `float` multiply) would plausibly get DSP down to just the 8 MAC-array DSPs — a ~43% drop from float32's 14, matching the kind of reduction the FF/BRAM numbers already show. **Not attempted this pass** — replacing the float rescale with a fixed-point one is a real architectural change (a shift-based or `ap_fixed`-based rescale instead of `float`), not a drop-in fix like the reciprocal-multiply was, and changes the rounding behavior enough that the Python reference and exact-match property would need to be re-derived around whatever fixed-point rescale scheme is chosen.

### Follow-up — genuine fixed-point requantization (`conv_c1_int8_fixedpoint.cpp`, new)

Implements the fixed-point rescale flagged as the next lever above: the requantization step (int32 accumulator → int8 output) is now a real **int64 multiply + rounding right-shift**, with zero floating-point operations anywhere in the per-pixel path — not `float`, not `fmul`/`fdiv`/`fadd`, nothing. Follows the same normalized-significand "quantization multiplier" technique TFLite/gemmlowp use (`tensorflow/lite/kernels/internal/quantization_util.cc`'s `QuantizeMultiplier()`), decomposing the real-valued ratio `combined_scale/out_scale` OFFLINE, once, in Python, into a Q31 fixed-point mantissa `M` and a shift `S` such that `(acc * M) >> S` (rounded) approximates `acc * (combined_scale/out_scale)`. `conv_c1_int8_fixedpoint.cpp`'s runtime interface carries only `M` (`ap_int<32>`) and `S` (`ap_int<8>`) — no float scale parameters at all, matching how real int8 accelerators actually work (the multiplier/shift are derived at model-conversion time, never recomputed by the chip).

**Files added (again all new — `conv_c1_int8.cpp` from the previous stage stays untouched as its own reference point, same convention throughout):**
- `hls/conv_c1/src/conv_c1_int8_fixedpoint.h`, `hls/conv_c1/src/conv_c1_int8_fixedpoint.cpp`
- `hls/conv_c1/tb/conv_c1_int8_fixedpoint_tb.cpp`, `hls/conv_c1/tb/generate_test_data_int8_fixedpoint.py`, `hls/conv_c1/tb/conv_c1_int8_fixedpoint_test_data.h`
- `hls/conv_c1/run_hls_int8_fixedpoint_pynqz2.tcl` — new solution, `conv_c1_int8_fixedpoint_proj`, same PYNQ-Z2 part

**`quantize_multiplier(real_multiplier)`** (in the generator script): `math.frexp(real_multiplier)` decomposes it into `significand * 2**exponent` with `significand ∈ [0.5, 1.0)`; `M = round(significand * 2**31)` (renormalized if rounding pushes it to exactly `2**31`) is a Q31 fixed-point mantissa that **always fits comfortably inside signed int32** (`0 < M < 2**31`, never close to overflow); `S = 31 - exponent` folds both the Q31 descale and the significand's own exponent into one combined shift, applied over a 64-bit intermediate. (This combines gemmlowp's own two split steps — a fixed 31-bit "doubling high mul" plus a separate variable "rounding divide by power-of-two" — into one wider shift; gemmlowp splits them for ARM NEON SIMD performance reasons, not because it's numerically required.) For this layer's real data: **M = 1,947,178,031, S = 40** — M sits comfortably below `2**31 - 1 = 2,147,483,647`, nowhere near overflow, exactly as intended.

**Recovering the reference accumulator, verified not assumed**: `conv2d_int()` computes the raw int32 accumulator internally but only returns the dequantized float (by its own design — see its docstring). The generator recovers it via the *exact inverse* of `conv2d_int()`'s own `acc.astype(np.float32) * combined_scale` step, then **checks** the round-trip is lossless before trusting it as the fixed-point reference's foundation:
```
Accumulator recovery verified exact. acc range: [-80059, 71713]
```
(an `assert` in the script, not a comment — it would have failed loudly rather than silently producing a wrong reference).

**Bit-exactness was checked, not assumed — again.** The reciprocal-multiply version's rounding was already known to disagree with true division on this data (documented above: 530/4704 elements at the float level, 0/4704 after rounding). The fixed-point version's rounding rule (`(acc*M + 2^(S-1)) >> S`, round-half-up on a non-negative product) is a *third*, independently-derived arithmetic path — nothing was assumed equivalent to either earlier version. The Python reference was regenerated from scratch with this exact integer arithmetic (`quantize_multiplier()` + `requantize_fixed()` in `generate_test_data_int8_fixedpoint.py`), and only then was the identical logic ported to `conv_c1_int8_fixedpoint.cpp`.

**C-simulation: exact match, confirmed independently of the previous two stages.**
```
Total output elements: 4704
Mismatches: 0
Max abs diff: 0
TEST PASSED -- HLS fixed-point INT8 output matches Python fixed-point reference exactly
```

**C-synthesis: succeeded. No floating-point core is generated at all** — `vitis_hls`'s RTL generation log shows exactly one arithmetic core (`mul_31ns_32s_63_2_1`, a plain 31×32-bit signed integer multiplier) plus the usual muxes; no `fmul`/`fadd`/`fdiv`/`sitofp`, confirming the design is genuinely, entirely integer now.

### Full three-way comparison: float32 → int8 (reciprocal-multiply) → int8 (fixed-point)

| Metric | float32 (`conv_c1_systolic`) | int8, reciprocal-multiply (`conv_c1_int8`) | int8, **fixed-point** (`conv_c1_int8_fixedpoint`) |
|---|---|---|---|
| BRAM | 18 (6%) | 5 (1%) | **5 (1%)** |
| DSP | 14 (6%) | 13 (5%) | **11 (5%)** |
| FF | 28,886 (27%) | 9,950 (9%) | **9,889 (9%)** |
| LUT | 19,572 (36%) | 16,461 (30%) | **16,124 (30%)** |
| Latency (cycles) | 144,591 | 223,780 | **124,991** |
| Timing (10ns/100MHz target) | Slack **−2.36ns** (violated) | Slack 0.00ns (met) | Slack **0.00ns** (met) |
| Estimated Fmax | 104.80 MHz | 136.99 MHz | **136.99 MHz** |

**The fixed-point version beats float32 `conv_c1_systolic` on every single metric** — the first INT8 variant of C1 to do so:
- BRAM −72%, DSP −21%, FF −66%, LUT −18%, **and latency −13.6%** (124,991 vs 144,591 cycles) — the earlier reciprocal-multiply version was still +54.8% *worse* than float32 on latency despite its resource wins; removing the last floating-point ops from the per-pixel path doesn't just save area, it recovers (and then some) the latency float32 held.
- Versus the reciprocal-multiply stage specifically: latency dropped a further 44.1% (223,780→124,991), and DSP dropped another 2 (13→11) — see the breakdown below for exactly where those 2 DSPs went.
- Timing still closes cleanly (0.00ns slack, same 136.99MHz Fmax as the reciprocal-multiply stage) — removing the float chain didn't cost anything on the timing side either.

**DSP breakdown (Bind Op Report) — now genuinely almost all MAC array:**

| DSPs | Source |
|---|---|
| 8 | The PE_COUNT=8 MAC array (`mac_muladd_8s_8s_32s_32_4_1` ×8, unchanged from both earlier INT8 stages) |
| 3 | `mul_31ns_32s_63_2_1` — the **one** integer multiply left in the whole design: `acc_relu (32-bit) × requant_mult (32-bit) → 64-bit` for the fixed-point rescale |
| 0 | Every add/sub/shift (bias add, ReLU compare, the rounding `+half`, the `>>` itself) — all fabric-only, no DSP |
| **11** | **Total** — down from the reciprocal-multiply version's 13 (8 MAC + 3 shared `fmul` + 2 `fadd`) |

This directly closes the loop on the earlier open question ("would a fully-integer rescale get DSP down to just the 8 MAC-array DSPs?"): **almost, not quite** — 8 of 11 (73%) is the MAC array, but the fixed-point rescale's own 32×32→64 integer multiply still needs 3 DSPs of its own (a wide multiply is a wide multiply, whether its operands are meant as floats or fixed-point integers — DSP48 slices are general MAC primitives, not float-specific hardware). What actually disappeared going from 13→11 is the **`fadd`** (2 DSP, the float rounding step's `+0.5f`/`-0.5f`) — the fixed-point version's rounding (`+half` before the shift) is a plain integer add, free on fabric, with no dedicated adder core needed at all. The remaining 3-DSP multiplier is the practical floor for this design unless the rescale multiply itself were narrowed (e.g. a smaller `M` with fewer significant bits) or replaced with shift-and-add — not attempted here, as `M`'s full 31-bit precision is what makes the fixed-point rescale numerically match its Python reference exactly in the first place.

**Next step**: `conv_c1_int8_fixedpoint.cpp` is the most hardware-realistic and best-performing C1 variant produced so far across all three axes (resources, latency, timing) — a reasonable candidate to standardize on when this pattern is extended to C3, rather than starting C3 from the float-requantize version.



## S2 Pooling — INT8 Quantization (new, PYNQ-Z2)

Extends the INT8 quantization work to S2 max pooling, per the project guide's INT8-conversion requirement. Unlike every one of conv_c1's INT8 stages, MaxPool needs **no scale factor and no requantization at all**: every value inside a 2×2 pooling window comes from the same activation tensor and therefore shares the exact same (positive) scale factor, so comparing raw int8 values directly gives exactly the same ordering — and so the same max — as comparing the real dequantized values would (max commutes with any positive affine rescale). This makes `pool_s2_int8` a much smaller, purely-structural change than `conv_c1_int8*`: same max-pooling logic, same `ARRAY_PARTITION` fix, `ap_int<8>` in place of `float`, nothing else.

**Files added (float32 `pool_s2.cpp`/`.h` stay untouched, same convention as every stage above):**
- `hls/pool/src/pool_s2_int8.h`, `hls/pool/src/pool_s2_int8.cpp`
- `hls/pool/tb/pool_s2_int8_tb.cpp`, `hls/pool/tb/generate_test_data_int8.py`, `hls/pool/tb/pool_s2_int8_test_data.h`
- `hls/pool/run_hls_int8_pynqz2.tcl` — new solution, `pool_s2_int8_proj`, same PYNQ-Z2 part (`xc7z020clg400-1`) as C1's INT8 work

**Test data**: rather than arbitrary hand-picked values, the input is a *real* int8 C1 activation map, produced by literally running C1's own proven int8 pipeline (`generate_test_data_int8.py` here re-derives it: real `Models/lenet5_relu.keras` weights, real MNIST test[0] image, `quantize_int_real()`/`quantize_activation()`, the verified accumulator recovery, ReLU on the accumulator, and `conv_c1_int8_fixedpoint`'s own `quantize_multiplier()`/`requantize_fixed()` fixed-point requantization) — so `pool_s2_int8` is tested on genuinely realistic quantized activations, not synthetic data. The reference output is simply `np.max()` over int8 values directly (no multiplier, no shift — see above).

**C-simulation: exact match.**
```
Total output elements: 1176
Mismatches: 0
Max abs diff: 0
TEST PASSED -- HLS INT8 pool_s2 output matches Python int8 reference exactly
```

**C-synthesis: succeeded**, all loop constraints satisfied, auto-pipelined at II=1 with no explicit `PIPELINE` pragma needed (same as the float32 version).

### Utilization — float32 (`pool_s2`) vs int8 (`pool_s2_int8`)

| Metric | float32 (`pool_s2`, per this doc's existing S2 log) | int8 (`pool_s2_int8`, PYNQ-Z2) | Change |
|---|---|---|---|
| Latency (cycles) | 1,186 | 1,181 | −0.4% (essentially unchanged) |
| II | 1 | 1 | unchanged |
| DSP | 0 | 0 | unchanged |
| FF | 576 | **158** | **−72.6%** |
| LUT | 589 | **370** | **−37.2%** |
| Timing / Fmax | not recorded in the earlier float32 run | Slack **+0.21ns** (met), Fmax **140.95MHz** | int8 confirmed meets timing; no float32 baseline was logged for this to compare against |

(Float32 numbers are the ones already on record above under "S2 Pooling — HLS Optimization Log (final, corrected)" — that run didn't log an Fmax/timing figure, so this table reports "not recorded" honestly rather than inventing one; every other column there was a real synthesis result, not an estimate.)

**Findings — matches the "should be simpler" expectation exactly:**

1. **DSP stays at 0 in both.** Confirms the existing doc's own note above ("DSP=0 because pure comparisons need no MAC hardware") holds regardless of datatype — max-pooling was never MAC-shaped, so there was never any DSP to remove.
2. **FF drops sharply (−73%) and LUT drops substantially (−37%)** — both entirely from the comparator itself, not from any structural change (the loop nest, the `ARRAY_PARTITION` fix, and the pipelining are byte-for-byte the same as float32). A float32 `>` comparison needs sign/exponent/mantissa-aware compare logic (and the registers to hold 32-bit operands through the pipeline); an `ap_int<8>` comparison is a trivial 8-bit magnitude compare with 4x narrower registers. This is the cleanest, least-confounded resource comparison of any layer converted so far — literally nothing but the datatype changed.
3. **Latency is essentially identical (1,181 vs 1,186 cycles, both II=1)** — pooling's pipeline depth is governed by the loop structure and the `ARRAY_PARTITION` fix, not by how wide the compare is; both datatypes were already fully pipelined at II=1, so there was no float-specific latency cost to remove in the first place (unlike C1, where the float32 requantization chain's multi-cycle `fmul`/`fadd`/`fdiv` cores were the actual latency cost — pooling never had an equivalent).

**Next step**: `pool_s2_int8` is done and validated. The guide's remaining named layers (C3, C5, F6, output dense) still need their own INT8 conversions; C3 in particular will need conv_c1_int8_fixedpoint's full multiplier+shift requantization pattern (it's a real MAC layer, not a comparison-only one like S2/S4), while S4 pooling can very likely reuse `pool_s2_int8.cpp`'s pattern directly, the same way the float32 S4 IP already reused S2's `ARRAY_PARTITION` fix with zero rework.



## S4 Pooling — INT8 Quantization (new, PYNQ-Z2)

Extends INT8 pooling to S4, following `pool_s2_int8`'s exact pattern (predicted above and confirmed here) — same max-pooling logic and `ARRAY_PARTITION cyclic factor=2` fix, resized for S4's actual dimensions (C3's 10×10×16 output → 5×5×16), `ap_int<8>` throughout, no rescaling needed for the same reason S2 needed none. Also mirrors the float32 `pool_s4.cpp`'s one small addition over `pool_s2.cpp`: an explicit `#pragma HLS PIPELINE II=1` on the inner channel loop, kept since the float32 version was **built correctly from the start** with it (no baseline-then-fix cycle — S4 reused S2's proven fix directly) — same approach applied here, no baseline int8 build attempted first either.

**Files added (float32 `pool_s4.cpp`/`.h` stay untouched):**
- `hls/pool_s4/src/pool_s4_int8.h`, `hls/pool_s4/src/pool_s4_int8.cpp`
- `hls/pool_s4/tb/pool_s4_int8_tb.cpp`, `hls/pool_s4/tb/generate_test_data_int8.py`, `hls/pool_s4/tb/pool_s4_int8_test_data.h`
- `hls/pool_s4/run_hls_int8_pynqz2.tcl` — new solution, `pool_s4_int8_proj`, same PYNQ-Z2 part

**Test data — chaining further than S2 needed to:** S4 sits after *both* C1 and C3, and no HLS C3 IP exists yet (only C1 has been converted to INT8 hardware so far), so producing a real (non-synthetic) 10×10×16 input meant running the full int8 pipeline in **Python**, chaining four stages in `generate_test_data_int8.py`:
1. C1 conv (int8×int8→int32 MAC, ReLU on the accumulator, fixed-point requantize — same steps as `conv_c1_int8_fixedpoint`'s own generator) → real 28×28×6 int8 activations.
2. S2 max pool (int8, no rescale) → 14×14×6. Its output shares C1's exact `out_scale` untouched, since pooling never changes the scale factor.
3. C3 conv (`padding="valid"`, C1's `out_scale` as C3's *input* scale since S2 didn't change it, C3's own weights quantized fresh, same MAC → verified-accumulator-recovery → ReLU → fixed-point-requantize pattern as C1) → real 10×10×16 int8 activations. This is pure-Python int8 arithmetic reusing the proven pattern — no synthesizable C3 kernel needed for this purpose, exactly the "simpler proxy" question resolved in favor of doing the real chain, since the Python-only arithmetic was cheap to build correctly (verified exact accumulator recovery at C3 too, same `assert` pattern as every prior stage).
4. That real C3 output is what `pool_s4_int8` is actually tested against — genuinely representative data, not synthetic.

**C-simulation: exact match.**
```
Total output elements: 400
Mismatches: 0
Max abs diff: 0
TEST PASSED -- HLS INT8 pool_s4 output matches Python int8 reference exactly
```

**C-synthesis: succeeded on the first attempt** (as expected — no baseline-then-fix cycle needed, same as float32 S4), all loop constraints satisfied.

### Utilization — float32 (`pool_s4`) vs int8 (`pool_s4_int8`)

| Metric | float32 (`pool_s4`, "built correctly from start") | int8 (`pool_s4_int8`, PYNQ-Z2) | Change |
|---|---|---|---|
| Latency (cycles) | 409 (411 total) | 406 (407 total) | −0.7% (essentially unchanged) |
| II | 1 | 1 | unchanged |
| DSP | 0 | 0 | unchanged |
| FF | 565 | **147** | **−74.0%** |
| LUT | 545 | **326** | **−40.2%** |
| Fmax | 140.71 MHz | **142.57 MHz** | +1.3% |
| Timing | clears 100MHz target | Slack **+0.29ns** (met) | both close timing |

**Findings — same story as S2, confirmed at a second (differently-shaped, differently-sourced) layer:**

1. **DSP stays at 0 in both**, latency is essentially unchanged (409→406 cycles, both II=1) — pooling's cost structure is set by the loop/partition structure, not the datatype, exactly as S2 already showed.
2. **FF drops even further proportionally than S2's did (−74% vs −73%), LUT similarly (−40% vs −37%)** — consistent with the same underlying cause (a trivial 8-bit magnitude compare vs. a float32 sign/exponent/mantissa-aware compare, at 4x narrower registers), just at S4's own scale (16 channels vs S2's 6, smaller spatial extent).
3. **Confirms the reusability prediction from S2's own write-up** ("S4 pooling can very likely reuse `pool_s2_int8.cpp`'s pattern directly, the same way the float32 S4 IP already reused S2's fix with zero rework") — it did, with the same resource-savings profile showing up again almost exactly.

**Next step**: both pooling layers are now done in INT8. C3 (a real MAC layer) is the next natural target and is the only remaining piece needed before S4's own Python-side test-data chain could be replaced with a genuine HLS-verified C3 IP output; C5, F6, and the output dense layer remain unconverted.



## C3 Convolution — INT8 Quantization (new, PYNQ-Z2)

Converts C3 to INT8, combining two already-proven pieces rather than starting from scratch: the **architecture** from `conv_c3_partialsum_lut.cpp` (the kept, superior stage of C3's own float32 optimization arc — PE_COUNT=6 partial-sum split with division-free LUT-based `(i,kr,kc)` decode, same flattened `mac_idx`/`(m,p)` loop nest that lets HLS auto-flatten the outer `o`/`r`/`c` loop into the inner pipeline), and the **requantization** from `conv_c1_int8_fixedpoint.cpp` (genuine fixed-point: int64 multiply by a Q31 mantissa + rounding right-shift, zero floating point anywhere in the per-pixel path).

**Files added (all existing float32 C3 variants — `conv_c3.cpp`, `conv_c3_partialsum.cpp`, `conv_c3_partialsum_roundrobin.cpp`, `conv_c3_partialsum_lut.cpp` — stay untouched):**
- `hls/conv_c3/src/conv_c3_int8_fixedpoint.h`, `hls/conv_c3/src/conv_c3_int8_fixedpoint.cpp`
- `hls/conv_c3/tb/conv_c3_int8_fixedpoint_tb.cpp`, `hls/conv_c3/tb/generate_test_data_int8_fixedpoint.py`, `hls/conv_c3/tb/conv_c3_int8_fixedpoint_test_data.h`
- `hls/conv_c3/run_hls_int8_fixedpoint_pynqz2.tcl` — new solution, `conv_c3_int8_fixedpoint_proj`, PYNQ-Z2 part (`xc7z020clg400-1`)

### Python reference built and verified FIRST, independently of C1's

Per the same rigor as every prior INT8 stage: the Python fixed-point reference was built and checked before any C++ was written. C3 sits after both C1 and S2, so producing a real (non-synthetic) 14×14×6 input meant re-running C1's own int8 pipeline first (duplicated fresh in this script, not imported — see the script's own docstring) purely to get a real activation map to pool and convolve over. But **C3's own quantization is derived and verified completely independently**, not reused from C1:

- C3's weights quantized fresh via `quantize_int_real()`; C3's *input* scale is C1's `out_scale` carried through S2 unchanged (pooling doesn't rescale).
- C3's raw int32 accumulator recovered from `conv2d_int()`'s float output via its exact inverse, and the round-trip **verified exact** — C3's own `assert`, independent of C1's:
  ```
  C3 accumulator recovery verified exact. acc range: [-78100, 53653]
  ```
- C3's own multiplier derived via `quantize_multiplier()`:
  ```
  C3 real_multiplier = combined_scale/out_scale = 0.002367062526883068
  C3 quantize_multiplier -> M = 1301306386, S = 39
  ```
  (M comfortably inside signed int32, same as C1's M was — nowhere near `2**31 - 1`.)

**A bug was caught at this stage, before it ever reached C++.** The first cross-check (comparing the fixed-point path against an independently-computed float path) showed 471/1600 mismatches — not the "small number at rounding boundaries" expected. Traced it: the cross-check itself double-applied `combined_scale` (multiplying the already-dequantized `relu_out` by `M/2^S ≈ combined_scale/out_scale` a *second* time, instead of just dividing by `out_scale` directly) — a bug in the verification script, not in `requantize_fixed()` or the actual reference. Fixed the cross-check to divide the already-dequantized value by `out_scale` directly (genuinely independent of `M`/`S`, analogous to the reciprocal-multiply-vs-division check done for C1) and re-ran:
```
Cross-check vs. independent float division (no M/S involved): 0/1600 int8 buckets differ; max abs diff in the pre-clip requantized value: 0.0
```
Exact agreement, confirmed **before** writing any HLS C++ — exactly the "confirm bit-exact match in Python before touching HLS" this stage asked for, and a concrete example of why that check matters even when the underlying technique is already proven.

### C-simulation: exact match

```
Total output elements: 1600
Mismatches: 0
Max abs diff: 0
TEST PASSED -- HLS fixed-point INT8 C3 output matches Python fixed-point reference exactly
```

### C-synthesis

The known accumulator-latency floor from C3's own float32 log reproduces here too (expected, not a regression): `Loop Constraint Status: All loop constraints were NOT satisfied` — same mechanism documented in C3's float32 optimization log above (the MAC reduction's carried dependency caps II below the II=1 target). The int8 version's inner MAC pipeline (`VITIS_LOOP_61_5`, trip 25 = `MACS_PER_PE`) achieves **II=3**, an improvement over float32's **II=4** — a narrower int32 accumulator add resolves faster than a 32-bit float `fadd`'s multi-cycle latency, the same mechanism C1's stages already showed.

### Utilization — float32 (`conv_c3_partialsum_lut`) vs int8 (`conv_c3_int8_fixedpoint`)

| Metric | float32 (`conv_c3_partialsum_lut`, `xc7z020-clg484-1`) | int8, fixed-point (`conv_c3_int8_fixedpoint`, PYNQ-Z2) | Change |
|---|---|---|---|
| BRAM | 0 | 0 | unchanged |
| DSP | 12 (5%) | **9 (4%)** | **−25.0%** |
| FF | 3,013 (2%) | **1,523 (1%)** | **−49.4%** |
| LUT | 6,250 (11%) | **4,919 (9%)** | **−21.3%** |
| Latency (cycles) | 160,047 | **150,401** | **−6.0%** |
| Inner-loop II (achieved/target) | 4/1 | **3/1** | improved, target still not met |
| Timing (10ns/100MHz target) | Slack **−1.66ns** (violated) | Slack **0.00ns** (met) | int8 closes timing, float32 doesn't |
| Estimated Fmax | 111.66 MHz | **136.99 MHz** | +22.7% |

**C3 beats float32 on every single metric, same as C1's fixed-point stage did** — the pattern holds at a second, larger, more complex layer (PE_COUNT=6 vs C1's PE_COUNT=8, 16 output channels vs 6, an existing LUT-decode optimization already in play rather than a from-scratch baseline).

**DSP breakdown (Bind Op Report) — confirms exactly where the 12→9 DSPs went:**

| DSPs | float32 (`conv_c3_partialsum_lut`) | int8 (`conv_c3_int8_fixedpoint`) |
|---|---|---|
| PE array (PE_COUNT=6) | 12 (2 DSP/lane — float MAC's `fmul`+`fadd`) | **6** (1 DSP/lane — `mac_muladd_8s_8s_20s`, genuine int8×int8→int32) |
| Rescale | 0 (float ReLU is a free same-cycle compare) | **3** (`mul_31ns_32s_63`, the one int64 fixed-point rescale multiply — same single core as C1's) |
| **Total** | **12** | **9** |

This is a cleaner accounting than C1's was: float32's 6 MAC lanes cost 2 DSP each (12 total) with *no* separate rescale cost (ReLU was free); int8's 6 MAC lanes cost 1 DSP each (6 total, halved) but add back 3 DSP for the one rescale multiply the float32 version never needed — netting a 25% reduction rather than a larger one, because narrowing the MAC itself (float→int8) saves proportionally more here than the fixed 3-DSP rescale multiply costs.

**Findings, consistent with C1's own fixed-point stage:**

1. **The fixed-point technique transfers directly to a more complex, already-optimized layer with zero surprises** — same architecture-plus-requantization composition, same exact-match discipline, same result shape (beats float32 everywhere).
2. **Latency actually improved (−6.0%)** despite C3's accumulator-limited floor persisting in both versions — the narrower/faster int8 MAC chain (II=3 vs II=4) more than compensates for whatever overhead the rescale step adds, unlike C1's *first* INT8 stage (the float-division version), which was initially *slower* than float32 before the reciprocal-multiply and fixed-point fixes closed that gap. Going straight to the fixed-point version here (skipping the float-requantize baseline entirely, since it was already known to be inferior) avoided repeating that detour.
3. **BRAM stays at 0 in both** — C3 was never BRAM-bound (no AXI/line-buffer machinery here, unlike C1's standalone-IP interface style; C3 is called as a sub-function, plain `ap_memory` arrays only).

**Next step**: C1 and C3 (the two real MAC/conv layers) and both pooling layers (S2, S4) are now all done in INT8, all following the same proven pattern (fixed-point PE-array MAC + `quantize_multiplier()`-derived rescale for conv layers, plain int8 compare for pooling). C5, F6, and the output dense layer (softmax) remain — C5/F6 should extend the same conv-layer pattern (MAC + fixed-point requantize, no LUT-decode needed since dense layers don't have `(i,kr,kc)` coordinates to decode), while the output layer's softmax stage will need its own INT8-appropriate treatment (softmax doesn't requantize to int8 the same way ReLU layers do). With S4's real C3-sourced test data no longer needing a Python-only C3 stand-in, S4's testbench could optionally be regenerated against the real `conv_c3_int8_fixedpoint` HLS output instead — not required (the Python arithmetic is already proven identical), but available if end-to-end HLS-to-HLS chaining is ever wanted.



## C5 Dense — INT8 Quantization (new, PYNQ-Z2)

Combines two already-proven pieces, same composition approach as C3: the **architecture** from `dense_c5_partialsum.cpp` (the kept PE_COUNT=4 partial-sum split — 400-term reduction across 4 independent accumulator chains, flattened `mac_idx`/`(m,p)` loop nest), and the **requantization** from `conv_c1_int8_fixedpoint.cpp`/`conv_c3_int8_fixedpoint.cpp` (genuine fixed-point: int64 multiply by a Q31 mantissa + rounding right-shift, zero floating point anywhere in the per-neuron path).

**Files added (existing float32 variants — `dense_c5.cpp`, `dense_c5_partialsum.cpp` — stay untouched):**
- `hls/dense_c5/src/dense_c5_int8_fixedpoint.h`, `hls/dense_c5/src/dense_c5_int8_fixedpoint.cpp`
- `hls/dense_c5/tb/dense_c5_int8_fixedpoint_tb.cpp`, `hls/dense_c5/tb/generate_test_data_int8_fixedpoint.py`, `hls/dense_c5/tb/dense_c5_int8_fixedpoint_test_data.h`
- `hls/dense_c5/run_hls_int8_fixedpoint_pynqz2.tcl` — new solution, `dense_c5_int8_fixedpoint_proj`, PYNQ-Z2 part (`xc7z020clg400-1`)

### Python reference built and verified FIRST, chaining the real C1→S2→C3→S4 int8 pipeline

Per the same rigor as every prior INT8 stage: the Python fixed-point reference was built and checked before any C++ was written. C5 sits after C1, S2, C3, and S4, so producing a real (non-synthetic) 400-element input meant re-running the full C1→S2→C3→S4 int8 pipeline first (duplicated fresh in this script, not imported — see the script's own docstring), then flattening S4's 5×5×16 output with NumPy's default C-order `.flatten()` (the same call `hls/lenet5_top/tb/generate_test_data.py` uses, matching Keras' `Flatten()` layer) into the 400-element vector `dense_c5_int8_fixedpoint` actually consumes. **C5's own quantization is derived and verified completely independently** of every earlier layer:

- C5's weights quantized fresh via `quantize_int_real()`; C5's *input* scale is C3's `out_scale` carried through S4 and the flatten unchanged (neither pooling nor flattening rescales).
- C5's raw int32 accumulator recovered from `dense_int()`'s float output via its exact inverse, and the round-trip **verified exact** — C5's own `assert`, independent of every earlier layer's:
  ```
  C5 accumulator recovery verified exact. acc range: [-95548, 58826]
  ```
- C5's own multiplier derived via `quantize_multiplier()`:
  ```
  C5 real_multiplier = combined_scale/out_scale = 0.0021589092456742068
  C5 quantize_multiplier -> M = 1186872909, S = 39
  ```
  (M comfortably inside signed int32.)
- Cross-check vs. an independently-computed float division (no `M`/`S` involved at all): **0/120 int8 buckets differ**, max abs diff in the pre-clip requantized value 0.0 — confirmed exact agreement before writing any HLS C++.

### C-simulation: exact match

```
Total output elements: 120
Mismatches: 0
Max abs diff: 0
TEST PASSED -- HLS fixed-point INT8 C5 output matches Python fixed-point reference exactly
```

### C-synthesis

**A different bottleneck class than every earlier INT8 stage.** C1's and C3's `Loop Constraint Status: NOT satisfied` came from the accumulator's own recurrence (still true here too, in principle), but the HLS scheduler log for C5 instead reports the *actual* limiter directly:
```
WARNING: [HLS 200-885] The II Violation ...: Unable to schedule 'load' operation ('input_r_load_1') on
array 'input_r' due to limited memory ports (II = 1). ... Pipelining result : Target II = 1, Final II = 2
```
Unlike the float32 version (accumulator-latency-limited at II=4 by the `fadd`'s 4-cycle latency — see this layer's own optimization log above), int8's integer add resolves in a single cycle, so the accumulator recurrence stops being the bottleneck; instead, the PE_COUNT=4-way unrolled reads of `input`/`weights` per iteration now saturate the plain `ap_memory` interface's 2 read ports first, capping II at 2 rather than 1. Still a clear win over float32's II=4 floor either way.

### Utilization — float32 (`dense_c5_partialsum`) vs int8 (`dense_c5_int8_fixedpoint`)

| Metric | float32 (`dense_c5_partialsum`, `xc7z020-clg484-1`) | int8, fixed-point (`dense_c5_int8_fixedpoint`, PYNQ-Z2) | Change |
|---|---|---|---|
| BRAM | 0 | 0 | unchanged |
| DSP | 7 (3%) | 7 (3%) | unchanged |
| FF | 1,863 (1%) | **740 (~0%)** | **−60.3%** |
| LUT | 2,253 (4%) | **1,827 (3%)** | **−18.9%** |
| Latency (cycles) | 48,031 | **25,801** | **−46.3%** |
| Inner-loop II (achieved/target) | 4/1 | **2/1** | improved, target still not met |
| Timing (10ns/100MHz target) | Slack **−1.66ns** (violated) | Slack **0.14ns** (met) | int8 closes timing, float32 doesn't |
| Estimated Fmax | 111.66 MHz | **139.72 MHz** | +25.1% |

**DSP breakdown (Bind/Instance report):**

| DSPs | float32 (`dense_c5_partialsum`) | int8 (`dense_c5_int8_fixedpoint`) |
|---|---|---|
| PE array (PE_COUNT=4) | — (not broken out in the float32 report) | **4** (1 DSP/lane — `mac_muladd_8s_8s_22s`, genuine int8×int8→int32) |
| Rescale | 0 (float ReLU is a free same-cycle compare) | **3** (`mul_31ns_32s_63`, the same single fixed-point rescale core as C1's/C3's) |
| **Total** | **7** | **7** |

DSP total lands exactly unchanged (7→7) — the first INT8 stage in this project where that happens: narrowing the 4 MAC lanes from float to int8 drops their cost to 1 DSP/lane (4 total), but that saving is exactly offset by the 3 DSP the one fixed-point rescale multiply adds (which the float32 version, with its free ReLU compare, never needed) — a coincidence of this layer's specific PE_COUNT/lane-cost combination, not a general rule.

**Findings, consistent with C1's and C3's own fixed-point stages:**

1. **The fixed-point technique transfers directly to a dense (fully-connected) layer with zero surprises** — same architecture-plus-requantization composition, same exact-match discipline, same result shape (beats or matches float32 on every metric, never worse).
2. **The bottleneck mechanism itself changed, not just its severity** — this is the first INT8 stage where the II floor comes from memory ports rather than the accumulator's arithmetic latency, because int8 addition is fast enough that the recurrence is no longer the critical path. Confirms the layer's own float32 diagnosis (`fadd`'s 4-cycle latency) really was arithmetic-specific, not structural.
3. **Latency drops nearly 2x (−46.3%)** — both because II improved (4→2) and because the fixed-point requantize is cheaper per neuron than float32's ReLU-then-nothing (no dequant/requant round-trip needed at all, the accumulator's sign check IS the ReLU).

**Next step**: F6 is the next natural target — same PE_COUNT=4 architecture at F6's own dimensions (120→84), chaining C5's own real int8 output as F6's real input.



## F6 Dense — INT8 Quantization (new, PYNQ-Z2)

Identical composition to C5: the **architecture** from `dense_c5_int8_fixedpoint.cpp`'s own PE_COUNT=4 partial-sum split (itself `dense_c5_partialsum.cpp`'s proven pattern, applied directly at F6's dimensions — 120 inputs, 84 outputs — same reusable-IP principle the float32 `dense_f6.cpp` already used against `dense_c5_partialsum.cpp`), and the same fixed-point requantization as every INT8 stage above.

**Files added (existing float32 variant — `dense_f6.cpp` — stays untouched):**
- `hls/dense_f6/src/dense_f6_int8_fixedpoint.h`, `hls/dense_f6/src/dense_f6_int8_fixedpoint.cpp`
- `hls/dense_f6/tb/dense_f6_int8_fixedpoint_tb.cpp`, `hls/dense_f6/tb/generate_test_data_int8_fixedpoint.py`, `hls/dense_f6/tb/dense_f6_int8_fixedpoint_test_data.h`
- `hls/dense_f6/run_hls_int8_fixedpoint_pynqz2.tcl` — new solution, `dense_f6_int8_fixedpoint_proj`, PYNQ-Z2 part (`xc7z020clg400-1`)

### Python reference built and verified FIRST, chaining the real C1→S2→C3→S4→C5 int8 pipeline

Same rigor as every prior stage: F6's real 120-element input is C5's own real int8 output, produced by re-running the full C1→S2→C3→S4→flatten→C5 int8 pipeline fresh in this script (not imported — C5's own M/S are re-derived here too, independently of `dense_c5`'s own generator script). **F6's own quantization is derived and verified completely independently** of every earlier layer:

- F6's weights quantized fresh via `quantize_int_real()`; F6's *input* scale is C5's own `out_scale`.
- F6's raw int32 accumulator recovered from `dense_int()`'s float output via its exact inverse, and the round-trip **verified exact** — F6's own `assert`:
  ```
  F6 accumulator recovery verified exact. acc range: [-29327, 35373]
  ```
- F6's own multiplier derived via `quantize_multiplier()`:
  ```
  F6 real_multiplier = combined_scale/out_scale = 0.0035903089928803603
  F6 quantize_multiplier -> M = 1973793242, S = 39
  ```
- Cross-check vs. an independently-computed float division: **0/84 int8 buckets differ**, max abs diff 0.0 — confirmed exact before writing any HLS C++.

### C-simulation: exact match

```
Total output elements: 84
Mismatches: 0
TEST PASSED -- HLS fixed-point INT8 F6 output matches Python fixed-point reference exactly
```

### C-synthesis

Same memory-port-limited mechanism as C5 (not the accumulator recurrence): `Unable to schedule 'load' operation ('input_r_load_1') ... due to limited memory ports`, `Final II = 2` against a target of 1 — confirms C5's diagnosis generalizes, independent of N_IN (120 here vs. C5's 400).

### Utilization — float32 (`dense_f6`) vs int8 (`dense_f6_int8_fixedpoint`)

| Metric | float32 (`dense_f6`, `xc7z020-clg484-1`) | int8, fixed-point (`dense_f6_int8_fixedpoint`, PYNQ-Z2) | Change |
|---|---|---|---|
| BRAM | 0 | 0 | unchanged |
| DSP | 11 (5%) | 11 (5%) | unchanged |
| FF | 1,673 (1%) | **887 (~0%)** | **−47.0%** |
| LUT | 1,988 (3%) | **1,749 (3%)** | **−12.0%** |
| Latency (cycles) | 10,113 | **6,469** | **−36.0%** |
| Inner-loop II (achieved/target) | 4/1 | **2/1** | improved, target still not met |
| Timing (10ns/100MHz target) | Slack **−1.66ns** (violated) | Slack **0.25ns** (met) | int8 closes timing, float32 doesn't |
| Estimated Fmax | 111.66 MHz | **141.79 MHz** | +27.0% |

**DSP breakdown (Bind/Instance report):**

| DSPs | float32 (`dense_f6`) | int8 (`dense_f6_int8_fixedpoint`) |
|---|---|---|
| PE array (PE_COUNT=4) | — (not broken out in the float32 report) | **8** (2 DSP/lane) |
| Rescale | 0 (float ReLU is a free same-cycle compare) | **3** (`mul_31ns_32s_63`, same rescale core) |
| **Total** | **11** | **11** |

DSP total again lands exactly unchanged (11→11) — same coincidence as C5, though the split differs (F6's PE array binds at 2 DSP/lane here vs. C5's 1 DSP/lane for the identical PE_COUNT=4 architecture; a scheduling/binding artifact of the different N_IN/MACS_PER_PE trip count, not a difference in the source code).

**Findings, consistent with C5's own fixed-point stage:**

1. **The fixed-point technique and the PE_COUNT=4 architecture both transfer unchanged from C5 to F6**, exactly the "only dimensions changed" reuse the float32 `dense_f6.cpp` already demonstrated against `dense_c5_partialsum.cpp` — now confirmed true for the INT8/fixed-point version too.
2. **The memory-port-limited II=2 floor is independent of N_IN** (120 here vs. C5's 400) — same mechanism, same final II, consistent with C5's own finding that this is a port-count property of the `ap_memory` interface, not something that scales with problem size.
3. **DSP staying exactly flat (11→11) at both C5 and F6, despite the two designs splitting those DSPs differently between the MAC array and the rescale multiply**, is the first time in this project's INT8 conversion that resource neutrality (rather than a clear reduction) is the headline DSP result — still a strict win overall given the simultaneous latency, FF, LUT, and timing improvements.



## Output Dense (Softmax) — INT8 Quantization (new, PYNQ-Z2)

Unlike every earlier INT8 layer (C1, C3, C5, F6), this is deliberately **not** a "`_fixedpoint`" design, and the reasoning is worth stating before the implementation: fixed-point rescaling exists in this project specifically to avoid floating-point cost on a *repeated, per-pixel* datapath that feeds another quantized layer — that's what justified paying for `quantize_multiplier()`'s Q31-mantissa-plus-shift machinery at every earlier stage. The Output layer's softmax is the network's **final** activation: its result is read as a probability distribution by a human or a downstream host, not fed forward into another int8 MAC, and it runs on only `N_OUT=10` values once per inference, not once per pixel. Paying fixed-point complexity for that would add hardware cost for no benefit. So this IP keeps the proven float32 `dense_output.cpp`'s own two-stage structure exactly, changing only Stage 1's datapath:

- **Stage 1 (MAC accumulation)** — genuine int8×int8→int32 integer arithmetic, same `PE_COUNT=4` partial-sum architecture as `dense_c5_int8_fixedpoint.cpp`/`dense_f6_int8_fixedpoint.cpp`, into a local `ap_int<32> logits[N_OUT]` array. No activation, no requantization to int8 — this accumulator is never consumed by another quantized layer, so unlike every earlier layer there is nothing to rescale it *for*.
- **Stage 2 (softmax)** — dequantizes Stage 1's accumulator to float via the same `combined_scale = x_scale*w_scale` every conv/dense layer already computes, then reuses `dense_output.cpp`'s own proven max-subtract/exp/sum/divide **unchanged**. This mirrors `src/integer_layers.py`'s own `dense_int(..., activation="softmax")`, which does exactly this: integer MAC, then float dequantize+softmax. `x_scale`/`w_scale` are passed in as plain float scalars, the same convention `conv_c1_int8.h` (the project's own precedent for an "int8 MAC + float finish" design, before `_fixedpoint` existed) already used.

**Files added (existing float32 IP — `dense_output.cpp` — stays untouched):**
- `hls/dense_output/src/dense_output_int8.h`, `hls/dense_output/src/dense_output_int8.cpp`
- `hls/dense_output/tb/dense_output_int8_tb.cpp`, `hls/dense_output/tb/generate_test_data_int8.py`, `hls/dense_output/tb/dense_output_int8_test_data.h`
- `hls/dense_output/run_hls_int8_pynqz2.tcl` — new solution, `dense_output_int8_proj`, PYNQ-Z2 part (`xc7z020clg400-1`)

### Python reference built and verified FIRST, chaining the full real C1→S2→C3→S4→C5→F6 int8 pipeline

Same rigor as every prior stage: the Output layer's real 84-element input is F6's own real int8 output, produced by re-running the full C1→S2→C3→S4→flatten→C5→F6 int8 pipeline fresh in this script (not imported — every earlier layer's own M/S is re-derived here too, independently of its own generator script). **The Output layer's own quantization is derived and verified completely independently** of every earlier layer:

- Output's weights quantized fresh via `quantize_int_real()`; Output's *input* scale is F6's own `out_scale`.
- Output's raw int32 accumulator (the pre-softmax logits) recovered from `dense_int()`'s float output via its exact inverse, and the round-trip **verified exact** — Output's own `assert`:
  ```
  Output accumulator recovery verified exact. acc (logits) range: [-60804, 65305]
  ```
- Unlike every earlier layer, there is **no `quantize_multiplier()` call here at all** — the reference re-dequantizes the verified accumulator with `combined_scale` directly (`acc_recovered.astype(float32) * combined_scale`, asserted equal to `dense_int()`'s own float output) and runs the exact same NumPy max-subtract/exp/sum/divide the network's own float32 reference uses:
  ```
  Output softmax reference: [4.04e-09, 7.13e-07, 2.39e-06, 7.28e-05, 5.21e-12, 3.07e-07, 9.44e-12, 9.9992e-01, 8.00e-08, 4.10e-08]
  Sum: 0.99999994, predicted label = 7 (true label = 7)
  ```

### C-simulation: matches within tight float tolerance (not bit-exact — by design)

Stage 1's int32 accumulator is genuine integer arithmetic and deterministic — any error there would show up as a large discrepancy, not a small one. Stage 2 is intentionally float, so (same standard this project's own `lenet5_top_tb.cpp` already uses for its softmax output, which matched its NumPy reference to 6.5e-11, not bit-for-bit) the testbench checks a tight numerical tolerance rather than exact equality:
```
output[7] = 0.999924  (expected 0.999924)
Sum of outputs: 1
Predicted label: 7 (true label: 7)
Max abs diff vs. Python reference: 1.19209e-07
TEST PASSED -- HLS INT8 Output softmax matches Python reference within tolerance
```
Max absolute difference across all 10 outputs is 1.19e-7 — single-precision float rounding noise from `std::exp()` vs. NumPy's `exp()`, not an arithmetic error — and the predicted class (7) matches the true MNIST label, exactly like the float32 baseline.

### C-synthesis

**Per-stage breakdown** (same structure as the float32 baseline's own table above):

| Stage | Latency (cycles) | II (achieved/target) | Trip count | DSP | Notes |
|---|---|---|---|---|---|
| MAC accumulation | 520 (outer, not flattened) / inner pipeline 47 | **2 / 1** | 21×10 = 210 | 4 | Same PE_COUNT=4 architecture as C5/F6; memory-port-limited at II=2 like both (not accumulator-limited — int8 add is fast enough), same mechanism, unflattened outer loop this time (10 trips of the inner pipeline, vs. float32's fully-flattened single 210-iteration pipeline) |
| Dequantize (int32→float ×10) | 23 | — | 10 | 0 | New stage, not present in float32 (its accumulator was already float) — cheap, not a per-pixel cost |
| Find max logit | 22 | 2 / 1 | 9 | 0 | Identical cycle count and II to float32 — this stage's code and datapath are completely unchanged |
| exp + sum | 70 | 5 / 1 | 10 | 9 | Identical cycle count and II to float32 (same `std::exp` code) — DSP differs (9 vs. float32's 7), a resource-binding effect of the new dequantize stage changing what gets shared, not a code change |
| Normalize (divide) | 29 | 1 / 1 | 10 | 0 | Identical to float32 — unchanged code |

**Loop Constraint Status: NOT satisfied** (MAC stage's II=2 floor, same as C5/F6) — expected, not a regression.

### Utilization — float32 (`dense_output`) vs int8 (`dense_output_int8`)

| Metric | float32 (`dense_output`, `xc7z020-clg484-1`) | int8 (`dense_output_int8`, PYNQ-Z2) | Change |
|---|---|---|---|
| BRAM | 0 | 0 | unchanged |
| DSP | 14 (6%) | **16 (7%)** | **+14.3% (worse)** |
| FF | 2,778 (2%) | **1,834 (1%)** | **−34.0%** |
| LUT | 3,721 (6%) | **2,973 (5%)** | **−20.1%** |
| Latency (cycles) | 999 | **678** | **−32.1%** |
| Timing (10ns/100MHz target) | Slack **−1.66ns** (violated) | Slack **0.04ns** (met) | int8 closes timing, float32 doesn't (barely — smallest margin of any layer so far) |
| Estimated Fmax | 111.66 MHz | **137.82 MHz** | +23.4% |

**DSP breakdown (Instance report) — the one metric that got WORSE, and exactly why:**

| DSPs | float32 (`dense_output`) | int8 (`dense_output_int8`) |
|---|---|---|
| MAC stage (PE_COUNT=4) | 5 | **4** |
| Dequantize (new stage) | — (didn't exist) | **0** |
| Find max | 0 | 0 |
| exp + sum | 7 | **9** |
| Normalize | 0 | 0 |
| Shared instance (`faddfsub`/`fmul`, resource-shared across stages) | 2 | **3** |
| **Total** | **14** | **16** |

**Findings:**

1. **This is the first INT8 layer in this project where total DSP goes up, not down or flat** — and the cause is fully accounted for: the MAC stage *does* shrink as expected (5→4, int8's narrower multiply), but that's a small saving here because `N_OUT=10` was already cheap for float32 to resource-share efficiently (5 DSP for 4 lanes, not the naive 8). The increase comes entirely from Stage 2 — the `exp`+`sum` stage costs 2 more DSP (7→9) purely from a resource-binding side effect of adding the new dequantize stage ahead of it (same `std::exp` source code, confirmed by find-max and normalize being byte-for-byte identical in cycle count to float32), plus the shared floating-point instance costing 1 more DSP (2→3). None of this is a design flaw — it's the honest cost of the design decision explained at the top: keeping Stage 2 in float (cheap in LUT/FF/latency terms, and correct) has a small DSP price at this layer's tiny N_OUT=10 scale, and it's worth paying since Stage 2 was never the resource bottleneck for either version.
2. **Every other metric still improves, most by a wide margin**: FF −34.0%, LUT −20.1%, latency −32.1% (largely from Stage 1's MAC stage no longer needing full loop-flattening to beat float32's flattened-but-II=4 baseline), and timing flips from a −1.66ns violation to a (barely) met 0.04ns slack — the smallest positive margin of any INT8 conversion in this project so far, worth flagging if this layer is ever pushed further (e.g. more slack headroom might be wanted before board bring-up).
3. **Stage 2's untouched stages (find max, normalize) reproduce float32's exact cycle counts and II**, confirming the two-stage architecture transplant was clean — the only real changes are Stage 1's datapath (float MAC → int8 MAC) and the one new dequantize stage Stage 2 needed to bridge them, exactly as designed.
4. **The design choice to keep softmax in float, rather than inventing a fixed-point exp/normalize, is validated by these numbers**: the "cost" is 2 DSP at the whole design's smallest layer, in exchange for reusing proven, already-optimized float32 code unchanged for two of Stage 2's three sub-stages — a good trade for a stage that runs once per inference on 10 values, not once per pixel.

**Next step**: all seven layers of the network (C1, C3 — conv/MAC; S2, S4 — pooling; C5, F6, Output — dense) now have INT8 implementations, each independently verified against its own from-scratch Python reference and chained through the real upstream int8 pipeline rather than synthetic data. The natural next step is assembling an INT8 `lenet5_top` that chains all seven INT8 IPs end-to-end (mirroring the existing float32 `lenet5_top.cpp`), to get one real utilization/timing/latency number for the complete quantized network on the PYNQ-Z2 target, the same way the float32 design's combined top-level numbers were reported above.

**Next step**: C5 and F6 (both dense ReLU layers) are now done in INT8, joining C1, C3 (conv/MAC layers) and S2, S4 (pooling) — six of the network's seven layers. Only the output dense layer (softmax) remains; its softmax stage will need its own INT8-appropriate treatment (softmax doesn't requantize to int8 the same way ReLU layers do, per this project's own float32 optimization log for that layer above).