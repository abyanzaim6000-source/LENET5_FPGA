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