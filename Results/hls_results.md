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

| **KEPT — Partial-sum split (PE_COUNT=6, mac_idx decode)**, `conv_c3_partialsum.cpp` | **160,078** | **160,079** | **4 / 1** | **45** | **41,020** | **31,870** | **0** | **111.66** | csim: TEST PASSED. Same `partial_sum[PE_COUNT]` pattern as `conv_c1_systolic.cpp`: 150-term reduction (IN_C×K×K = 6×5×5) split across 6 independent accumulator chains (25 terms each), combined at the end. HLS auto-flattened the outer o/r/c loops into the pipeline (trip count 40,000 = 1,600×25). Latency dropped ~7.66x vs. baseline. II only improved 5→4 (still not 1) — same accumulator-latency floor as C1's PE_COUNT=8 case (Final II=4 there too). Most of the DSP/FF growth (33 of 45 DSP; `urem_*` modules ~2,100-2,300 FF each ×16) comes from decoding the flattened `mac_idx` back into `(i,kr,kc)` via `/`/`%` by K=5 (non-power-of-2), not from the MAC math itself — see the rejected div/mod-free attempt below for why that overhead was kept rather than removed. **This is the version kept as C3's optimized stage** — it's the only one of the two fix attempts that clears the 100MHz-class timing target |
| **EXPLORED, REJECTED — div/mod-free round-robin**, `conv_c3_partialsum_roundrobin.cpp` | 1,030,401 | 1,030,402 | 4 / 1 | 5 | 964 | 2,050 | 0 | **93.03** | csim: TEST PASSED. Attempted to remove the div/mod overhead above by using natural nested `i`/`kr`/`kc` loops (zero decode cost) and an increment-and-wrap counter (compare+reset, not a divider) to pick which of `PE_COUNT` partial-sum registers each term hits. **It works**: DSP/FF/LUT drop back to near-baseline levels (DSP 45→5, FF 41,020→964, LUT 31,870→2,050), confirming the div/mod was indeed the dominant resource cost, not the MACs. **But** the `pe`-counter's compare/reset in the loop latch breaks the "perfect loop nest" property HLS needs to auto-flatten o/r/c into the pipeline (log: "Cannot flatten loop ... the outer loop is not a perfect loop because there is nontrivial logic in the loop latch") — so the 150-iteration inner pipeline now drains/refills separately 1,600 times with zero overlap, making overall latency ~6.4x **worse** than the kept version above (still ~16% better than the un-split baseline). Fmax also fell to 93.03MHz, missing the 100MHz-class target — the `pe`-select mux landed on the same critical path as `input_r_load`→`fmul`. Kept in the repo and documented here (not deleted) as a real cost/latency trade worth revisiting, mirroring how C1's rejected intermediate attempts (forced-full-unroll bug, 150-port AXI explosion) were kept visible in this log rather than erased |

**Next step**: `conv_c3_partialsum.cpp` (PE_COUNT=6, mac_idx decode) is C3's current optimized stage. Continue C1's methodology from here: `ARRAY_PARTITION` on `input`/`weights`, then line buffer → systolic PE variant → AXI conversion. The div/mod-free approach could be revisited later if the outer-loop flattening can be recovered without reintroducing division (e.g. static PE offsets instead of one runtime counter), but isn't blocking further progress.



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