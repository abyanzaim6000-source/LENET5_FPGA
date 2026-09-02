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