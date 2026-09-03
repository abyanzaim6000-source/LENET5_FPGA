## [Today's date] — HLS optimization + first Vivado integration (C1 convolution)

Took the C1 convolution layer through the full HLS optimization pipeline the paper describes: 
started from a plain, unoptimized baseline (II=13, only 2 DSP-equivalent units), diagnosed the 
real memory-port bottleneck (input array, not weights — an assumption that turned out wrong and 
cost some time to discover), built a line-buffer streaming architecture matching the paper's Fig 
2(b)/(c), and reached full II=1 pipelining with 150 fully-parallel MAC units.

Then built a genuine 8-PE systolic variant, matching the paper's exact PE count — first attempt 
accidentally re-created full parallelism due to a PIPELINE pragma placement mistake, second attempt 
correctly time-multiplexed down to just 2 physical multiply/add units, at the cost of II rising to 4 
and Fmax dropping from 137.82MHz to 104.80MHz. This tradeoff is the central resource-vs-throughput 
finding the paper's Section IV-B describes, and now I have real numbers to back it, not just the 
paper's claims.

Converted the IP to AXI interfaces (m_axi + s_axilite) for system integration — hit a serious pitfall 
here: fully-partitioned arrays cannot use m_axi interfaces, and my first attempt generated 150+ 
redundant AXI masters, taking ~20 minutes to synthesize. Fixed by copying AXI-facing arrays into 
local on-chip buffers first, then doing all fast/parallel computation on the local copies — this 
is apparently the standard HLS pattern for this situation.

Successfully built and validated a Zynq block design in Vivado: PS + AXI interconnects + the 
conv_c1_systolic IP, including working through an address-overlap validation error (fixed via 
Address Editor's Auto Assign Address applied across the whole design at once, rather than 
incrementally).

**Not yet done**: PE_COUNT=1/2/4 sweep, bitstream generation, applying this same process to C3, 
generalizing to a reusable/parameterized core, and adding a DMA block to more closely match the 
paper's exact architecture.

**Biggest lessons**: (1) always verify which array is the real bottleneck via the II-violation 
warning rather than assuming; (2) PIPELINE II=1 forces complete unrolling of everything inside it, 
which can silently defeat resource-sharing goals; (3) AXI interfaces and array partitioning are 
fundamentally incompatible on the same array — burst-copy to local buffers is the fix.

## 2026-09-03 — Bootstrap C3 convolution baseline IP

Started the next item on the "not yet done" list from the entry above: applying the C1 process to 
C3. Added `hls/conv_c3/` following the exact same pattern as C1's original baseline and S2's 
pooling IP — a plain, unoptimized `conv_c3.cpp`/`.h` (no pragmas) plus a hand-computable testbench.

C3 differs from C1 in two structural ways that matter for the coming optimization pass: input is 
14×14×6 (S2's pooled output, not the raw 28×28×1 image) and padding is "valid" instead of "same" 
(matches `lenet5_relu.py` — C3 has no `padding=` argument, so it defaults to valid). The wider 
input channel fan-in (6 vs 1) means the inner MAC loop is already 6x deeper than C1's before any 
unrolling, and valid padding means every output pixel — including corners — sees a full K×K×IN_C 
window, which simplified the testbench (no edge-vs-center distinction needed, just check any pixel 
against 5×5×6=150 with all-ones inputs/weights).

Verified functionally with a local `g++` compile + run of the testbench (TEST PASSED) — this is 
not yet a Vitis HLS C-simulation/synthesis run, so no real latency/II/DSP/FF/LUT/Fmax numbers exist 
yet for C3. That's the immediate next step.

**Not yet done**: run C-simulation and baseline synthesis for C3 in Vitis HLS to get real numbers 
(mirrors the very first C1 baseline step), then repeat the rest of the arc that worked for C1 — 
diagnose the real memory bottleneck, line-buffer streaming, systolic PE-count variant, AXI 
conversion. Still also outstanding from before: PE_COUNT=1/2/4 sweep on C1, bitstream generation, 
reusable/parameterized core, DMA block.

## 2026-09-03 — Bootstrap C5 dense (fully-connected) baseline IP

Added `hls/dense_c5/`, the first fully-connected layer IP and the first departure from the conv/pool
pattern — same plain-baseline + hand-computable-testbench structure as C1/C3/S2, but the loop nest
is now a matrix-vector multiply (one MAC loop per output neuron) instead of a sliding-window
convolution, and there's no spatial dimension at all: 400 flattened inputs (S4's pooled 5×5×16
output) straight to 120 outputs.

Activation is **tanh**, not ReLU — deliberately matching `lenet5.py` (the original tanh+AvgPool
network) rather than `lenet5_relu.py`, which C1/C3/S2 have been following so far. This means the
dense layers are being ported from a different variant than the convs; worth remembering when it's
time to chain layers together into a full pipeline, since a tanh C5 can't feed cleanly from a ReLU
C3/S4 without picking one variant end-to-end (or explicitly supporting both).

Testbench uses the same all-ones input/weights, zero-bias trick as before, but the "hand-computable"
check now relies on tanh saturation rather than passthrough: pre-activation sum is exactly
N_IN = 400 for every neuron, and tanh saturates to 1.0f in float32 well before an input that large,
so every output should be exactly 1.0. Verified via local `g++` build (TEST PASSED); no Vitis HLS
numbers yet.

**Not yet done**: Vitis HLS C-simulation/synthesis for C5's baseline numbers, then F6 (120→84,
tanh) and Output (84→10, softmax) to complete the dense stack, plus resolving the ReLU-vs-tanh
variant mismatch noted above before any full-network integration.