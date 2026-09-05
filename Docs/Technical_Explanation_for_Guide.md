# LeNet-5 FPGA Project: Complete Technical Explanation

This document explains, in plain language, everything done during the HLS (High-Level
Synthesis) phase of this project — what each technique means, why it was needed, what
happened when we tried it, and what the results mean. It's written so that someone with
no FPGA background can follow the whole story from start to finish.

---

## Part 1: The Absolute Basics

### What is HLS, and why does this project need it?

Your Python/Keras model runs on a CPU, which executes instructions one at a time (or a
handful in parallel, at most). An FPGA is different: it's a piece of hardware you can
configure to have thousands of tiny calculators all working *at the same time*. The
challenge is that FPGAs are normally programmed in languages like Verilog or VHDL, which
describe *circuits*, not *programs* — a much harder skill to learn from scratch.

**High-Level Synthesis (HLS)** is a compiler that takes C/C++ code and automatically
converts it into that circuit-level hardware description. You write something that looks
like a normal C++ function; the HLS tool (Vitis HLS, in our case) turns it into real
digital logic. This is why every one of our building blocks — `conv_c1`, `pool_s2`,
`dense_c5`, etc. — is written as plain-looking C++ functions, even though the end goal is
a physical circuit.

### What is a "pragma"?

A pragma is a special comment-like instruction, written as `#pragma HLS ...`, that doesn't
change what the C++ code *computes* — it changes *how the hardware is built* to compute
it. Think of it as a set of construction instructions layered on top of a blueprint: the
blueprint (your C++ logic) stays the same, but the pragma tells the builder "put these two
walls next to each other" or "build 4 of these rooms instead of 1."

### What is "Initiation Interval" (II)?

This is the single most important number in this whole project, so it's worth being
precise about it.

Imagine a factory assembly line. Each station does one step of building a car. If a new
car can enter the line every 1 minute, the line's "interval" is 1 minute — regardless of
how long it takes any single car to travel all the way through.

**II is exactly this, but for hardware loops.** If a loop has `II = 1`, it means: a new
iteration of the loop *starts* every single clock cycle, even though any *one* iteration
might take several cycles to fully finish (just like a car takes many minutes to travel
the whole assembly line, even though a new car enters every 1 minute).

- **Lower II is better** — it means the hardware can start new work more often, so it
  finishes processing a whole image faster.
- **`II = 1` is the theoretical best case** for a simple loop — a new result started every
  single cycle.
- Our whole optimization story, for almost every layer, has been about **discovering why
  II was stuck at some higher number, and whether/how we could bring it down.**

### What is "Latency"?

The total number of clock cycles a function takes to run from start to finish, once.
Latency and II are related but different: II tells you how *often* new work starts;
Latency tells you how long the *whole job* takes overall (which depends on II, plus how
many loop iterations there are, plus some fixed startup/drain overhead).

### What does "DSP", "FF", "LUT", "BRAM" mean?

These are the four basic resources every FPGA is built from — think of them as the raw
construction materials, and every design we build "spends" some amount of each:

- **DSP (Digital Signal Processor)**: a small, dedicated hardware block built specifically
  for fast multiplication and addition. Every multiply-accumulate (MAC) operation in a
  neural network wants to use one of these if possible, since they're much faster and more
  efficient than building a multiplier out of general-purpose logic.
- **LUT (Look-Up Table)**: the basic general-purpose building block of an FPGA. Almost any
  small logical function (comparisons, simple math, control logic) gets built from LUTs.
- **FF (Flip-Flop)**: a single bit of memory — a register. Anything that needs to
  "remember" a value from one clock cycle to the next uses flip-flops. More parallelism
  and more pipeline stages generally means more flip-flops are needed to hold intermediate
  values.
- **BRAM (Block RAM)**: small, dedicated on-chip memory blocks, used to store larger
  arrays (like a full image) more efficiently than using individual flip-flops for every
  value.

Every FPGA chip (ours is a Xilinx Zynq-7020) has a *fixed, limited* supply of each of
these. Part of good hardware design is using as few of them as possible while still
hitting your performance target — using every single DSP/LUT/FF available isn't a badge
of honor, it just means there's no room left for anything else (like the rest of the
network).

### What does "Fmax" mean, and why 100MHz?

Fmax is the fastest clock speed the synthesized hardware can reliably run at, as estimated
by the tool. We picked a **target of 100MHz (a 10-nanosecond clock period)** at the very
start of this project as a conservative, standard choice for this FPGA chip, and kept it
fixed throughout every experiment — this is important, because it means every comparison
in this document (this design vs. that design) is a fair one: same target speed every
time, only the design itself changes.

---

## Part 2: The Pragmas We Actually Used, Explained One at a Time

### `#pragma HLS PIPELINE II=1`

**What it does:** tells the HLS tool "make this loop start a new iteration every single
clock cycle, no exceptions." This is a *demand*, not just a hint — the tool will
restructure the hardware however necessary to try to satisfy it.

**Important side-effect we discovered the hard way:** `PIPELINE II=1` forces the tool to
**completely unroll** (build separate, parallel hardware for) everything nested inside
that loop. This bit us directly in the first attempt at C1's systolic design — putting
`PIPELINE II=1` around what was supposed to be a slow, resource-saving loop instead forced
it to build 150 parallel multipliers, defeating the entire point of that experiment. The
fix was moving the pipeline pragma to a different, smaller loop, so only the intended part
got the "must be II=1" treatment.

### `#pragma HLS ARRAY_PARTITION`

**What it does:** normally, an array (like your weights or an image row) is stored in one
memory block with only 1-2 "ports" — meaning only 1-2 values can be read out of it in a
single clock cycle. If your calculation needs to read, say, 25 values from that array *at
the same time* (like all 25 weights in a 5×5 kernel), a single memory block can't do that
— it becomes a bottleneck. `ARRAY_PARTITION` splits one array into several independent
smaller memories (or even individual registers), each with its own port, so many values
can be read simultaneously.

**`complete dim=N`**: split every single element along dimension N into its own
independent register. Best for small arrays (like our 150-element convolution weights)
where full parallelism is cheap.

**`cyclic factor=N`**: split the array into N interleaved groups, rather than one register
per element. Used for the pooling layers, where full "complete" partitioning would have
been overkill for what was needed.

**The critical lesson we learned (the hard way, more than once):** you must identify
*which* array is actually the bottleneck before partitioning it. In C1's very first
optimization attempt, we partitioned the `weights` array and got **zero improvement** —
because it turned out `input` was the real bottleneck, not weights. The way to know for
sure is reading the actual warning message HLS prints, which literally names the array
causing the problem (e.g. *"Unable to schedule 'load' operation... on array 'weights' due
to limited memory ports"*).

### `#pragma HLS UNROLL`

**What it does:** takes a loop and builds separate, independent hardware for *every*
iteration, instead of reusing the same hardware over and over across cycles. If a loop
runs 8 times and you fully unroll it, you get 8 physical copies of whatever's inside,
all able to work simultaneously — at the cost of needing 8× the hardware resources.

### `#pragma HLS INTERFACE m_axi` and `#pragma HLS INTERFACE s_axilite`

**What these do:** these aren't about making the *computation* faster — they're about
connecting your custom hardware block to the rest of the system (the ARM processor and
memory). Think of your `conv_c1` or `dense_c5` block as a specialist worker in a factory —
these pragmas are what actually wire that worker into the factory's conveyor belts and
telephone lines so it can receive materials and instructions.

- **`m_axi`** ("AXI Master") gives your hardware block its own direct connection to main
  memory (DDR RAM), so it can read/write large arrays (images, weights) on its own,
  without needing the processor to hand it every single value one at a time.
- **`s_axilite`** ("AXI Slave, Lite version") is a small, simple control channel that lets
  the ARM processor start your hardware block, check if it's finished, and read/write
  small configuration values.

**A serious mistake we made and fixed:** you cannot combine `m_axi` with full
`ARRAY_PARTITION complete` on the *same* array. Partitioning says "split this into many
independent on-chip registers"; `m_axi` says "this data actually lives off-chip in DDR and
streams in over one shared connection." These are contradictory instructions for the same
array. Our first attempt at this produced **150+ redundant hardware connections** and took
nearly 20 minutes just to generate a report (should take under a minute). The fix: keep
the DDR-facing arrays *unpartitioned*, copy their contents into small *local* on-chip
arrays first, do all the fast/parallel work on those local copies, then copy the results
back out. This is a standard, well-known HLS pattern once you know to look for it.

---

## Part 3: The "Accumulator Latency" Problem — The Single Biggest Recurring Lesson

This came up independently in **three different layers** (C1's convolution, C3's
convolution, and every dense layer) — worth understanding thoroughly, since it's the
project's central hardware-design finding.

### The problem, explained with an analogy

Imagine you're adding up a long list of numbers one at a time on a calculator that takes
4 seconds to actually display each result: `2 + 3 =` (wait 4 seconds) `... + 5 =` (wait 4
seconds) `... + 7 =` (wait 4 seconds), and so on. Even though *entering* each new number
only takes an instant, you're stuck waiting for the calculator's display before you can
add the next number, because each step needs the *previous* step's answer.

This is exactly what happens with a line of code like:
```
acc = acc + (input * weight);
```
repeated many times in a loop. A hardware floating-point adder isn't instant — ours took
about 4-5 clock cycles internally to actually produce its answer. Since each loop
iteration *needs* the previous iteration's answer before it can proceed, the loop is
forced to wait, and **II gets stuck at roughly the adder's own latency** (we consistently
measured this landing at II=4 or II=5), no matter what else you try.

### Why array partitioning doesn't fix this

This is a completely different kind of bottleneck than the "not enough memory ports" issue
partitioning solves. Partitioning fixes "I can't *read* enough values per cycle."
Accumulator latency is "I *can* read the values fine, but I can't finish *adding* them
fast enough, because each addition depends on the one before it." No amount of extra
memory ports changes how long a single addition takes to compute.

### The fix: splitting one long chain into several shorter, independent chains

Instead of one accumulator doing `acc = acc + x1; acc = acc + x2; acc = acc + x3; ...` all
in one dependent sequence, we use **several separate accumulators** (we called this
"PE_COUNT", short for "Processing Element count"), each handling a subset of the terms
independently:
```
partial_sum[0] = x1 + x5 + x9 + ...
partial_sum[1] = x2 + x6 + x10 + ...
partial_sum[2] = x3 + x7 + x11 + ...
partial_sum[3] = x4 + x8 + x12 + ...
```
Since these four chains don't depend on each other at all, they can all make progress at
the same time, and only get added together (`partial_sum[0] + partial_sum[1] + ...`) once,
right at the very end. This is directly inspired by the paper's own "8 PE systolic array"
concept, and is exactly why we built a `PE_COUNT`-based partial-sum design for C1, C3,
`dense_c5`, and `dense_f6`.

### The surprising discovery: more PE_COUNT doesn't automatically mean lower II

We proved, by actually testing it (not just assuming), that increasing `PE_COUNT` beyond
the adder's own latency (4, in our case) **does not** get you all the way down to `II=1`.
We tested `PE_COUNT=4`, `5`, and `6` on different layers and every single one landed at
`II=4` — never lower.

**The real reason (confirmed by careful reading of the tool's own scheduling report):**
when you write `#pragma HLS UNROLL` across the PE dimension, all the partial-sum updates
happen *within the same loop iteration*. Each individual `partial_sum[p]` value still only
gets revisited once per iteration — a "gap" of exactly 1 iteration between one update and
the next, regardless of how many separate partial sums exist. Since a single iteration
still needs to fully account for the adder's 4-cycle latency, `II = 4` is the floor no
matter how many parallel chains you add. Adding more `PE_COUNT` only reduces the *total
number of iterations needed* (since each iteration now does more work at once) — it
doesn't change the *per-iteration* limit.

**What this means in practice:** once you hit this floor, throwing more hardware resources
at the problem stops helping the metric that matters (II), while still costing more DSP,
FF, and LUT for no further benefit. We confirmed this directly: `PE_COUNT=5` used 71% more
DSP than `PE_COUNT=4` for only 20% more speed improvement, with *identical* II — so we
kept `PE_COUNT=4` as the better choice.

---

## Part 4: Layer-by-Layer Story

### C1 — First Convolution Layer (28×28×1 input → 28×28×6 output)

This was the very first layer built, and the one where every technique in this document
was discovered for the first time (the rest of the layers benefited from lessons learned
here).

1. **Baseline** (no pragmas at all): `II = 13`, meaning a new pixel's worth of computation
   only started every 13 clock cycles. Very slow.
2. **First fix attempt — partition `weights`**: no improvement at all (`II` stayed at 13).
   This taught us the critical lesson: you must find the *actual* bottleneck array before
   partitioning, not guess.
3. **Line buffer + full partitioning**: instead of trying to have instant access to the
   *entire* 28×28 image at once (wasteful — most of it isn't needed at any given moment),
   we built a "line buffer" that only keeps the most recent 5 rows of the image on hand at
   once (since a 5×5 kernel only ever needs 5 rows), and made *that* small buffer fully
   partitioned. Combined with also partitioning the output array (since 6 output channels
   were being written simultaneously), this achieved the ideal **`II = 1`** — the best
   possible result — using 150 parallel multiply/add units.
4. **Systolic PE array (matching the paper's "8 PE" design)**: rather than 150 parallel
   MAC units, we tried to match the paper's approach of just 8 processing elements sharing
   the work over multiple cycles. First attempt accidentally rebuilt full parallelism
   (the `PIPELINE II=1` placement mistake described in Part 2). Second, corrected attempt
   genuinely used only 2 physical multiply/add units (thanks to further sharing the tool
   found automatically) — a **75× reduction in DSP usage** — at the cost of `II` rising to
   4 (the accumulator-latency floor described in Part 3) and a roughly 24% drop in maximum
   clock speed. This is a completely real, honest engineering trade-off: much cheaper
   hardware, meaningfully slower per-pixel throughput.
5. **AXI interface conversion (for connecting to the rest of the system)**: first attempt
   combined `m_axi` with full array partitioning on the same arrays, producing the 150+
   redundant connections mistake described in Part 2. Fixed by copying data into local
   on-chip buffers first, and computing there instead.
6. **Vivado integration**: successfully built and validated a complete system diagram
   containing the ARM processor, the necessary connection hardware (AXI Interconnect), and
   our custom C1 hardware block — confirmed with no errors by Vivado's own validation
   check.

### S2 — First Pooling Layer (28×28×6 → 14×14×6)

Pooling has no weights and no multiplication at all — it's just "look at a small window of
values, keep the largest one" (max pooling) or "average them" (average pooling).

- **A real bug was caught and fixed here**: the layer was initially built using *average*
  pooling, matching an earlier (tanh-based) version of the model, before the whole project
  standardized on the ReLU+MaxPool variant. This was corrected to genuine max pooling.
- **Resource result**: max pooling uses **zero DSPs** — since it only needs comparisons,
  not multiplication or addition, it's inherently cheaper than average pooling (which
  needed a divide-by-4 operation).
- **Optimization**: same idea as C1 — the 2×2 window needs to read 4 values from the input
  array at once, so the input needed partitioning (specifically, `cyclic factor=2` on the
  row and column dimensions). This achieved `II = 1`.

### S4 — Second Pooling Layer (10×10×16 → 5×5×16)

Built using the exact same proven max-pooling + partitioning pattern as S2, just resized
for different input dimensions (since it processes C3's output, not C1's). Reaching
`II = 1` on the very first attempt — no trial-and-error needed, since the fix was already
known. This is a nice, concrete demonstration of the paper's central idea: a proven
hardware design, reused across multiple layers.

### C3 — Second Convolution Layer (14×14×6 → 10×10×16)

- **Baseline**: `II = 5` — but this time, the cause was different from C1's baseline
  bottleneck. C1's problem was memory ports (not enough simultaneous reads); C3's was
  the accumulator-latency problem described in Part 3, since C3's convolution sums over
  150 terms per output (5×5×6) in one long dependent chain.
- **Attempted fix #1 — partial-sum split with a "flattened" index (PE_COUNT=6)**: worked,
  and worked *very* well — latency dropped by 7.66×. But it came with a hidden, unwanted
  cost: converting a single loop-counter number back into three separate coordinates
  (row, column, channel) required genuine integer division hardware (since the kernel size
  5 isn't a "round" binary number like 4 or 8), which used up far more DSPs and flip-flops
  than the partial-sum idea alone should have needed.
- **Attempted fix #2 — remove the division by using natural nested loops instead**: this
  *did* remove the expensive division hardware entirely (confirmed: DSP dropped right back
  down to normal levels). But it had its own hidden cost: removing that "flattened index"
  math also broke a different optimization the tool had been doing for free (merging
  multiple loops into one continuous pipeline), and the design ended up **6.4× slower**
  overall than fix #1, with a lower maximum clock speed too.
- **Decision**: we kept fix #1 (the version with the division hardware) as the actual
  final design, specifically **because it was the only one of the two that still met our
  100MHz timing target** — fix #2, despite using less hardware, was measured at only
  93MHz, meaning it would not actually run correctly at our intended clock speed. This is
  an important, general lesson: a "resource-cheaper" design is not automatically a
  "better" design if it fails to meet the required speed.

### Dense Layers — C5 (400→120), F6 (120→84), Output (84→10)

Dense (fully-connected) layers are conceptually simpler than convolution — no sliding
window, just "multiply every input by its own weight, add them all up, repeat for every
output neuron." But they hit the *exact same* accumulator-latency problem as C3, for the
same underlying reason (one long dependent addition chain per output).

- **C5 and F6**: both built with the proven `PE_COUNT=4` partial-sum split from the start
  (no baseline-then-fix cycle needed, since the fix was already known from C3/C1). Both
  landed at the same `II = 4` floor, confirming this really is a general property of
  floating-point accumulation, not something specific to convolution.
- **Output layer — a genuinely different structure, on purpose**: this layer uses
  "softmax" activation, which is different from every other activation in this project
  (ReLU) in one important way — it needs to know the value of *every* output neuron before
  it can correctly compute *any single one* (it works by comparing each neuron's value
  against the total). This meant it could not use the same simple "compute one value, then
  immediately apply activation, then move to the next" pattern the other layers use. We
  explicitly split it into two clearly separate stages: first, calculate all 10 raw
  numbers into a small temporary list; second, run the actual softmax steps (find the
  largest value, calculate exponentials, sum them, divide) as their own separate pass
  over that completed list. This produced a correct, clean design — not one continuously
  pipelined loop, but four short, purpose-built stages in sequence, which is the
  structurally correct shape for this specific kind of computation.
- **A genuine software bug was found and fixed here too**: the tool's version of the C++
  standard library didn't include a specific function (`expf`) we initially used for the
  exponential calculation; switched to the equivalent, available `exp` function instead.

---

## Part 5: Summary Table (All Layers)

| Layer | Function | Final II | DSP | Notes |
|---|---|---|---|---|
| C1 (systolic) | 1st Convolution | 4 | 2 | 75x DSP reduction vs. fully-parallel version; matches paper's 8-PE concept |
| S2 | 1st Max Pooling | 1 | 0 | No multiplication needed at all |
| C3 (partial-sum) | 2nd Convolution | 4 | 45 (mostly index-decode hardware) | Kept despite higher DSP, because the "cheaper" alternative failed timing |
| S4 | 2nd Max Pooling | 1 | 0 | Reused S2's proven fix directly, no rediscovery needed |
| Dense C5 | 1st Fully-Connected | 4 | 7 | Same accumulator-latency floor as convolution layers |
| Dense F6 | 2nd Fully-Connected | 4 | 11 | Same floor again — confirms it's independent of layer size |
| Dense Output | Final classification | (4-stage design, not single II) | 14 | Softmax's global-normalization requirement needed a different structure entirely |

## Part 6: What This Demonstrates, Overall

1. **We independently rediscovered a real, general hardware-design principle** (that
   floating-point accumulation chains have a latency floor no amount of parallelism alone
   can remove) across three completely different layer types, through actual measurement
   rather than assumption — and then correctly explained *why* it happens at a mechanical
   level.
2. **We correctly diagnosed and distinguished two different kinds of hardware bottleneck**
   — memory port contention (fixed by array partitioning) versus computational dependency
   (fixed by splitting into independent parallel chains) — and learned that applying the
   wrong fix to the wrong problem produces no benefit at all (C1's first weights-partition
   attempt).
3. **We learned that "using fewer resources" and "being a better design" are not the same
   thing** — a design must be judged by whether it meets its timing requirement first, and
   resource cost second (C3's rejected division-free alternative).
4. **We demonstrated the paper's central "reusable IP" claim directly**, not just by
   reading about it: S4 reused S2's exact fix with zero rework, and F6 reused C5's exact
   PE_COUNT=4 pattern with zero rework — proving that a properly-designed hardware
   template really can transfer cleanly across layers of the same type.