# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

Reproduces/extends a paper's template-based methodology for running LeNet-5 inference on a
Zynq-7020 FPGA (ZedBoard / PYNQ-Z2): train in Keras → quantize to integer/fixed-point →
port each layer to HLS C++ → synthesize per-layer IP cores → integrate into a Vivado block
design → (eventual) PS/PL deployment. The base paper is Swati/Banerjee/Engineer, "A
Template-Based Methodology for Efficient DNNs Inference on FPGA Devices With HW-SW
Co-Design" (IEEE Embedded Systems Letters, 2025). Known divergences from the paper's exact
spec (32×32 input, 8-channel C1, `<13,3>` fixed-point, AXI-stream+DMA) are tracked in
`Docs/Project journal.md` and `Results/hls_results.md` — check there before assuming a
layer's dimensions or precision match the paper.

There are two parallel, independent tracks in this repo:
- **`src/`** — Python/Keras/NumPy: model definition, training, quantization simulation,
  weight export. No FPGA tools required.
- **`hls/`** — HLS C++: per-layer hardware IP cores, built and simulated with AMD Vitis
  HLS. Requires Vitis HLS installed and on `PATH` (`vitis_hls`).
These meet at the weight/test-data files each side exports for the other, not through any
shared build system — there is no top-level Makefile/CMake tying Python and HLS together.

## Long-term goal

Get a real handwritten-digit image classified end-to-end on physical PYNQ-Z2 hardware,
using integer/fixed-point arithmetic throughout the PL side, at a resource/timing profile
documented against the base paper's own numbers. "Done" means: all 7 layers in HLS
fixed-point, chained into one `lenet5_top` network, integrated into a Vivado block design
on the PYNQ-Z2, bitstream built, and a downloaded/hand-written digit image classified on
the board (or via PYNQ's Python API) with results matching the Python reference.

## Current phase & status

**Phase: INT8 fixed-point quantization, rolling out layer by layer.** The project already
has a complete, working float32 pipeline (see "Completed" below); the current work is
replacing each layer's float32 HLS kernel with a bit-exact INT8 fixed-point one (see
`_int8_fixedpoint` in the HLS section below), verified against a chained Python integer
reference before every HLS run.

**Completed:**
- Full float32 HLS pipeline for all 7 layers (C1, S2, C3, S4, C5, F6, Output), each
  individually optimized (line-buffer/systolic C1, partial-sum PE splits for C3/C5/F6,
  array-partitioned S2/S4) and C-sim/C-synth verified.
- Full `lenet5_top` float32 network assembled and synthesized — fits the chip budget (LUT
  71%, DSP 11%, FF 37%, BRAM 82%, Fmax 103.49MHz on the ZedBoard part).
- Vivado block design (PS + AXI interconnect + HLS IP) validated; bitstreams built for both
  the ZedBoard part (`xc7z020iclg484-1L`) and PYNQ-Z2 (`xc7z020clg400-1`).
- INT8 fixed-point conversion done and C-sim/C-synth verified for: `conv_c1`, `pool_s2`,
  `pool_s4`, `conv_c3`, `dense_c5`, `dense_f6` — each shown to *beat* its float32
  counterpart on FF/LUT/latency and close timing where float32 didn't, at roughly flat DSP
  cost. Pooling layers (`pool_s2`/`pool_s4`) stop at plain INT8 deliberately — max-pooling
  is comparison-only, so there's no multiply/accumulate for fixed-point rescaling to
  improve.
- `dense_output`'s INT8 fixed-point kernel is written and algorithm-verified against the
  chained Python int8 reference (correct class, matching probabilities); its real Vitis HLS
  C-sim/C-synth run is the one piece still pending (see Next steps).

**Important decisions/constraints:**
- HLS build/optimization work happens on a separate PYNQ-Z2-equipped machine — Vitis HLS
  is not available in this dev environment, so new `_int8_fixedpoint` kernels are algorithm-
  verified locally (plain g++, `ap_int` stubbed) before being handed off for the real run.
- Requantization uses the TFLite/gemmlowp technique (Q31 mantissa + shift, no runtime float
  division) for every ReLU layer (C1/C3/C5/F6). `dense_output`'s softmax stage is the one
  deliberate exception — it stays float (single one-shot pass over 10 values, not a hot
  loop, so fixed-point buys nothing there but reference complexity).
- The HLS side is ReLU+MaxPool only (`lenet5_relu.py`); the tanh+AvgPool `lenet5.py`
  variant is Python-only and intentionally has no HLS counterpart (see below).
- No DMA block in the Vivado design yet — `m_axi` straight through an AXI Interconnect to
  PS memory, a deliberate simplification vs. the paper's PS↔DMA↔AXI-interconnect↔PL design.

**Next planned steps:**
1. Run the real Vitis HLS C-sim/C-synth for `dense_output_int8_fixedpoint` on the PYNQ-Z2
   machine and record results in `Results/hls_results.md` — the last individual-layer gap.
2. Chain all 7 INT8 fixed-point layers into one INT8 `lenet5_top` network (mirroring the
   existing float32 `lenet5_top`) and confirm it still fits the PYNQ-Z2 chip budget.
3. Integrate the INT8 `lenet5_top` into the PYNQ-Z2 Vivado block design and rebuild the
   bitstream.
4. Classify a real downloaded/handwritten digit image (`Data/downloaded_test/`,
   `src/test_downloaded_image.py`) end-to-end on hardware and compare against the Python
   integer reference.

## Python side (`src/`)

Run everything from the project root (scripts use relative paths like `models/...`,
`weights/...`). A `venv/` exists at the repo root with tensorflow/numpy/matplotlib installed.

```bash
source venv/bin/activate
python3 src/train.py              # trains tanh+AvgPool LeNet-5, saves models/lenet5.keras
python3 src/train_relu.py         # trains ReLU+MaxPool variant, saves models/lenet5_relu.keras
python3 src/train_3x3.py          # smaller 3x3-kernel comparison CNN (tanh variant)
python3 src/train_3x3_relu.py     # smaller 3x3-kernel comparison CNN (ReLU variant)

python3 src/verify.py             # accuracy check, tanh variant
python3 src/verify_relu.py        # accuracy check, ReLU variant
python3 src/test_single_image.py  # visual single-image prediction check
python3 src/compare_variants.py   # accuracy comparison across model variants

python3 src/export_weights.py             # float32 weights → weights/float32/*.npy,*.dat
python3 src/export_quantized_weights.py   # INT8 + INT4 weights → weights/int{8,4}/<model>/*
python3 src/quantized_inference.py        # float-simulated quantization accuracy (tanh)
python3 src/quantized_inference_relu.py   # float-simulated quantization accuracy (ReLU)
python3 src/integer_inference.py          # REAL int8/int4 MAC arithmetic accuracy (ReLU) —
                                           # the genuine hardware-facing correctness proof,
                                           # not float pretending to be integer
```

Note: scripts consistently load/save to `models/...` (lowercase) but the tracked directory
is `Models/`. This only works because macOS's default filesystem is case-insensitive —
don't assume it will work unmodified on Linux/CI.

`models/lenet5.keras` (tanh + AveragePooling, `lenet5.py`) and `models/lenet5_relu.keras`
(ReLU + MaxPooling, `lenet5_relu.py`) are **separate architectures kept side by side**, not
one superseding the other — don't merge them. However, **the HLS side has standardized on
ReLU + MaxPool only** (matching `lenet5_relu.py`); the tanh/AvgPool variant is Python-only
and has no HLS counterpart. When porting a new layer to HLS, always match `lenet5_relu.py`'s
activation/pooling choice, not `lenet5.py`'s.

Quantization has two distinct meanings in this codebase, both in `src/quantization.py` —
don't confuse them:
- `quantize_fixed` / `FORMATS["fixed16_paper"]`: fixed-point simulation (`<13,3>`, the
  paper's format) — still stored as float, values rounded/clipped to what fixed-point could
  represent.
- `quantize_int_real` / `quantize_int_dynamic`: true INTn (8 or 4-bit), data-driven scale
  factor per tensor, actual `int8`/`int32` arrays exported to `weights/int{8,4}/`. This is
  what `integer_inference.py` and the HLS `*_int8*` kernels are based on.

## HLS side (`hls/`)

One directory per layer: `conv_c1`, `pool` (S2), `conv_c3`, `pool_s4`, `dense_c5`,
`dense_f6`, `dense_output`, plus `lenet5_top` for whole-network integration. Each layer
directory typically contains multiple **variants** of the same layer, progressively closer
to the paper's spec — read the filename suffix to know which one you're looking at:
- plain (e.g. `conv_c1.cpp`) — unoptimized float baseline, no pragmas.
- `_linebuf` / `_systolic` / `_partialsum*` — intermediate optimization experiments for that
  layer (line-buffer streaming, systolic PE array, LUT/round-robin partial-sum variants).
- `_int8` — INT8 quantized, still float division for dequantization.
- `_int8_fixedpoint` — INT8 quantized using genuine fixed-point (int64 multiply + rounding
  right-shift), no float division anywhere — the variant closest to real hardware and the
  current frontier of the project.

Build/simulate a variant with Vitis HLS via its `run_hls*.tcl` script:

```bash
cd hls/conv_c1
vitis_hls -f run_hls_int8_fixedpoint_pynqz2.tcl   # runs csim_design + csynth_design
```

Each `run_hls*.tcl` opens a project, sets the top function, adds `src/`+`tb/` files, targets
`xc7z020clg400-1` (or check the specific script — parts vary slightly across files) at a
10ns/100MHz clock (kept fixed across every experiment in this project for fair comparison),
then runs C-simulation and C-synthesis. Do the C-sim/synth results land in a generated
`*_proj/` directory next to the script — these are gitignored (`**/*proj*/` in `.gitignore`).

Testbenches can also be smoke-tested without Vitis HLS, with plain `g++`, for INT8/float
variants that don't use `ap_int`/`ap_fixed` types:
```bash
g++ -std=c++11 -I src -o /tmp/tb tb/conv_c1_tb.cpp && /tmp/tb
```
Variants using `ap_int<N>`/`ap_fixed<N,M>` (the `_int8_fixedpoint` ones) need Vitis HLS's
`ap_int.h`/`ap_fixed.h` headers on the include path to compile outside Vitis HLS itself.

Each layer's `tb/generate_test_data*.py` script regenerates that layer's `*_test_data.h`
from real exported weights (`weights/int8/lenet5_relu/...`) — regenerate test data with
these scripts if the trained model or its quantized weights change, don't hand-edit the
generated `.h` files.

### Hard-won HLS lessons (see `Docs/Technical_Explanation_for_Guide.md` for full detail)

- **`#pragma HLS PIPELINE II=1` forces complete unrolling of everything nested inside it.**
  This silently defeats resource-sharing designs (bit the systolic C1 variant once — a
  misplaced pragma turned an intended 8-PE time-multiplexed design back into 150 parallel
  MACs). Place `PIPELINE` pragmas carefully, on the loop actually meant to be II=1.
- **Identify the real bottleneck array before partitioning.** HLS's own scheduling warning
  names the array causing an II violation (e.g. `"...on array 'weights' due to limited
  memory ports"`) — trust that message over assumptions. Partitioning the wrong array (e.g.
  weights when input was the bottleneck) costs FF/LUT for zero latency improvement.
- **`m_axi` interfaces and full array partitioning are incompatible on the same array.**
  Fully-partitioned arrays exposed via `m_axi` generate one AXI master per partition (seen:
  150+ masters, ~20 min synth). Fix: burst-copy AXI-facing arrays into local on-chip buffers
  first, then do all fast/parallel computation on the local copies only.
- Every layer's optimization arc is logged in `Results/hls_results.md` (latency/II/Fmax/
  DSP/FF/LUT per experiment) — check it before re-deriving a layer's bottleneck from
  scratch; it may already be documented with the exact HLS warning that identified it.

## Vivado integration (`Lenet5_fpga_top*/`)

`Lenet5_fpga_top/` (ZedBoard) and `Lenet5_fpga_top_pynqz2/` (PYNQ-Z2) hold Vivado block-
design sources (`*.srcs/sources_1/bd/`). Current integration connects an HLS IP's `m_axi`
ports through an AXI Interconnect straight to PS memory — **no DMA block yet**, unlike the
paper's architecture (PS ↔ DMA ↔ AXI-interconnect ↔ PL accelerators). This is a deliberate,
tracked simplification, not an oversight.

## Directory notes

- `Models/`, `Results/`, `Data/`, `Docs/` are capitalized; `src/`, `hls/`, `weights/` are
  lowercase — no consistent convention, just match what's there.
- `experiments.md`, `Learning_notes.md`, `Project journal.md`, `Report_draft.md` at the repo
  root are **empty stray directories** (not files) — the real, current project journal is
  `Docs/Project journal.md` and the real technical writeup is
  `Docs/Technical_Explanation_for_Guide.md`.
- `weights/` is gitignored; `Models/*.keras` are gitignored per `Models/.gitignore`. Both are
  regenerated by the training/export scripts above, not meant to be hand-edited or assumed
  present in a fresh clone.
