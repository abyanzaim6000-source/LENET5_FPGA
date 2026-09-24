#ifndef DENSE_OUTPUT_INT8_FIXEDPOINT_H
#define DENSE_OUTPUT_INT8_FIXEDPOINT_H

#include <ap_int.h>

// Same fixed dimensions as dense_output.h -- 84 inputs (F6's output),
// 10 outputs. Kept as its own header (not shared with dense_output.h)
// so the float32 IP is never touched by this work, per the project's
// "old version stays untouched" convention.
#define N_IN 84
#define N_OUT 10

// Same PE_COUNT=4 partial-sum split as dense_c5_int8_fixedpoint.h/.cpp
// and dense_f6_int8_fixedpoint.h/.cpp for Stage 1's MAC reduction.
#define PE_COUNT 4
#define MACS_PER_PE ((N_IN + PE_COUNT - 1) / PE_COUNT)

// INT8 dense_output -- NOT a drop-in copy of dense_c5/dense_f6's int8
// pattern. Softmax needs the sum of every output neuron's exponential
// before any one of them can be normalized, so (exactly like the float32
// dense_output.cpp) this stays two explicit stages, now with an integer
// Stage 1 and a float Stage 2:
//
//   Stage 1 -- MAC accumulation: the same proven PE_COUNT=4 partial-sum
//              split as dense_c5_int8_fixedpoint.cpp/dense_f6_int8_fixedpoint.cpp,
//              pure int8xint8->int32 arithmetic, into a small local int32
//              `logits_int` array. Deliberately NO ReLU and NO
//              requantize-to-int8 here (unlike C5/F6): softmax needs the
//              FULL-PRECISION accumulator, not an int8-rounded value, and
//              there is no next INT8 layer downstream to requantize FOR --
//              this is the network's last layer.
//   Stage 2 -- softmax: dequantizes each of the 10 raw accumulators to
//              float with ONE multiply by `combined_scale` (x_scale*
//              w_scale, precomputed offline exactly like requant_mult/
//              requant_shift are precomputed offline for the other INT8
//              layers), then runs the exact same max-subtract / exp /
//              sum / divide as the proven float32 dense_output.cpp.
//
//              This deliberately does NOT use the TFLite/gemmlowp Q31
//              mantissa-and-shift technique C1/C3/C5/F6 use for their
//              ReLU requantize. That machinery exists to keep floating-
//              point hardware OUT of a repeated, per-cycle-critical loop
//              (per-pixel for a conv, per-tap x N_IN/N_OUT for a dense
//              MAC). Stage 2 here runs exactly ONCE per inference over
//              just 10 values -- a dequantize multiply, a max, 10 exps, a
//              sum, 10 divides -- negligible resource/timing cost either
//              way, so forcing it through int64 fixed-point buys nothing
//              and only makes the reference harder to verify. A plain
//              float dequantize + float softmax is the actually-simpler,
//              actually-cheaper design at this specific point in the
//              pipeline (see Results/hls_results.md for the full
//              reasoning and the Python reference that verified it).
//
//   input, weights           : ap_int<8>
//   bias                     : ap_int<32>, PRE-QUANTIZED as
//                               round(bias_float / (x_scale*w_scale))
//   combined_scale            : float, x_scale*w_scale -- Stage 2's ONE
//                               dequantize multiplier (no Q31 mantissa/
//                               shift needed -- see rationale above)
//   output                    : float, final softmax probabilities --
//                               the network's last layer, so there is
//                               nothing further to requantize to int8 FOR;
//                               the float32 dense_output.cpp already
//                               returns float probabilities too
void dense_output_int8_fixedpoint(
    ap_int<8>  input[N_IN],
    ap_int<8>  weights[N_IN][N_OUT],
    ap_int<32> bias[N_OUT],
    float      combined_scale,
    float      output[N_OUT]
);

#endif
