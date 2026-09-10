#ifndef DENSE_OUTPUT_INT8_H
#define DENSE_OUTPUT_INT8_H

#include <ap_int.h>

// Same fixed dimensions as dense_output.h -- 84 inputs (F6's output), 10
// outputs, softmax activation. Kept as its own header (not shared with
// dense_output.h) so the float32 IP is never touched, per the project's
// "old version stays untouched" convention.
#define N_IN 84
#define N_OUT 10

// Same PE_COUNT=4 partial-sum split as dense_c5_int8_fixedpoint.h/
// dense_f6_int8_fixedpoint.h for the MAC stage.
#define PE_COUNT 4
#define MACS_PER_PE ((N_IN + PE_COUNT - 1) / PE_COUNT)

// INT8 dense_output, structured as the SAME two-stage design as the
// proven float32 dense_output.cpp -- softmax needs every neuron's raw
// accumulation before any one output can be normalized, so it can't fit
// the single-pass "compute acc, activate, write output[j]" loop the
// ReLU layers use:
//
//   Stage 1 -- int8 MAC accumulation (PE_COUNT=4 partial-sum split,
//              same architecture as dense_c5_int8_fixedpoint.cpp/
//              dense_f6_int8_fixedpoint.cpp): int8 x int8 -> int32,
//              into a small local int32 logits array. No activation,
//              no requantization to int8 -- unlike every earlier INT8
//              layer, this accumulator is never consumed by another
//              quantized layer, so there is nothing to rescale it FOR.
//   Stage 2 -- softmax normalization. Deliberately NOT fixed-point:
//              this is the network's final activation (read as a
//              probability distribution by a human/host, not fed
//              forward), and it runs once over just N_OUT=10 values, so
//              there is no repeated-per-pixel cost to justify avoiding
//              float here the way the ReLU layers' rescale step does.
//              Dequantizes Stage 1's verified-exact int32 accumulator to
//              float via the same combined_scale = x_scale*w_scale every
//              conv/dense layer already uses, then reuses the proven
//              float32 dense_output.cpp's own max-subtract/exp/sum/divide
//              unchanged -- matching src/integer_layers.py's own
//              dense_int(..., activation="softmax").
//
//   input, weights : ap_int<8>
//   bias           : ap_int<32>, PRE-QUANTIZED as
//                     round(bias_float / (x_scale*w_scale))
//   x_scale, w_scale : input/weight quantization scales (plain float
//                     scalars, same convention as conv_c1_int8.h -- this
//                     is an "int8 MAC + float finish" design, not a
//                     "_fixedpoint" one, since there is no int8 output
//                     to rescale into)
//   output          : float[N_OUT], the final softmax probabilities
void dense_output_int8(
    ap_int<8>  input[N_IN],
    ap_int<8>  weights[N_IN][N_OUT],
    ap_int<32> bias[N_OUT],
    float x_scale,
    float w_scale,
    float output[N_OUT]
);

#endif
