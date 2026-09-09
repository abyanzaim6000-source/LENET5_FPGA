#ifndef DENSE_F6_INT8_FIXEDPOINT_H
#define DENSE_F6_INT8_FIXEDPOINT_H

#include <ap_int.h>

// Same fixed dimensions as dense_f6.h -- 120 inputs (C5's output), 84
// outputs. Kept as its own header (not shared with dense_f6.h) so none
// of the float32 IPs are ever touched by this work, per the project's
// "old version stays untouched" convention.
#define N_IN 120
#define N_OUT 84

// Same PE_COUNT=4 partial-sum split as dense_c5_int8_fixedpoint.h/.cpp --
// only the dimensions changed (400->120 inputs, 120->84 outputs), same
// reusable-IP principle as the float32 dense_f6.cpp reusing
// dense_c5_partialsum.cpp's pattern.
#define PE_COUNT 4
#define MACS_PER_PE ((N_IN + PE_COUNT - 1) / PE_COUNT)

// INT8 dense_f6, combining TWO already-proven pieces:
//   - the architecture: dense_c5_int8_fixedpoint.cpp's PE_COUNT=4
//     partial-sum split (itself dense_c5_partialsum.cpp's proven
//     architecture), applied directly at F6's own dimensions
//   - the requantization: conv_c1_int8_fixedpoint.cpp's genuine
//     fixed-point (TFLite/gemmlowp quantization-multiplier) rescale --
//     no floating point anywhere in the per-neuron path
//
//   input, weights, output : ap_int<8>
//   bias                   : ap_int<32>, PRE-QUANTIZED as
//                             round(bias_float / (x_scale*w_scale))
//   requant_mult            : ap_int<32>, Q31 fixed-point significand of
//                             (x_scale*w_scale)/out_scale -- F6's OWN
//                             multiplier, independently derived (not
//                             reused from any earlier layer's)
//   requant_shift            : ap_int<8>, F6's own rounding right-shift
//                             amount paired with requant_mult
void dense_f6_int8_fixedpoint(
    ap_int<8>  input[N_IN],
    ap_int<8>  weights[N_IN][N_OUT],
    ap_int<32> bias[N_OUT],
    ap_int<32> requant_mult,
    ap_int<8>  requant_shift,
    ap_int<8>  output[N_OUT]
);

#endif
