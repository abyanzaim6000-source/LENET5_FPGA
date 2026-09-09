#ifndef DENSE_C5_INT8_FIXEDPOINT_H
#define DENSE_C5_INT8_FIXEDPOINT_H

#include <ap_int.h>

// Same fixed dimensions as dense_c5.h -- 400 inputs (S4's flattened
// 5x5x16 output), 120 outputs. Kept as its own header (not shared with
// dense_c5.h) so none of the float32 IPs are ever touched by this work,
// per the project's "old version stays untouched" convention.
#define N_IN 400
#define N_OUT 120

// INT8 dense_c5, combining TWO already-proven pieces:
//   - the architecture: dense_c5_partialsum.cpp's kept, superior
//     PE_COUNT=4 partial-sum split (independent partial_sum[] accumulator
//     chains over a flattened mac_idx=m*PE_COUNT+p index, same as
//     conv_c3_partialsum_lut.cpp's (m,p) loop nest)
//   - the requantization: conv_c1_int8_fixedpoint.cpp's genuine
//     fixed-point (TFLite/gemmlowp quantization-multiplier) rescale --
//     no floating point anywhere in the per-neuron path
//
//   input, weights, output : ap_int<8>
//   bias                   : ap_int<32>, PRE-QUANTIZED as
//                             round(bias_float / (x_scale*w_scale))
//   requant_mult            : ap_int<32>, Q31 fixed-point significand of
//                             (x_scale*w_scale)/out_scale -- C5's OWN
//                             multiplier, independently derived (not
//                             reused from C1's or C3's)
//   requant_shift            : ap_int<8>, C5's own rounding right-shift
//                             amount paired with requant_mult
void dense_c5_int8_fixedpoint(
    ap_int<8>  input[N_IN],
    ap_int<8>  weights[N_IN][N_OUT],
    ap_int<32> bias[N_OUT],
    ap_int<32> requant_mult,
    ap_int<8>  requant_shift,
    ap_int<8>  output[N_OUT]
);

#endif
