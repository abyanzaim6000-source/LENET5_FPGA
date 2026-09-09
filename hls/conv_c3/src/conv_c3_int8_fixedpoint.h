#ifndef CONV_C3_INT8_FIXEDPOINT_H
#define CONV_C3_INT8_FIXEDPOINT_H

#include <ap_int.h>

// Same fixed dimensions as conv_c3.h -- 14x14x6 input (S2's pooled
// output), 5x5 kernel, 16 output channels, "valid" padding. Kept as its
// own header (not shared with conv_c3.h) so none of the float32 IPs are
// ever touched by this work, per the project's "old version stays
// untouched" convention.
#define IN_H 14
#define IN_W 14
#define IN_C 6
#define OUT_C 16
#define K 5
#define OUT_H (IN_H - K + 1)
#define OUT_W (IN_W - K + 1)

// INT8 conv_c3, combining TWO already-proven pieces:
//   - the architecture: conv_c3_partialsum_lut.cpp's kept, superior
//     PE_COUNT=6 partial-sum split with LUT-based (i,kr,kc) mac_idx
//     decode (division-free, same loop-flattening/timing as the
//     div/mod version at far lower DSP/FF/LUT -- see this IP's own
//     optimization log in Results/hls_results.md)
//   - the requantization: conv_c1_int8_fixedpoint.cpp's genuine
//     fixed-point (TFLite/gemmlowp quantization-multiplier) rescale --
//     no floating point anywhere in the per-pixel path
//
//   input, weights, output : ap_int<8>
//   bias                   : ap_int<32>, PRE-QUANTIZED as
//                             round(bias_float / (x_scale*w_scale))
//   requant_mult            : ap_int<32>, Q31 fixed-point significand of
//                             (x_scale*w_scale)/out_scale -- C3's OWN
//                             multiplier, independently derived (not
//                             reused from C1's)
//   requant_shift            : ap_int<8>, C3's own rounding right-shift
//                             amount paired with requant_mult
void conv_c3_int8_fixedpoint(
    ap_int<8>  input[IN_H][IN_W][IN_C],
    ap_int<8>  weights[K][K][IN_C][OUT_C],
    ap_int<32> bias[OUT_C],
    ap_int<32> requant_mult,
    ap_int<8>  requant_shift,
    ap_int<8>  output[OUT_H][OUT_W][OUT_C]
);

#endif
