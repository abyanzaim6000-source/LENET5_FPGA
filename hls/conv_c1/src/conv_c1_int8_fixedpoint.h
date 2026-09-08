#ifndef CONV_C1_INT8_FIXEDPOINT_H
#define CONV_C1_INT8_FIXEDPOINT_H

#include <ap_int.h>

// Same fixed dimensions as conv_c1_int8.h -- 28x28x1 input, 5x5 kernel,
// 6 output channels, "same" padding. Kept as its own header (not shared
// with conv_c1.h / conv_c1_int8.h) so neither the float32 IP nor the
// earlier INT8-with-float-requantize IP is ever touched by this work,
// per the project's "old version stays untouched" convention.
#define IN_H 28
#define IN_W 28
#define IN_C 1
#define OUT_C 6
#define K 5
#define PAD 2   // (K/2), for "same" padding

// Same INT8 conv_c1 as conv_c1_int8.cpp, but the requantization step
// (int32 accumulator -> int8 output) is now GENUINE fixed-point integer
// arithmetic -- a real int64 multiply + rounding right-shift -- instead
// of the float32 multiply/divide conv_c1_int8.cpp still uses at the
// layer's activation boundary. This is the TFLite/gemmlowp "quantization
// multiplier" technique: the real-valued ratio
// (x_scale*w_scale)/out_scale is decomposed OFFLINE and ONCE, in Python
// (see generate_test_data_int8_fixedpoint.py's quantize_multiplier()),
// into a Q31 fixed-point mantissa and a shift -- never on this kernel's
// per-pixel critical path.
//
//   input, weights, output : ap_int<8>
//   bias                   : ap_int<32>, PRE-QUANTIZED as before
//                             (round(bias_float / (x_scale*w_scale)))
//   requant_mult            : ap_int<32>, Q31 fixed-point significand of
//                             (x_scale*w_scale)/out_scale -- always fits
//                             comfortably inside signed int32
//   requant_shift            : ap_int<8>, total rounding right-shift
//                             amount paired with requant_mult
//
// No float scale parameters remain on this kernel's interface at all --
// every runtime operation, MAC through requantization, is integer only.
void conv_c1_int8_fixedpoint(
    ap_int<8>  input[IN_H][IN_W][IN_C],
    ap_int<8>  weights[K][K][IN_C][OUT_C],
    ap_int<32> bias[OUT_C],
    ap_int<32> requant_mult,
    ap_int<8>  requant_shift,
    ap_int<8>  output[IN_H][IN_W][OUT_C]
);

#endif
