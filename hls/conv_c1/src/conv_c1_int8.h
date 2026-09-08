#ifndef CONV_C1_INT8_H
#define CONV_C1_INT8_H

#include <ap_int.h>

// Same fixed dimensions as conv_c1.h (float32 version) -- 28x28x1 input,
// 5x5 kernel, 6 output channels, "same" padding. Kept as a separate header
// (not shared with conv_c1.h) so the float32 IP is never touched by this
// INT8 work, per the project's "old version stays untouched" convention.
#define IN_H 28
#define IN_W 28
#define IN_C 1
#define OUT_C 6
#define K 5
#define PAD 2   // (K/2), for "same" padding

// INT8 quantized conv_c1, matching the data flow proven in
// src/integer_layers.py's conv2d_int(): int8 input x int8 weight
// accumulated into int32, then rescaled back down to int8 using a single
// combined scale factor -- the same thing real quantized-int8 hardware
// does at a layer boundary.
//
//   input, weights, output : ap_int<8>  (INT8 activations/weights)
//   bias                   : ap_int<32>, PRE-QUANTIZED as
//                             round(bias_float / (x_scale*w_scale)) --
//                             matches conv2d_int()'s own bias handling,
//                             just computed once in software ahead of time
//                             instead of re-derived (via float division)
//                             inside the datapath every call.
//   x_scale, w_scale        : input/weight quantization scales, as
//                             returned by quantize_activation() /
//                             quantize_int_real().
//   out_scale                : this layer's OUTPUT activation scale (from
//                             quantize_activation() applied to the ReLU'd
//                             float output) -- the int32 accumulator is
//                             rescaled through combined_scale/out_scale,
//                             rounded, and saturated to produce the int8
//                             output that feeds the next quantized layer.
void conv_c1_int8(
    ap_int<8>  input[IN_H][IN_W][IN_C],
    ap_int<8>  weights[K][K][IN_C][OUT_C],
    ap_int<32> bias[OUT_C],
    float x_scale,
    float w_scale,
    float out_scale,
    ap_int<8>  output[IN_H][IN_W][OUT_C]
);

#endif
