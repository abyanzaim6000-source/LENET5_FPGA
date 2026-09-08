#ifndef POOL_S2_INT8_H
#define POOL_S2_INT8_H

#include <ap_int.h>

// Same dimensions as pool_s2.h -- input is C1's output (28x28x6), 2x2 max
// pooling, stride 2. Kept as its own header (not shared with pool_s2.h)
// so the float32 IP is never touched, per the project's "old version
// stays untouched" convention.
#define IN_H 28
#define IN_W 28
#define CHANNELS 6
#define POOL_SIZE 2
#define STRIDE 2
#define OUT_H (IN_H / STRIDE)
#define OUT_W (IN_W / STRIDE)

// INT8 max pooling. Unlike conv_c1's INT8 stages, this needs NO scale
// factor and NO requantization at all: every value in a pooling window
// shares the exact same (positive) scale, so comparing raw int8 values
// directly gives exactly the same result as comparing their real
// (dequantized) values would -- max() commutes with any positive affine
// rescale. Straight ap_int<8> in, ap_int<8> out.
void pool_s2_int8(
    ap_int<8> input[IN_H][IN_W][CHANNELS],
    ap_int<8> output[OUT_H][OUT_W][CHANNELS]
);

#endif
