#ifndef POOL_S4_INT8_H
#define POOL_S4_INT8_H

#include <ap_int.h>

// Dimensions for S4: input is C3's output (10x10x16), 2x2 max pooling,
// stride 2. Kept as its own header (not shared with pool_s4.h) so the
// float32 IP is never touched, per the project's "old version stays
// untouched" convention.
#define IN_H 10
#define IN_W 10
#define CHANNELS 16
#define POOL_SIZE 2
#define STRIDE 2
#define OUT_H (IN_H / STRIDE)
#define OUT_W (IN_W / STRIDE)

// INT8 max pooling, same as pool_s2_int8.h/.cpp: no scale factor and no
// requantization needed at all -- every value in a pooling window shares
// the exact same positive scale, so comparing raw int8 values directly
// gives exactly the same result as comparing their real (dequantized)
// values would.
void pool_s4_int8(
    ap_int<8> input[IN_H][IN_W][CHANNELS],
    ap_int<8> output[OUT_H][OUT_W][CHANNELS]
);

#endif
