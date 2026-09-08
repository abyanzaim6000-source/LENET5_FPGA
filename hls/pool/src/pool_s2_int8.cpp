#include "pool_s2_int8.h"

// Same max-pooling logic and ARRAY_PARTITION fix as the proven float32
// pool_s2.cpp (cyclic factor=2 on both spatial dims of `input`, splitting
// each 2x2 window's four loads across four separate memory banks -- the
// fix that resolved pool_s2's II=2 port-contention bottleneck). Only the
// datapath type changes: ap_int<8> compare/select instead of float
// compare/select -- no multiply, no accumulate, no rescale anywhere.
void pool_s2_int8(
    ap_int<8> input[IN_H][IN_W][CHANNELS],
    ap_int<8> output[OUT_H][OUT_W][CHANNELS]
) {
#pragma HLS ARRAY_PARTITION variable=input cyclic factor=2 dim=1
#pragma HLS ARRAY_PARTITION variable=input cyclic factor=2 dim=2
    for (int out_r = 0; out_r < OUT_H; out_r++) {
        for (int out_c = 0; out_c < OUT_W; out_c++) {
            for (int ch = 0; ch < CHANNELS; ch++) {
                ap_int<8> max_val = input[out_r * STRIDE][out_c * STRIDE][ch];
                for (int pr = 0; pr < POOL_SIZE; pr++) {
                    for (int pc = 0; pc < POOL_SIZE; pc++) {
                        int in_r = out_r * STRIDE + pr;
                        int in_c = out_c * STRIDE + pc;
                        if (input[in_r][in_c][ch] > max_val) {
                            max_val = input[in_r][in_c][ch];
                        }
                    }
                }
                output[out_r][out_c][ch] = max_val;
            }
        }
    }
}
