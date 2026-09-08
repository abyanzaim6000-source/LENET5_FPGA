#include "pool_s4_int8.h"

// Same max-pooling logic, same ARRAY_PARTITION fix, and same explicit
// PIPELINE II=1 as the proven float32 pool_s4.cpp -- built correctly
// from the start, same as the float32 S4 did (reusing S2's proven cyclic
// partition fix directly, no baseline-then-fix cycle). Only the datapath
// type changes: ap_int<8> compare/select instead of float compare/select.
void pool_s4_int8(
    ap_int<8> input[IN_H][IN_W][CHANNELS],
    ap_int<8> output[OUT_H][OUT_W][CHANNELS]
) {
    #pragma HLS ARRAY_PARTITION variable=input cyclic factor=2 dim=1
    #pragma HLS ARRAY_PARTITION variable=input cyclic factor=2 dim=2

    for (int out_r = 0; out_r < OUT_H; out_r++) {
        for (int out_c = 0; out_c < OUT_W; out_c++) {
            for (int ch = 0; ch < CHANNELS; ch++) {
                #pragma HLS PIPELINE II=1
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
