#include "pool_s4.h"

// Max pooling: for each 2x2 window, output = maximum of the 4 values.
// Same logic as pool_s2.cpp, resized for S4's input (C3's 10x10x16 output).
//
// ARRAY_PARTITION included from the start this time, unlike S2's baseline --
// S2 hit an II=2 port-contention violation (the 2x2 window's four loads
// competing for the same memory bank) and fixed it by cyclic-partitioning
// the input array on dims 1 and 2 (row, col) with factor 2, so each of the
// 4 window taps lands in a separate bank. Same fix applied here up front.
void pool_s4(
    float input[IN_H][IN_W][CHANNELS],
    float output[OUT_H][OUT_W][CHANNELS]
) {
    #pragma HLS ARRAY_PARTITION variable=input cyclic factor=2 dim=1
    #pragma HLS ARRAY_PARTITION variable=input cyclic factor=2 dim=2

    for (int out_r = 0; out_r < OUT_H; out_r++) {
        for (int out_c = 0; out_c < OUT_W; out_c++) {
            for (int ch = 0; ch < CHANNELS; ch++) {
                #pragma HLS PIPELINE II=1
                float max_val = input[out_r * STRIDE][out_c * STRIDE][ch];
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
