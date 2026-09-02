#include "pool_s2.h"

// Average pooling: for each 2x2 window, output = mean of the 4 values.
// No weights, no MAC -- just accumulate and divide.
void pool_s2(
    float input[IN_H][IN_W][CHANNELS],
    float output[OUT_H][OUT_W][CHANNELS]
) {
    for (int out_r = 0; out_r < OUT_H; out_r++) {
        for (int out_c = 0; out_c < OUT_W; out_c++) {
            for (int ch = 0; ch < CHANNELS; ch++) {
                float sum = 0;
                for (int pr = 0; pr < POOL_SIZE; pr++) {
                    for (int pc = 0; pc < POOL_SIZE; pc++) {
                        int in_r = out_r * STRIDE + pr;
                        int in_c = out_c * STRIDE + pc;
                        sum += input[in_r][in_c][ch];
                    }
                }
                output[out_r][out_c][ch] = sum / (POOL_SIZE * POOL_SIZE);
            }
        }
    }
}