#include "pool_s2.h"

// Max pooling: for each 2x2 window, output = maximum of the 4 values.
// Matches lenet5_relu.py's MaxPooling2D (project standardized on ReLU+MaxPool).
void pool_s2(
    float input[IN_H][IN_W][CHANNELS],
    float output[OUT_H][OUT_W][CHANNELS]
) {
    for (int out_r = 0; out_r < OUT_H; out_r++) {
        for (int out_c = 0; out_c < OUT_W; out_c++) {
            for (int ch = 0; ch < CHANNELS; ch++) {
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