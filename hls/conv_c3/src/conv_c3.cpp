#include "conv_c3.h"

// This loop structure is DELIBERATELY identical to conv_c1.cpp's baseline --
// output channel -> row -> col -> input channel -> kernel row -> kernel col.
// Unlike C1, C3 uses "valid" padding (no zero-padding, no bounds check
// needed) and a wider input channel fan-in (6, from S2's pooled output).
void conv_c3(
    float input[IN_H][IN_W][IN_C],
    float weights[K][K][IN_C][OUT_C],
    float bias[OUT_C],
    float output[OUT_H][OUT_W][OUT_C]
) {
    for (int o = 0; o < OUT_C; o++) {
        for (int r = 0; r < OUT_H; r++) {
            for (int c = 0; c < OUT_W; c++) {
                float acc = bias[o];
                for (int i = 0; i < IN_C; i++) {
                    for (int kr = 0; kr < K; kr++) {
                        for (int kc = 0; kc < K; kc++) {
                            acc += input[r + kr][c + kc][i] * weights[kr][kc][i][o];
                        }
                    }
                }
                // ReLU activation, applied inline (matches lenet5_relu.py)
                output[r][c][o] = (acc > 0) ? acc : 0;
            }
        }
    }
}
