#include "conv_c1.h"

// This loop structure is DELIBERATELY identical to your manual_layers.py
// conv2d() function -- output channel -> row -> col -> input channel ->
// kernel row -> kernel col. Same math, different language.
void conv_c1(
    float input[IN_H][IN_W][IN_C],
    float weights[K][K][IN_C][OUT_C],
    float bias[OUT_C],
    float output[IN_H][IN_W][OUT_C]
) {
    for (int o = 0; o < OUT_C; o++) {
        for (int r = 0; r < IN_H; r++) {
            for (int c = 0; c < IN_W; c++) {
                float acc = bias[o];
                for (int i = 0; i < IN_C; i++) {
                    for (int kr = 0; kr < K; kr++) {
                        for (int kc = 0; kc < K; kc++) {
                            int in_r = r + kr - PAD;
                            int in_c = c + kc - PAD;
                            // Zero-padding: skip (treat as 0) if outside bounds
                            if (in_r >= 0 && in_r < IN_H && in_c >= 0 && in_c < IN_W) {
                                acc += input[in_r][in_c][i] * weights[kr][kc][i][o];
                            }
                        }
                    }
                }
                // ReLU activation, applied inline (matches your ReLU model)
                output[r][c][o] = (acc > 0) ? acc : 0;
            }
        }
    }
}