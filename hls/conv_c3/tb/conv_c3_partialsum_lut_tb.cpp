#include <iostream>
#include <cmath>
#include "../src/conv_c3.h"

int main() {
    static float input[IN_H][IN_W][IN_C];
    static float weights[K][K][IN_C][OUT_C];
    static float bias[OUT_C];
    static float output[OUT_H][OUT_W][OUT_C];

    // Same hand-computable setup as conv_c3_tb.cpp / conv_c3_partialsum_tb.cpp.
    for (int r = 0; r < IN_H; r++)
        for (int c = 0; c < IN_W; c++)
            for (int i = 0; i < IN_C; i++)
                input[r][c][i] = 1.0f;

    for (int kr = 0; kr < K; kr++)
        for (int kc = 0; kc < K; kc++)
            for (int i = 0; i < IN_C; i++)
                for (int o = 0; o < OUT_C; o++)
                    weights[kr][kc][i][o] = 1.0f;

    for (int o = 0; o < OUT_C; o++)
        bias[o] = 0.0f;

    conv_c3_partialsum_lut(input, weights, bias, output);

    float expected = 150.0f;
    float actual = output[0][0][0];

    std::cout << "Expected corner value: " << expected << std::endl;
    std::cout << "Actual corner value:   " << actual << std::endl;

    bool pass = std::fabs(expected - actual) < 1e-5;
    if (pass) {
        std::cout << "Uniform-value TEST PASSED" << std::endl;
    } else {
        std::cout << "Uniform-value TEST FAILED" << std::endl;
    }

    // Uniform inputs can't catch an (i, kr, kc) permutation bug in the LUTs
    // -- the reduction sum is invariant to term order. Cross-check against
    // conv_c3_partialsum.cpp (division-based decode, already proven correct)
    // using distinct per-(kr,kc,i) weight values, so any LUT mismatch shows
    // up as a numeric difference.
    for (int kr = 0; kr < K; kr++)
        for (int kc = 0; kc < K; kc++)
            for (int i = 0; i < IN_C; i++)
                for (int o = 0; o < OUT_C; o++)
                    weights[kr][kc][i][o] = (float)(kr * 100 + kc * 10 + i) + (float)o * 0.01f;

    for (int r = 0; r < IN_H; r++)
        for (int c = 0; c < IN_W; c++)
            for (int i = 0; i < IN_C; i++)
                input[r][c][i] = (float)((r + c + i) % 7) * 0.5f;

    for (int o = 0; o < OUT_C; o++)
        bias[o] = (float)o * 0.25f;

    static float output_ref[OUT_H][OUT_W][OUT_C];
    conv_c3_partialsum(input, weights, bias, output_ref);
    conv_c3_partialsum_lut(input, weights, bias, output);

    float max_diff = 0.0f;
    for (int r = 0; r < OUT_H; r++)
        for (int c = 0; c < OUT_W; c++)
            for (int o = 0; o < OUT_C; o++) {
                float diff = std::fabs(output[r][c][o] - output_ref[r][c][o]);
                if (diff > max_diff) max_diff = diff;
            }

    std::cout << "Cross-check max diff vs conv_c3_partialsum: " << max_diff << std::endl;
    bool cross_pass = max_diff < 1e-3f;
    if (cross_pass) {
        std::cout << "Cross-check TEST PASSED" << std::endl;
    } else {
        std::cout << "Cross-check TEST FAILED" << std::endl;
    }

    if (pass && cross_pass) {
        std::cout << "TEST PASSED" << std::endl;
        return 0;
    } else {
        std::cout << "TEST FAILED" << std::endl;
        return 1;
    }
}
