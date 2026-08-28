#include <iostream>
#include <cmath>
#include "../src/conv_c1.h"

int main() {
    static float input[IN_H][IN_W][IN_C];
    static float weights[K][K][IN_C][OUT_C];
    static float bias[OUT_C];
    static float output[IN_H][IN_W][OUT_C];

    // Simple test: all-ones input, all-ones weights, zero bias.
    // With "same" padding, the CENTER pixel (away from edges) should sum
    // exactly K*K = 25 (all 25 kernel taps see a 1, times weight 1).
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

    conv_c1(input, weights, bias, output);

    // Check a center pixel (row 14, col 14) -- should be exactly 25.0
    float expected = 25.0f;
    float actual = output[14][14][0];

    std::cout << "Expected center value: " << expected << std::endl;
    std::cout << "Actual center value:   " << actual << std::endl;

    if (std::fabs(expected - actual) < 1e-5) {
        std::cout << "TEST PASSED" << std::endl;
        return 0;
    } else {
        std::cout << "TEST FAILED" << std::endl;
        return 1;
    }
}