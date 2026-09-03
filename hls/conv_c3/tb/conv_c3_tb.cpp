#include <iostream>
#include <cmath>
#include "../src/conv_c3.h"

int main() {
    static float input[IN_H][IN_W][IN_C];
    static float weights[K][K][IN_C][OUT_C];
    static float bias[OUT_C];
    static float output[OUT_H][OUT_W][OUT_C];

    // Simple test: all-ones input, all-ones weights, zero bias.
    // With "valid" padding (no zero-padding), EVERY output pixel -- even
    // the corners -- sees a full K*K*IN_C window, so every output should
    // be exactly K*K*IN_C = 5*5*6 = 150.
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

    conv_c3(input, weights, bias, output);

    // Check the top-left corner (row 0, col 0) -- should be exactly 150.0
    float expected = 150.0f;
    float actual = output[0][0][0];

    std::cout << "Expected corner value: " << expected << std::endl;
    std::cout << "Actual corner value:   " << actual << std::endl;

    if (std::fabs(expected - actual) < 1e-5) {
        std::cout << "TEST PASSED" << std::endl;
        return 0;
    } else {
        std::cout << "TEST FAILED" << std::endl;
        return 1;
    }
}
