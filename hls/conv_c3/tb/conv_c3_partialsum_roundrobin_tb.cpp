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

    conv_c3_partialsum_roundrobin(input, weights, bias, output);

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
