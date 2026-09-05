#include <iostream>
#include <cmath>
#include "../src/dense_c5.h"

int main() {
    static float input[N_IN];
    static float weights[N_IN][N_OUT];
    static float bias[N_OUT];
    static float output[N_OUT];

    // Same hand-computable setup as dense_c5_tb.cpp: all-ones input,
    // all-ones weights, zero bias. Pre-activation sum for every output
    // neuron = N_IN = 400, ReLU passes positive values through unchanged,
    // so every output should be exactly 400.0f.
    for (int i = 0; i < N_IN; i++)
        input[i] = 1.0f;

    for (int i = 0; i < N_IN; i++)
        for (int j = 0; j < N_OUT; j++)
            weights[i][j] = 1.0f;

    for (int j = 0; j < N_OUT; j++)
        bias[j] = 0.0f;

    dense_c5_partialsum(input, weights, bias, output);

    // Check the first output neuron -- should be exactly 400.0 (ReLU passthrough)
    float expected = 400.0f;
    float actual = output[0];

    std::cout << "Expected output[0]: " << expected << std::endl;
    std::cout << "Actual output[0]:   " << actual << std::endl;

    if (std::fabs(expected - actual) < 1e-5) {
        std::cout << "TEST PASSED" << std::endl;
        return 0;
    } else {
        std::cout << "TEST FAILED" << std::endl;
        return 1;
    }
}
