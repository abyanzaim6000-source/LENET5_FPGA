#include <iostream>
#include <cmath>
#include "../src/dense_output.h"

int main() {
    static float input[N_IN];
    static float weights[N_IN][N_OUT];
    static float bias[N_OUT];
    static float output[N_OUT];

    // Simple test: all-ones input, all-ones weights, zero bias.
    // Every output neuron gets the identical pre-activation logit
    // (bias 0 + 84 taps of 1*1 = 84.0), so softmax over 10 EQUAL
    // logits collapses to a uniform distribution: each output should
    // be exactly 1/10 = 0.1, independent of the max-subtraction
    // detail inside the normalization stage.
    for (int i = 0; i < N_IN; i++)
        input[i] = 1.0f;

    for (int i = 0; i < N_IN; i++)
        for (int j = 0; j < N_OUT; j++)
            weights[i][j] = 1.0f;

    for (int j = 0; j < N_OUT; j++)
        bias[j] = 0.0f;

    dense_output(input, weights, bias, output);

    // All 10 outputs should be exactly 0.1, and they must sum to 1.0.
    float expected = 0.1f;
    bool pass = true;
    float sum = 0.0f;

    for (int j = 0; j < N_OUT; j++) {
        sum += output[j];
        if (std::fabs(output[j] - expected) > 1e-5) pass = false;
    }

    std::cout << "Expected output[j] for all j: " << expected << std::endl;
    std::cout << "Actual output[0]:             " << output[0] << std::endl;
    std::cout << "Sum of all outputs:           " << sum << std::endl;

    if (pass && std::fabs(sum - 1.0f) < 1e-5) {
        std::cout << "TEST PASSED" << std::endl;
        return 0;
    } else {
        std::cout << "TEST FAILED" << std::endl;
        return 1;
    }
}
