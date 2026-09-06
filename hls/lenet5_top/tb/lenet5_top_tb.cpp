#include <iostream>
#include <cmath>
#include "../src/lenet5_top.h"
#include "lenet5_test_data.h"

int main() {
    static float image[28][28][1];
    static float c1_weights[5][5][1][6];
    static float c1_bias[6];
    static float c3_weights[5][5][6][16];
    static float c3_bias[16];
    static float c5_weights[400][120];
    static float c5_bias[120];
    static float f6_weights[120][84];
    static float f6_bias[84];
    static float output_weights[84][10];
    static float output_bias[10];
    static float result[10];

    // Load the real MNIST test image + real trained lenet5_relu.keras
    // weights from the auto-generated header (see generate_test_data.py).
    for (int r = 0; r < 28; r++)
        for (int c = 0; c < 28; c++)
            image[r][c][0] = test_image[r][c][0];

    for (int kr = 0; kr < 5; kr++)
        for (int kc = 0; kc < 5; kc++)
            for (int i = 0; i < 1; i++)
                for (int o = 0; o < 6; o++)
                    c1_weights[kr][kc][i][o] = c1_weights_data[kr][kc][i][o];
    for (int o = 0; o < 6; o++)
        c1_bias[o] = c1_bias_data[o];

    for (int kr = 0; kr < 5; kr++)
        for (int kc = 0; kc < 5; kc++)
            for (int i = 0; i < 6; i++)
                for (int o = 0; o < 16; o++)
                    c3_weights[kr][kc][i][o] = c3_weights_data[kr][kc][i][o];
    for (int o = 0; o < 16; o++)
        c3_bias[o] = c3_bias_data[o];

    for (int i = 0; i < 400; i++)
        for (int j = 0; j < 120; j++)
            c5_weights[i][j] = c5_weights_data[i][j];
    for (int j = 0; j < 120; j++)
        c5_bias[j] = c5_bias_data[j];

    for (int i = 0; i < 120; i++)
        for (int j = 0; j < 84; j++)
            f6_weights[i][j] = f6_weights_data[i][j];
    for (int j = 0; j < 84; j++)
        f6_bias[j] = f6_bias_data[j];

    for (int i = 0; i < 84; i++)
        for (int j = 0; j < 10; j++)
            output_weights[i][j] = output_weights_data[i][j];
    for (int j = 0; j < 10; j++)
        output_bias[j] = output_bias_data[j];

    lenet5_top(image, c1_weights, c1_bias, c3_weights, c3_bias,
               c5_weights, c5_bias, f6_weights, f6_bias,
               output_weights, output_bias, result);

    std::cout << "HLS result:      ";
    float sum = 0.0f;
    int argmax = 0;
    for (int j = 0; j < 10; j++) {
        std::cout << result[j] << " ";
        sum += result[j];
        if (result[j] > result[argmax]) argmax = j;
    }
    std::cout << std::endl;

    std::cout << "Reference result: ";
    for (int j = 0; j < 10; j++) std::cout << expected_output[j] << " ";
    std::cout << std::endl;

    std::cout << "Sum of HLS outputs:    " << sum << std::endl;
    std::cout << "HLS predicted class:   " << argmax << std::endl;
    std::cout << "Python predicted class (manual_layers.py reference): " << EXPECTED_CLASS << std::endl;
    std::cout << "MNIST true label:      " << TRUE_LABEL << std::endl;

    float max_abs_diff = 0.0f;
    for (int j = 0; j < 10; j++) {
        float diff = std::fabs(result[j] - expected_output[j]);
        if (diff > max_abs_diff) max_abs_diff = diff;
    }
    std::cout << "Max abs diff vs. Python reference: " << max_abs_diff << std::endl;

    bool sum_ok = std::fabs(sum - 1.0f) < 1e-3f;
    bool class_ok = (argmax == EXPECTED_CLASS);

    if (sum_ok && class_ok) {
        std::cout << "TEST PASSED" << std::endl;
        return 0;
    } else {
        if (!sum_ok) std::cout << "TEST FAILED: output does not sum to 1.0" << std::endl;
        if (!class_ok) std::cout << "TEST FAILED: predicted class does not match Python reference" << std::endl;
        return 1;
    }
}
