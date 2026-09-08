#include <iostream>
#include "../src/conv_c1_int8_fixedpoint.h"
#include "conv_c1_int8_fixedpoint_test_data.h"

// Feeds the same real int8-quantized weights/input as conv_c1_int8_tb.cpp
// through the fixed-point INT8 HLS kernel, and checks the result against
// the Python fixed-point reference (generate_test_data_int8_fixedpoint.py:
// same int32 MAC, ReLU on the accumulator, then a genuine int64 multiply
// + rounding right-shift -- NOT the float division or reciprocal-multiply
// used in the earlier two INT8 stages) EXACTLY -- integer equality, not a
// floating-point tolerance.
int main() {
    static ap_int<8>  input[IN_H][IN_W][IN_C];
    static ap_int<8>  weights[K][K][IN_C][OUT_C];
    static ap_int<32> bias[OUT_C];
    static ap_int<8>  output[IN_H][IN_W][OUT_C];

    for (int r = 0; r < IN_H; r++)
        for (int c = 0; c < IN_W; c++)
            for (int i = 0; i < IN_C; i++)
                input[r][c][i] = input_data[r][c][i];

    for (int kr = 0; kr < K; kr++)
        for (int kc = 0; kc < K; kc++)
            for (int i = 0; i < IN_C; i++)
                for (int o = 0; o < OUT_C; o++)
                    weights[kr][kc][i][o] = weights_data[kr][kc][i][o];

    for (int o = 0; o < OUT_C; o++)
        bias[o] = bias_int_data[o];

    conv_c1_int8_fixedpoint(input, weights, bias, REQUANT_MULT, REQUANT_SHIFT, output);

    int mismatches = 0;
    int max_abs_diff = 0;
    for (int r = 0; r < IN_H; r++) {
        for (int c = 0; c < IN_W; c++) {
            for (int o = 0; o < OUT_C; o++) {
                int actual = output[r][c][o].to_int();
                int expected = expected_output[r][c][o];
                int diff = actual - expected;
                if (diff < 0) diff = -diff;
                if (diff > max_abs_diff) max_abs_diff = diff;
                if (actual != expected) {
                    if (mismatches < 10) {
                        std::cout << "Mismatch at [" << r << "][" << c << "][" << o
                                  << "]: expected " << expected << ", got " << actual << std::endl;
                    }
                    mismatches++;
                }
            }
        }
    }

    std::cout << "Total output elements: " << (IN_H * IN_W * OUT_C) << std::endl;
    std::cout << "Mismatches: " << mismatches << std::endl;
    std::cout << "Max abs diff: " << max_abs_diff << std::endl;

    if (mismatches == 0) {
        std::cout << "TEST PASSED -- HLS fixed-point INT8 output matches Python fixed-point reference exactly" << std::endl;
        return 0;
    } else {
        std::cout << "TEST FAILED" << std::endl;
        return 1;
    }
}
