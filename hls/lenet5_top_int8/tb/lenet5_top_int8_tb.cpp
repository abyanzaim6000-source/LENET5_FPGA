#include <iostream>
#include <cmath>
#include <ap_int.h>
#include "../src/lenet5_top_int8.h"
#include "lenet5_top_int8_test_data.h"

// Forward declarations of the 7 already-proven INT8 layer IPs -- same
// literal-dimension style lenet5_top_int8.cpp itself uses (their own
// headers reuse macro names like IN_H/N_IN for different values, so they
// can't all be #included into this one translation unit). Declared again
// here so this testbench can call each stage DIRECTLY, verifying every
// intermediate activation bit-exact against generate_test_data_int8.py's
// chained Python reference -- not just the final softmax output -- the
// same standard used throughout this INT8 phase.
void conv_c1_int8_fixedpoint(
    ap_int<8>  input[28][28][1],
    ap_int<8>  weights[5][5][1][6],
    ap_int<32> bias[6],
    ap_int<32> requant_mult,
    ap_int<8>  requant_shift,
    ap_int<8>  output[28][28][6]
);

void pool_s2_int8(
    ap_int<8> input[28][28][6],
    ap_int<8> output[14][14][6]
);

void conv_c3_int8_fixedpoint(
    ap_int<8>  input[14][14][6],
    ap_int<8>  weights[5][5][6][16],
    ap_int<32> bias[16],
    ap_int<32> requant_mult,
    ap_int<8>  requant_shift,
    ap_int<8>  output[10][10][16]
);

void pool_s4_int8(
    ap_int<8> input[10][10][16],
    ap_int<8> output[5][5][16]
);

void dense_c5_int8_fixedpoint(
    ap_int<8>  input[400],
    ap_int<8>  weights[400][120],
    ap_int<32> bias[120],
    ap_int<32> requant_mult,
    ap_int<8>  requant_shift,
    ap_int<8>  output[120]
);

void dense_f6_int8_fixedpoint(
    ap_int<8>  input[120],
    ap_int<8>  weights[120][84],
    ap_int<32> bias[84],
    ap_int<32> requant_mult,
    ap_int<8>  requant_shift,
    ap_int<8>  output[84]
);

int main() {
    // ---- Static top-level buffers, loaded once from the auto-generated
    // header -- the SAME data feeds both the per-stage direct calls below
    // and the full lenet5_top_int8() call. ----
    static ap_int<8>  image[28][28][1];
    static ap_int<8>  c1_weights[5][5][1][6];
    static ap_int<32> c1_bias[6];
    static ap_int<8>  c3_weights[5][5][6][16];
    static ap_int<32> c3_bias[16];
    static ap_int<8>  c5_weights[400][120];
    static ap_int<32> c5_bias[120];
    static ap_int<8>  f6_weights[120][84];
    static ap_int<32> f6_bias[84];
    static ap_int<8>  output_weights[84][10];
    static ap_int<32> output_bias[10];
    static float      result[10];

    for (int r = 0; r < 28; r++)
        for (int c = 0; c < 28; c++)
            image[r][c][0] = image_data[r][c][0];

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

    bool all_stages_exact = true;

    // ---- Stage-by-stage bit-exact verification, calling each already-
    // proven INT8 kernel directly with the SAME real weights/activations
    // generate_test_data_int8.py chained through in Python. ----
    static ap_int<8> c1_out[28][28][6];
    conv_c1_int8_fixedpoint(image, c1_weights, c1_bias,
                             C1_REQUANT_MULT, C1_REQUANT_SHIFT, c1_out);
    {
        int mismatches = 0;
        for (int r = 0; r < 28; r++)
            for (int c = 0; c < 28; c++)
                for (int o = 0; o < 6; o++)
                    if (c1_out[r][c][o].to_int() != c1_expected[r][c][o]) mismatches++;
        std::cout << "C1 stage mismatches: " << mismatches << " / " << (28 * 28 * 6) << std::endl;
        if (mismatches != 0) all_stages_exact = false;
    }

    static ap_int<8> s2_out[14][14][6];
    pool_s2_int8(c1_out, s2_out);
    {
        int mismatches = 0;
        for (int r = 0; r < 14; r++)
            for (int c = 0; c < 14; c++)
                for (int o = 0; o < 6; o++)
                    if (s2_out[r][c][o].to_int() != s2_expected[r][c][o]) mismatches++;
        std::cout << "S2 stage mismatches: " << mismatches << " / " << (14 * 14 * 6) << std::endl;
        if (mismatches != 0) all_stages_exact = false;
    }

    static ap_int<8> c3_out[10][10][16];
    conv_c3_int8_fixedpoint(s2_out, c3_weights, c3_bias,
                             C3_REQUANT_MULT, C3_REQUANT_SHIFT, c3_out);
    {
        int mismatches = 0;
        for (int r = 0; r < 10; r++)
            for (int c = 0; c < 10; c++)
                for (int o = 0; o < 16; o++)
                    if (c3_out[r][c][o].to_int() != c3_expected[r][c][o]) mismatches++;
        std::cout << "C3 stage mismatches: " << mismatches << " / " << (10 * 10 * 16) << std::endl;
        if (mismatches != 0) all_stages_exact = false;
    }

    static ap_int<8> s4_out[5][5][16];
    pool_s4_int8(c3_out, s4_out);
    {
        int mismatches = 0;
        for (int r = 0; r < 5; r++)
            for (int c = 0; c < 5; c++)
                for (int o = 0; o < 16; o++)
                    if (s4_out[r][c][o].to_int() != s4_expected[r][c][o]) mismatches++;
        std::cout << "S4 stage mismatches: " << mismatches << " / " << (5 * 5 * 16) << std::endl;
        if (mismatches != 0) all_stages_exact = false;
    }

    static ap_int<8> flat[400];
    for (int r = 0; r < 5; r++)
        for (int c = 0; c < 5; c++)
            for (int ch = 0; ch < 16; ch++)
                flat[r * 5 * 16 + c * 16 + ch] = s4_out[r][c][ch];

    static ap_int<8> c5_out[120];
    dense_c5_int8_fixedpoint(flat, c5_weights, c5_bias,
                              C5_REQUANT_MULT, C5_REQUANT_SHIFT, c5_out);
    {
        int mismatches = 0;
        for (int j = 0; j < 120; j++)
            if (c5_out[j].to_int() != c5_expected[j]) mismatches++;
        std::cout << "C5 stage mismatches: " << mismatches << " / 120" << std::endl;
        if (mismatches != 0) all_stages_exact = false;
    }

    static ap_int<8> f6_out[84];
    dense_f6_int8_fixedpoint(c5_out, f6_weights, f6_bias,
                              F6_REQUANT_MULT, F6_REQUANT_SHIFT, f6_out);
    {
        int mismatches = 0;
        for (int j = 0; j < 84; j++)
            if (f6_out[j].to_int() != f6_expected[j]) mismatches++;
        std::cout << "F6 stage mismatches: " << mismatches << " / 84" << std::endl;
        if (mismatches != 0) all_stages_exact = false;
    }

    std::cout << (all_stages_exact
                  ? "All intermediate stages (C1/S2/C3/S4/C5/F6) match the Python int8 reference chain bit-exact."
                  : "At least one intermediate stage did NOT match the Python reference exactly.")
              << std::endl;

    // ---- Full top-level call: the ACTUAL lenet5_top_int8() IP, fed the
    // same real image/weights, checked against the same softmax
    // reference to a tight tolerance (softmax's exp()/summation-order
    // can differ trivially from NumPy's -- same 1e-5 standard as
    // dense_output_int8_tb.cpp and the float32 lenet5_top_tb.cpp). ----
    lenet5_top_int8(image, c1_weights, c1_bias, C1_REQUANT_MULT, C1_REQUANT_SHIFT,
                     c3_weights, c3_bias, C3_REQUANT_MULT, C3_REQUANT_SHIFT,
                     c5_weights, c5_bias, C5_REQUANT_MULT, C5_REQUANT_SHIFT,
                     f6_weights, f6_bias, F6_REQUANT_MULT, F6_REQUANT_SHIFT,
                     output_weights, output_bias, OUTPUT_X_SCALE, OUTPUT_W_SCALE,
                     result);

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

    float max_abs_diff = 0.0f;
    for (int j = 0; j < 10; j++) {
        float diff = std::fabs(result[j] - expected_output[j]);
        if (diff > max_abs_diff) max_abs_diff = diff;
    }

    std::cout << "Sum of HLS outputs:  " << sum << std::endl;
    std::cout << "HLS predicted class: " << argmax
               << " (Python int8 chain predicted: " << EXPECTED_CLASS
               << ", MNIST true label: " << TRUE_LABEL << ")" << std::endl;
    std::cout << "Max abs diff vs. Python reference: " << max_abs_diff << std::endl;

    bool sum_ok = std::fabs(sum - 1.0f) < 1e-5f;
    bool tol_ok = max_abs_diff < 1e-5f;
    bool class_ok = (argmax == EXPECTED_CLASS);

    bool pass = all_stages_exact && sum_ok && tol_ok && class_ok;

    if (pass) {
        std::cout << "TEST PASSED -- every intermediate INT8 stage matches the Python "
                     "reference chain bit-exact, final softmax matches within tolerance"
                  << std::endl;
        return 0;
    } else {
        if (!all_stages_exact) std::cout << "TEST FAILED: intermediate stage mismatch" << std::endl;
        if (!sum_ok) std::cout << "TEST FAILED: output does not sum to 1.0" << std::endl;
        if (!tol_ok) std::cout << "TEST FAILED: softmax output outside tolerance" << std::endl;
        if (!class_ok) std::cout << "TEST FAILED: predicted class does not match Python reference" << std::endl;
        return 1;
    }
}
