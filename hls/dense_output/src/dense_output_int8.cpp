#include "dense_output_int8.h"
#include <cmath>

// See dense_output_int8.h for the two-stage rationale (identical
// reasoning to the proven float32 dense_output.cpp, just with an int8
// MAC stage in place of a float one).
void dense_output_int8(
    ap_int<8>  input[N_IN],
    ap_int<8>  weights[N_IN][N_OUT],
    ap_int<32> bias[N_OUT],
    float x_scale,
    float w_scale,
    float output[N_OUT]
) {
    ap_int<32> logits[N_OUT];

    // ---- Stage 1: int8 MAC accumulation (PE_COUNT=4 partial-sum split),
    // same architecture as dense_c5_int8_fixedpoint.cpp/
    // dense_f6_int8_fixedpoint.cpp. No activation, no requantization --
    // this accumulator feeds Stage 2's float dequantize directly. ----
    for (int j = 0; j < N_OUT; j++) {
        ap_int<32> partial_sum[PE_COUNT];
        #pragma HLS ARRAY_PARTITION variable=partial_sum complete dim=0
        for (int p = 0; p < PE_COUNT; p++) {
            #pragma HLS UNROLL
            partial_sum[p] = 0;
        }
        for (int m = 0; m < MACS_PER_PE; m++) {
            #pragma HLS PIPELINE II=1
            for (int p = 0; p < PE_COUNT; p++) {
                #pragma HLS UNROLL
                int mac_idx = m * PE_COUNT + p;
                if (mac_idx < N_IN) {
                    // PURE INTEGER MULTIPLY-ACCUMULATE: int8 x int8 -> int32
                    partial_sum[p] += (ap_int<32>)input[mac_idx] * (ap_int<32>)weights[mac_idx][j];
                }
            }
        }
        ap_int<32> acc = bias[j];
        for (int p = 0; p < PE_COUNT; p++) {
            #pragma HLS UNROLL
            acc += partial_sum[p];
        }
        logits[j] = acc;
    }

    // ---- Stage 2: dequantize the int32 accumulator to float (single
    // combined-scale multiply per neuron, 10 total -- not a per-pixel
    // cost), then the SAME softmax normalization as dense_output.cpp:
    // max-subtraction for numerical stability, exp, sum, divide. ----
    float combined_scale = x_scale * w_scale;

    float dequant_logits[N_OUT];
    for (int j = 0; j < N_OUT; j++) {
        dequant_logits[j] = (float)logits[j] * combined_scale;
    }

    float max_logit = dequant_logits[0];
    for (int j = 1; j < N_OUT; j++) {
        if (dequant_logits[j] > max_logit) max_logit = dequant_logits[j];
    }

    float exp_vals[N_OUT];
    float sum_exp = 0;
    for (int j = 0; j < N_OUT; j++) {
        exp_vals[j] = std::exp(dequant_logits[j] - max_logit);
        sum_exp += exp_vals[j];
    }

    for (int j = 0; j < N_OUT; j++) {
        output[j] = exp_vals[j] / sum_exp;
    }
}
