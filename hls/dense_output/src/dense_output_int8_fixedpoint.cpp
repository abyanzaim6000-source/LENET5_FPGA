#include "dense_output_int8_fixedpoint.h"
#include <cmath>

void dense_output_int8_fixedpoint(
    ap_int<8>  input[N_IN],
    ap_int<8>  weights[N_IN][N_OUT],
    ap_int<32> bias[N_OUT],
    float      combined_scale,
    float      output[N_OUT]
) {
    ap_int<32> logits_int[N_OUT];

    // ---- Stage 1: MAC accumulation (PE_COUNT=4 partial-sum split, pure
    //      integer -- same architecture as dense_c5_int8_fixedpoint.cpp/
    //      dense_f6_int8_fixedpoint.cpp, but no ReLU/requantize: softmax
    //      needs the full-precision accumulator). ----
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
        logits_int[j] = acc;
    }

    // ---- Stage 2: dequantize (one float multiply per neuron -- see the
    //      header for why the Q31 fixed-point technique isn't used here),
    //      then float softmax: max-subtract / exp / sum / divide, the
    //      exact same sequence as the proven float32 dense_output.cpp. ----
    float logits_float[N_OUT];
    for (int j = 0; j < N_OUT; j++) {
        logits_float[j] = (float)logits_int[j] * combined_scale;
    }

    float max_logit = logits_float[0];
    for (int j = 1; j < N_OUT; j++) {
        if (logits_float[j] > max_logit) max_logit = logits_float[j];
    }

    float exp_vals[N_OUT];
    float sum_exp = 0;
    for (int j = 0; j < N_OUT; j++) {
        exp_vals[j] = std::exp(logits_float[j] - max_logit);
        sum_exp += exp_vals[j];
    }

    for (int j = 0; j < N_OUT; j++) {
        output[j] = exp_vals[j] / sum_exp;
    }
}
