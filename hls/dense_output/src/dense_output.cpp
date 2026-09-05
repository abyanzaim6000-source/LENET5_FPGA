#include "dense_output.h"
#include <cmath>

// Softmax(logit_j) = exp(logit_j) / sum_k(exp(logit_k)) needs every
// neuron's raw accumulation before ANY single output can be normalized
// -- it doesn't fit the single-pass "compute acc, activate, write
// output[j]" loop the ReLU layers use. So this is deliberately two
// separate stages instead of one fused loop:
//
//   Stage 1 -- MAC accumulation, same proven partial-sum pattern as
//              dense_c5_partialsum.cpp/dense_f6.cpp (PE_COUNT=4,
//              accumulator-limited by the float adder's latency, not
//              a memory-port issue -- so only the local `partial_sum[]`
//              array is partitioned, not `input`/`weights`), into a
//              small local `logits` array. No activation applied yet.
//   Stage 2 -- softmax normalization pass over the completed `logits`
//              array: max-subtraction for numerical stability, exp,
//              sum, then divide. Only runs once Stage 1 has produced
//              all N_OUT=10 values.
void dense_output(
    float input[N_IN],
    float weights[N_IN][N_OUT],
    float bias[N_OUT],
    float output[N_OUT]
) {
    float logits[N_OUT];

    // ---- Stage 1: MAC accumulation (PE_COUNT=4 partial-sum split) ----
    for (int j = 0; j < N_OUT; j++) {
        float partial_sum[PE_COUNT];
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
                    partial_sum[p] += input[mac_idx] * weights[mac_idx][j];
                }
            }
        }

        float acc = bias[j];
        for (int p = 0; p < PE_COUNT; p++) {
            #pragma HLS UNROLL
            acc += partial_sum[p];
        }
        logits[j] = acc;
    }

    // ---- Stage 2: softmax normalization over the completed logits ----
    float max_logit = logits[0];
    for (int j = 1; j < N_OUT; j++) {
        if (logits[j] > max_logit) max_logit = logits[j];
    }

    float exp_vals[N_OUT];
    float sum_exp = 0;
    for (int j = 0; j < N_OUT; j++) {
        exp_vals[j] = std::expf(logits[j] - max_logit);
        sum_exp += exp_vals[j];
    }

    for (int j = 0; j < N_OUT; j++) {
        output[j] = exp_vals[j] / sum_exp;
    }
}
