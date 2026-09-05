#include "dense_f6.h"

// Same partial-sum pattern as dense_c5_partialsum.cpp (PE_COUNT=4),
// applied directly instead of building a serial-accumulator baseline
// first: C5's baseline showed the bottleneck is the loop-carried
// dependency on a single scalar `acc` (the float adder's latency),
// not a memory-port conflict -- so only the local `partial_sum[]`
// array is partitioned; `input`/`weights` are left as plain arrays
// since ARRAY_PARTITION on them would do nothing for this bottleneck.
// The 120-term reduction per output neuron is split across 4
// independent accumulator chains (30 terms each via the flattened
// `mac_idx`), combined into one final sum right before the bias add.
void dense_f6(
    float input[N_IN],
    float weights[N_IN][N_OUT],
    float bias[N_OUT],
    float output[N_OUT]
) {
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

        // ReLU activation, applied inline (matches lenet5_relu.py)
        output[j] = (acc > 0) ? acc : 0;
    }
}
