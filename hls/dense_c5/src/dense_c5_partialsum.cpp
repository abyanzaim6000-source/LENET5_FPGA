#include "dense_c5.h"

#define PE_COUNT 4
#define MACS_PER_PE ((N_IN + PE_COUNT - 1) / PE_COUNT)

// KEPT / FINAL for this stage (see Results/hls_results.md). Fix for the
// accumulator-limited baseline: dense_c5.cpp pipelines the inner i-loop
// MAC but is capped at II=5 because every iteration reads back the SAME
// scalar `acc` before adding (the float adder's 4-cycle latency). Same
// partial-sum-splitting pattern as conv_c3_partialsum.cpp: split the
// N_IN=400-term reduction across PE_COUNT independent partial-sum
// registers, each accumulating its own subset of terms via a flattened
// index, combined into one final sum at the end.
//
// PE_COUNT=4, 5, and (by extrapolation from C3) 6 all land at the SAME
// II=4 floor -- confirmed by actually running PE_COUNT=4 (DSP 7, II=4)
// and PE_COUNT=5 (DSP 12, II=4): more parallel chains only shrinks
// MACS_PER_PE (so total latency drops) and adds DSPs, it does not lower
// II. That's because #pragma UNROLL over `p` puts all PE_COUNT adds in
// the SAME pipeline iteration, so any one partial_sum[p] is revisited
// every 1 iteration regardless of PE_COUNT -- the recurrence distance
// that must absorb the 4-cycle fadd latency is fixed at 1 iteration by
// construction, not by PE_COUNT. Getting a true PE_COUNT-iteration
// revisit gap would need temporal (round-robin) interleaving instead of
// spatial unroll, which is exactly what conv_c3_partialsum_roundrobin.cpp
// tried for C3 -- and that hit a different wall (variable array index
// defeats HLS's dependence analysis, forcing the same conservative II=4
// anyway, plus it broke loop flattening). Since PE_COUNT=4 already meets
// the 100MHz target at the lowest resource cost among the II=4 options,
// PE_COUNT=6 was not run -- it would cost more DSP for the same II=4.
void dense_c5_partialsum(
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
        output[j] = (acc > 0) ? acc : 0;
    }
}
