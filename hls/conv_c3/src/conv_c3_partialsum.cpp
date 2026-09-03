#include "conv_c3.h"

#define PE_COUNT 6
#define TOTAL_MACS (IN_C * K * K)
#define MACS_PER_PE ((TOTAL_MACS + PE_COUNT - 1) / PE_COUNT)

// KEPT / FINAL for this stage (see Results/hls_results.md). Fix for the
// accumulator-limited baseline: conv_c3.cpp pipelines the inner i/kr/kc MAC
// loop but is capped at II=5 because every iteration reads back the SAME
// scalar `acc` before adding (the float adder's latency). This applies the
// same partial-sum-splitting pattern already proven for this project's
// accumulator bottleneck in conv_c1_systolic.cpp: split the 150-term
// reduction (IN_C*K*K = 6*5*5) across PE_COUNT independent partial-sum
// registers, each accumulating its own subset of terms via a flattened
// mac_idx, combined into one final sum at the end.
//
// A div/mod-free rewrite was tried (conv_c3_partialsum_roundrobin.cpp) to
// avoid decoding mac_idx via '/'/'%' by K=5 -- it succeeds at that (DSP
// dropped to near-baseline levels) but loses the "perfect loop nest"
// property that lets HLS auto-flatten the outer o/r/c loop into the
// pipeline, so overall latency got ~6.4x WORSE and Fmax dropped below the
// 100MHz target. This version is kept as C3's optimized stage instead: it's
// the only one of the two that meets timing (111.66MHz > 100MHz), and the
// mac_idx decode cost, while real, is a secondary concern next to that.
void conv_c3_partialsum(
    float input[IN_H][IN_W][IN_C],
    float weights[K][K][IN_C][OUT_C],
    float bias[OUT_C],
    float output[OUT_H][OUT_W][OUT_C]
) {
    for (int o = 0; o < OUT_C; o++) {
        for (int r = 0; r < OUT_H; r++) {
            for (int c = 0; c < OUT_W; c++) {
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
                        if (mac_idx < TOTAL_MACS) {
                            int i  = mac_idx / (K * K);
                            int kr = (mac_idx / K) % K;
                            int kc = mac_idx % K;
                            partial_sum[p] += input[r + kr][c + kc][i] * weights[kr][kc][i][o];
                        }
                    }
                }
                float acc = bias[o];
                for (int p = 0; p < PE_COUNT; p++) {
                    #pragma HLS UNROLL
                    acc += partial_sum[p];
                }
                output[r][c][o] = (acc > 0) ? acc : 0;
            }
        }
    }
}
