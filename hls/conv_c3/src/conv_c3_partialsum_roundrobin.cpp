#include "conv_c3.h"

#define PE_COUNT 6
// TOTAL_MACS = IN_C*K*K = 6*5*5 = 150, evenly divisible by PE_COUNT (150/6=25).

// EXPLORED, REJECTED (see Results/hls_results.md) -- kept for the record,
// same as C1's rejected intermediate systolic attempts. This rewrite of
// conv_c3_partialsum.cpp's mac_idx decode was meant to eliminate the
// division/modulo-by-K=5 hardware that decode required (mac_idx/(K*K),
// (mac_idx/K)%K, mac_idx%K -- 33 of 45 DSPs in that version). It succeeds at
// that goal: i/kr/kc come straight from natural nested loop counters (zero
// decode cost), and which of the PE_COUNT partial-sum registers each term
// accumulates into is tracked by a simple increment-and-wrap counter (a
// compare + reset) instead of a literal '% PE_COUNT'.
//
// BUT: the `pe` counter's compare/reset logic in the loop latch breaks the
// "perfect loop nest" property HLS needs to auto-flatten the outer o/r/c
// loop into the inner pipeline (log: "Cannot flatten loop ... the outer
// loop is not a perfect loop because there is nontrivial logic in the loop
// latch"). Without that flattening, the 150-iteration inner pipeline (still
// only II=4, same accumulator-latency floor as the kept version) drains and
// refills separately for all 1,600 (o,r,c) combinations with zero overlap,
// so overall latency is ~6.4x WORSE than conv_c3_partialsum.cpp despite the
// per-iteration II being identical, and Fmax (93.03MHz) falls below the
// 100MHz target -- the `pe`-select mux lands on the same critical path as
// the input load + multiply. Resource cost is genuinely near-baseline
// (DSP 5, FF 964, LUT 2050), so this is a real cost/latency trade worth
// revisiting later (e.g. if the outer loop's flattening can be recovered
// without reintroducing division), but conv_c3_partialsum.cpp is what's
// kept as C3's optimized stage for now.
void conv_c3_partialsum_roundrobin(
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

                int pe = 0;
                for (int i = 0; i < IN_C; i++) {
                    for (int kr = 0; kr < K; kr++) {
                        for (int kc = 0; kc < K; kc++) {
                            #pragma HLS PIPELINE II=1
                            partial_sum[pe] += input[r + kr][c + kc][i] * weights[kr][kc][i][o];
                            pe = (pe == PE_COUNT - 1) ? 0 : pe + 1;
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
