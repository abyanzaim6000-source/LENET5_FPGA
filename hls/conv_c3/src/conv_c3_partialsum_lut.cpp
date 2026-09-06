#include "conv_c3.h"

#define PE_COUNT 6
#define TOTAL_MACS (IN_C * K * K)
#define MACS_PER_PE ((TOTAL_MACS + PE_COUNT - 1) / PE_COUNT)

// EXPLORED: LUT-based mac_idx decode. Same flattened mac_idx/(m,p) loop
// structure as conv_c3_partialsum.cpp (the kept version) -- this is
// deliberate: that structure is exactly what lets HLS auto-flatten the
// outer o/r/c loop into the inner pipeline (see conv_c3_partialsum.cpp's own
// header comment and Results/hls_results.md). The only change is HOW
// mac_idx gets decoded back into (i, kr, kc): instead of computing it with
// '/' and '%' by K=5 (non-power-of-2 division, ~33 of 45 DSPs and large
// urem_* FF cost in the kept version), it's looked up in three small
// TOTAL_MACS=150-entry constant tables. Each table entry is just the
// natural nested-loop (i, kr, kc) value for that flattened position -- the
// same values conv_c3_partialsum_roundrobin.cpp gets for free from its loop
// counters, but here delivered via ROM read (ordinary array indexing, same
// cost class as reading `weights`) instead of a stateful runtime counter.
// The tables are ARRAY_PARTITION'd complete so each of the PE_COUNT
// unrolled lookups is a single mux instead of contending for 1-2 BRAM
// ports (which would just add a new bottleneck in place of the divider).
void conv_c3_partialsum_lut(
    float input[IN_H][IN_W][IN_C],
    float weights[K][K][IN_C][OUT_C],
    float bias[OUT_C],
    float output[OUT_H][OUT_W][OUT_C]
) {
    static const int i_lut[TOTAL_MACS] = {
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
        3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
        4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
        5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5
    };
    static const int kr_lut[TOTAL_MACS] = {
        0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4,
        0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4,
        0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4,
        0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4,
        0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4,
        0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4
    };
    static const int kc_lut[TOTAL_MACS] = {
        0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,
        0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,
        0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,
        0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,
        0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,
        0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4
    };
    #pragma HLS ARRAY_PARTITION variable=i_lut complete dim=0
    #pragma HLS ARRAY_PARTITION variable=kr_lut complete dim=0
    #pragma HLS ARRAY_PARTITION variable=kc_lut complete dim=0

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
                            int i  = i_lut[mac_idx];
                            int kr = kr_lut[mac_idx];
                            int kc = kc_lut[mac_idx];
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
