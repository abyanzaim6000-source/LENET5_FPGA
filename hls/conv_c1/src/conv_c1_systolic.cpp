#include "conv_c1.h"

// Number of parallel Processing Elements (PEs) in the systolic MAC chain.
#define PE_COUNT 8
#define TOTAL_MACS (K * K * IN_C)           // 5*5*1 = 25 MACs per output channel
#define MACS_PER_PE ((TOTAL_MACS + PE_COUNT - 1) / PE_COUNT)

void conv_c1_systolic(
    float input[IN_H][IN_W][IN_C],
    float weights[K][K][IN_C][OUT_C],
    float bias[OUT_C],
    float output[IN_H][IN_W][OUT_C]
) {
    // AXI4 Master interfaces: each array gets its own bus to DDR memory
    #pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=weights offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi port=bias offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem3

    // AXI4-Lite control interface: lets the ARM processor start/stop/check status
    #pragma HLS INTERFACE s_axilite port=input bundle=control
    #pragma HLS INTERFACE s_axilite port=weights bundle=control
    #pragma HLS INTERFACE s_axilite port=bias bundle=control
    #pragma HLS INTERFACE s_axilite port=output bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    #pragma HLS ARRAY_PARTITION variable=weights complete dim=0

    static float line_buf[K][IN_W + 2*PAD][IN_C];
    #pragma HLS ARRAY_PARTITION variable=line_buf complete dim=0
    #pragma HLS ARRAY_PARTITION variable=output complete dim=3

    float window[K][K][IN_C];
    #pragma HLS ARRAY_PARTITION variable=window complete dim=0

    float partial_sum[PE_COUNT];
    #pragma HLS ARRAY_PARTITION variable=partial_sum complete dim=0

    for (int r = 0; r < K; r++)
        for (int c = 0; c < IN_W + 2*PAD; c++)
            for (int i = 0; i < IN_C; i++)
                line_buf[r][c][i] = 0;

    for (int out_r = 0; out_r < IN_H; out_r++) {

        int src_row = out_r + K - 1 - PAD;
        for (int shift = 0; shift < K - 1; shift++) {
            #pragma HLS UNROLL
            for (int c = 0; c < IN_W + 2*PAD; c++)
                for (int i = 0; i < IN_C; i++)
                    line_buf[shift][c][i] = line_buf[shift + 1][c][i];
        }
        for (int c = 0; c < IN_W + 2*PAD; c++) {
            for (int i = 0; i < IN_C; i++) {
                int src_col = c - PAD;
                bool valid = (src_row >= 0 && src_row < IN_H &&
                              src_col >= 0 && src_col < IN_W);
                line_buf[K-1][c][i] = valid ? input[src_row][src_col][i] : (float)0;
            }
        }

        for (int out_c = 0; out_c < IN_W; out_c++) {

            for (int kr = 0; kr < K; kr++)
                for (int kc = 0; kc < K; kc++)
                    for (int i = 0; i < IN_C; i++)
                        window[kr][kc][i] = line_buf[kr][out_c + kc][i];

            for (int o = 0; o < OUT_C; o++) {

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
                            int kr = mac_idx / (K * IN_C);
                            int kc = (mac_idx / IN_C) % K;
                            int i  = mac_idx % IN_C;
                            partial_sum[p] += window[kr][kc][i] * weights[kr][kc][i][o];
                        }
                    }
                }

                float acc = bias[o];
                for (int p = 0; p < PE_COUNT; p++) {
                    #pragma HLS UNROLL
                    acc += partial_sum[p];
                }

                output[out_r][out_c][o] = (acc > 0) ? acc : 0;
            }
        }
    }
}