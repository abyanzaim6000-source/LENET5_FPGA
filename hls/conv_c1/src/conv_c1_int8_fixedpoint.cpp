#include "conv_c1_int8_fixedpoint.h"

// Same systolic PE-array architecture as conv_c1_int8.cpp (line buffer +
// AXI interfaces + local on-chip buffers + PE_COUNT-way partial-sum
// split) -- only the requantization step at the very end of each output
// pixel changes, from float32 multiply/divide to a genuine int64
// multiply + rounding right-shift. Isolating that one change keeps the
// float-vs-fixed-point comparison apples-to-apples with both earlier
// INT8 stages.
#define PE_COUNT 8
#define TOTAL_MACS (K * K * IN_C)
#define MACS_PER_PE ((TOTAL_MACS + PE_COUNT - 1) / PE_COUNT)

void conv_c1_int8_fixedpoint(
    ap_int<8>  input[IN_H][IN_W][IN_C],
    ap_int<8>  weights[K][K][IN_C][OUT_C],
    ap_int<32> bias[OUT_C],
    ap_int<32> requant_mult,
    ap_int<8>  requant_shift,
    ap_int<8>  output[IN_H][IN_W][OUT_C]
) {
    #pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=weights offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi port=bias offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem3
    #pragma HLS INTERFACE s_axilite port=input bundle=control
    #pragma HLS INTERFACE s_axilite port=weights bundle=control
    #pragma HLS INTERFACE s_axilite port=bias bundle=control
    #pragma HLS INTERFACE s_axilite port=output bundle=control
    #pragma HLS INTERFACE s_axilite port=requant_mult bundle=control
    #pragma HLS INTERFACE s_axilite port=requant_shift bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    // Local on-chip copies. THESE get partitioned, NOT the AXI arguments.
    static ap_int<8>  local_input[IN_H][IN_W][IN_C];
    static ap_int<8>  local_weights[K][K][IN_C][OUT_C];
    #pragma HLS ARRAY_PARTITION variable=local_weights complete dim=0
    static ap_int<32> local_bias[OUT_C];
    static ap_int<8>  local_output[IN_H][IN_W][OUT_C];

    // Burst-copy in from DDR (over AXI) to local buffers
    for (int r = 0; r < IN_H; r++)
        for (int c = 0; c < IN_W; c++)
            for (int i = 0; i < IN_C; i++)
                local_input[r][c][i] = input[r][c][i];

    for (int kr = 0; kr < K; kr++)
        for (int kc = 0; kc < K; kc++)
            for (int i = 0; i < IN_C; i++)
                for (int o = 0; o < OUT_C; o++)
                    local_weights[kr][kc][i][o] = weights[kr][kc][i][o];

    for (int o = 0; o < OUT_C; o++)
        local_bias[o] = bias[o];

    // ---- Same systolic computation as conv_c1_int8.cpp, now with a
    // fully-integer requantization step ----
    static ap_int<8> line_buf[K][IN_W + 2*PAD][IN_C];
    #pragma HLS ARRAY_PARTITION variable=line_buf complete dim=0

    ap_int<8> window[K][K][IN_C];
    #pragma HLS ARRAY_PARTITION variable=window complete dim=0

    ap_int<32> partial_sum[PE_COUNT];
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
                line_buf[K-1][c][i] = valid ? local_input[src_row][src_col][i] : (ap_int<8>)0;
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
                            // PURE INTEGER MULTIPLY-ACCUMULATE: int8 x int8 -> int32
                            partial_sum[p] += (ap_int<32>)window[kr][kc][i] * (ap_int<32>)local_weights[kr][kc][i][o];
                        }
                    }
                }
                ap_int<32> acc = local_bias[o];
                for (int p = 0; p < PE_COUNT; p++) {
                    #pragma HLS UNROLL
                    acc += partial_sum[p];
                }

                // ReLU directly on the integer accumulator: sign(acc) ==
                // sign of the (never-computed) dequantized value, since
                // the true scale factor is always positive -- exact, no
                // float dequant step needed at all.
                ap_int<32> acc_relu = (acc < 0) ? ap_int<32>(0) : acc;

                // Fixed-point requantize: genuine int64 multiply by the
                // Q31 fixed-point mantissa, then a rounding right-shift
                // by requant_shift bits -- no floating point anywhere in
                // this per-pixel path. acc_relu >= 0 and requant_mult > 0
                // always (by construction), so the product is always
                // non-negative and a plain "add half the LSB, then
                // shift" round-to-nearest is unambiguous.
                ap_int<64> prod = (ap_int<64>)acc_relu * (ap_int<64>)requant_mult;
                ap_int<64> half = (requant_shift > 0)
                                     ? (ap_int<64>(1) << (requant_shift - 1))
                                     : ap_int<64>(0);
                ap_int<64> shifted = (prod + half) >> requant_shift;

                ap_int<32> rounded = shifted;
                if (rounded > 127) rounded = 127;
                if (rounded < -128) rounded = -128;
                local_output[out_r][out_c][o] = (ap_int<8>)rounded;
            }
        }
    }

    // Burst-copy result back out to DDR
    for (int r = 0; r < IN_H; r++)
        for (int c = 0; c < IN_W; c++)
            for (int o = 0; o < OUT_C; o++)
                output[r][c][o] = local_output[r][c][o];
}
