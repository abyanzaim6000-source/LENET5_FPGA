#include "conv_c1_int8.h"

// Same systolic PE-array architecture as conv_c1_systolic.cpp (line buffer
// + AXI interfaces + local on-chip buffers + PE_COUNT-way partial-sum
// split), so the float32-vs-int8 utilization comparison isolates the
// effect of the numeric representation, not the architecture. Only the
// datapath changes: ap_int<8>/ap_int<32> MACs instead of float MACs, and
// an explicit dequantize->ReLU->requantize step (in place of a plain
// float ReLU) at each output pixel.
#define PE_COUNT 8
#define TOTAL_MACS (K * K * IN_C)
#define MACS_PER_PE ((TOTAL_MACS + PE_COUNT - 1) / PE_COUNT)

void conv_c1_int8(
    ap_int<8>  input[IN_H][IN_W][IN_C],
    ap_int<8>  weights[K][K][IN_C][OUT_C],
    ap_int<32> bias[OUT_C],
    float x_scale,
    float w_scale,
    float out_scale,
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
    #pragma HLS INTERFACE s_axilite port=x_scale bundle=control
    #pragma HLS INTERFACE s_axilite port=w_scale bundle=control
    #pragma HLS INTERFACE s_axilite port=out_scale bundle=control
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

    // Combined input*weight dequantization scale -- applied once per
    // output pixel after the integer MAC, exactly like conv2d_int()'s
    // `acc.astype(np.float32) * combined_scale`.
    float combined_scale = x_scale * w_scale;

    // Reciprocal precomputed ONCE (loop-invariant), so the per-pixel
    // requantization step is a multiply instead of a division -- the
    // per-pixel `dequant / out_scale` divide was a 15-cycle fdiv that
    // prevented the output-channel loop from pipelining/flattening with
    // the MAC reduction (see Results/hls_results.md's C1 INT8 log). Only
    // this ONE division happens per call, not once per output pixel.
    float inv_out_scale = 1.0f / out_scale;

    // ---- Same systolic computation as conv_c1_systolic.cpp, now integer ----
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

                // Dequantize -> ReLU -> requantize, same order and same
                // math as integer_layers.py's conv2d_int() + relu() +
                // quantize_activation() chain:
                //   dequant = acc * (x_scale*w_scale)          [conv2d_int]
                //   dequant = max(0, dequant)                   [relu]
                //   requant = round(dequant * (1/out_scale))    [quantize_activation]
                // clamped to the signed INT8 range. Multiply by the
                // precomputed reciprocal instead of dividing by out_scale
                // every pixel -- NOT bit-identical to true division in
                // general IEEE754 (a/b != a*(1/b) exactly), so the Python
                // golden reference was regenerated with this exact same
                // reciprocal-multiply to keep the exact-match property.
                float dequant = (float)acc * combined_scale;
                if (dequant < 0.0f) dequant = 0.0f;
                float requant = dequant * inv_out_scale;

                int rounded = (int)(requant >= 0.0f ? requant + 0.5f : requant - 0.5f);
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
