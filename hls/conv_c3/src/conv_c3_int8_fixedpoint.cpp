#include "conv_c3_int8_fixedpoint.h"

// Same PE_COUNT=6 partial-sum split with LUT-based mac_idx decode as
// conv_c3_partialsum_lut.cpp (division-free (i,kr,kc) lookup instead of
// '/'/'%' by K=5 -- see that file's own header comment and this IP's
// optimization log in Results/hls_results.md), and the same flattened
// mac_idx=m*PE_COUNT+p / (m,p) loop nest that lets HLS auto-flatten the
// outer o/r/c loop into the inner pipeline. Only the datapath changes:
// ap_int<8>/ap_int<32> MACs instead of float, and conv_c1_int8_fixedpoint's
// proven fixed-point (int64 multiply + rounding right-shift) requantize
// in place of a plain float ReLU.
#define PE_COUNT 6
#define TOTAL_MACS (IN_C * K * K)
#define MACS_PER_PE ((TOTAL_MACS + PE_COUNT - 1) / PE_COUNT)

void conv_c3_int8_fixedpoint(
    ap_int<8>  input[IN_H][IN_W][IN_C],
    ap_int<8>  weights[K][K][IN_C][OUT_C],
    ap_int<32> bias[OUT_C],
    ap_int<32> requant_mult,
    ap_int<8>  requant_shift,
    ap_int<8>  output[OUT_H][OUT_W][OUT_C]
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
                        if (mac_idx < TOTAL_MACS) {
                            int i  = i_lut[mac_idx];
                            int kr = kr_lut[mac_idx];
                            int kc = kc_lut[mac_idx];
                            // PURE INTEGER MULTIPLY-ACCUMULATE: int8 x int8 -> int32
                            partial_sum[p] += (ap_int<32>)input[r + kr][c + kc][i] * (ap_int<32>)weights[kr][kc][i][o];
                        }
                    }
                }
                ap_int<32> acc = bias[o];
                for (int p = 0; p < PE_COUNT; p++) {
                    #pragma HLS UNROLL
                    acc += partial_sum[p];
                }

                // ReLU directly on the integer accumulator (sign(acc) ==
                // sign of the never-computed dequantized value, since
                // the true scale factor is always positive), then the
                // same fixed-point requantize as conv_c1_int8_fixedpoint.cpp:
                // a genuine int64 multiply by the Q31 mantissa, then a
                // rounding right-shift -- no floating point anywhere in
                // this per-pixel path.
                ap_int<32> acc_relu = (acc < 0) ? ap_int<32>(0) : acc;

                ap_int<64> prod = (ap_int<64>)acc_relu * (ap_int<64>)requant_mult;
                ap_int<64> half = (requant_shift > 0)
                                     ? (ap_int<64>(1) << (requant_shift - 1))
                                     : ap_int<64>(0);
                ap_int<64> shifted = (prod + half) >> requant_shift;

                ap_int<32> rounded = shifted;
                if (rounded > 127) rounded = 127;
                if (rounded < -128) rounded = -128;
                output[r][c][o] = (ap_int<8>)rounded;
            }
        }
    }
}
