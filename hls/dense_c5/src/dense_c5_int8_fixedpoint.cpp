#include "dense_c5_int8_fixedpoint.h"

// Same PE_COUNT=4 partial-sum split as dense_c5_partialsum.cpp (only
// partial_sum[] partitioned, per that IP's own bind-report diagnosis
// that the bottleneck is the accumulator's loop-carried recurrence, not
// a memory-port conflict), and the same flattened mac_idx=m*PE_COUNT+p /
// (m,p) loop nest used by conv_c3_int8_fixedpoint.cpp. Only the datapath
// changes: ap_int<8>/ap_int<32> MACs instead of float, and
// conv_c1_int8_fixedpoint's proven fixed-point (int64 multiply +
// rounding right-shift) requantize in place of a plain float ReLU.
#define PE_COUNT 4
#define MACS_PER_PE ((N_IN + PE_COUNT - 1) / PE_COUNT)

void dense_c5_int8_fixedpoint(
    ap_int<8>  input[N_IN],
    ap_int<8>  weights[N_IN][N_OUT],
    ap_int<32> bias[N_OUT],
    ap_int<32> requant_mult,
    ap_int<8>  requant_shift,
    ap_int<8>  output[N_OUT]
) {
    for (int j = 0; j < N_OUT; j++) {
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
                if (mac_idx < N_IN) {
                    // PURE INTEGER MULTIPLY-ACCUMULATE: int8 x int8 -> int32
                    partial_sum[p] += (ap_int<32>)input[mac_idx] * (ap_int<32>)weights[mac_idx][j];
                }
            }
        }
        ap_int<32> acc = bias[j];
        for (int p = 0; p < PE_COUNT; p++) {
            #pragma HLS UNROLL
            acc += partial_sum[p];
        }

        // ReLU directly on the integer accumulator (sign(acc) == sign of
        // the never-computed dequantized value, since the true scale
        // factor is always positive), then the same fixed-point
        // requantize as conv_c1_int8_fixedpoint.cpp / conv_c3_int8_fixedpoint.cpp:
        // a genuine int64 multiply by the Q31 mantissa, then a rounding
        // right-shift -- no floating point anywhere in this per-neuron path.
        ap_int<32> acc_relu = (acc < 0) ? ap_int<32>(0) : acc;

        ap_int<64> prod = (ap_int<64>)acc_relu * (ap_int<64>)requant_mult;
        ap_int<64> half = (requant_shift > 0)
                             ? (ap_int<64>(1) << (requant_shift - 1))
                             : ap_int<64>(0);
        ap_int<64> shifted = (prod + half) >> requant_shift;

        ap_int<32> rounded = shifted;
        if (rounded > 127) rounded = 127;
        if (rounded < -128) rounded = -128;
        output[j] = (ap_int<8>)rounded;
    }
}
