#ifndef LENET5_TOP_INT8_H
#define LENET5_TOP_INT8_H

#include <ap_int.h>

// INT8 combined top-level: mirrors lenet5_top.cpp's own combined design
// exactly (same DATAFLOW-free sequential chain of 7 already-proven layer
// IPs, one m_axi bundle per layer, local burst-copy buffers -- see that
// file's own header comment for the full rationale, unchanged here) but
// chains the INT8 kernels instead:
//   C1(conv_c1_int8_fixedpoint, 28x28x1->28x28x6)
//   -> S2(pool_s2_int8, ->14x14x6)
//   -> C3(conv_c3_int8_fixedpoint, LUT-decode architecture, ->10x10x16)
//   -> S4(pool_s4_int8, ->5x5x16)
//   -> flatten(400)
//   -> C5(dense_c5_int8_fixedpoint, ->120)
//   -> F6(dense_f6_int8_fixedpoint, ->84)
//   -> Output(dense_output_int8, ->10, int8 MAC + float softmax).
//
// Every requant_mult/requant_shift pair is THAT LAYER'S OWN, independently
// derived Q31 fixed-point (TFLite/gemmlowp quantize_multiplier) rescale --
// never reused across layers, since each layer's (x_scale, w_scale,
// out_scale) triple differs and is derived fresh from the real activation
// range flowing through it. output_x_scale/output_w_scale are the plain
// float scales dense_output_int8 needs for its own float softmax stage
// (F6's own out_scale and the Output layer's own weight scale,
// respectively -- NOT reused from any earlier layer's scale). See
// tb/generate_test_data_int8.py for exactly how all of these are derived,
// chained end-to-end from a real MNIST image and real trained weights.
void lenet5_top_int8(
    ap_int<8>  image[28][28][1],
    ap_int<8>  c1_weights[5][5][1][6],
    ap_int<32> c1_bias[6],
    ap_int<32> c1_requant_mult,
    ap_int<8>  c1_requant_shift,
    ap_int<8>  c3_weights[5][5][6][16],
    ap_int<32> c3_bias[16],
    ap_int<32> c3_requant_mult,
    ap_int<8>  c3_requant_shift,
    ap_int<8>  c5_weights[400][120],
    ap_int<32> c5_bias[120],
    ap_int<32> c5_requant_mult,
    ap_int<8>  c5_requant_shift,
    ap_int<8>  f6_weights[120][84],
    ap_int<32> f6_bias[84],
    ap_int<32> f6_requant_mult,
    ap_int<8>  f6_requant_shift,
    ap_int<8>  output_weights[84][10],
    ap_int<32> output_bias[10],
    float      output_x_scale,
    float      output_w_scale,
    float      result[10]
);

#endif
