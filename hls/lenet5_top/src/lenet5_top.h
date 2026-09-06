#ifndef LENET5_TOP_H
#define LENET5_TOP_H

// Combined top-level: chains all 7 existing layer IPs (each already proven
// individually, see Results/hls_results.md) into one forward pass.
// C1(28x28x1->28x28x6) -> S2(->14x14x6) -> C3(->10x10x16) -> S4(->5x5x16)
// -> flatten(400) -> C5(->120) -> F6(->84) -> Output(->10, softmax).
void lenet5_top(
    float image[28][28][1],
    float c1_weights[5][5][1][6],
    float c1_bias[6],
    float c3_weights[5][5][6][16],
    float c3_bias[16],
    float c5_weights[400][120],
    float c5_bias[120],
    float f6_weights[120][84],
    float f6_bias[84],
    float output_weights[84][10],
    float output_bias[10],
    float result[10]
);

// EXPLORED, REVERTED variant with #pragma HLS DATAFLOW for image-to-image
// pipelining -- exceeds the xc7z020's LUT/BRAM budget (see
// lenet5_top_dataflow.cpp and Results/hls_results.md). Kept for reference,
// not the active design.
void lenet5_top_dataflow(
    float image[28][28][1],
    float c1_weights[5][5][1][6],
    float c1_bias[6],
    float c3_weights[5][5][6][16],
    float c3_bias[16],
    float c5_weights[400][120],
    float c5_bias[120],
    float f6_weights[120][84],
    float f6_bias[84],
    float output_weights[84][10],
    float output_bias[10],
    float result[10]
);

#endif
