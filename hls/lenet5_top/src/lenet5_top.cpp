#include "lenet5_top.h"

// Forward declarations of the 7 existing, individually-proven layer IPs
// (see Results/hls_results.md for each one's own optimization log). Their
// own headers (conv_c1.h, pool_s2.h, conv_c3.h, pool_s4.h, dense_c5.h,
// dense_f6.h, dense_output.h) all reuse the SAME macro names (IN_H, N_IN,
// etc.) for different values, so they can't all be #included into this one
// translation unit -- these declarations spell out the literal dimensions
// instead (identical decayed array types to what each macro expands to in
// its own header, so they link against the real definitions unchanged).
void conv_c1_systolic(
    float input[28][28][1],
    float weights[5][5][1][6],
    float bias[6],
    float output[28][28][6]
);

void pool_s2(
    float input[28][28][6],
    float output[14][14][6]
);

void conv_c3_partialsum_lut(
    float input[14][14][6],
    float weights[5][5][6][16],
    float bias[16],
    float output[10][10][16]
);

void pool_s4(
    float input[10][10][16],
    float output[5][5][16]
);

void dense_c5_partialsum(
    float input[400],
    float weights[400][120],
    float bias[120],
    float output[120]
);

void dense_f6(
    float input[120],
    float weights[120][84],
    float bias[84],
    float output[84]
);

void dense_output(
    float input[84],
    float weights[84][10],
    float bias[10],
    float output[10]
);

// Combined top-level LeNet-5 forward pass -- PRIMARY WORKING BASELINE.
// Runs the 7 layers strictly sequentially (no DATAFLOW): image N fully
// completes before image N+1 starts. See Results/hls_results.md for why:
// the DATAFLOW variant (lenet5_top_dataflow.cpp, kept for reference, not
// active) pipelines successive images across layers but exceeds the
// xc7z020's LUT/BRAM budget, mainly because DATAFLOW forces every
// producer/consumer buffer -- including the weight arrays, which never
// change between images -- into double-buffered (ping-pong) memory. This
// version fits the chip; see the results log for the actual utilization.
//
// AXI interface: one m_axi bundle per LAYER (each layer's weights+bias
// share a bundle; image and result get their own), all control through
// s_axilite -- see Results/hls_results.md for why this replaced the
// earlier one-bundle-per-array scheme. Same local-buffer burst-copy
// pattern -- every AXI array gets copied into a plain on-chip buffer
// before use; only the LOCAL copies (and the purely-internal inter-layer
// buffers, which were never AXI arguments) are ever candidates for
// partitioning, and that partitioning already lives inside each layer's
// own proven source file, not here.
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
) {
    // One bundle PER LAYER (weights+bias share a bundle), not per array --
    // 7 bundles total instead of 12. Safe without the multi-process-per-
    // bundle conflict hit during the DATAFLOW exploration (see
    // Results/hls_results.md): that error only arises when DATAFLOW splits
    // each burst-copy loop into its own concurrent hardware process, and
    // this sequential version has no DATAFLOW, hence no concurrent
    // processes competing for one AXI port. `image` and `result` are the
    // network's own I/O rather than any single layer's weights, so each
    // keeps its own bundle.
    #pragma HLS INTERFACE m_axi port=image           offset=slave bundle=gmem_img
    #pragma HLS INTERFACE m_axi port=c1_weights      offset=slave bundle=gmem_c1
    #pragma HLS INTERFACE m_axi port=c1_bias         offset=slave bundle=gmem_c1
    #pragma HLS INTERFACE m_axi port=c3_weights      offset=slave bundle=gmem_c3
    #pragma HLS INTERFACE m_axi port=c3_bias         offset=slave bundle=gmem_c3
    #pragma HLS INTERFACE m_axi port=c5_weights      offset=slave bundle=gmem_c5
    #pragma HLS INTERFACE m_axi port=c5_bias         offset=slave bundle=gmem_c5
    #pragma HLS INTERFACE m_axi port=f6_weights      offset=slave bundle=gmem_f6
    #pragma HLS INTERFACE m_axi port=f6_bias         offset=slave bundle=gmem_f6
    #pragma HLS INTERFACE m_axi port=output_weights  offset=slave bundle=gmem_out
    #pragma HLS INTERFACE m_axi port=output_bias     offset=slave bundle=gmem_out
    #pragma HLS INTERFACE m_axi port=result          offset=slave bundle=gmem_res
    #pragma HLS INTERFACE s_axilite port=image           bundle=control
    #pragma HLS INTERFACE s_axilite port=c1_weights      bundle=control
    #pragma HLS INTERFACE s_axilite port=c1_bias         bundle=control
    #pragma HLS INTERFACE s_axilite port=c3_weights      bundle=control
    #pragma HLS INTERFACE s_axilite port=c3_bias         bundle=control
    #pragma HLS INTERFACE s_axilite port=c5_weights      bundle=control
    #pragma HLS INTERFACE s_axilite port=c5_bias         bundle=control
    #pragma HLS INTERFACE s_axilite port=f6_weights      bundle=control
    #pragma HLS INTERFACE s_axilite port=f6_bias         bundle=control
    #pragma HLS INTERFACE s_axilite port=output_weights  bundle=control
    #pragma HLS INTERFACE s_axilite port=output_bias     bundle=control
    #pragma HLS INTERFACE s_axilite port=result          bundle=control
    #pragma HLS INTERFACE s_axilite port=return          bundle=control

    // ---- Local on-chip copies of every AXI argument. THESE (never the
    // AXI arguments themselves) are what could be partitioned -- none of
    // them need to be here, since each layer already partitions what it
    // needs internally on ITS OWN local copies. ----
    static float local_image[28][28][1];
    static float local_c1_weights[5][5][1][6];
    static float local_c1_bias[6];
    static float local_c3_weights[5][5][6][16];
    static float local_c3_bias[16];
    static float local_c5_weights[400][120];
    static float local_c5_bias[120];
    static float local_f6_weights[120][84];
    static float local_f6_bias[84];
    static float local_output_weights[84][10];
    static float local_output_bias[10];
    static float local_result[10];

    // Burst-copy every AXI array in to its local buffer.
    copy_image:
    for (int r = 0; r < 28; r++)
        for (int c = 0; c < 28; c++)
            local_image[r][c][0] = image[r][c][0];

    copy_c1_weights:
    for (int kr = 0; kr < 5; kr++)
        for (int kc = 0; kc < 5; kc++)
            for (int i = 0; i < 1; i++)
                for (int o = 0; o < 6; o++)
                    local_c1_weights[kr][kc][i][o] = c1_weights[kr][kc][i][o];
    copy_c1_bias:
    for (int o = 0; o < 6; o++)
        local_c1_bias[o] = c1_bias[o];

    copy_c3_weights:
    for (int kr = 0; kr < 5; kr++)
        for (int kc = 0; kc < 5; kc++)
            for (int i = 0; i < 6; i++)
                for (int o = 0; o < 16; o++)
                    local_c3_weights[kr][kc][i][o] = c3_weights[kr][kc][i][o];
    copy_c3_bias:
    for (int o = 0; o < 16; o++)
        local_c3_bias[o] = c3_bias[o];

    copy_c5_weights:
    for (int i = 0; i < 400; i++)
        for (int j = 0; j < 120; j++)
            local_c5_weights[i][j] = c5_weights[i][j];
    copy_c5_bias:
    for (int j = 0; j < 120; j++)
        local_c5_bias[j] = c5_bias[j];

    copy_f6_weights:
    for (int i = 0; i < 120; i++)
        for (int j = 0; j < 84; j++)
            local_f6_weights[i][j] = f6_weights[i][j];
    copy_f6_bias:
    for (int j = 0; j < 84; j++)
        local_f6_bias[j] = f6_bias[j];

    copy_output_weights:
    for (int i = 0; i < 84; i++)
        for (int j = 0; j < 10; j++)
            local_output_weights[i][j] = output_weights[i][j];
    copy_output_bias:
    for (int j = 0; j < 10; j++)
        local_output_bias[j] = output_bias[j];

    // ---- Purely internal inter-layer buffers. Never AXI arguments, so
    // free to partition if a future stage needs it -- none currently do,
    // since each layer function partitions its own inputs where required
    // (e.g. pool_s2/pool_s4 cyclic-partition their `input` parameter
    // internally). Sized to each layer's ACTUAL output shape per its .h. ----
    static float c1_out[28][28][6];
    static float s2_out[14][14][6];
    static float c3_out[10][10][16];
    static float s4_out[5][5][16];
    static float flat[400];
    static float c5_out[120];
    static float f6_out[84];

    conv_c1_systolic(local_image, local_c1_weights, local_c1_bias, c1_out);
    pool_s2(c1_out, s2_out);
    conv_c3_partialsum_lut(s2_out, local_c3_weights, local_c3_bias, c3_out);
    pool_s4(c3_out, s4_out);

    // Flatten S4's (5,5,16) output to a 400-vector. Matches Keras
    // Flatten()'s row-major (H,W,C) order exactly -- same order NumPy's
    // default x.flatten() uses on an (H,W,C) array in manual_layers.py,
    // which is what dense_c5's weights were trained against.
    flatten:
    for (int r = 0; r < 5; r++)
        for (int c = 0; c < 5; c++)
            for (int ch = 0; ch < 16; ch++)
                flat[r * 5 * 16 + c * 16 + ch] = s4_out[r][c][ch];

    dense_c5_partialsum(flat, local_c5_weights, local_c5_bias, c5_out);
    dense_f6(c5_out, local_f6_weights, local_f6_bias, f6_out);
    dense_output(f6_out, local_output_weights, local_output_bias, local_result);

    // Burst-copy the final result back out to DDR.
    copy_result:
    for (int j = 0; j < 10; j++)
        result[j] = local_result[j];
}
