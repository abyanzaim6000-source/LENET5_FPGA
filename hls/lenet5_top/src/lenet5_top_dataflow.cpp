#include "lenet5_top.h"

// EXPLORED, REVERTED -- see Results/hls_results.md ("LeNet-5 Combined
// Top-Level" section). Identical to lenet5_top.cpp except for one extra
// #pragma HLS DATAFLOW, added so successive images could pipeline across
// the 7 layers. It works functionally (csim TEST PASSED, same real-MNIST
// result as the non-dataflow version) and csynth completes, but the
// resulting design EXCEEDS the xc7z020's actual capacity: LUT 140%
// (74,562 / 53,200) and BRAM_18K 158% (444 / 280) of budget -- it would
// fail Vivado place-and-route even though Vitis HLS's csynth step reports
// success. Root cause: DATAFLOW requires every buffer crossing a
// producer/consumer boundary to be double-buffered (ping-pong) so image
// N+1's producer can write while image N's consumer still reads. That's
// a reasonable cost for the small per-image feature-map buffers, but the
// *weight* arrays (400x120, 120x84, 5x5x6x16, ...) never change between
// images and pay the exact same doubling tax purely because their
// burst-copy-in loops live inside the same DATAFLOW region as everything
// else. A second, separate issue: conv_c3_partialsum's own outer loop
// stops auto-flattening once it's a callee inside this DATAFLOW region
// (confirmed in its own sub-report: outer o/r/c loop shows "Pipelined:
// no" here, vs. one continuous 40,000-iteration pipeline when C3 is its
// own top-level function) -- its instance latency nearly doubles
// (160,078 -> 291,201 cycles) purely from that, becoming 58% of total
// latency and gating the whole design's achievable throughput.
//
// Reverted to the plain sequential version (lenet5_top.cpp, no DATAFLOW)
// as the primary working baseline. Kept here, not deleted, as a real,
// reproducible exploration -- same project convention as
// conv_c3_partialsum_roundrobin.cpp being kept alongside the version
// that was actually kept. Possible follow-ups if image-to-image pipelining
// is revisited: move the weight burst-copy-in loops outside the DATAFLOW
// region (they only need to run once, not per image), and investigate why
// C3's loop flattening doesn't survive being called as a DATAFLOW-region
// callee.
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

void conv_c3_partialsum(
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
) {
    #pragma HLS INTERFACE m_axi port=image           offset=slave bundle=gmem_img
    #pragma HLS INTERFACE m_axi port=c1_weights      offset=slave bundle=gmem_c1w
    #pragma HLS INTERFACE m_axi port=c1_bias         offset=slave bundle=gmem_c1b
    #pragma HLS INTERFACE m_axi port=c3_weights      offset=slave bundle=gmem_c3w
    #pragma HLS INTERFACE m_axi port=c3_bias         offset=slave bundle=gmem_c3b
    #pragma HLS INTERFACE m_axi port=c5_weights      offset=slave bundle=gmem_c5w
    #pragma HLS INTERFACE m_axi port=c5_bias         offset=slave bundle=gmem_c5b
    #pragma HLS INTERFACE m_axi port=f6_weights      offset=slave bundle=gmem_f6w
    #pragma HLS INTERFACE m_axi port=f6_bias         offset=slave bundle=gmem_f6b
    #pragma HLS INTERFACE m_axi port=output_weights  offset=slave bundle=gmem_outw
    #pragma HLS INTERFACE m_axi port=output_bias     offset=slave bundle=gmem_outb
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

    #pragma HLS DATAFLOW

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

    static float c1_out[28][28][6];
    static float s2_out[14][14][6];
    static float c3_out[10][10][16];
    static float s4_out[5][5][16];
    static float flat[400];
    static float c5_out[120];
    static float f6_out[84];

    conv_c1_systolic(local_image, local_c1_weights, local_c1_bias, c1_out);
    pool_s2(c1_out, s2_out);
    conv_c3_partialsum(s2_out, local_c3_weights, local_c3_bias, c3_out);
    pool_s4(c3_out, s4_out);

    flatten:
    for (int r = 0; r < 5; r++)
        for (int c = 0; c < 5; c++)
            for (int ch = 0; ch < 16; ch++)
                flat[r * 5 * 16 + c * 16 + ch] = s4_out[r][c][ch];

    dense_c5_partialsum(flat, local_c5_weights, local_c5_bias, c5_out);
    dense_f6(c5_out, local_f6_weights, local_f6_bias, f6_out);
    dense_output(f6_out, local_output_weights, local_output_bias, local_result);

    copy_result:
    for (int j = 0; j < 10; j++)
        result[j] = local_result[j];
}
