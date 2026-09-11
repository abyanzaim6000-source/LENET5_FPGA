#include "lenet5_top_int8.h"

// Forward declarations of the 7 already-proven INT8 layer IPs (see
// Results/hls_results.md for each one's own optimization log). Their own
// headers (conv_c1_int8_fixedpoint.h, pool_s2_int8.h,
// conv_c3_int8_fixedpoint.h, pool_s4_int8.h, dense_c5_int8_fixedpoint.h,
// dense_f6_int8_fixedpoint.h, dense_output_int8.h) all reuse the SAME
// macro names (IN_H, N_IN, etc.) for different values, so they can't all
// be #included into this one translation unit -- same reasoning, and
// same fix, as lenet5_top.cpp's own forward declarations for the float32
// layers: these declarations spell out the literal dimensions instead
// (identical decayed array types to what each macro expands to in its
// own header, so they link against the real definitions unchanged).
void conv_c1_int8_fixedpoint(
    ap_int<8>  input[28][28][1],
    ap_int<8>  weights[5][5][1][6],
    ap_int<32> bias[6],
    ap_int<32> requant_mult,
    ap_int<8>  requant_shift,
    ap_int<8>  output[28][28][6]
);

void pool_s2_int8(
    ap_int<8> input[28][28][6],
    ap_int<8> output[14][14][6]
);

void conv_c3_int8_fixedpoint(
    ap_int<8>  input[14][14][6],
    ap_int<8>  weights[5][5][6][16],
    ap_int<32> bias[16],
    ap_int<32> requant_mult,
    ap_int<8>  requant_shift,
    ap_int<8>  output[10][10][16]
);

void pool_s4_int8(
    ap_int<8> input[10][10][16],
    ap_int<8> output[5][5][16]
);

void dense_c5_int8_fixedpoint(
    ap_int<8>  input[400],
    ap_int<8>  weights[400][120],
    ap_int<32> bias[120],
    ap_int<32> requant_mult,
    ap_int<8>  requant_shift,
    ap_int<8>  output[120]
);

void dense_f6_int8_fixedpoint(
    ap_int<8>  input[120],
    ap_int<8>  weights[120][84],
    ap_int<32> bias[84],
    ap_int<32> requant_mult,
    ap_int<8>  requant_shift,
    ap_int<8>  output[84]
);

void dense_output_int8(
    ap_int<8>  input[84],
    ap_int<8>  weights[84][10],
    ap_int<32> bias[10],
    float      x_scale,
    float      w_scale,
    float      output[10]
);

// Combined INT8 top-level LeNet-5 forward pass. Runs the 7 layers
// strictly sequentially (no DATAFLOW), one m_axi bundle PER LAYER (weights
// + bias + that layer's own requant scalars share a bundle/control port),
// all control through s_axilite -- the exact same structure
// lenet5_top.cpp's proven float32 design uses, carried over unchanged;
// only the layers underneath are now INT8. Every individual INT8 layer
// already uses far fewer LUT/FF/DSP/BRAM than its float32 counterpart
// (see Results/hls_results.md's per-layer INT8 conversion logs), so this
// combined design starts from a much smaller total footprint than
// lenet5_top.cpp's -- confirmed against the actual C-synthesis numbers,
// not assumed, exactly like every other build in this project.
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
) {
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
    #pragma HLS INTERFACE s_axilite port=image            bundle=control
    #pragma HLS INTERFACE s_axilite port=c1_weights       bundle=control
    #pragma HLS INTERFACE s_axilite port=c1_bias          bundle=control
    #pragma HLS INTERFACE s_axilite port=c1_requant_mult  bundle=control
    #pragma HLS INTERFACE s_axilite port=c1_requant_shift bundle=control
    #pragma HLS INTERFACE s_axilite port=c3_weights       bundle=control
    #pragma HLS INTERFACE s_axilite port=c3_bias          bundle=control
    #pragma HLS INTERFACE s_axilite port=c3_requant_mult  bundle=control
    #pragma HLS INTERFACE s_axilite port=c3_requant_shift bundle=control
    #pragma HLS INTERFACE s_axilite port=c5_weights       bundle=control
    #pragma HLS INTERFACE s_axilite port=c5_bias          bundle=control
    #pragma HLS INTERFACE s_axilite port=c5_requant_mult  bundle=control
    #pragma HLS INTERFACE s_axilite port=c5_requant_shift bundle=control
    #pragma HLS INTERFACE s_axilite port=f6_weights       bundle=control
    #pragma HLS INTERFACE s_axilite port=f6_bias          bundle=control
    #pragma HLS INTERFACE s_axilite port=f6_requant_mult  bundle=control
    #pragma HLS INTERFACE s_axilite port=f6_requant_shift bundle=control
    #pragma HLS INTERFACE s_axilite port=output_weights   bundle=control
    #pragma HLS INTERFACE s_axilite port=output_bias      bundle=control
    #pragma HLS INTERFACE s_axilite port=output_x_scale   bundle=control
    #pragma HLS INTERFACE s_axilite port=output_w_scale   bundle=control
    #pragma HLS INTERFACE s_axilite port=result           bundle=control
    #pragma HLS INTERFACE s_axilite port=return           bundle=control

    // ---- Local on-chip copies of every AXI argument. THESE (never the
    // AXI arguments themselves) are what could be partitioned -- none of
    // them need to be here, since each layer already partitions what it
    // needs internally on ITS OWN local copies. ----
    static ap_int<8>  local_image[28][28][1];
    static ap_int<8>  local_c1_weights[5][5][1][6];
    static ap_int<32> local_c1_bias[6];
    static ap_int<8>  local_c3_weights[5][5][6][16];
    static ap_int<32> local_c3_bias[16];
    static ap_int<8>  local_c5_weights[400][120];
    static ap_int<32> local_c5_bias[120];
    static ap_int<8>  local_f6_weights[120][84];
    static ap_int<32> local_f6_bias[84];
    static ap_int<8>  local_output_weights[84][10];
    static ap_int<32> local_output_bias[10];
    static float      local_result[10];

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
    // since each layer partitions its own inputs where required (e.g.
    // pool_s2_int8/pool_s4_int8 cyclic-partition their `input` parameter
    // internally). Sized to each layer's ACTUAL output shape per its .h. ----
    static ap_int<8> c1_out[28][28][6];
    static ap_int<8> s2_out[14][14][6];
    static ap_int<8> c3_out[10][10][16];
    static ap_int<8> s4_out[5][5][16];
    static ap_int<8> flat[400];
    static ap_int<8> c5_out[120];
    static ap_int<8> f6_out[84];

    conv_c1_int8_fixedpoint(local_image, local_c1_weights, local_c1_bias,
                             c1_requant_mult, c1_requant_shift, c1_out);
    pool_s2_int8(c1_out, s2_out);
    conv_c3_int8_fixedpoint(s2_out, local_c3_weights, local_c3_bias,
                             c3_requant_mult, c3_requant_shift, c3_out);
    pool_s4_int8(c3_out, s4_out);

    // Flatten S4's (5,5,16) output to a 400-vector. Matches Keras
    // Flatten()'s row-major (H,W,C) order exactly -- same order NumPy's
    // default x.flatten() uses on an (H,W,C) array, which is what
    // dense_c5's weights were trained against (identical to
    // lenet5_top.cpp's own flatten).
    flatten:
    for (int r = 0; r < 5; r++)
        for (int c = 0; c < 5; c++)
            for (int ch = 0; ch < 16; ch++)
                flat[r * 5 * 16 + c * 16 + ch] = s4_out[r][c][ch];

    dense_c5_int8_fixedpoint(flat, local_c5_weights, local_c5_bias,
                              c5_requant_mult, c5_requant_shift, c5_out);
    dense_f6_int8_fixedpoint(c5_out, local_f6_weights, local_f6_bias,
                              f6_requant_mult, f6_requant_shift, f6_out);
    dense_output_int8(f6_out, local_output_weights, local_output_bias,
                       output_x_scale, output_w_scale, local_result);

    // Burst-copy the final result back out to DDR.
    copy_result:
    for (int j = 0; j < 10; j++)
        result[j] = local_result[j];
}
