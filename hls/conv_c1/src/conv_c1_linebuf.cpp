#include "conv_c1.h"

// Line-buffer + window-register based convolution.
// Instead of random-accessing the full input array, this streams the
// image ROW BY ROW, keeping only K=5 rows in small buffers at any time.
void conv_c1_linebuf(
    float input[IN_H][IN_W][IN_C],
    float weights[K][K][IN_C][OUT_C],
    float bias[OUT_C],
    float output[IN_H][IN_W][OUT_C]
) {
    #pragma HLS ARRAY_PARTITION variable=weights complete dim=0

    // Line buffer: holds the last K rows of the (padded) input.
    // Fully partitioned so all K*K window values are available every cycle.
    static float line_buf[K][IN_W + 2*PAD][IN_C];
    #pragma HLS ARRAY_PARTITION variable=line_buf complete dim=1
    #pragma HLS ARRAY_PARTITION variable=line_buf complete dim=3

    // Window register: the current K x K patch the kernel is looking at.
    float window[K][K][IN_C];
    #pragma HLS ARRAY_PARTITION variable=window complete dim=0

    // Initialize line buffer to zero (handles top padding rows)
    for (int r = 0; r < K; r++)
        for (int c = 0; c < IN_W + 2*PAD; c++)
            for (int i = 0; i < IN_C; i++)
                line_buf[r][c][i] = 0;

    // Row pointer into the padded input space
    for (int out_r = 0; out_r < IN_H; out_r++) {

        // Shift in a new row: drop the oldest buffered row, load the next
        // real input row (or zero, if beyond image bounds -- bottom padding)
        int src_row = out_r + K - 1 - PAD;  // which real input row is newest
        for (int shift = 0; shift < K - 1; shift++) {
            #pragma HLS UNROLL
            for (int c = 0; c < IN_W + 2*PAD; c++) {
                for (int i = 0; i < IN_C; i++) {
                    line_buf[shift][c][i] = line_buf[shift + 1][c][i];
                }
            }
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
            #pragma HLS PIPELINE II=1

            // Load the current K x K window from the line buffer
            for (int kr = 0; kr < K; kr++)
                for (int kc = 0; kc < K; kc++)
                    for (int i = 0; i < IN_C; i++)
                        window[kr][kc][i] = line_buf[kr][out_c + kc][i];

            // MAC: same math as before, but now reading from the
            // fully-parallel window register instead of the full image
            for (int o = 0; o < OUT_C; o++) {
                float acc = bias[o];
                for (int i = 0; i < IN_C; i++) {
                    for (int kr = 0; kr < K; kr++) {
                        for (int kc = 0; kc < K; kc++) {
                            acc += window[kr][kc][i] * weights[kr][kc][i][o];
                        }
                    }
                }
                output[out_r][out_c][o] = (acc > 0) ? acc : 0;
            }
        }
    }
}