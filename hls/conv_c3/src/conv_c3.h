#ifndef CONV_C3_H
#define CONV_C3_H

// Fixed dimensions for C3: 14x14x6 input (S2's pooled output), 5x5 kernel,
// 16 output channels, "valid" padding (matches lenet5_relu.py exactly).
#define IN_H 14
#define IN_W 14
#define IN_C 6
#define OUT_C 16
#define K 5
#define OUT_H (IN_H - K + 1)
#define OUT_W (IN_W - K + 1)

void conv_c3(
    float input[IN_H][IN_W][IN_C],
    float weights[K][K][IN_C][OUT_C],
    float bias[OUT_C],
    float output[OUT_H][OUT_W][OUT_C]
);

void conv_c3_partialsum(
    float input[IN_H][IN_W][IN_C],
    float weights[K][K][IN_C][OUT_C],
    float bias[OUT_C],
    float output[OUT_H][OUT_W][OUT_C]
);

void conv_c3_partialsum_roundrobin(
    float input[IN_H][IN_W][IN_C],
    float weights[K][K][IN_C][OUT_C],
    float bias[OUT_C],
    float output[OUT_H][OUT_W][OUT_C]
);

#endif
