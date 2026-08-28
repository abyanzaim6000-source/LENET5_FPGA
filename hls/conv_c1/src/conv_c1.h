#ifndef CONV_C1_H
#define CONV_C1_H

// Fixed dimensions for C1: 28x28x1 input, 5x5 kernel, 6 output channels,
// "same" padding (matches your lenet5_relu.py architecture exactly).
#define IN_H 28
#define IN_W 28
#define IN_C 1
#define OUT_C 6
#define K 5
#define PAD 2   // (K/2), for "same" padding

void conv_c1(
    float input[IN_H][IN_W][IN_C],
    float weights[K][K][IN_C][OUT_C],
    float bias[OUT_C],
    float output[IN_H][IN_W][OUT_C]
);

#endif