#ifndef POOL_S2_H
#define POOL_S2_H

// Dimensions for S2: input is C1's output (28x28x6), 2x2 average pooling, stride 2
#define IN_H 28
#define IN_W 28
#define CHANNELS 6
#define POOL_SIZE 2
#define STRIDE 2
#define OUT_H (IN_H / STRIDE)
#define OUT_W (IN_W / STRIDE)

void pool_s2(
    float input[IN_H][IN_W][CHANNELS],
    float output[OUT_H][OUT_W][CHANNELS]
);

#endif