#ifndef POOL_S4_H
#define POOL_S4_H

// Dimensions for S4: input is C3's output (10x10x16), 2x2 max pooling, stride 2
#define IN_H 10
#define IN_W 10
#define CHANNELS 16
#define POOL_SIZE 2
#define STRIDE 2
#define OUT_H (IN_H / STRIDE)
#define OUT_W (IN_W / STRIDE)

void pool_s4(
    float input[IN_H][IN_W][CHANNELS],
    float output[OUT_H][OUT_W][CHANNELS]
);

#endif
