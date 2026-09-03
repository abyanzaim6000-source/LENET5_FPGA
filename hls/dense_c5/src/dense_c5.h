#ifndef DENSE_C5_H
#define DENSE_C5_H

// Fixed dimensions for C5: 400 inputs (S4's flattened 5x5x16 output),
// 120 outputs, tanh activation (matches lenet5.py's C5 layer exactly).
#define N_IN 400
#define N_OUT 120

void dense_c5(
    float input[N_IN],
    float weights[N_IN][N_OUT],
    float bias[N_OUT],
    float output[N_OUT]
);

#endif
