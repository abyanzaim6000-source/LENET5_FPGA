#ifndef DENSE_OUTPUT_H
#define DENSE_OUTPUT_H

// Fixed dimensions for Output: 84 inputs (F6's output), 10 outputs,
// softmax activation (matches lenet5_relu.py's Output layer exactly).
#define N_IN 84
#define N_OUT 10

// PE_COUNT=4 partial-sum accumulator split for the MAC stage -- same
// proven pattern as dense_c5_partialsum.cpp/dense_f6.cpp. Softmax
// itself is NOT part of this split: it needs every neuron's raw MAC
// result before it can normalize any one of them, so it runs as a
// separate second pass (see dense_output.cpp).
#define PE_COUNT 4
#define MACS_PER_PE ((N_IN + PE_COUNT - 1) / PE_COUNT)

void dense_output(
    float input[N_IN],
    float weights[N_IN][N_OUT],
    float bias[N_OUT],
    float output[N_OUT]
);

#endif
