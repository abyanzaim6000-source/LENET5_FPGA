#ifndef DENSE_F6_H
#define DENSE_F6_H

// Fixed dimensions for F6: 120 inputs (C5's output), 84 outputs,
// ReLU activation (matches lenet5_relu.py's F6 layer exactly).
#define N_IN 120
#define N_OUT 84

// PE_COUNT=4 partial-sum accumulator split -- same proven MAC-stage
// pattern as dense_c5_partialsum.cpp (accumulator-limited by the float
// adder's latency, confirmed lowest resource cost among the II=4
// options), applied directly instead of starting from a serial-
// accumulator baseline.
#define PE_COUNT 4
#define MACS_PER_PE ((N_IN + PE_COUNT - 1) / PE_COUNT)

void dense_f6(
    float input[N_IN],
    float weights[N_IN][N_OUT],
    float bias[N_OUT],
    float output[N_OUT]
);

#endif
