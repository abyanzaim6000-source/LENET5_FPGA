#include "dense_c5.h"

// This loop structure is DELIBERATELY identical to manual_layers.py's
// dense() -- one MAC loop per output neuron, bias first, then activation
// applied once the full accumulation is done.
void dense_c5(
    float input[N_IN],
    float weights[N_IN][N_OUT],
    float bias[N_OUT],
    float output[N_OUT]
) {
    for (int j = 0; j < N_OUT; j++) {
        float acc = bias[j];
        for (int i = 0; i < N_IN; i++) {
            acc += input[i] * weights[i][j];
        }
        // ReLU activation, applied inline (matches lenet5_relu.py)
        output[j] = (acc > 0) ? acc : 0;
    }
}
