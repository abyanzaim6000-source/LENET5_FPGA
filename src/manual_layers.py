"""
Manual (from-scratch) layer implementations for LeNet-5 inference.
No TensorFlow/Keras layer calls are used here — only NumPy.
"""

import numpy as np


def conv2d(x, w, b, padding="valid"):
    """
    x : input feature map,  shape (H, W, C_in)
    w : kernel weights,     shape (kH, kW, C_in, C_out)
    b : bias,                shape (C_out,)
    padding : "valid" or "same"

    Implements:  y[o][r][c] = bias[o] + sum_i sum_k sum_l  x[i][r+k][c+l] * w[o][i][k][l]
    (indices rearranged here to match NumPy's (H, W, C) convention instead of (C, H, W))
    """
    H, W, C_in = x.shape
    kH, kW, _, C_out = w.shape

    if padding == "same":
        pad_h = kH // 2
        pad_w = kW // 2
        x = np.pad(x, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)))
        H, W, _ = x.shape

    out_H = H - kH + 1
    out_W = W - kW + 1
    y = np.zeros((out_H, out_W, C_out), dtype=np.float32)

    for o in range(C_out):                     # for each output channel
        for r in range(out_H):                  # for each output row
            for c in range(out_W):               # for each output column
                acc = b[o]                        # start accumulator with bias
                for i in range(C_in):              # for each input channel
                    for kr in range(kH):            # for each kernel row
                        for kc in range(kW):          # for each kernel column
                            acc += x[r + kr, c + kc, i] * w[kr, kc, i, o]
                y[r, c, o] = acc

    return y
def avgpool2d(x, pool_size=2, stride=2):
    """
    x : input feature map, shape (H, W, C)
    Implements: y[r][c][ch] = mean of the pool_size x pool_size window
                              in x, at stride 'stride', for each channel.
    """
    H, W, C = x.shape
    out_H = (H - pool_size) // stride + 1
    out_W = (W - pool_size) // stride + 1
    y = np.zeros((out_H, out_W, C), dtype=np.float32)

    for ch in range(C):                         # for each channel
        for r in range(out_H):                    # for each output row
            for c in range(out_W):                   # for each output column
                r0 = r * stride
                c0 = c * stride
                window = x[r0:r0 + pool_size, c0:c0 + pool_size, ch]
                y[r, c, ch] = np.mean(window)

    return y
def dense(x, w, b, activation=None):
    """
    x : input vector,  shape (N_in,)
    w : weights,       shape (N_in, N_out)
    b : bias,          shape (N_out,)
    activation : None, "tanh", or "softmax"

    Implements: y[j] = activation( bias[j] + sum_i x[i] * w[i][j] )
    This is a matrix-vector multiply, one MAC loop per output neuron.
    """
    N_in = x.shape[0]
    N_out = w.shape[1]
    y = np.zeros((N_out,), dtype=np.float32)

    for j in range(N_out):                 # for each output neuron
        acc = b[j]                           # start with bias
        for i in range(N_in):                 # for each input neuron
            acc += x[i] * w[i, j]
        y[j] = acc

    if activation == "tanh":
        y = np.tanh(y)
    elif activation == "softmax":
        # subtract max for numerical stability, standard trick
        shifted = y - np.max(y)
        exp_y = np.exp(shifted)
        y = exp_y / np.sum(exp_y)

    return y

def relu(x):
    """
    Implements: y = max(0, x), element-wise.
    Unlike tanh, ReLU has no smooth curve to compute — just a comparison.
    This is actually CHEAPER in hardware than tanh (no lookup table needed).
    """
    return np.maximum(0, x)


def maxpool2d(x, pool_size=2, stride=2):
    """
    x : input feature map, shape (H, W, C)
    Implements: y[r][c][ch] = max value in the pool_size x pool_size
                              window in x, at stride 'stride', per channel.
    Same loop structure as avgpool2d -- only the reduction (mean -> max) differs.
    """
    H, W, C = x.shape
    out_H = (H - pool_size) // stride + 1
    out_W = (W - pool_size) // stride + 1
    y = np.zeros((out_H, out_W, C), dtype=np.float32)

    for ch in range(C):
        for r in range(out_H):
            for c in range(out_W):
                r0 = r * stride
                c0 = c * stride
                window = x[r0:r0 + pool_size, c0:c0 + pool_size, ch]
                y[r, c, ch] = np.max(window)

    return y