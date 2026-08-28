"""
Integer-arithmetic layer implementations.
Unlike manual_layers.py (float32 math), these functions perform the
actual multiply-accumulate using INTEGER values only, matching what
real fixed-point/quantized hardware computes.
"""

import numpy as np


def quantize_activation(x, n_bits):
    """
    Quantizes an activation tensor to n_bits (int8 container), returns
    (int_array, scale). Same idea as quantize_int_real in quantization.py,
    but activations are re-quantized fresh at every layer (their range
    changes each time), unlike weights which are quantized once and saved.
    """
    qmax = 2 ** (n_bits - 1) - 1
    qmin = -(2 ** (n_bits - 1))

    max_abs = np.max(np.abs(x))
    scale = max_abs / qmax if max_abs != 0 else 1.0

    scaled = np.round(x / scale)
    scaled = np.clip(scaled, qmin, qmax)

    return scaled.astype(np.int8), scale


def conv2d_int(x_int, w_int, bias_float, x_scale, w_scale, padding="valid"):
    """
    x_int : quantized input,  shape (H, W, C_in), dtype int8
    w_int : quantized kernel, shape (kH, kW, C_in, C_out), dtype int8
    bias_float : ORIGINAL float32 bias (not pre-quantized -- we requantize
                 it here using the combined scale, as explained above)
    x_scale, w_scale : scale factors for input and weights

    Returns: (output_float, ) -- real-valued float32 output, ready for
             activation. The MAC loop itself is pure integer arithmetic.
    """
    H, W, C_in = x_int.shape
    kH, kW, _, C_out = w_int.shape

    if padding == "same":
        pad_h = kH // 2
        pad_w = kW // 2
        x_int = np.pad(x_int, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)))
        H, W, _ = x_int.shape

    out_H = H - kH + 1
    out_W = W - kW + 1

    # Accumulator in int32 -- wider than the int8 inputs, to hold the
    # growing sum without overflowing (this mirrors real hardware, where
    # the MAC accumulator register is always wider than the input width).
    acc = np.zeros((out_H, out_W, C_out), dtype=np.int32)

    combined_scale = x_scale * w_scale
    bias_int = np.round(bias_float / combined_scale).astype(np.int32)

    for o in range(C_out):
        for r in range(out_H):
            for c in range(out_W):
                total = np.int32(0)
                for i in range(C_in):
                    for kr in range(kH):
                        for kc in range(kW):
                            # PURE INTEGER MULTIPLY-ACCUMULATE -- no floats here
                            total += np.int32(x_int[r + kr, c + kc, i]) * np.int32(w_int[kr, kc, i, o])
                acc[r, c, o] = total + bias_int[o]

    # Single dequantization step, applied once at the very end
    output_float = acc.astype(np.float32) * combined_scale
    return output_float

def dense_int(x_int, w_int, bias_float, x_scale, w_scale, activation=None):
    """
    x_int : quantized input,  shape (N_in,), dtype int8
    w_int : quantized weight, shape (N_in, N_out), dtype int8
    bias_float : original float32 bias, requantized here using combined scale
    x_scale, w_scale : scale factors

    Returns: real-valued float32 output (after optional activation).
    """
    N_in = x_int.shape[0]
    N_out = w_int.shape[1]

    combined_scale = x_scale * w_scale
    bias_int = np.round(bias_float / combined_scale).astype(np.int32)

    y = np.zeros((N_out,), dtype=np.int32)

    for j in range(N_out):
        total = np.int32(0)
        for i in range(N_in):
            # PURE INTEGER MULTIPLY-ACCUMULATE
            total += np.int32(x_int[i]) * np.int32(w_int[i, j])
        y[j] = total + bias_int[j]

    output_float = y.astype(np.float32) * combined_scale

    if activation == "tanh":
        output_float = np.tanh(output_float)
    elif activation == "relu":
        output_float = np.maximum(0, output_float)
    elif activation == "softmax":
        shifted = output_float - np.max(output_float)
        exp_y = np.exp(shifted)
        output_float = exp_y / np.sum(exp_y)

    return output_float