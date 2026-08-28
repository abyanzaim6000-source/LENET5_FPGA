"""
Fixed-point quantization utilities.
Each format is defined explicitly: total bits, integer bits, fractional
bits, scale factor, and range — nothing is assumed.
"""

import numpy as np


def quantize_fixed(x, int_bits, frac_bits, signed=True):
    """
    Converts a float32 array to a fixed-point representation, simulated
    IN FLOAT (we store the *effect* of quantization, not actual packed
    integers yet — that comes later for HLS).

    Format: Q(int_bits).(frac_bits), signed two's complement style.
    Total bits = int_bits + frac_bits (+ 1 implicit sign bit if signed).

    Returns: (quantized_float_array, scale, min_val, max_val)
    """
    scale = 2 ** frac_bits

    if signed:
        min_val = -(2 ** int_bits)
        max_val = (2 ** int_bits) - (1 / scale)
    else:
        min_val = 0
        max_val = (2 ** (int_bits + 1)) - (1 / scale)

    # Step 1: scale up to integer domain, round to nearest integer
    scaled = np.round(x * scale)

    # Step 2: clip (saturate) to representable range in the integer domain
    int_min = min_val * scale
    int_max = max_val * scale
    scaled = np.clip(scaled, int_min, int_max)

    # Step 3: scale back down to recover the "dequantized" float value
    # This is what the hardware would actually compute with, represented
    # back in float for us to measure error against the original.
    quantized = scaled / scale

    return quantized.astype(np.float32), scale, min_val, max_val


def quantization_error(original, quantized):
    """Returns (max_abs_error, mean_abs_error, rmse) between two arrays."""
    diff = np.abs(original - quantized)
    max_err = diff.max()
    mean_err = diff.mean()
    rmse = np.sqrt(np.mean(diff ** 2))
    return max_err, mean_err, rmse


# ---- Named formats, explicitly defined ----
# (paper's format — OUR extension formats are separate, added in the next step)
FORMATS = {
    "fixed16_paper":  {"int_bits": 12, "frac_bits": 3},  # paper's <13,3>, sign bit implicit
}

def quantize_layer_weights(weights_dict, int_bits, frac_bits):
    """
    Applies quantize_fixed() to every weight/bias array in a dict like
    {"C1": [w, b], "C3": [w, b], ...}. Returns a new dict, same structure.
    """
    quantized = {}
    for name, (w, b) in weights_dict.items():
        qw, _, _, _ = quantize_fixed(w, int_bits, frac_bits)
        qb, _, _, _ = quantize_fixed(b, int_bits, frac_bits)
        quantized[name] = [qw, qb]
    return quantized

def quantize_layer_weights_int(weights_dict, n_bits):
    """Like quantize_layer_weights, but using true INTn (data-driven scale)."""
    quantized = {}
    for name, (w, b) in weights_dict.items():
        qw, _, _, _ = quantize_int_dynamic(w, n_bits)
        qb, _, _, _ = quantize_int_dynamic(b, n_bits)
        quantized[name] = [qw, qb]
    return quantized

def quantize_int_dynamic(x, n_bits):
    """
    True INTn quantization: finds the scale from the actual data range,
    rather than a fixed number of fractional bits. This is the standard
    approach used by INT8 quantization in mainstream ML tooling.

    n_bits : total bits, e.g. 8 for INT8, 4 for INT4 (signed, symmetric)
    """
    qmax = 2 ** (n_bits - 1) - 1   # e.g. 127 for INT8, 7 for INT4
    qmin = -(2 ** (n_bits - 1))     # e.g. -128 for INT8, -8 for INT4

    max_abs = np.max(np.abs(x))
    if max_abs == 0:
        scale = 1.0
    else:
        scale = max_abs / qmax   # so that max_abs maps exactly to qmax

    scaled = np.round(x / scale)
    scaled = np.clip(scaled, qmin, qmax)
    quantized = scaled * scale

    return quantized.astype(np.float32), scale, qmin, qmax

def quantize_int_real(x, n_bits):
    """
    TRUE integer quantization: returns actual integer values (as numpy
    int8/int32 arrays, not dequantized floats), plus the scale needed
    to reconstruct approximate float values later.

    This is what actually gets exported to hardware -- real integers,
    not floats pretending to be integers.
    """
    qmax = 2 ** (n_bits - 1) - 1
    qmin = -(2 ** (n_bits - 1))

    max_abs = np.max(np.abs(x))
    scale = max_abs / qmax if max_abs != 0 else 1.0

    scaled = np.round(x / scale)
    scaled = np.clip(scaled, qmin, qmax)

    # Store as the smallest integer type that fits (int8 for INT8,
    # int8 also used for INT4 since numpy has no native 4-bit type --
    # the VALUES are still constrained to the 4-bit range [-8,7])
    if n_bits <= 8:
        int_array = scaled.astype(np.int8)
    else:
        int_array = scaled.astype(np.int32)

    return int_array, scale