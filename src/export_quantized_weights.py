"""
Exports REAL integer-quantized weights (INT8 and INT4) for all trained
models -- actual integer arrays + scale factors, not simulated floats.
Saves both .npy (for our own reuse) and .dat (for the eventual HLS side).
Run from project root: python3 src/export_quantized_weights.py
"""

import os
import numpy as np
import tensorflow as tf

from quantization import quantize_int_real

MODELS = {
    "lenet5":        "models/lenet5.keras",
    "lenet5_relu":    "models/lenet5_relu.keras",
    "cnn3x3":         "models/cnn3x3.keras",
    "cnn3x3_relu":    "models/cnn3x3_relu.keras",
}

for model_name, model_path in MODELS.items():
    print(f"\nExporting: {model_name}")
    model = tf.keras.models.load_model(model_path)

    for bit_width in [8, 4]:
        out_dir = f"weights/int{bit_width}/{model_name}"
        os.makedirs(out_dir, exist_ok=True)

        for layer in model.layers:
            w = layer.get_weights()
            if not w:
                continue
            weights, bias = w
            name = layer.name

            qw, w_scale = quantize_int_real(weights, bit_width)
            qb, b_scale = quantize_int_real(bias, bit_width)

            # .npy -- exact integer array + scale, for our own Python reuse
            np.save(f"{out_dir}/{name}_weights_int.npy", qw)
            np.save(f"{out_dir}/{name}_bias_int.npy", qb)

            # .dat -- plain text integers, one per line (hardware-facing)
            np.savetxt(f"{out_dir}/{name}_weights.dat", qw.flatten(), fmt="%d")
            np.savetxt(f"{out_dir}/{name}_bias.dat", qb.flatten(), fmt="%d")

            # scale factors saved separately -- hardware needs these to
            # interpret the integers back into meaningful values
            with open(f"{out_dir}/{name}_scales.txt", "w") as f:
                f.write(f"weight_scale={w_scale}\n")
                f.write(f"bias_scale={b_scale}\n")

            print(f"  {name:10s} INT{bit_width}  weights={qw.shape} (range {qw.min()} to {qw.max()})  bias={qb.shape}")

print("\nAll models exported to weights/int8/ and weights/int4/")