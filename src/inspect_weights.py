"""
Prints the shape of every trained weight/bias in the saved LeNet-5 model.
"""

import tensorflow as tf

model = tf.keras.models.load_model("models/lenet5.keras")

for layer in model.layers:
    weights = layer.get_weights()
    if weights:
        w_shape = weights[0].shape
        b_shape = weights[1].shape if len(weights) > 1 else None
        print(f"{layer.name:10s}  weights={w_shape}  bias={b_shape}")