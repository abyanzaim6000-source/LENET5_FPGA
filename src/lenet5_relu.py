"""
LeNet-5 variant: ReLU activation + Max Pooling.
This is a SEPARATE architecture from lenet5.py (tanh + avg pooling),
kept side-by-side for comparison, not a replacement.
"""

import tensorflow as tf
from tensorflow.keras import layers, models

def build_lenet5_relu():

    model = models.Sequential(name="LeNet5_ReLU")

    model.add(layers.Input(shape=(28, 28, 1)))

    model.add(layers.Conv2D(
        filters=6,
        kernel_size=(5,5),
        activation="relu",
        padding="same",
        name="C1"
    ))

    model.add(layers.MaxPooling2D(
        pool_size=(2,2),
        strides=2,
        name="S2"
    ))

    model.add(layers.Conv2D(
        filters=16,
        kernel_size=(5,5),
        activation="relu",
        name="C3"
    ))

    model.add(layers.MaxPooling2D(
        pool_size=(2,2),
        strides=2,
        name="S4"
    ))

    model.add(layers.Flatten())

    model.add(layers.Dense(120, activation="relu", name="C5"))

    model.add(layers.Dense(84, activation="relu", name="F6"))

    model.add(layers.Dense(10, activation="softmax", name="Output"))

    return model