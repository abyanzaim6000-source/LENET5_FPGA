"""
LeNet-5 Architecture
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def build_lenet5():

    model = models.Sequential(name="LeNet5")

    # Input Layer
    model.add(layers.Input(shape=(28, 28, 1)))

    # C1 - Convolution
    model.add(layers.Conv2D(
        filters=6,
        kernel_size=(5,5),
        activation="tanh",
        padding="same",
        name="C1"
    ))

    # S2 - Average Pooling
    model.add(layers.AveragePooling2D(
        pool_size=(2,2),
        strides=2,
        name="S2"
    ))

    # C3 - Convolution
    model.add(layers.Conv2D(
        filters=16,
        kernel_size=(5,5),
        activation="tanh",
        name="C3"
    ))

    # S4 - Average Pooling
    model.add(layers.AveragePooling2D(
        pool_size=(2,2),
        strides=2,
        name="S4"
    ))

    # Flatten
    model.add(layers.Flatten())

    # Fully Connected
    model.add(layers.Dense(120, activation="tanh", name="C5"))

    model.add(layers.Dense(84, activation="tanh", name="F6"))

    # Output Layer
    model.add(layers.Dense(10, activation="softmax", name="Output"))

    return model