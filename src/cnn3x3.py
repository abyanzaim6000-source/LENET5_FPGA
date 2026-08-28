"""
Custom CNN with 3x3 kernels, mirroring the paper's second validation
network. Same MNIST task as LeNet-5, only the kernel size differs --
this isolates kernel size as the one variable being compared.
"""

import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn3x3():

    model = models.Sequential(name="CNN_3x3")

    model.add(layers.Input(shape=(28, 28, 1)))

    # C1 - 3x3 conv, same padding to preserve 28x28
    model.add(layers.Conv2D(
        filters=6,
        kernel_size=(3,3),
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

    # C3 - 3x3 conv, valid padding
    model.add(layers.Conv2D(
        filters=16,
        kernel_size=(3,3),
        activation="tanh",
        name="C3"
    ))

    # S4 - Average Pooling
    model.add(layers.AveragePooling2D(
        pool_size=(2,2),
        strides=2,
        name="S4"
    ))

    model.add(layers.Flatten())

    model.add(layers.Dense(120, activation="tanh", name="C5"))
    model.add(layers.Dense(84, activation="tanh", name="F6"))
    model.add(layers.Dense(10, activation="softmax", name="Output"))

    return model