"""
Custom CNN with 3x3 kernels, ReLU activation + Max Pooling variant.
Same as cnn3x3.py, only activation/pooling changed -- kernel size (3x3)
stays constant so this isolates activation/pooling as the one variable,
same approach as lenet5_relu.py vs lenet5.py.
"""

import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn3x3_relu():

    model = models.Sequential(name="CNN_3x3_ReLU")

    model.add(layers.Input(shape=(28, 28, 1)))

    model.add(layers.Conv2D(
        filters=6,
        kernel_size=(3,3),
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
        kernel_size=(3,3),
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