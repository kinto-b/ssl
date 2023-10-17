"""
A library of networks
"""

import tensorflow as tf


def get_model_nn_single_layer(in_shape, out_units):
    """A very simple feed-forward NN"""
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input(in_shape),
            tf.keras.layers.GaussianNoise(0.01),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(out_units),
        ]
    )
    return model


def get_model_nn_multi_layer(in_shape, out_units):
    """A very simple feed-forward NN"""
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input(in_shape),
            tf.keras.layers.GaussianNoise(0.1),
            tf.keras.layers.Dense(64, activation="relu"),
            # tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(32, activation="relu"),
            # tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(16, activation="relu"),
            # tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(out_units),
        ]
    )
    return model


def get_model_toy_cnn(in_shape, out_units):
    """A very simple CNN"""
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input(in_shape),
            # tf.keras.layers.GaussianNoise(0.01),
            tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=(3, 3),
                activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                kernel_initializer="he_uniform",
            ),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(out_units),
        ]
    )

    return model


def get_model_laine(in_shape, out_units):
    """The model described by Laine and Aila"""
    model = tf.keras.Sequential(
        [
            # Augmentation
            tf.keras.layers.GaussianNoise(0.15),
            tf.keras.layers.RandomFlip(),
            tf.keras.layers.RandomRotation((-0.1, 0.1)),
            tf.keras.layers.Conv2D(
                filters=128,
                kernel_size=(3, 3),
                padding="same",
                activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                kernel_initializer="he_uniform",
                input_shape=in_shape,
            ),
            tf.keras.layers.Conv2D(
                filters=128,
                kernel_size=(3, 3),
                padding="same",
                activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                kernel_initializer="he_uniform",
            ),
            tf.keras.layers.Conv2D(
                filters=128,
                kernel_size=(3, 3),
                padding="same",
                activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                kernel_initializer="he_uniform",
            ),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Conv2D(
                filters=256,
                kernel_size=(3, 3),
                padding="same",
                activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                kernel_initializer="he_uniform",
            ),
            tf.keras.layers.Conv2D(
                filters=256,
                kernel_size=(3, 3),
                padding="same",
                activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                kernel_initializer="he_uniform",
            ),
            tf.keras.layers.Conv2D(
                filters=256,
                kernel_size=(3, 3),
                padding="same",
                activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                kernel_initializer="he_uniform",
            ),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Conv2D(
                filters=512,
                kernel_size=(3, 3),
                padding="valid",
                activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                kernel_initializer="he_uniform",
            ),
            tf.keras.layers.Conv2D(
                filters=256,
                kernel_size=(1, 1),
                activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                kernel_initializer="he_uniform",
            ),
            tf.keras.layers.Conv2D(
                filters=128,
                kernel_size=(1, 1),
                activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                kernel_initializer="he_uniform",
            ),
            tf.keras.layers.GlobalAvgPool2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(out_units),
        ]
    )

    return model


def get_model_kaggle(in_shape, out_units):
    """
    A modified version of an architecture I saw on Kaggle.

    https://www.kaggle.com/code/hadeerismail/skin-cancer-prediction-cnn-acc-98
    """

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.GaussianNoise(0.15),
            tf.keras.layers.Conv2D(
                32,
                kernel_size=(3, 3),
                activation="relu",
                padding="Same",
                input_shape=in_shape,
            ),
            tf.keras.layers.Conv2D(
                32,
                kernel_size=(3, 3),
                activation="relu",
                padding="Same",
            ),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="Same"),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="Same"),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.40),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(out_units),
        ]
    )

    return model
