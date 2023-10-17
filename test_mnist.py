"""
Check model performance on the held-aside test set.
"""

import logging
import os
import re

import numpy as np
import pandas as pd
import tensorflow as tf


def main():
    # Evaluate models
    model_names = os.listdir("data/models")

    for model_name in model_names:
        if not "mnist" in model_name:
            continue

        print(f"Evaluating {model_name}...")
        model = tf.keras.models.load_model(f"data/models/{model_name}")

        # Load corresponding test data
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

        # We used a custom training loop, so we have to compile the model now
        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )

        model.evaluate(x_test, y_test)


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # FATAL
    logging.getLogger("tensorflow").setLevel(logging.FATAL)
    main()
