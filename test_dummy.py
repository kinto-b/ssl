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
        # if not "circles" in model_name:
        if "stars" in model_name or "mnist" in model_name:
            continue

        print(f"Evaluating {model_name}...")
        model = tf.keras.models.load_model(f"data/models/{model_name}")

        # Load corresponding test data
        x = np.array(pd.read_csv(f"data/grid.csv"))

        # We used a custom training loop, so we have to compile the model now
        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )
        pred = model.predict(x)
        pred = np.argmax(pred, 1)
        np.savetxt(f"data/predictions/{model_name}.csv", pred, delimiter=",")


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # FATAL
    logging.getLogger("tensorflow").setLevel(logging.FATAL)
    main()
