"""
Check model performance on the held-aside test set.
"""

import logging
import os

import numpy as np
import pandas as pd
import tensorflow as tf


def main():
    # Initialise results file
    results_fp = "data/results.csv"
    with open(results_fp, "w", encoding="utf8") as f:
        f.write("model,loss,accuracy\n")

    # Load test data
    test = pd.read_csv("data/stars/prepared-test.csv")
    test["class"] = test["class"].replace({pd.NA: -1, "GALAXY": 0, "STAR": 1, "QSO": 2})
    x = np.array(test.drop(columns=["class"]))
    y = np.array(test["class"]).astype(np.int32)

    # Evaluate models
    model_names = os.listdir("data/models")
    for model_name in model_names:
        print(f"Evaluating {model_name}...")
        model = tf.keras.models.load_model(f"data/models/{model_name}")

        # We used a custom training loop, so we have to compile the model now
        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )

        loss, acc = model.evaluate(x, y)

        with open(results_fp, "a", encoding="utf8") as f:
            f.write(f"{model_name},{loss},{acc}\n")


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # FATAL
    logging.getLogger("tensorflow").setLevel(logging.FATAL)
    main()
