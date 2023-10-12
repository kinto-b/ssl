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
    # Initialise results file
    results_fp = "data/results.csv"
    with open(results_fp, "w", encoding="utf8") as f:
        f.write("model,loss,accuracy\n")

    # Evaluate models
    model_names = os.listdir("data/models")
    folds = [re.findall(r"-f(\d)-", nm)[0] for nm in model_names]

    for model_name, fold in zip(model_names, folds):
        print(f"Evaluating {model_name}...")
        model = tf.keras.models.load_model(f"data/models/{model_name}")

        # Load corresponding test data
        test = pd.read_csv(f"data/stars/prepared-fold{fold}.csv")
        test["class"] = test["class"].replace({"GALAXY": 0, "STAR": 1, "QSO": 2})
        x = np.array(test.drop(columns=["class"]))
        y = np.array(test["class"]).astype(np.int32)

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
