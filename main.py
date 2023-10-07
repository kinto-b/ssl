"""Consistency regularisation

An implementation of the Pi-model using tensorflow.
"""

import logging
import os

import tensorflow as tf

from pylib.loaders import load_stars
from pylib.network import get_model_toy_nn
from pylib.ssl_models import BaseModel, PiModel, TemporalEnsembleModel


def main(epochs, prop_labeled=0.1):
    """Train an SSL model"""
    logging.basicConfig(
        level=logging.INFO,
        format="\n%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    logger.info("Loading data...")
    train_ds, test_ds = load_stars(prop_labeled)
    in_shape = (6,)
    out_units = 3

    optimizer = tf.keras.optimizers.Adam()

    logger.info("Fitting supervised-only...")
    model_sl = get_model_toy_nn(in_shape, out_units)
    fit_sl = BaseModel(model_sl, optimizer, epochs)
    fit_sl.train(train_ds, test_ds, gamma=10)
    model_sl.save(f"data/stars-{prop_labeled}-toy_nn-sl.keras")

    logger.info("Fitting pi model...")
    model_pi = get_model_toy_nn(in_shape, out_units)
    fit_pi = PiModel(model_pi, optimizer, epochs)
    fit_pi.train(train_ds, test_ds, gamma=10, tau=epochs // 3)
    model_pi.save(f"data/stars-{prop_labeled}-toy_nn-pi.keras")

    logger.info("Fitting temporal ensemble...")
    model_te = get_model_toy_nn(in_shape, out_units)
    fit_te = TemporalEnsembleModel(model_te, optimizer, epochs)
    fit_te.train(train_ds, test_ds, alpha=0.5, gamma=10, tau=epochs // 3)
    model_pi.save(f"data/stars-{prop_labeled}-toy_nn-te.keras")


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # FATAL
    logging.getLogger("tensorflow").setLevel(logging.FATAL)
    main(epochs=30, prop_labeled=0.001)
