"""
Applying consistency regularisation to MNIST
"""

import logging
import os

import tensorflow as tf

from pylib.loaders import load_mnist
from pylib.network import get_model_toy_cnn
from pylib.ssl_models import BaseModel, PiModel, TemporalEnsembleModel


def main(epochs, prop_labeled):
    """Train an SSL model"""
    logging.basicConfig(
        level=logging.INFO,
        format="\n%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    logger.info("Loading data...")
    train_ds, test_ds = load_mnist(prop_labeled)
    in_shape = (28, 28, 1)
    out_units = 10

    optimizer = tf.keras.optimizers.Adam()

    logger.info("Fitting supervised-only...")
    model_sl = get_model_toy_cnn(in_shape, out_units)
    fit_sl = BaseModel(model_sl, optimizer, epochs)
    fit_sl.train(train_ds, test_ds, gamma=10)
    model_sl.save(f"data/models/mnist-toy_cnn-sl.keras")

    logger.info("Fitting pi model...")
    model_pi = get_model_toy_cnn(in_shape, out_units)
    fit_pi = PiModel(model_pi, optimizer, epochs)
    fit_pi.train(train_ds, test_ds, gamma=20, tau=epochs // 3)
    model_pi.save(f"data/models/mnist-toy_cnn-pi.keras")

    logger.info("Fitting temporal ensemble...")
    model_te = get_model_toy_cnn(in_shape, out_units)
    fit_te = TemporalEnsembleModel(model_te, optimizer, epochs)
    fit_te.train(train_ds, test_ds, alpha=0.05, gamma=20, tau=epochs // 3)
    model_te.save(f"data/models/mnist-toy_cnn-te.keras")


if __name__ == "__main__":
    # os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # FATAL
    # logging.getLogger("tensorflow").setLevel(logging.FATAL)
    main(epochs=30, prop_labeled=0.01)
