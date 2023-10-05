import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


def load_stars(prop_unlabeled=0.9, prop_test=0.1):
    """Load prepared stellar classification data"""
    df = pd.read_csv("data/stars/prepared.csv")

    # Separate features and labels
    x = np.array(df.drop(columns=["class"]))
    y = np.array(df["class"])

    # Labels need to be signed 32 bit int
    y = y.astype(np.int32)

    # Split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=prop_test, random_state=2023
    )

    return as_ssl_data(x_train, y_train, x_test, y_test, prop_unlabeled)


def load_mnist(prop_unlabeled=0.9):
    """Load (Fashion) MNIST"""
    # Get data
    mnist = tf.keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Add 'channel'
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    # Labels need to be signed integers
    y_train = y_train.astype(dtype=np.int32)
    y_test = y_test.astype(dtype=np.int32)

    # Original data is read only
    y_train = y_train.copy()

    return as_ssl_data(x_train, y_train, x_test, y_test, prop_unlabeled)


def load_ham(prop_unlabeled=0.9, prop_test=0.1):
    """Load a subset of the skin lesion data"""
    df = pd.read_csv("data/ham/hmnist_28_28_RGB.csv")

    # Focus on nv, mel and bkl for now
    df = df[df["label"].isin([2, 4, 6])]
    df["label"] = df["label"].replace({2: 0, 4: 1, 6: 2})

    # Reshape
    x = df.drop(columns=["label"])
    x = np.array(x).reshape(-1, 28, 28, 3)
    y = np.array(df["label"])

    # Rescale
    x = x / 255

    # Labels need to be signed 32 bit int
    y = y.astype(np.int32)

    # Split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=prop_test, random_state=2023
    )

    return as_ssl_data(x_train, y_train, x_test, y_test, prop_unlabeled)


def as_ssl_data(x_train, y_train, x_test, y_test, prop_unlabeled):
    """
    Convert matrices to tf datasets and make some data 'unlabeled' by replacing
    labels with the sentinel value -1.
    """
    # Use -1 as an unlabeled sentinel value
    y_train[np.random.uniform(0, 1, size=y_train.shape) < prop_unlabeled] = -1
    report_unlabeled_count(y_train)

    # Create Datasets
    idx_train = tf.range(len(y_train))
    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train, idx_train)
    ).shuffle(100000)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    return train_ds, test_ds


def report_unlabeled_count(y):
    n_total = y.shape[0]
    n_labeled = np.sum(np.not_equal(y, -1))
    print(
        f"\nData loaded: {n_labeled} of {n_total} ({100*n_labeled/n_total:0.1f}%) are labeled!"
    )


if __name__ == "__main__":
    # Test that loaders are working as expected
    train, test = load_stars(0)
    i = 0
    for x, y, idx in train:
        print(x)
        print(y)
        print(idx)
        i += 1
        if i > 2:
            break
