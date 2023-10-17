from random import sample

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split


def load_dummy(which="moons"):
    df = pd.read_csv(f"data/{which}.csv")
    y = df.loc[:, ["class", "labeled"]]
    x = df.loc[:, ["x1", "x2"]]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=15102023
    )
    np.savetxt(f"data/tmpx.csv", x_test, delimiter=",")
    np.savetxt(f"data/tmpy.csv", y_test["class"], delimiter=",")

    # Remove labels
    y_train.loc[np.logical_not(y_train["labeled"]), "class"] = -1
    report_unlabeled_count(y_train["class"])

    y_train = y_train["class"].astype(np.int32)
    y_test = y_test["class"].astype(np.int32)

    # Create Datasets
    idx_train = tf.range(len(y_train))
    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train, idx_train)
    ).shuffle(100000, reshuffle_each_iteration=False)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    return train_ds, test_ds


def load_mnist(prop_labeled=0.1):
    """Load (Fashion) MNIST"""
    # Get data
    mnist = tf.keras.datasets.mnist
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

    # Remove labels
    is_unlabeled = np.random.uniform(0, 1, size=y_train.shape) > prop_labeled
    y_train[is_unlabeled] = -1

    # Duplicate labeled data so we get 50/50 labeled/unlabeled
    n_duplicates = int((1 - prop_labeled) / prop_labeled)
    idx_l = np.tile(np.where(np.logical_not(is_unlabeled)), n_duplicates)
    idx_l = idx_l.reshape((idx_l.shape[1],))

    x_train = np.concatenate([x_train[is_unlabeled], x_train[idx_l]])
    y_train = np.concatenate([y_train[is_unlabeled], y_train[idx_l]])

    # Create Datasets
    idx_train = tf.range(len(y_train))
    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train, idx_train)
    ).shuffle(100000)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    return train_ds, test_ds


def load_stars(holdout, prop_labeled, mechanism="car"):
    """Load prepared stellar classification data"""
    # Use holdout as test
    train = [pd.read_csv(f"data/stars/prepared-fold{i}.csv") for i in range(1, 6)]
    test = train[holdout - 1]
    del train[holdout - 1]
    train = pd.concat(train)

    # Convert class to numeric
    mapping = {"GALAXY": 0, "STAR": 1, "QSO": 2}
    train["class"] = train["class"].replace(mapping)
    test["class"] = test["class"].replace(mapping)

    # Remove labels
    is_labeled = train[f"{mechanism}-{prop_labeled}"]
    train.loc[np.logical_not(is_labeled), "class"] = -1
    train = train.loc[:, "u":"class"]
    test = test.loc[:, "u":"class"]

    report_unlabeled_count(train["class"])

    # Resample labeled data for the sake of efficiency
    train_u = train.loc[np.logical_not(is_labeled), :]
    train_l = train.loc[is_labeled, :]
    train_l = train_l.sample(len(train_u), replace=True)
    train = pd.concat([train_u, train_l])

    # Separate features and labels
    x_train = np.array(train.drop(columns=["class"]))
    x_test = np.array(test.drop(columns=["class"]))
    y_train = np.array(train["class"])
    y_test = np.array(test["class"])

    # Labels need to be signed 32 bit int
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)

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
        f"Data loaded: {n_labeled} of {n_total} ({100*n_labeled/n_total:0.1f}%) are labeled!"
    )


if __name__ == "__main__":
    # Test that loaders are working as expected
    train, test = load_mnist()
    i = 0
    for x, y, idx in train:
        print(x.shape)
        print(y)
        # print(idx)
        i += 1
        if i > 10:
            break
