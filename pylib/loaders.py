import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


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
    train, test = load_stars(1, 0.001)
    i = 0
    for x, y, idx in train:
        print(x)
        print(y)
        print(idx)
        i += 1
        if i > 2:
            break
