import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset(path, target_column):
    data = pd.read_csv(path)
    data = data.loc[:, ~data.columns.str.match(r"^Unnamed")]
    if target_column not in data.columns:
        raise ValueError("Target column '{}' was not found in {}".format(target_column, path))

    features = data.drop(columns=[target_column]).select_dtypes(include=[np.number])
    target = pd.to_numeric(data[target_column], errors="coerce")
    valid = features.notna().all(axis=1) & target.notna()

    if valid.sum() != len(data):
        print("Warning: removed {} rows containing missing values".format(len(data) - valid.sum()))

    features = features.loc[valid].copy()
    target = target.loc[valid].copy()
    if len(features) < 5:
        raise ValueError("At least 5 valid rows are required")
    return features, target


def split_dataset(features, target, dataset_dir="dataset", test_size=0.20,
                  random_seed=42):
    os.makedirs(dataset_dir, exist_ok=True)
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size,
        random_state=random_seed, shuffle=True
    )

    train = pd.concat([X_train, y_train.rename(target.name)], axis=1)
    test = pd.concat([X_test, y_test.rename(target.name)], axis=1)
    train.to_csv(os.path.join(dataset_dir, "train.csv"), index=False)
    test.to_csv(os.path.join(dataset_dir, "test.csv"), index=False)

    print("Train rows: {} ({}%)".format(len(train), round(len(train) * 100.0 / len(features), 1)))
    print("Test rows:  {} ({}%)".format(len(test), round(len(test) * 100.0 / len(features), 1)))
    return X_train, X_test, y_train, y_test
