import os
import tempfile

import pandas as pd
import numpy as np

from pipeline_steps import load_data, build_features, split_chronological, get_X_y, train_ridge, evaluate


DATA_PATH = os.path.join("data", "LD2011_2014.txt")


def small_df(nrows=2000, nclients=3):
    # load a small slice for testing speed
    df = load_data(path=DATA_PATH, nrows=nrows)
    cols = list(df.columns[:nclients])
    return df[cols]


def test_features_present():
    df = small_df()
    feat = build_features(df)
    for col in ["lag_1d", "lag_7d", "lag_30d", "rolling_mean_30d"]:
        assert col in feat.columns


def test_split_chronological():
    df = small_df()
    feat = build_features(df)
    train, test = split_chronological(feat, train_frac=0.6)
    assert len(train) > 0 and len(test) > 0
    # ensure no timestamp appears in train after test starts
    assert train.index.max() < test.index.min()


def test_training_reproducible():
    # use more rows so lag/rolling features have enough history
    df = small_df(nrows=10000, nclients=5)
    feat = build_features(df)
    train, test = split_chronological(feat, train_frac=0.7)
    features = ["lag_1d", "lag_7d", "lag_30d", "rolling_mean_30d"]
    Xtr, ytr = get_X_y(train.dropna(subset=features + ["consumption"]), features)
    Xte, yte = get_X_y(test.dropna(subset=features + ["consumption"]), features)

    m1 = train_ridge(Xtr, ytr, alpha=1.0)
    r1 = evaluate(m1, Xte, yte)["rmse"]

    m2 = train_ridge(Xtr, ytr, alpha=1.0)
    r2 = evaluate(m2, Xte, yte)["rmse"]

    assert np.isclose(r1, r2)
