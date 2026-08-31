import os
import tempfile

import pandas as pd
import numpy as np

from pipeline_steps import load_data, build_features, split_chronological, get_X_y, train_ridge, evaluate

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


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


def test_model_beats_naive_baseline():
    """Ridge with lag features must not be worse than the naive 'same time yesterday' baseline."""
    df = small_df(nrows=15000, nclients=10)
    features = ["lag_1d", "lag_7d", "lag_30d", "rolling_mean_30d"]
    feat = build_features(df)
    train, test = split_chronological(feat, train_frac=0.7)
    clean_train = train.dropna(subset=features + ["consumption"])
    clean_test  = test.dropna(subset=features + ["consumption"])
    Xtr, ytr = get_X_y(clean_train, features)
    Xte, yte = get_X_y(clean_test, features)

    model = train_ridge(Xtr, ytr, alpha=1.0)
    model_rmse = evaluate(model, Xte, yte)["rmse"]

    # naive baseline: predict lag_1d (same time yesterday)
    naive_errors = yte.to_numpy() - clean_test["lag_1d"].to_numpy()
    naive_rmse = float(np.sqrt((naive_errors ** 2).mean()))

    # allow 1% tolerance in case of numerical near-equality on small slices
    assert model_rmse <= naive_rmse * 1.01, (
        f"Model RMSE ({model_rmse:.4f}) significantly worse than naive baseline ({naive_rmse:.4f})"
    )
