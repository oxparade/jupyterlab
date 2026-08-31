import argparse
from pathlib import Path

import pandas as pd

from pipeline_steps import (
    load_data,
    build_features,
    split_chronological,
    get_X_y,
    train_ridge,
    evaluate,
    save_model,
)


def main():
    parser = argparse.ArgumentParser(description="Train pipeline for electricity forecasting")
    parser.add_argument("--input", default=None, help="Path to LD2011_2014.txt (CSV)")
    parser.add_argument("--nrows", type=int, default=None, help="Limit rows when loading (for fast dev)")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--model-out", default="model.pkl")
    args = parser.parse_args()

    df = load_data(args.input, nrows=args.nrows)
    features_df = build_features(df)
    train, test = split_chronological(features_df, train_frac=args.train_fraction)

    features = ["lag_1d", "lag_7d", "lag_30d", "rolling_mean_30d"]
    Xtr, ytr = get_X_y(train.dropna(subset=features + ["consumption"]), features)
    Xte, yte = get_X_y(test.dropna(subset=features + ["consumption"]), features)

    model = train_ridge(Xtr, ytr, alpha=args.alpha)
    metrics = evaluate(model, Xte, yte)
    print(f"test RMSE {metrics['rmse']:.3f} | MAE {metrics['mae']:.3f}")

    save_model(model, args.model_out)


if __name__ == "__main__":
    main()
