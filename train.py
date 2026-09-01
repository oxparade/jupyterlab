import argparse
import logging
import os
from pathlib import Path

import mlflow
import pandas as pd

from pipeline_steps import (
    load_data,
    build_features,
    split_chronological,
    select_best_alpha_time_cv,
    get_X_y,
    train_ridge,
    evaluate,
    save_model,
)


logger = logging.getLogger("train")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Train pipeline for electricity forecasting")
    parser.add_argument("--input", default=None, help="Path to LD2011_2014.txt (CSV)")
    parser.add_argument("--nrows", type=int, default=None, help="Limit rows when loading (for fast dev)")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument(
        "--alphas",
        default="0.1,1.0,10.0",
        help="Comma-separated alpha candidates for chronological CV",
    )
    parser.add_argument("--cv-splits", type=int, default=3, help="Number of expanding CV splits")
    parser.add_argument("--artifact-path", default="model", help="MLflow logged-model name")
    parser.add_argument("--tracking-uri", default=None, help="MLflow tracking URI")
    parser.add_argument("--experiment-name", default="electricity-load-baseline", help="MLflow experiment")
    parser.add_argument("--run-name", default="ridge-baseline", help="MLflow run name")
    parser.add_argument(
        "--registered-model-name",
        default=None,
        help="Optional MLflow registered model name",
    )
    parser.add_argument("--model-out", dest="artifact_path", help=argparse.SUPPRESS)
    args = parser.parse_args()

    tracking_uri = (
        args.tracking_uri
        or os.getenv("MLFLOW_TRACKING_URI")
        or f"sqlite:///{(Path.cwd() / 'mlflow.db').resolve()}"
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    df = load_data(args.input, nrows=args.nrows)
    features_df = build_features(df)
    train, test = split_chronological(features_df, train_frac=args.train_fraction)

    features = ["lag_1d", "lag_7d", "lag_30d", "rolling_mean_30d"]

    alpha_candidates = tuple(float(value.strip()) for value in args.alphas.split(",") if value.strip())
    if not alpha_candidates:
        alpha_candidates = (args.alpha,)

    best_alpha, cv_rmse = select_best_alpha_time_cv(
        train,
        features=features,
        alphas=alpha_candidates,
        n_splits=args.cv_splits,
    )
    for alpha, rmse in sorted(cv_rmse.items(), key=lambda item: item[0]):
        logger.info("cv_rmse alpha=%.4f rmse=%.4f", alpha, rmse)
    logger.info("selected_alpha=%.4f (lowest mean cv RMSE)", best_alpha)

    Xtr, ytr = get_X_y(train.dropna(subset=features + ["consumption"]), features)
    Xte, yte = get_X_y(test.dropna(subset=features + ["consumption"]), features)

    model = train_ridge(Xtr, ytr, alpha=best_alpha)
    metrics = evaluate(model, Xte, yte)
    print(f"test RMSE {metrics['rmse']:.3f} | MAE {metrics['mae']:.3f}")
    logger.info("test_rmse=%.4f test_mae=%.4f", metrics["rmse"], metrics["mae"])

    with mlflow.start_run(run_name=args.run_name):
        mlflow.set_tags(
            {
                "pipeline": "simple_train",
                "model_type": "ridge",
            }
        )
        mlflow.log_params(
            {
                "train_fraction": args.train_fraction,
                "selected_alpha": best_alpha,
                "alpha_candidates": ",".join(str(alpha) for alpha in alpha_candidates),
                "cv_splits": args.cv_splits,
                "nrows": args.nrows if args.nrows is not None else "full",
                "features": ",".join(features),
            }
        )
        mlflow.log_metrics(metrics)
        model_uri = save_model(
            model,
            artifact_path=args.artifact_path,
            input_example=Xte.head(5),
            registered_model_name=args.registered_model_name,
        )

    logger.info("model_logged_to_mlflow rmse=%.4f model_uri=%s", metrics["rmse"], model_uri)
    print(f"model logged to MLflow: {model_uri}")

    # Reproducibility self-check: train a second time and compare RMSE
    import numpy as _np
    model2 = train_ridge(Xtr, ytr, alpha=best_alpha)
    metrics2 = evaluate(model2, Xte, yte)
    assert _np.isclose(metrics["rmse"], metrics2["rmse"]), (
        f"Reproducibility check FAILED: run1={metrics['rmse']:.4f} run2={metrics2['rmse']:.4f}"
    )
    print("reproducibility check OK (RMSE identical on two identical runs)")


if __name__ == "__main__":
    main()
