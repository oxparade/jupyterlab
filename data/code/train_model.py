from __future__ import annotations

import gc
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

import mlflow
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:  # pragma: no cover
    from . import ml_pipeline
except ImportError:  # pragma: no cover
    import ml_pipeline


TARGET = ml_pipeline.TARGET_COLUMN
DEFAULT_MLFLOW_TRACKING_URI = "https://mlflow.10-53-101-61.nip.io"
MLFLOW_EXPERIMENT = "electricity-load-tp02"
REGISTER_MODEL = os.getenv("MLFLOW_REGISTER_MODEL", "false").lower() in {"1", "true", "yes"}
REGISTERED_MODEL_NAME = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "modelregistrytest")
PROMOTE_CHAMPION = os.getenv("MLFLOW_PROMOTE_CHAMPION", "false").lower() in {"1", "true", "yes"}
CHAMPION_ALIAS = os.getenv("MLFLOW_CHAMPION_ALIAS", "champion")
WRITE_DVC_SPLITS = os.getenv("WRITE_DVC_SPLITS", "false").lower() in {"1", "true", "yes"}
FEATURE_STRATEGIES: dict[str, dict[str, list[str] | str]] = {
    "short_term": {
        "features": ["lag_1d", "lag_7d"],
        "hypothesis": "La consommation dépend surtout des jours et semaines immédiatement précédents.",
    },
    "medium_term": {
        "features": ["lag_7d", "lag_30d"],
        "hypothesis": "La mémoire hebdomadaire et mensuelle capture une partie importante de la dynamique.",
    },
    "trend_and_seasonality": {
        "features": ml_pipeline.DEFAULT_FEATURES,
        "hypothesis": "Combiner mémoire courte, tendance et lissage améliore la prédiction.",
    },
}


def regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    y_true_arr = np.asarray(list(y_true))
    y_pred_arr = np.asarray(list(y_pred))
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))),
        "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "r2": float(r2_score(y_true_arr, y_pred_arr)),
    }


def coefficient_dict(model: Any, feature_names: list[str]) -> dict[str, float]:
    if not hasattr(model, "coef_"):
        return {}
    return {name: float(value) for name, value in zip(feature_names, np.ravel(model.coef_))}


def maybe_sample(frame: pd.DataFrame, max_rows: int | None, seed: int = 42) -> pd.DataFrame:
    if max_rows is None or len(frame) <= max_rows:
        return frame
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(frame), size=max_rows, replace=False)
    return frame.iloc[idx].copy()


def run_one_experiment(
    split: dict[str, pd.DataFrame],
    strategy_name: str,
    feature_names: list[str],
    estimator: Any,
    model_type: str,
    run_name: str,
    smoke_test: bool,
    train_rows: int | None,
    valid_rows: int | None,
    extra_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    train_df = maybe_sample(split["train"], train_rows, seed=42) if smoke_test else split["train"]
    valid_df = maybe_sample(split["validation"], valid_rows, seed=42) if smoke_test else split["validation"]

    X_train = train_df[feature_names]
    y_train = train_df[TARGET]
    X_valid = valid_df[feature_names]
    y_valid = valid_df[TARGET]

    model = clone(estimator)
    with mlflow.start_run(run_name=run_name, nested=mlflow.active_run() is not None):
        mlflow.set_tags(
            {
                "run_level": "child",
                "split_name": str(split["name"]),
                "strategy_name": strategy_name,
                "model_type": model_type,
            }
        )
        mlflow.log_params(
            {
                "model_type": model_type,
                "strategy_name": strategy_name,
                "features": ",".join(feature_names),
                "split_name": str(split["name"]),
                "split_strategy": split["description"],
                "smoke_test": smoke_test,
                "train_rows": len(train_df),
                "validation_rows": len(valid_df),
            }
        )
        if extra_params:
            mlflow.log_params(extra_params)

        model.fit(X_train, y_train)
        pred = model.predict(X_valid)
        metrics = regression_metrics(y_valid, pred)
        mlflow.log_metrics({f"val_{key}": value for key, value in metrics.items()})

        coefs = coefficient_dict(model, feature_names)
        if coefs:
            mlflow.log_dict(coefs, "coefficients.json")

        mlflow.sklearn.log_model(model, name="model", input_example=X_valid.head(5))
        run_id = mlflow.active_run().info.run_id

    gc.collect()
    return {
        "run_id": run_id,
        "split": split["name"],
        "strategy": strategy_name,
        "model_type": model_type,
        "features": feature_names,
        **{f"val_{key}": value for key, value in metrics.items()},
    }


def ensure_experiment_ready(experiment_name: str) -> None:
    client = MlflowClient()
    for experiment in client.search_experiments(view_type=ViewType.ALL):
        if experiment.name == experiment_name and experiment.lifecycle_stage == "deleted":
            client.restore_experiment(experiment.experiment_id)


def promote_latest_model_version(registered_model_name: str, alias: str) -> None:
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{registered_model_name}'")
    if not versions:
        print(f"No versions found for registered model '{registered_model_name}', skip alias '{alias}'.")
        return
    latest = max(versions, key=lambda version: int(version.version))
    client.set_registered_model_alias(
        name=registered_model_name,
        alias=alias,
        version=latest.version,
    )
    print(f"Registered model alias '{alias}' -> version {latest.version} ({registered_model_name})")


def main() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_MLFLOW_TRACKING_URI)
    mlflow.set_tracking_uri(tracking_uri)
    ensure_experiment_ready(MLFLOW_EXPERIMENT)
    try:
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
    except MlflowException as exc:
        if "deleted experiment" not in str(exc).lower():
            raise
        ensure_experiment_ready(MLFLOW_EXPERIMENT)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)

    frame = ml_pipeline.load_modeling_frame()
    required_columns = sorted({TARGET, *ml_pipeline.DEFAULT_FEATURES})
    frame = frame.dropna(subset=required_columns)
    split_v1 = ml_pipeline.build_temporal_split(frame, split_name="v1")
    split_v2 = ml_pipeline.build_temporal_split(frame, split_name="v2")

    if WRITE_DVC_SPLITS:
        split_v1_dir = ml_pipeline.materialize_split(split_v1, output_dir="data/splits")
        split_v2_dir = ml_pipeline.materialize_split(split_v2, output_dir="data/splits")
        print(f"DVC split materialized: {split_v1_dir}")
        print(f"DVC split materialized: {split_v2_dir}")
        print("Run: dvc add data/splits/split_v1_2011-2012_2013_2014")
        print("Run: dvc add data/splits/split_v2_2013_2014H1_2014H2")

    smoke_test = True
    train_rows = 600_000
    valid_rows = 300_000

    print("MLflow tracking URI :", mlflow.get_tracking_uri())
    print("Split v1:", split_v1["description"])
    print("Split v2:", split_v2["description"])

    with mlflow.start_run(run_name="tp02_training_session"):
        mlflow.set_tags(
            {
                "run_level": "parent",
                "pipeline": "training_tp02",
                "split_v1_name": str(split_v1["name"]),
                "split_v2_name": str(split_v2["name"]),
            }
        )
        mlflow.log_params(
            {
                "smoke_test": smoke_test,
                "train_rows": train_rows,
                "validation_rows": valid_rows,
                "split_v1_description": split_v1["description"],
                "split_v2_description": split_v2["description"],
            }
        )

        part1_results = []
        for strategy_name, spec in FEATURE_STRATEGIES.items():
            part1_results.append(
                run_one_experiment(
                    split=split_v1,
                    strategy_name=strategy_name,
                    feature_names=list(spec["features"]),
                    estimator=LinearRegression(),
                    model_type="linear_regression",
                    run_name=f"linear_{strategy_name}",
                    smoke_test=smoke_test,
                    train_rows=train_rows,
                    valid_rows=valid_rows,
                    extra_params={"hypothesis": spec["hypothesis"]},
                )
            )

        part1_df = pd.DataFrame(part1_results).sort_values("val_rmse").reset_index(drop=True)

        part2_results = []
        for strategy_name, spec in FEATURE_STRATEGIES.items():
            part2_results.append(
                run_one_experiment(
                    split=split_v2,
                    strategy_name=strategy_name,
                    feature_names=list(spec["features"]),
                    estimator=LinearRegression(),
                    model_type="linear_regression",
                    run_name=f"linear_{strategy_name}",
                    smoke_test=smoke_test,
                    train_rows=train_rows,
                    valid_rows=valid_rows,
                    extra_params={"hypothesis": spec["hypothesis"]},
                )
            )

        part2_df = pd.DataFrame(part2_results).sort_values("val_rmse").reset_index(drop=True)

        best_strategy_name = str(part1_df.loc[0, "strategy"])
        best_features = list(FEATURE_STRATEGIES[best_strategy_name]["features"])

        ridge_results = []
        for alpha in (1.0, 1e3, 1e9):
            ridge_results.append(
                run_one_experiment(
                    split=split_v1,
                    strategy_name=best_strategy_name,
                    feature_names=best_features,
                    estimator=Ridge(alpha=alpha),
                    model_type="ridge",
                    run_name=f"ridge_alpha_{alpha}",
                    smoke_test=smoke_test,
                    train_rows=train_rows,
                    valid_rows=valid_rows,
                    extra_params={"alpha": alpha},
                )
            )

        ridge_df = pd.DataFrame(ridge_results).sort_values("val_rmse").reset_index(drop=True)
        best_alpha = float(ridge_df.iloc[0]["run_id"] and 0)
        client = mlflow.tracking.MlflowClient()
        best_alpha = float(client.get_run(ridge_df.iloc[0]["run_id"]).data.params["alpha"])

        train_valid = pd.concat([split_v1["train"], split_v1["validation"]])
        train_valid = maybe_sample(train_valid, train_rows, seed=42) if smoke_test else train_valid
        test_df = maybe_sample(split_v1["test"], valid_rows, seed=42) if smoke_test else split_v1["test"]

        final_model = Ridge(alpha=best_alpha)
        final_model.fit(train_valid[best_features], train_valid[TARGET])
        test_pred = final_model.predict(test_df[best_features])
        test_metrics = regression_metrics(test_df[TARGET], test_pred)

        with mlflow.start_run(run_name="final_ridge", nested=True):
            mlflow.set_tags(
                {
                    "run_level": "child",
                    "split_name": str(split_v1["name"]),
                    "strategy_name": best_strategy_name,
                    "model_type": "ridge_final",
                }
            )
            mlflow.log_params(
                {
                    "model_type": "ridge_final",
                    "split_name": str(split_v1["name"]),
                    "split_strategy": split_v1["description"],
                    "best_strategy": best_strategy_name,
                    "best_features": ",".join(best_features),
                    "best_alpha": best_alpha,
                    "smoke_test": smoke_test,
                }
            )
            mlflow.log_metrics({f"test_{key}": value for key, value in test_metrics.items()})
            log_model_kwargs: dict[str, Any] = {
                "name": "model",
                "input_example": test_df[best_features].head(5),
            }
            if REGISTER_MODEL:
                log_model_kwargs["registered_model_name"] = REGISTERED_MODEL_NAME
            model_info = mlflow.sklearn.log_model(final_model, **log_model_kwargs)
            if REGISTER_MODEL:
                print(f"Registered model: {REGISTERED_MODEL_NAME} (uri={model_info.model_uri})")

        if REGISTER_MODEL and PROMOTE_CHAMPION:
            promote_latest_model_version(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)

        print("\nBest split v1:", part1_df.iloc[0]["strategy"], part1_df.iloc[0]["val_rmse"])
        print("Best split v2:", part2_df.iloc[0]["strategy"], part2_df.iloc[0]["val_rmse"])
        print("Best Ridge alpha:", best_alpha)
        print("Test metrics:", test_metrics)

        bonus = run_one_experiment(
            split=split_v1,
            strategy_name=best_strategy_name,
            feature_names=best_features,
            estimator=HistGradientBoostingRegressor(
                learning_rate=0.08,
                max_iter=150,
                max_leaf_nodes=31,
                random_state=42,
            ),
            model_type="hist_gradient_boosting",
            run_name="bonus_hist_gradient_boosting",
            smoke_test=smoke_test,
            train_rows=train_rows,
            valid_rows=valid_rows,
            extra_params={"learning_rate": 0.08, "max_iter": 150, "max_leaf_nodes": 31},
        )
        print("Bonus:", bonus["val_rmse"])


if __name__ == "__main__":
    main()