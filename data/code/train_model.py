from __future__ import annotations

import gc
import importlib.metadata
import json
import os
import sqlite3
import sys
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterable

import mlflow
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.entities import ViewType
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from mlflow.sklearn import log_model as log_sklearn_model
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:  # pragma: no cover
    from . import export
    from . import ml_pipeline
except ImportError:  # pragma: no cover
    import export
    import ml_pipeline


TARGET = ml_pipeline.TARGET_COLUMN
DEFAULT_MLFLOW_TRACKING_URI = "https://mlflow.10-53-101-61.nip.io"
MLFLOW_EXPERIMENT = "electricity-load-tp02"
REGISTER_MODEL = os.getenv("MLFLOW_REGISTER_MODEL", "true").lower() in {"1", "true", "yes"}
REGISTERED_MODEL_NAME = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "modelregistrytest")
PROMOTE_CHAMPION = os.getenv("MLFLOW_PROMOTE_CHAMPION", "true").lower() in {"1", "true", "yes"}
CHAMPION_ALIAS = os.getenv("MLFLOW_CHAMPION_ALIAS", "champion")
WRITE_DVC_SPLITS = os.getenv("WRITE_DVC_SPLITS", "false").lower() in {"1", "true", "yes"}
LOCAL_ARTIFACT_ROOT = Path("data/models/artifacts")
LOCAL_MODEL_DIR = LOCAL_ARTIFACT_ROOT / "models"
LOCAL_FIGURE_DIR = LOCAL_ARTIFACT_ROOT / "figures"
LOCAL_REPORT_DIR = LOCAL_ARTIFACT_ROOT / "reports"
LOCAL_METADATA_DIR = LOCAL_ARTIFACT_ROOT / "metadata"
LOCAL_METADATA_DB = LOCAL_ARTIFACT_ROOT / "metadata_models.db"
LOCAL_MLFLOW_DB = LOCAL_ARTIFACT_ROOT / "mlflow.db"
LOCAL_CONDA_YAML = LOCAL_ARTIFACT_ROOT / "conda.yaml"
FEATURE_STRATEGIES: dict[str, dict[str, list[str] | str]] = {
    "minimal": {
        "features": ["lag_1d"],
        "hypothesis": "Une seule mémoire courte sert de baseline simple.",
    },
    "weekly": {
        "features": ["lag_1d", "lag_7d"],
        "hypothesis": "La consommation dépend surtout des informations les plus récentes et du cycle hebdomadaire.",
    },
    "rolling": {
        "features": ["lag_1d", "lag_7d", "rolling_mean_7d", "rolling_mean_30d"],
        "hypothesis": "Les moyennes glissantes apportent une vue lissée de la tendance récente.",
    },
    "full": {
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
    coef_values = np.asarray(model.coef_, dtype=float).ravel()
    return {name: float(value) for name, value in zip(feature_names, coef_values)}


def maybe_sample(frame: pd.DataFrame, max_rows: int | None, seed: int = 42) -> pd.DataFrame:
    if max_rows is None or len(frame) <= max_rows:
        return frame
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(frame), size=max_rows, replace=False)
    return frame.iloc[idx].copy()


def log_html_artifact(html_content: str, artifact_name: str, artifact_dir: str = "reports") -> None:
    base_dir = LOCAL_REPORT_DIR if artifact_dir == "reports" else LOCAL_ARTIFACT_ROOT / artifact_dir
    local_path = base_dir / artifact_name
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_text(html_content, encoding="utf-8")
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / artifact_name
        temp_path.write_text(html_content, encoding="utf-8")
        mlflow.log_artifact(str(temp_path), artifact_path=artifact_dir)


def log_json_artifact(payload: Any, artifact_name: str, artifact_dir: str = "metadata") -> None:
    base_dir = LOCAL_METADATA_DIR if artifact_dir == "metadata" else LOCAL_ARTIFACT_ROOT / artifact_dir
    local_path = base_dir / artifact_name
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / artifact_name
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        mlflow.log_artifact(str(temp_path), artifact_path=artifact_dir)


def log_plotly_figure(fig: go.Figure, artifact_stem: str, artifact_dir: str = "figures") -> None:
    local_dir = LOCAL_FIGURE_DIR if artifact_dir == "figures" else LOCAL_ARTIFACT_ROOT / artifact_dir
    local_dir.mkdir(parents=True, exist_ok=True)
    try:
        export.export_plotly(fig, local_dir, artifact_stem)
    except Exception:
        fig.write_html(local_dir / f"{artifact_stem}.html", include_plotlyjs="cdn")
    with TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        try:
            export.export_plotly(fig, temp_dir_path, artifact_stem)
        except Exception:
            fig.write_html(temp_dir_path / f"{artifact_stem}.html", include_plotlyjs="cdn")
        mlflow.log_artifacts(str(temp_dir_path), artifact_path=artifact_dir)


def build_conda_environment_yaml(requirements_path: Path = Path("requirements.txt")) -> str:
    requirements: list[str] = []
    if requirements_path.exists():
        for line in requirements_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            requirements.append(line)

    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    lines = [
        "name: electricity-load-tp02",
        "channels:",
        "  - conda-forge",
        "dependencies:",
        f"  - python={python_version}",
        "  - pip",
        "  - pip:",
    ]
    lines.extend(f"      - {requirement}" for requirement in requirements)
    resolved_packages = ["mlflow", "pandas", "numpy", "scikit-learn", "joblib", "skops", "plotly", "pyarrow"]
    lines.append("# resolved package versions from the current environment")
    for package_name in resolved_packages:
        try:
            version = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            continue
        lines.append(f"# {package_name}=={version}")
    return "\n".join(lines) + "\n"


def write_conda_environment_file(output_path: Path = LOCAL_CONDA_YAML) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_conda_environment_yaml(), encoding="utf-8")


def write_metadata_database(records: list[dict[str, Any]], output_path: Path = LOCAL_METADATA_DB) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(output_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS model_metadata (
                run_id TEXT PRIMARY KEY,
                run_name TEXT,
                split_name TEXT,
                strategy_name TEXT,
                model_type TEXT,
                feature_names_json TEXT,
                metrics_json TEXT,
                extra_params_json TEXT,
                model_uri TEXT,
                payload_json TEXT
            )
            """
        )
        connection.execute("DELETE FROM model_metadata")
        rows = []
        for record in records:
            rows.append(
                (
                    str(record.get("run_id", "")),
                    str(record.get("run_name", "")),
                    str(record.get("split_name", "")),
                    str(record.get("strategy_name", "")),
                    str(record.get("model_type", "")),
                    json.dumps(record.get("features", []), ensure_ascii=False, default=str),
                    json.dumps(record.get("metrics", {}), ensure_ascii=False, default=str),
                    json.dumps(record.get("extra_params", {}), ensure_ascii=False, default=str),
                    str(record.get("model_uri", "")) if record.get("model_uri") else None,
                    json.dumps(record, ensure_ascii=False, default=str),
                )
            )
        connection.executemany(
            """
            INSERT OR REPLACE INTO model_metadata (
                run_id,
                run_name,
                split_name,
                strategy_name,
                model_type,
                feature_names_json,
                metrics_json,
                extra_params_json,
                model_uri,
                payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        connection.commit()


def save_model_serializations(model: Any, artifact_stem: str) -> None:
    LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, str(LOCAL_MODEL_DIR / f"{artifact_stem}.joblib"))
    try:
        import skops.io as sio

        sio.dump(model, LOCAL_MODEL_DIR / f"{artifact_stem}.skops")
    except Exception as exc:  # pragma: no cover
        print(f"Skipping skops export for {artifact_stem}: {exc}")


def build_prediction_figure(y_true: pd.Series, y_pred: np.ndarray, title: str) -> go.Figure:
    frame = pd.DataFrame({"actual": y_true.to_numpy(), "predicted": np.asarray(y_pred)})
    fig = px.scatter(
        frame,
        x="actual",
        y="predicted",
        title=title,
        labels={"actual": "Observed", "predicted": "Predicted"},
        opacity=0.55,
    )
    low = float(min(frame["actual"].min(), frame["predicted"].min()))
    high = float(max(frame["actual"].max(), frame["predicted"].max()))
    fig.add_shape(
        type="line",
        x0=low,
        y0=low,
        x1=high,
        y1=high,
        line={"color": "#B0B0B0", "dash": "dash"},
    )
    fig.update_layout(template="plotly_dark", xaxis_title="Observed", yaxis_title="Predicted")
    return fig


def build_residual_figure(y_true: pd.Series, y_pred: np.ndarray, title: str) -> go.Figure:
    residuals = pd.Series(y_true.to_numpy() - np.asarray(y_pred), name="residual")
    fig = px.histogram(residuals.to_frame(), x="residual", nbins=60, title=title)
    fig.update_layout(template="plotly_dark", xaxis_title="Residual", yaxis_title="Count")
    return fig


def build_metric_comparison_figure(frame: pd.DataFrame, label_column: str, metric: str, title: str) -> go.Figure:
    ordered = frame.sort_values(metric, ascending=True).reset_index(drop=True)
    fig = px.bar(
        ordered,
        x=metric,
        y=label_column,
        orientation="h",
        text=metric,
        title=title,
    )
    fig.update_layout(template="plotly_dark", xaxis_title=metric, yaxis_title="")
    return fig


def log_run_report(
    *,
    run_title: str,
    split_name: str,
    strategy_name: str,
    model_type: str,
    metrics: dict[str, float],
    extra_params: dict[str, Any] | None = None,
) -> None:
    run_rows: list[dict[str, Any]] = [
        {"group": "run", "key": "title", "value": run_title},
        {"group": "run", "key": "split", "value": split_name},
        {"group": "run", "key": "strategy", "value": strategy_name},
        {"group": "run", "key": "model_type", "value": model_type},
    ]
    metric_rows: list[dict[str, Any]] = []
    for key, value in metrics.items():
        metric_rows.append({"group": "metrics", "key": key, "value": value})
    param_rows: list[dict[str, Any]] = []
    if extra_params:
        for key, value in extra_params.items():
            param_rows.append({"group": "params", "key": key, "value": value})

    log_html_artifact(pd.DataFrame(run_rows).to_html(index=False, border=0), f"{run_title}_run.html")
    log_html_artifact(pd.DataFrame(metric_rows).to_html(index=False, border=0), f"{run_title}_metrics.html")
    log_html_artifact(pd.DataFrame(param_rows).to_html(index=False, border=0), f"{run_title}_params.html")


def build_metadata_record(
    *,
    run_id: str,
    run_name: str,
    split_name: str,
    strategy_name: str,
    model_type: str,
    feature_names: list[str],
    metrics: dict[str, float],
    extra_params: dict[str, Any] | None = None,
    model_uri: str | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "run_id": run_id,
        "run_name": run_name,
        "split_name": split_name,
        "strategy_name": strategy_name,
        "model_type": model_type,
        "features": feature_names,
        "metrics": metrics,
    }
    if extra_params:
        record["extra_params"] = extra_params
    if model_uri:
        record["model_uri"] = model_uri
    return record


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
    mlflow.start_run(run_name=run_name, nested=mlflow.active_run() is not None)
    try:
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
        signature = infer_signature(X_valid.head(5), pred[:5])
        mlflow.log_metrics({f"val_{key}": value for key, value in metrics.items()})

        coefs = coefficient_dict(model, feature_names)
        if coefs:
            mlflow.log_dict(coefs, "coefficients.json")

        model_info = log_sklearn_model(
            model,
            name="model",
            input_example=X_valid.head(5),
            signature=signature,
        )
        save_model_serializations(model, "model")
        mlflow.log_artifact(str(LOCAL_MODEL_DIR / "model.joblib"), artifact_path="model")
        mlflow.log_artifact(str(LOCAL_MODEL_DIR / "model.skops"), artifact_path="model")
        prediction_figure = build_prediction_figure(
            y_true=y_valid,
            y_pred=pred,
            title=f"{run_name} — observed vs predicted",
        )
        residual_figure = build_residual_figure(
            y_true=y_valid,
            y_pred=pred,
            title=f"{run_name} — residuals",
        )
        log_plotly_figure(prediction_figure, f"{run_name}_observed_vs_predicted")
        log_plotly_figure(residual_figure, f"{run_name}_residuals")
        log_run_report(
            run_title=run_name,
            split_name=str(split["name"]),
            strategy_name=strategy_name,
            model_type=model_type,
            metrics=metrics,
            extra_params=extra_params,
        )
        active_run = mlflow.active_run()
        assert active_run is not None
        run_id = active_run.info.run_id
        metadata_record = build_metadata_record(
            run_id=str(run_id),
            run_name=run_name,
            split_name=str(split["name"]),
            strategy_name=strategy_name,
            model_type=model_type,
            feature_names=feature_names,
            metrics=metrics,
            extra_params=extra_params,
            model_uri=model_info.model_uri,
        )
        log_json_artifact(metadata_record, f"{run_name}_metadata.json")
    finally:
        try:
            mlflow.end_run()
        except Exception:
            pass

    gc.collect()
    return {
        **metadata_record,
        "split": split["name"],
        "strategy": strategy_name,
        "model_type": model_type,
        "features": feature_names,
        **{f"val_{key}": value for key, value in metrics.items()},
        **(extra_params or {}),
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

    mlflow.start_run(run_name="tp02_training_session")
    try:
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

        metadata_records: list[dict[str, Any]] = []

        part1_results: list[dict[str, Any]] = []
        for strategy_name, spec in FEATURE_STRATEGIES.items():
            part1_results.append(
                run_one_experiment(
                    split=split_v1,
                    strategy_name=strategy_name,
                    feature_names=list(spec["features"]),
                    estimator=LinearRegression(),
                    model_type="linear_regression",
                    run_name="linear-regression",
                    smoke_test=smoke_test,
                    train_rows=train_rows,
                    valid_rows=valid_rows,
                    extra_params={"hypothesis": spec["hypothesis"]},
                )
            )

        metadata_records.extend(part1_results)

        part1_df = pd.DataFrame(part1_results).sort_values("val_rmse").reset_index(drop=True)
        part1_df["label"] = part1_df["strategy"] + " | " + part1_df["split"]

        log_plotly_figure(
            build_metric_comparison_figure(
                part1_df,
                label_column="label",
                metric="val_rmse",
                title="Split v1 — validation RMSE by strategy",
            ),
            "split_v1_validation_rmse",
        )
        log_plotly_figure(
            build_metric_comparison_figure(
                part1_df,
                label_column="label",
                metric="val_mae",
                title="Split v1 — validation MAE by strategy",
            ),
            "split_v1_validation_mae",
        )
        log_plotly_figure(
            build_metric_comparison_figure(
                part1_df,
                label_column="label",
                metric="val_r2",
                title="Split v1 — validation R² by strategy",
            ),
            "split_v1_validation_r2",
        )

        part2_results: list[dict[str, Any]] = []
        for strategy_name, spec in FEATURE_STRATEGIES.items():
            part2_results.append(
                run_one_experiment(
                    split=split_v2,
                    strategy_name=strategy_name,
                    feature_names=list(spec["features"]),
                    estimator=LinearRegression(),
                    model_type="linear_regression",
                    run_name="linear-regression",
                    smoke_test=smoke_test,
                    train_rows=train_rows,
                    valid_rows=valid_rows,
                    extra_params={"hypothesis": spec["hypothesis"]},
                )
            )

        metadata_records.extend(part2_results)

        part2_df = pd.DataFrame(part2_results).sort_values("val_rmse").reset_index(drop=True)
        part2_df["label"] = part2_df["strategy"] + " | " + part2_df["split"]

        log_plotly_figure(
            build_metric_comparison_figure(
                part2_df,
                label_column="label",
                metric="val_rmse",
                title="Split v2 — validation RMSE by strategy",
            ),
            "split_v2_validation_rmse",
        )
        log_plotly_figure(
            build_metric_comparison_figure(
                part2_df,
                label_column="label",
                metric="val_mae",
                title="Split v2 — validation MAE by strategy",
            ),
            "split_v2_validation_mae",
        )
        log_plotly_figure(
            build_metric_comparison_figure(
                part2_df,
                label_column="label",
                metric="val_r2",
                title="Split v2 — validation R² by strategy",
            ),
            "split_v2_validation_r2",
        )

        best_strategy_name = str(part1_df.loc[0, "strategy"])
        best_features = list(FEATURE_STRATEGIES[best_strategy_name]["features"])

        ridge_results: list[dict[str, Any]] = []
        for alpha in (1.0, 1e3, 1e9):
            ridge_results.append(
                run_one_experiment(
                    split=split_v1,
                    strategy_name=best_strategy_name,
                    feature_names=best_features,
                    estimator=Ridge(alpha=alpha),
                    model_type="ridge",
                    run_name="ridge",
                    smoke_test=smoke_test,
                    train_rows=train_rows,
                    valid_rows=valid_rows,
                    extra_params={"alpha": alpha},
                )
            )

        metadata_records.extend(ridge_results)

        ridge_df = pd.DataFrame(ridge_results).sort_values("val_rmse").reset_index(drop=True)
        ridge_display = ridge_df.copy()
        ridge_display["label"] = ridge_display["strategy"] + " | alpha=" + ridge_display["alpha"].astype(str)

        log_plotly_figure(
            build_metric_comparison_figure(
                ridge_display,
                label_column="label",
                metric="val_rmse",
                title="Ridge candidates — validation RMSE",
            ),
            "ridge_validation_rmse",
        )
        log_plotly_figure(
            build_metric_comparison_figure(
                ridge_display,
                label_column="label",
                metric="val_mae",
                title="Ridge candidates — validation MAE",
            ),
            "ridge_validation_mae",
        )
        log_plotly_figure(
            build_metric_comparison_figure(
                ridge_display,
                label_column="label",
                metric="val_r2",
                title="Ridge candidates — validation R²",
            ),
            "ridge_validation_r2",
        )

        best_alpha = float(ridge_df.iloc[0]["alpha"])

        train_valid = pd.concat([split_v1["train"], split_v1["validation"]])
        train_valid = maybe_sample(train_valid, train_rows, seed=42) if smoke_test else train_valid
        test_df = maybe_sample(split_v1["test"], valid_rows, seed=42) if smoke_test else split_v1["test"]

        final_model = Ridge(alpha=best_alpha)
        final_model.fit(train_valid[best_features], train_valid[TARGET])
        test_pred = final_model.predict(test_df[best_features])
        test_metrics = regression_metrics(test_df[TARGET], test_pred)
        final_signature = infer_signature(test_df[best_features].head(5), test_pred[:5])

        mlflow.start_run(run_name="ridge-final", nested=True)
        try:
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
            log_model_kwargs["signature"] = final_signature
            model_info = log_sklearn_model(final_model, **log_model_kwargs)
            champion_dir = Path("data/models/champions")
            champion_dir.mkdir(parents=True, exist_ok=True)
            champion_artifact_stem = f"champion_{best_strategy_name}_alpha_{best_alpha}"
            champion_path = champion_dir / f"{champion_artifact_stem}.joblib"
            joblib.dump(final_model, str(champion_path))
            save_model_serializations(final_model, champion_artifact_stem)
            mlflow.log_artifact(str(champion_path), artifact_path="champions")
            mlflow.log_artifact(str(LOCAL_MODEL_DIR / f"{champion_artifact_stem}.skops"), artifact_path="champions")
            active_run = mlflow.active_run()
            assert active_run is not None
            champion_metadata = build_metadata_record(
                run_id=str(active_run.info.run_id),
                run_name="ridge-final",
                split_name=str(split_v1["name"]),
                strategy_name=best_strategy_name,
                model_type="ridge_final",
                feature_names=best_features,
                metrics={f"test_{key}": value for key, value in test_metrics.items()},
                extra_params={"best_alpha": best_alpha, "best_features": ",".join(best_features)},
                model_uri=model_info.model_uri,
            )
            metadata_records.append(champion_metadata)
            log_json_artifact(champion_metadata, "champion_metadata.json", artifact_dir="metadata")
            if REGISTER_MODEL:
                active_run = mlflow.active_run()
                assert active_run is not None
                current_run_id = active_run.info.run_id
                versions = MlflowClient().search_model_versions(
                    f"name='{REGISTERED_MODEL_NAME}' and run_id='{current_run_id}'"
                )
                if versions:
                    registered_version = max(versions, key=lambda version: int(version.version))
                    client = MlflowClient()
                    version_tags = {
                        "split_name": str(split_v1["name"]),
                        "strategy_name": best_strategy_name,
                        "model_type": "ridge_final",
                        "best_alpha": str(best_alpha),
                        "best_features": ",".join(best_features),
                        "test_rmse": str(test_metrics["rmse"]),
                        "test_mae": str(test_metrics["mae"]),
                        "test_r2": str(test_metrics["r2"]),
                    }
                    for key, value in version_tags.items():
                        client.set_model_version_tag(
                            name=REGISTERED_MODEL_NAME,
                            version=registered_version.version,
                            key=key,
                            value=value,
                        )
            final_prediction_figure = build_prediction_figure(
                y_true=test_df[TARGET],
                y_pred=test_pred,
                title="Final ridge — observed vs predicted",
            )
            final_residual_figure = build_residual_figure(
                y_true=test_df[TARGET],
                y_pred=test_pred,
                title="Final ridge — residuals",
            )
            log_plotly_figure(final_prediction_figure, "final_ridge_observed_vs_predicted")
            log_plotly_figure(final_residual_figure, "final_ridge_residuals")
            log_run_report(
                run_title="ridge-final",
                split_name=str(split_v1["name"]),
                strategy_name=best_strategy_name,
                model_type="ridge_final",
                metrics={f"test_{key}": value for key, value in test_metrics.items()},
                extra_params={"best_alpha": best_alpha, "best_features": ",".join(best_features)},
            )
            if REGISTER_MODEL:
                print(f"Registered model: {REGISTERED_MODEL_NAME} (uri={model_info.model_uri})")
        finally:
            try:
                mlflow.end_run()
            except Exception:
                pass

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
            run_name="hist-gradient-boosting",
            smoke_test=smoke_test,
            train_rows=train_rows,
            valid_rows=valid_rows,
            extra_params={"learning_rate": 0.08, "max_iter": 150, "max_leaf_nodes": 31},
        )
        print("Bonus:", bonus["val_rmse"])

        metadata_records.append(
            build_metadata_record(
                run_id=bonus["run_id"],
                run_name="hist-gradient-boosting",
                split_name=str(split_v1["name"]),
                strategy_name=best_strategy_name,
                model_type="hist_gradient_boosting",
                feature_names=best_features,
                metrics={f"val_{key}": value for key, value in bonus.items() if key.startswith("val_")},
                extra_params={"learning_rate": 0.08, "max_iter": 150, "max_leaf_nodes": 31},
                model_uri=bonus.get("model_uri"),
            )
        )

        log_json_artifact(metadata_records, "metadata_models.json", artifact_dir="metadata")
        write_metadata_database(metadata_records)
        shutil.copyfile(LOCAL_METADATA_DB, LOCAL_MLFLOW_DB)
        mlflow.log_artifact(str(LOCAL_METADATA_DB), artifact_path="metadata")
        mlflow.log_artifact(str(LOCAL_MLFLOW_DB), artifact_path="metadata")
        write_conda_environment_file()

    finally:
        try:
            mlflow.end_run()
        except Exception:
            pass


if __name__ == "__main__":
    main()