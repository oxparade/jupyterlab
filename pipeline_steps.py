from pathlib import Path
from typing import List, Tuple, Dict, Optional

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge


def load_data(path: Optional[str] = None, nrows: Optional[int] = None) -> pd.DataFrame:
    """Load the dataset and convert to kWh per 15-min interval.

    Defaults to data/LD2011_2014.txt if no path provided.
    """
    if path is None:
        path = Path("data") / "LD2011_2014.txt"
    df = pd.read_csv(path, sep=";", decimal=",", index_col=0, parse_dates=True, nrows=nrows)
    df = df.astype("float32")
    # convert average kW over 15-min interval -> kWh in that interval
    df = df / 4.0
    return df


_PERIODS_PER_DAY = 96  # 15-min intervals in a day


def _periods_to_label(periods: int) -> str:
    """Convert a lag/window expressed in 15-min periods to a human-readable label (e.g. '7d')."""
    days = periods // _PERIODS_PER_DAY
    return f"{days}d"


def build_features(
    df: pd.DataFrame,
    lags: Tuple[int, ...] = (96, 96 * 7, 96 * 30, 96 * 365),
    rolling_windows: Tuple[int, ...] = (96 * 7, 96 * 30),
) -> pd.DataFrame:
    """Construct a stacked long DataFrame with features for every client.

    Column names are derived from *lags* and *rolling_windows*:
      - lag_{n}d          for each lag of n days
      - rolling_mean_{n}d for each rolling window of n days
    Index is the timestamp; rows are stacked for all clients.
    """
    frames = []
    for name in df.columns:
        one = df[[name]].copy()
        one.columns = ["consumption"]
        for lag in lags:
            one[f"lag_{_periods_to_label(lag)}"] = one["consumption"].shift(lag)
        # rolling means are computed on shifted consumption so the target cannot peek
        shifted = one["consumption"].shift(1)
        for window in rolling_windows:
            one[f"rolling_mean_{_periods_to_label(window)}"] = (
                shifted.rolling(window).mean().astype("float32")
            )
        one["client"] = name
        frames.append(one)

    tmp = pd.concat(frames).sort_index()
    tmp["client"] = tmp["client"].astype("category")
    return tmp


def split_chronological(data: pd.DataFrame, train_frac: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split by timestamp (no timestamp appears in both splits).

    We split on unique timestamps so that all client rows for a given timestamp
    belong to the same side of the split.
    """
    times = data.index.unique().sort_values()
    if len(times) < 2:
        raise ValueError("Not enough distinct timestamps to split")
    cutoff = int(len(times) * train_frac)
    if cutoff < 1:
        cutoff = 1
    if cutoff >= len(times):
        cutoff = len(times) - 1
    cutoff_time = times[cutoff - 1]
    train = data.loc[data.index <= cutoff_time]
    test = data.loc[data.index > cutoff_time]
    return train, test


def get_X_y(df: pd.DataFrame, features: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    X = df[features]
    y = df["consumption"]
    return X, y


def train_ridge(X: pd.DataFrame, y: pd.Series, alpha: float = 1.0) -> Ridge:
    model = Ridge(alpha=alpha)
    model.fit(X, y)
    return model


def _estimate_warmup_periods(features: List[str]) -> int:
    """Estimate minimum history (in 15-min periods) required by lag/rolling features."""
    warmup = 1
    for name in features:
        if name.startswith("lag_"):
            days = int(name.replace("lag_", "").replace("d", ""))
            warmup = max(warmup, days * _PERIODS_PER_DAY)
        elif name.startswith("rolling_mean_"):
            days = int(name.replace("rolling_mean_", "").replace("d", ""))
            warmup = max(warmup, days * _PERIODS_PER_DAY + 1)
    return warmup


def select_best_alpha_time_cv(
    data: pd.DataFrame,
    features: List[str],
    alphas: Tuple[float, ...] = (0.1, 1.0, 10.0),
    n_splits: int = 3,
) -> Tuple[float, Dict[float, float]]:
    """Select alpha with expanding-window CV, preserving chronology.

    The validation windows are always strictly after each training window.
    """
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")

    times = data.index.unique().sort_values()
    warmup_periods = _estimate_warmup_periods(features)
    if len(times) <= warmup_periods + n_splits:
        raise ValueError("Not enough timestamps for time-series CV with requested splits")

    usable_times = times[warmup_periods:]
    block = len(usable_times) // (n_splits + 1)
    if block < 1:
        raise ValueError("Not enough timestamps to build expanding CV folds")

    per_alpha_scores: Dict[float, List[float]] = {float(alpha): [] for alpha in alphas}

    for fold in range(1, n_splits + 1):
        train_end_idx = fold * block
        val_end_idx = (fold + 1) * block if fold < n_splits else len(usable_times)

        train_end_time = usable_times[train_end_idx - 1]
        val_start_time = usable_times[train_end_idx]
        val_end_time = usable_times[val_end_idx - 1]

        train_fold = data.loc[data.index <= train_end_time]
        val_fold = data.loc[(data.index >= val_start_time) & (data.index <= val_end_time)]

        clean_train = train_fold.dropna(subset=features + ["consumption"])
        clean_val = val_fold.dropna(subset=features + ["consumption"])
        if clean_train.empty or clean_val.empty:
            raise ValueError(
                "A CV fold is empty after dropna; increase data size or reduce split count"
            )

        Xtr, ytr = get_X_y(clean_train, features)
        Xva, yva = get_X_y(clean_val, features)

        for alpha in alphas:
            model = train_ridge(Xtr, ytr, alpha=float(alpha))
            rmse = evaluate(model, Xva, yva)["rmse"]
            per_alpha_scores[float(alpha)].append(rmse)

    mean_rmse = {alpha: float(np.mean(scores)) for alpha, scores in per_alpha_scores.items()}
    best_alpha = min(mean_rmse, key=mean_rmse.get)
    return best_alpha, mean_rmse


def evaluate(model, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    pred = model.predict(X)
    errors = y.to_numpy() - pred
    rmse = float(np.sqrt((errors ** 2).mean()))
    mae = float(np.abs(errors).mean())
    return {"rmse": rmse, "mae": mae}


def save_model(
    model,
    artifact_path: str = "model",
    *,
    input_example: Optional[pd.DataFrame] = None,
    registered_model_name: Optional[str] = None,
) -> str:
    import mlflow
    from mlflow.models import infer_signature
    from mlflow.sklearn import log_model as log_sklearn_model

    if mlflow.active_run() is None:
        raise RuntimeError("An active MLflow run is required to log the model.")

    log_kwargs: Dict[str, object] = {"name": artifact_path}
    if input_example is not None and not input_example.empty:
        predictions = model.predict(input_example)
        log_kwargs["input_example"] = input_example
        log_kwargs["signature"] = infer_signature(input_example, predictions)
    if registered_model_name:
        log_kwargs["registered_model_name"] = registered_model_name

    model_info = log_sklearn_model(model, **log_kwargs)
    return str(model_info.model_uri)
