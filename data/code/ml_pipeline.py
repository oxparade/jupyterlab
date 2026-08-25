from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

try:
    from . import paths
except ImportError:  # pragma: no cover
    import paths

TARGET_COLUMN = "consumption_kwh"
DEFAULT_FEATURES = [
    "lag_1d",
    "lag_7d",
    "lag_30d",
    "lag_365d",
    "rolling_mean_7d",
    "rolling_mean_30d",
]


def _validate_multiindex(frame: pd.DataFrame, label: str) -> None:
    if not isinstance(frame.index, pd.MultiIndex):
        raise TypeError(f"{label} index is not a MultiIndex: individual + timestamp expected.")
    expected_names = ["individual", "timestamp"]
    if list(frame.index.names) != expected_names:
        frame.index = frame.index.set_names(expected_names)


def load_modeling_frame(
    features_path: str | Path | None = None,
    target_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load the feature and target parquet files and join them on the MultiIndex."""
    if features_path is None:
        features_path = paths.MODELLING_DATA_DIR / "features.parquet"
    if target_path is None:
        target_path = paths.MODELLING_DATA_DIR / "target.parquet"

    features = pd.read_parquet(features_path)
    target = pd.read_parquet(target_path)

    _validate_multiindex(features, "FEATURES")
    _validate_multiindex(target, "TARGET")

    frame = features.join(target, how="inner")
    if TARGET_COLUMN not in frame.columns:
        raise KeyError(f"Target column '{TARGET_COLUMN}' not found after joining feature and target tables.")

    return frame.sort_index()


def _build_split_v1(frame: pd.DataFrame) -> Dict[str, Any]:
    ts: pd.DatetimeIndex = pd.DatetimeIndex(frame.index.get_level_values("timestamp"))
    years: pd.Index = ts.year
    train_mask: pd.Series | np.ndarray = np.isin(years, [2011, 2012])
    validation_mask: pd.Series | np.ndarray = years == 2013
    test_mask: pd.Series | np.ndarray = years == 2014

    train_df: pd.DataFrame = frame.loc[train_mask].copy()
    validation_df: pd.DataFrame = frame.loc[validation_mask].copy()
    test_df: pd.DataFrame = frame.loc[test_mask].copy()

    return {
        "name": "split_v1_2011-2012_2013_2014",
        "description": "train=2011-2012 | validation=2013 | test=2014",
        "train": train_df,
        "validation": validation_df,
        "test": test_df,
    }


def _build_split_v2(frame: pd.DataFrame, cutoff: str = "2014-07-01") -> Dict[str, Any]:
    ts: pd.DatetimeIndex = pd.DatetimeIndex(frame.index.get_level_values("timestamp"))
    years: pd.Index = ts.year
    cutoff_ts: pd.Timestamp = pd.Timestamp(cutoff)

    train_mask: pd.Series | np.ndarray = years == 2013
    validation_mask: pd.Series | np.ndarray = (ts >= pd.Timestamp("2014-01-01")) & (ts < cutoff_ts)
    test_mask: pd.Series | np.ndarray = ts >= cutoff_ts

    train_df: pd.DataFrame = frame.loc[train_mask].copy()
    validation_df: pd.DataFrame = frame.loc[validation_mask].copy()
    test_df: pd.DataFrame = frame.loc[test_mask].copy()

    return {
        "name": "split_v2_2013_2014H1_2014H2",
        "description": f"train=2013 | validation=2014-01-01..{cutoff_ts.date()} | test={cutoff_ts.date()}..fin-2014",
        "train": train_df,
        "validation": validation_df,
        "test": test_df,
    }


def build_temporal_split(frame: pd.DataFrame, split_name: str = "v1", cutoff: str = "2014-07-01") -> Dict[str, pd.DataFrame]:
    """Create a train/validation/test split on a time-based MultiIndex."""
    _validate_multiindex(frame, "FRAME")
    if split_name == "v1":
        return _build_split_v1(frame)
    if split_name == "v2":
        return _build_split_v2(frame, cutoff=cutoff)
    raise ValueError(f"Unsupported split_name={split_name!r}. Expected 'v1' or 'v2'.")


def build_time_series_cv(
    frame: pd.DataFrame,
    n_splits: int = 3,
    gap: int = 0,
    min_train_size: int | None = None,
    min_valid_size: int | None = None,
) -> List[Dict[str, Any]]:
    """Build time-ordered CV folds using timestamp as the temporal axis."""
    _validate_multiindex(frame, "FRAME")
    timestamps = pd.DatetimeIndex(frame.index.get_level_values("timestamp").drop_duplicates().sort_values())
    if len(timestamps) < n_splits + 1:
        raise ValueError(
            f"Not enough unique timestamps ({len(timestamps)}) to build {n_splits} folds. "
            "Need at least n_splits + 1 different timestamps."
        )

    if min_train_size is None:
        min_train_size = max(1, len(timestamps) // (n_splits + 2))
    if min_valid_size is None:
        min_valid_size = max(1, len(timestamps) // (n_splits + 2))

    splitter = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    unique_positions = np.arange(len(timestamps))
    folds: List[Dict[str, Any]] = []

    for fold_id, (train_idx, valid_idx) in enumerate(splitter.split(unique_positions), start=1):
        train_times = timestamps[train_idx]
        valid_times = timestamps[valid_idx]

        if len(train_times) < min_train_size or len(valid_times) < min_valid_size:
            continue

        train_frame = frame[frame.index.get_level_values("timestamp").isin(train_times)].copy()
        valid_frame = frame[frame.index.get_level_values("timestamp").isin(valid_times)].copy()

        folds.append(
            {
                "fold": fold_id,
                "name": f"cv_fold_{fold_id}",
                "description": (
                    f"fold={fold_id} | train={train_times.min().date()}..{train_times.max().date()} "
                    f"| validation={valid_times.min().date()}..{valid_times.max().date()}"
                ),
                "train": train_frame,
                "validation": valid_frame,
            }
        )

    if not folds:
        raise ValueError("No valid CV folds were created. Check n_splits, gap, and dataset size.")

    return folds


def make_X_y(
    split: Mapping[str, pd.DataFrame],
    feature_names: Iterable[str] | None = None,
    target_column: str = TARGET_COLUMN,
) -> Dict[str, Any]:
    """Return X_train, y_train, X_valid, y_valid, X_test, y_test."""
    if feature_names is None:
        feature_names = DEFAULT_FEATURES
    feature_names = list(feature_names)

    missing_train = [name for name in feature_names if name not in split["train"].columns]
    missing_valid = [name for name in feature_names if name not in split["validation"].columns]
    if missing_train or missing_valid:
        missing = sorted(set(missing_train) | set(missing_valid))
        raise KeyError(f"Missing feature(s) in split: {missing}")

    X_train = split["train"][feature_names]
    y_train = split["train"][target_column]
    X_valid = split["validation"][feature_names]
    y_valid = split["validation"][target_column]

    result: Dict[str, Any] = {
        "X_train": X_train,
        "y_train": y_train,
        "X_validation": X_valid,
        "y_validation": y_valid,
    }

    if "test" in split:
        missing_test = [name for name in feature_names if name not in split["test"].columns]
        if missing_test:
            raise KeyError(f"Missing feature(s) in test split: {sorted(set(missing_test))}")
        result["X_test"] = split["test"][feature_names]
        result["y_test"] = split["test"][target_column]

    return result


def maybe_sample(frame: pd.DataFrame, max_rows: int | None = None, seed: int = 42) -> pd.DataFrame:
    """Randomly subsample a frame for smoke tests only."""
    if max_rows is None or len(frame) <= max_rows:
        return frame
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(frame), size=max_rows, replace=False)
    return frame.iloc[idx].copy()


def materialize_split(split: Mapping[str, pd.DataFrame], output_dir: str | Path) -> Path:
    """Write a temporal split to parquet files for later DVC versioning."""
    out_dir: Path = Path(output_dir) / str(split["name"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for part in ("train", "validation", "test"):
        split[part].to_parquet(out_dir / f"{part}.parquet", compression="zstd")

    manifest: Dict[str, Any] = {
        "name": split["name"],
        "description": split["description"],
        "rows": {part: int(len(split[part])) for part in ("train", "validation", "test")},
        "columns": list(split["train"].columns),
    }
    manifest_path: Path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out_dir


def main() -> None:
    df = load_modeling_frame()
    split_v1 = build_temporal_split(df, split_name="v1")
    split_v2 = build_temporal_split(df, split_name="v2")
    cv_folds = build_time_series_cv(df, n_splits=3)

    print("Joined frame shape:", df.shape)
    print("MultiIndex names:", df.index.names)

    for split in (split_v1, split_v2):
        print(f"\n{split['description']}")
        for part in ("train", "validation", "test"):
            print(f"  {part:10s}: {len(split[part]):,} rows")

    print(f"\nTime-series CV folds: {len(cv_folds)}")
    for fold in cv_folds:
        print(f"  {fold['description']}")
        print(f"    train rows={len(fold['train']):,} | validation rows={len(fold['validation']):,}")

    Xy_v1 = make_X_y(split_v1)
    print("\nX_train shape:", Xy_v1["X_train"].shape)
    print("y_train shape:", Xy_v1["y_train"].shape)


if __name__ == "__main__":
    main()
