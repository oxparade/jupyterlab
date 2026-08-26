import logging

import pandas as pd

try:
    from . import processing as common_processing
    from . import loading as common_loading
    from . import paths
    from . import constants
except ImportError:
    import processing as common_processing
    import loading as common_loading
    import paths
    import constants

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)




def main():
    raw_file_path = paths.DATASET_DIR / "LD2011_2014.txt"
    if not raw_file_path.exists():
        raw_file_path = paths.RAW_DATA_DIR / "raw.txt"

    logger.info("Starting dataset preparation")
    logger.info(f"Reading raw dataset from {raw_file_path}")

    df = common_loading.read_raw_data(raw_file_path)

    logger.info(f"Raw dataset loaded with shape={df.shape}")
    logger.info("Normalizing consumption dataset")

    df_normalized = common_processing.normalize_consumption_dataset(df)

    filtered_individuals_file_path = paths.PROCESSED_DATA_DIR / "filtered_individuals.csv"
    logger.info(f"Load filtered individuals from {filtered_individuals_file_path}")
    kept_individuals: pd.DataFrame = pd.read_csv(filtered_individuals_file_path)

    logger.info(f"Filtering dataset to keep only individuals in {filtered_individuals_file_path}")
    df_features = df_normalized[df_normalized["individual"].isin(kept_individuals["individual"])].sort_values(["individual", "timestamp"]).copy()
    assert (
        df_features.groupby("individual")["timestamp"]
        .diff()
        .dropna()
        .dt.total_seconds()
        .eq(15 * 60)
        .all()
    ), "Dataset is not sampled at 15 minutes intervals. Lag features will be incorrect."
    assert (
        df_features
        .groupby("individual")["timestamp"]
        .apply(lambda s: s.is_monotonic_increasing)
        .all()
    ), "Dataset is not sorted by timestamp. Lag features will be incorrect."

    logger.info(f"Compute lag features: {list(constants.LAG_FEATURES.keys())}")
    consumption_by_individuals = df_features.groupby("individual")["consumption_kwh"]
    for lag_feature, nbr_intervals in constants.LAG_FEATURES.items():
        df_features[lag_feature] = consumption_by_individuals.shift(nbr_intervals)

    logger.info(f"Compute rolling features: {list(constants.ROLLING_MEAN_FEATURES.keys())}")
    for rolling_mean_feature, nbr_intervals in constants.ROLLING_MEAN_FEATURES.items():
        df_features[rolling_mean_feature] = consumption_by_individuals.rolling(nbr_intervals).mean().reset_index(level=0, drop=True)

    df_features = df_features.set_index(["individual", "timestamp"])
    df_features.drop(columns=["consumption", "consumption_kwh", "year_month"]).to_parquet(paths.MODELLING_DATA_DIR / "features.parquet")
    df_features[["consumption_kwh"]].to_parquet(paths.MODELLING_DATA_DIR / "target.parquet")


if __name__ == "__main__":
    main()