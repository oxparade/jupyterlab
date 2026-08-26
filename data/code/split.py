from __future__ import annotations

from typing import Generator

import numpy as np
import pandas as pd
from sklearn import model_selection


TARGET_COLUMN = "consumption_kwh"


def split_by_dates(
    df: pd.DataFrame,
    dates: list[tuple[int, ...]],
    timestamp_level: str = "timestamp",
) -> Generator[pd.DataFrame, None, None]:
    for date_group in dates:
        yield df[np.isin(df.index.get_level_values(timestamp_level).year, date_group)]


def split(
    df: pd.DataFrame,
    test_size: float,
    target_column: str = TARGET_COLUMN,
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_features_train, df_features_test, df_target_train, df_target_test = model_selection.train_test_split(
        df.drop(columns=[target_column]),
        df[target_column],
        test_size=test_size,
        **kwargs,
    )
    return (
        df_features_train.join(df_target_train),
        df_features_test.join(df_target_test),
    )
