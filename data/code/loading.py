from pathlib import Path

import pandas as pd


def read_raw_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, delimiter=";", index_col=0, decimal=",")
    df.index.name = "timestamp"
    return df
