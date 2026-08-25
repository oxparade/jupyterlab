import pandas as pd


def normalize_consumption_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df.index = pd.to_datetime(df.index, format="%Y-%m-%d %H:%M:%S")
    df_long = df.reset_index().melt(
        id_vars="timestamp",
        var_name="individual",
        value_name="consumption"
    )
    df_long["consumption_kwh"] = df_long["consumption"] / 4
    df_long["year_month"] = df_long["timestamp"].dt.to_period("M").astype(str)
    return df_long
