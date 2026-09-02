"""Module 3 demo pipeline: importer -> load_data -> build_features.

Self-contained components compatible with lightweight KFP v2 execution.
"""

from enum import StrEnum
from typing import NamedTuple

from kfp import dsl
from kfp.dsl import Dataset, Input

DATASET_URI = "s3://models/datasets/LD2011_2014_kwh.parquet"


class Feature(StrEnum):
    LAG_1D = "lag_1d"
    LAG_7D = "lag_7d"
    LAG_30D = "lag_30d"


class FeatureSet(StrEnum):
    LAG_ONLY = "lag_only"
    FULL = "full"


FEATURE_SETS: dict[FeatureSet, list[Feature]] = {
    FeatureSet.LAG_ONLY: [Feature.LAG_1D],
    FeatureSet.FULL: [Feature.LAG_1D, Feature.LAG_7D],
}


def feature_columns(feature_set: FeatureSet) -> list[str]:
    return sorted(map(str, FEATURE_SETS[feature_set]))


@dsl.component(base_image="python:3.14-slim-trixie", packages_to_install=["pandas", "pyarrow"])
def load_data(dataset: Input[Dataset]) -> dsl.Dataset:
    import pandas as pd

    dataframe = pd.read_parquet(dataset.path).set_index("timestamp")
    out_data = dsl.Dataset(uri=dsl.get_uri())
    dataframe.to_parquet(out_data.path)
    return out_data


@dsl.component(base_image="python:3.14-slim-trixie", packages_to_install=["pandas", "pyarrow"])
def build_features(in_data: Input[Dataset], features: list[str]) -> NamedTuple(
    "Features", [("out_data", Dataset), ("n_rows", int)]
):
    import pandas as pd

    class Features(NamedTuple):
        out_data: Dataset
        n_rows: int

    lag_steps = {"lag_1d": 96, "lag_7d": 96 * 7, "lag_30d": 96 * 30}

    dataframe = pd.read_parquet(in_data.path)
    long = dataframe.stack().rename("consumption").reset_index()
    long.columns = ["timestamp", "client", "consumption"]
    long = long.sort_values(["client", "timestamp"]).reset_index(drop=True)

    grouped = long.groupby("client")["consumption"]
    for feature_name in features:
        long[feature_name] = grouped.shift(lag_steps[feature_name])

    long = long.dropna().reset_index(drop=True)
    out_data = Dataset(uri=dsl.get_uri("out_data"))
    long.to_parquet(out_data.path)
    return Features(out_data=out_data, n_rows=int(long.shape[0]))


@dsl.pipeline(name="electricity-first-pipeline")
def first_pipeline(features: list[str] = feature_columns(FeatureSet.FULL)):
    source = dsl.importer(artifact_uri=DATASET_URI, artifact_class=Dataset, reimport=False)
    load_task = load_data(dataset=source.output)
    build_features(in_data=load_task.output, features=features)
