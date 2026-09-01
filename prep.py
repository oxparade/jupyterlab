"""TP M02TP01 — prep_data: construit les features et écrit train/validation/test en parquet."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from config import DATA_DIR, DATASET_SPLIT_DATES, MODELLING_FEATURES, TARGET, ModellingStrategy, SplitStrategy
from pipeline_steps import build_features, load_data

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def main(
    input: Path = DATA_DIR / "LD2011_2014.txt",
    output: Path = DATA_DIR / "processed",
    strategy: ModellingStrategy = ModellingStrategy.MIXED,
    split_strategy: SplitStrategy = SplitStrategy.FULL_HISTORY,
) -> None:
    """Prépare et persiste les trois jeux train/validation/test."""
    raw = load_data(str(input))
    stacked = build_features(raw)

    features = MODELLING_FEATURES[strategy]
    required_columns = [*features, TARGET]
    clean = stacked.dropna(subset=required_columns)

    output.mkdir(parents=True, exist_ok=True)
    for name, (start, end) in DATASET_SPLIT_DATES[split_strategy].items():
        mask = (clean.index >= pd.Timestamp(start)) & (
            clean.index < pd.Timestamp(end) + pd.DateOffset(days=1)
        )
        part = clean.loc[mask, required_columns].copy()
        destination = output / f"{name}.parquet"
        part.to_parquet(destination)
        logger.info("%s: %d rows, %d cols -> %s", name, part.shape[0], part.shape[1], destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Préparation des datasets train/validation/test")
    parser.add_argument("--input", type=Path, default=DATA_DIR / "LD2011_2014.txt")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "processed")
    parser.add_argument(
        "--strategy",
        choices=[strategy.value for strategy in ModellingStrategy],
        default=ModellingStrategy.MIXED.value,
    )
    parser.add_argument(
        "--split-strategy",
        choices=[split.value for split in SplitStrategy],
        default=SplitStrategy.FULL_HISTORY.value,
    )
    args = parser.parse_args()
    main(
        input=args.input,
        output=args.output,
        strategy=ModellingStrategy(args.strategy),
        split_strategy=SplitStrategy(args.split_strategy),
    )
