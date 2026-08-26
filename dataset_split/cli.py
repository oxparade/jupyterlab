from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_CODE_DIR = ROOT / "data" / "code"
if str(DATA_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_CODE_DIR))

try:  # pragma: no cover
    import ml_pipeline
except ImportError:  # pragma: no cover
    from data.code import ml_pipeline  # type: ignore[no-redef]

from .constants import HEAD_ROWS, RANDOM_SEED

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

OUTPUT_DIR = ROOT / "dataset"


def _write_head_csv(frame: pd.DataFrame, path: Path) -> None:
    sample_size = min(HEAD_ROWS, len(frame))
    sampled = frame.sample(n=sample_size, random_state=RANDOM_SEED).sort_index()
    sampled.to_csv(path)


def main() -> None:
    logger.info("Loading modelling frame")
    frame = ml_pipeline.load_modeling_frame()
    required_columns = sorted({ml_pipeline.TARGET_COLUMN, *ml_pipeline.DEFAULT_FEATURES})
    frame = frame.dropna(subset=required_columns)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    split_v1 = ml_pipeline.build_temporal_split(frame, split_name="v1")
    logger.info("Writing full split parquets and seed-controlled head CSVs")

    for split_name in ("train", "validation", "test"):
        split_frame = split_v1[split_name]
        split_frame.to_parquet(OUTPUT_DIR / f"{split_name}.parquet")
        _write_head_csv(split_frame, OUTPUT_DIR / f"{split_name}_head.csv")

    logger.info("Dataset split written in %s", OUTPUT_DIR)
    logger.info("RANDOM_SEED=%s", RANDOM_SEED)


if __name__ == "__main__":
    main()
