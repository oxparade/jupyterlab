from __future__ import annotations

from pathlib import Path

try:  # pragma: no cover
    from . import ml_pipeline
except ImportError:  # pragma: no cover
    import ml_pipeline


def main() -> None:
    frame = ml_pipeline.load_modeling_frame()
    required_columns = sorted({ml_pipeline.TARGET_COLUMN, *ml_pipeline.DEFAULT_FEATURES})
    frame = frame.dropna(subset=required_columns)

    split_v1 = ml_pipeline.build_temporal_split(frame, split_name="v1")
    split_v2 = ml_pipeline.build_temporal_split(frame, split_name="v2")

    output_dir = Path("data/splits")
    ml_pipeline.materialize_split(split_v1, output_dir=output_dir)
    ml_pipeline.materialize_split(split_v2, output_dir=output_dir)


if __name__ == "__main__":
    main()
