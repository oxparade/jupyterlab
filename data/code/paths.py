from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT

RAW_DATA_DIR = DATASET_DIR / "raw"
PROCESSED_DATA_DIR = DATASET_DIR / "processed"
MODELLING_DATA_DIR = DATASET_DIR / "modelling"
OUTLIER_DETECTION_OUTPUT = PROJECT_ROOT / "code" / "outlier_detection" / "outputs"
