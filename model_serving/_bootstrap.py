from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_CODE_DIR = ROOT / "data" / "code"


def ensure_data_code_path() -> Path:
    data_code_dir = str(DATA_CODE_DIR)
    if data_code_dir not in sys.path:
        sys.path.insert(0, data_code_dir)
    return DATA_CODE_DIR
