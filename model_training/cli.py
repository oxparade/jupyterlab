from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_CODE_DIR = ROOT / "data" / "code"
if str(DATA_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_CODE_DIR))

from train_model import main  # type: ignore[import-not-found]


if __name__ == "__main__":
    main()
