from __future__ import annotations

import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "model_serving.app:app",
        host=os.getenv("TP03_HOST", "127.0.0.1"),
        port=int(os.getenv("TP03_PORT", "8000")),
        reload=os.getenv("TP03_RELOAD", "false").lower() in {"1", "true", "yes"},
    )


if __name__ == "__main__":
    main()
