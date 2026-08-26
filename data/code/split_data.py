from __future__ import annotations

"""Entry point for dataset preparation and split materialization.

This script wraps the existing preparation logic used in `cli.py` so the lab can
be run as a plain Python script.
"""

try:  # pragma: no cover
    from .cli import main as _prepare_main
except ImportError:  # pragma: no cover
    from cli import main as _prepare_main


def main() -> None:
    _prepare_main()


if __name__ == "__main__":
    main()