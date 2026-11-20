"""Command entry point for ``python -m app.dense_cli``."""

from app.cli.dense_cli import main

__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
