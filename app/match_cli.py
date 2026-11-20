"""Command entry point for ``python -m app.match_cli``."""

from app.cli.match_cli import main

__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
