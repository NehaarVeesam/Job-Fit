"""Command entry point for ``python -m app.search_cli``."""

from app.cli.search_cli import main

__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
