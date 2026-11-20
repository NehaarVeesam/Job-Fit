"""Command entry point for ``python -m app.check_setup``."""

from app.cli.check_setup import main

__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
