"""Command entry point for ``python -m app.prepare_resume``."""

from app.cli.prepare_resume import main

__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
