"""Command entry point for ``python -m app.dataset_loader``."""

from app.corpus.dataset_loader import CorpusOptions, PreparedCorpus, main, prepare_corpus, write_corpus

__all__ = ["CorpusOptions", "PreparedCorpus", "main", "prepare_corpus", "write_corpus"]


if __name__ == "__main__":
    raise SystemExit(main())
