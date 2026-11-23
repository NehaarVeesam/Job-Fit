"""Rank prepared job descriptions with BM25 keyword search."""

import re
from pathlib import Path

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app.retrieval.documents import load_job_documents
from app.retrieval.filters import matches_preferences
from app.schemas import Preferences

TOKEN_PATTERN = re.compile(r"\.?\w+(?:[./-]\w+)*(?:\+\+|#)?")


def tokenize(text: str) -> list[str]:
    """Normalize casing and preserve punctuation inside technical terms."""
    return TOKEN_PATTERN.findall(text.casefold())


def search_bm25(
    query: str,
    top_k: int,
    jobs_path: Path,
    *,
    preferences: Preferences | None = None,
) -> tuple[int, list[tuple[Document, float]]]:
    """Return the catalog size and ranked document-score pairs."""
    if not query.strip():
        raise ValueError("The query must contain some text.")
    if top_k < 1:
        raise ValueError("--top-k must be at least 1.")

    documents = load_job_documents(jobs_path)
    query_tokens = tokenize(query)
    corpus = [tokenize(document.page_content) for document in documents]

    if not query_tokens or not any(corpus):
        return len(documents), []

    scores = BM25Okapi(corpus).get_scores(query_tokens)
    query_terms = set(query_tokens)

    # Keep jobs that share a query term and match the supplied filters.
    hits = [
        (document, float(score))
        for document, tokens, score in zip(documents, corpus, scores, strict=True)
        if query_terms.intersection(tokens)
        and matches_preferences(document, preferences)
    ]
    hits.sort(key=lambda hit: hit[1], reverse=True)

    return len(documents), hits[:top_k]
