"""Combine dense and BM25 job rankings with Reciprocal Rank Fusion."""

from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from langchain_core.documents import Document

from app.retrieval.dense import search_dense
from app.retrieval.keyword_search import search_bm25
from app.schemas import Preferences


SearchMode = Literal["dense", "bm25", "hybrid"]
RankedHits = list[tuple[Document, float]]
RRF_CONSTANT = 60
CANDIDATE_MULTIPLIER = 3


def fuse_rankings(rankings: Iterable[RankedHits], top_k: int) -> RankedHits:
    """Add rank contributions by job ID and return the top fused hits."""
    documents: dict[str, Document] = {}
    scores: dict[str, float] = {}

    for ranking in rankings:
        # Rank starts at one for each retrieval method, and each job has one total.
        for rank, (document, _) in enumerate(ranking, start=1):
            job_id = document.metadata["job_id"]
            documents.setdefault(job_id, document)
            scores[job_id] = scores.get(job_id, 0.0) + 1 / (RRF_CONSTANT + rank)

    # Break equal scores by job ID so the order is repeatable.
    ranked_ids = sorted(scores, key=lambda job_id: (-scores[job_id], job_id))
    return [(documents[job_id], scores[job_id]) for job_id in ranked_ids[:top_k]]


def retrieve_candidates(
    query: str,
    top_k: int,
    jobs_path: Path,
    *,
    mode: SearchMode = "hybrid",
    preferences: Preferences | None = None,
) -> tuple[int, RankedHits]:
    """Return the catalog size and ranked hits for the selected search mode."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if mode == "bm25":
        return search_bm25(query, top_k, jobs_path, preferences=preferences)
    if mode == "dense":
        return search_dense(query, top_k, preferences=preferences)
    if mode != "hybrid":
        raise ValueError(f"Unknown search mode: {mode!r}")

    # Give both methods room to contribute before taking the final top_k.
    candidate_limit = top_k * CANDIDATE_MULTIPLIER
    catalog_size, bm25_hits = search_bm25(
        query, candidate_limit, jobs_path, preferences=preferences
    )
    indexed_jobs, dense_hits = search_dense(
        query, candidate_limit, preferences=preferences
    )

    if indexed_jobs != catalog_size:
        raise ValueError("The index and catalog have different job counts. Rebuild the index.")

    return catalog_size, fuse_rankings((dense_hits, bm25_hits), top_k)
