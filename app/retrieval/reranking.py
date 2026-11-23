"""Score and reorder shortlisted jobs with a cross-encoder."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from torch import nn


RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
RERANKER_MAX_LENGTH = 512


def create_reranker() -> CrossEncoder:
    """Load the reranker on CPU and return raw relevance scores."""
    return CrossEncoder(
        RERANKER_MODEL_NAME,
        device="cpu",
        # The token limit applies to the query and description together.
        max_length=RERANKER_MAX_LENGTH,
        # Keep the model's raw score for ranking the pairs.
        activation_fn=nn.Identity(),
    )


@dataclass(frozen=True)
class RerankedHit:
    """A shortlisted job with its original rank and both scores."""

    document: Document
    retrieval_rank: int
    retrieval_score: float
    rerank_score: float


def rerank_hits(
    query: str,
    hits: Sequence[tuple[Document, float]],
    reranker: CrossEncoder,
    *,
    batch_size: int = 8,
) -> list[RerankedHit]:
    """Score each query–job pair and sort the shortlist by its new score."""
    query = query.strip()
    if not query:
        raise ValueError("The query must contain some text.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if not hits:
        return []

    pairs = [(query, document.page_content) for document, _ in hits]
    model_scores = reranker.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=False,
    )

    reranked: list[RerankedHit] = []
    # Require one model score per job before pairing scores with retrieval ranks.
    scored_hits = zip(hits, model_scores, strict=True)

    for rank, ((document, retrieval_score), model_score) in enumerate(
        scored_hits, start=1
    ):
        rerank_score = float(model_score)
        if not isfinite(rerank_score):
            raise ValueError(f"Invalid reranker score for job at rank {rank}.")

        reranked.append(
            RerankedHit(
                document=document,
                retrieval_rank=rank,
                retrieval_score=retrieval_score,
                rerank_score=rerank_score,
            )
        )

    # Equal reranker scores keep their original retrieval order.
    return sorted(reranked, key=lambda hit: hit.rerank_score, reverse=True)
