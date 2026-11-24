"""Measure retrieval quality when only some jobs have judgments."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import log2

from app.evaluation.fixtures import JobJudgment


@dataclass(frozen=True)
class RetrievalMetrics:
    """Ranking metrics and the number of results they are based on."""

    returned_count: int
    judged_count: int
    judged_precision: float | None
    known_relevant_recall: float | None
    judged_ndcg: float | None


def _dcg(grades: Sequence[int]) -> float:
    """Give higher grades more credit when they appear earlier."""
    return sum(
        grade / log2(rank + 1)
        for rank, grade in enumerate(grades, start=1)
    )


def evaluate_ranking(
    ranked_ids: Sequence[str],
    judgments: Mapping[str, JobJudgment],
    top_k: int,
) -> RetrievalMetrics:
    """Score the judged results within the first top_k returned positions."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    top_ids = ranked_ids[:top_k]
    grades = [
        judgments[job_id].relevance
        for job_id in top_ids
        if job_id in judgments
    ]
    relevant_hits = sum(grade > 0 for grade in grades)
    known_relevant = sum(item.relevance > 0 for item in judgments.values())

    # Compare the judged results with the best labels at the same list length.
    ideal_grades = sorted(
        (item.relevance for item in judgments.values()),
        reverse=True,
    )[:len(grades)]
    ideal_gain = _dcg(ideal_grades)

    return RetrievalMetrics(
        returned_count=len(top_ids),
        judged_count=len(grades),
        judged_precision=relevant_hits / len(grades) if grades else None,
        known_relevant_recall=relevant_hits / known_relevant if known_relevant else None,
        judged_ndcg=_dcg(grades) / ideal_gain if ideal_gain else None,
    )

def panel_ndcg_at_k(
    ranked_ids: Sequence[str],
    grades: Mapping[str, int],
    top_k: int,
) -> float | None:
    """Compare the returned ranking with the ideal ranking on a labeled panel."""
    observed = [grades[job_id] for job_id in ranked_ids[:top_k]]
    ideal = sorted(grades.values(), reverse=True)[:top_k]
    ideal_gain = _dcg(ideal)

    return _dcg(observed) / ideal_gain if ideal_gain else None