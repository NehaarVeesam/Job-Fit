"""Compare four retrieval variants on the original dev-branch benchmark."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory
from typing import Literal

from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder

from app.config import Settings
from app.dataset.dataset_loader import load_jobs, write_jsonl
from app.evaluation.fixtures import EvaluationCase
from app.evaluation.metrics import panel_ndcg_at_k
from app.matching.matcher import match_candidates
from app.resumes.profile import load_skill_vocabulary, prepare_resume
from app.retrieval.dense import search_dense
from app.retrieval.hybrid import CANDIDATE_MULTIPLIER, RankedHits, fuse_rankings
from app.retrieval.keyword_search import search_bm25
from app.retrieval.reranking import create_reranker, rerank_hits
from app.retrieval.vector_index import build_index
from app.schemas import ParsedResume, PreparedResume, ResumePage


TOP_K = 5
VARIANTS = ("bm25", "dense", "hybrid", "hybrid_reranked")


class PanelCase(EvaluationCase):
    """One synthetic resume with compact suitability labels."""

    split: Literal["dev", "heldout"]
    suitable: set[str]
    partial: set[str]

    def grades(self, panel_ids: list[str]) -> dict[str, int]:
        """Expand the case into one grade per panel job."""
        if self.suitable & self.partial:
            raise ValueError(f"Overlapping grades for {self.case_id}.")
        if (self.suitable | self.partial) - set(panel_ids):
            raise ValueError(f"Unknown panel job in {self.case_id}.")
        if self.expected_no_match == bool(self.suitable):
            raise ValueError(f"No-match expectation conflicts with {self.case_id}.")

        return {
            job_id: 2 if job_id in self.suitable else int(job_id in self.partial)
            for job_id in panel_ids
        }


class PanelSuite(BaseModel):
    """The fixed job panel and resume cases."""

    panel_job_ids: list[str] = Field(min_length=17, max_length=17)
    cases: list[PanelCase] = Field(min_length=12, max_length=12)


@dataclass(frozen=True)
class Outcome:
    """Scores and outcomes for one case and one retrieval variant."""

    ndcg: float | None
    accepted_grades: list[int]
    no_match_correct: bool


def _prepare(case: PanelCase, vocabulary: list[str]) -> PreparedResume:
    """Build the usual JobFit profile from the synthetic resume text."""
    pages = [
        ResumePage(page=number, text=text)
        for number, text in enumerate(case.resume_pages, start=1)
    ]
    return prepare_resume(ParsedResume(pages=pages), vocabulary, case.preferences)


def _rank(
    prepared: PreparedResume,
    jobs_path: Path,
    index_path: Path,
    reranker: CrossEncoder,
) -> dict[str, RankedHits]:
    """Retrieve and rerank candidates from the same job panel."""
    query = prepared.retrieval_query
    if query is None:
        return {variant: [] for variant in VARIANTS}

    limit = TOP_K * CANDIDATE_MULTIPLIER
    _, keywords = search_bm25(query, limit, jobs_path, preferences=prepared.preferences)
    _, dense = search_dense(query, limit, index_path, preferences=prepared.preferences)

    # Compare the same hybrid pool before and after cross-encoder scoring.
    hybrid = fuse_rankings((dense, keywords), limit)
    reranked = rerank_hits(query, hybrid, reranker)[:TOP_K]

    return {
        "bm25": keywords[:TOP_K],
        "dense": dense[:TOP_K],
        "hybrid": hybrid[:TOP_K],
        "hybrid_reranked": [(hit.document, hit.rerank_score) for hit in reranked],
    }


def _print_summary(outcomes: dict[str, list[Outcome]]) -> None:
    """Summarize retrieval, strict accepted precision, and no-match behavior."""
    print("\nVariant | Mean suitability nDCG@5 | Strict accepted precision | Match outcomes")
    for variant, rows in outcomes.items():
        ndcg = [row.ndcg for row in rows if row.ndcg is not None]
        accepted = [grade for row in rows for grade in row.accepted_grades]
        ndcg_text = f"{mean(ndcg):.3f}" if ndcg else "n/a"
        strict_text = (
            f"{sum(grade == 2 for grade in accepted) / len(accepted):.3f}"
            if accepted
            else "n/a"
        )
        print(
            f"{variant} | {ndcg_text} | {strict_text} "
            f"| {sum(row.no_match_correct for row in rows)}/{len(rows)}"
        )


def run_benchmark(split: str, settings: Settings) -> None:
    """Evaluate the requested split and print disagreements for review."""
    suite = PanelSuite.model_validate_json(
        Path("evaluation/panel_benchmark.json").read_text(encoding="utf-8")
    )
    if len(set(suite.panel_job_ids)) != 17:
        raise ValueError("Panel job IDs must be unique.")
    if len({case.case_id for case in suite.cases}) != 12:
        raise ValueError("Benchmark case IDs must be unique.")
    if sum(case.split == "dev" for case in suite.cases) != 4:
        raise ValueError("Expected four development and eight held-out cases.")

    labels = {case.case_id: case.grades(suite.panel_job_ids) for case in suite.cases}
    cases = [case for case in suite.cases if case.split == split]

    with settings.prepared_jobs_path.open(encoding="utf-8") as source:
        jobs, rejected = load_jobs(source)
    if rejected:
        raise ValueError("Prepare a valid job catalog before running the benchmark.")

    catalog = {job.job_id: job for job in jobs}
    panel = {job_id: catalog[job_id] for job_id in suite.panel_job_ids}
    vocabulary = load_skill_vocabulary(settings.prepared_data_dir / "skill_evidence.jsonl")
    outcomes: dict[str, list[Outcome]] = {variant: [] for variant in VARIANTS}

    print(f"Suite: {len(suite.cases)} resumes, {len(panel)} jobs, 204 graded pairs")
    print(f"Evaluating {len(cases)} {split} resumes")

    with TemporaryDirectory() as folder:
        jobs_path = Path(folder) / "panel.jsonl"
        index_path = Path(folder) / "qdrant"
        write_jsonl(jobs_path, panel.values())
        build_index(jobs_path, index_path)
        reranker = create_reranker()

        for case in cases:
            prepared = _prepare(case, vocabulary)
            grades = labels[case.case_id]

            for variant, hits in _rank(prepared, jobs_path, index_path, reranker).items():
                ranked_ids = [document.metadata["job_id"] for document, _ in hits]
                response = match_candidates(
                    (panel[job_id] for job_id in ranked_ids), prepared
                )
                accepted = {
                    item.job.job_id: grades[item.job.job_id]
                    for item in response.recommendations
                }
                outcomes[variant].append(
                    Outcome(
                        ndcg=panel_ndcg_at_k(ranked_ids, grades, TOP_K),
                        accepted_grades=list(accepted.values()),
                        no_match_correct=(not accepted) == case.expected_no_match,
                    )
                )

                for job_id, grade in accepted.items():
                    if grade < 2:
                        print(f"{case.case_id} | {variant} | accepted {job_id} with grade {grade}")
                for job_id in sorted(case.suitable - accepted.keys()):
                    stage = "matching" if job_id in ranked_ids else "retrieval"
                    print(f"{case.case_id} | {variant} | missed {job_id} at {stage}")

    _print_summary(outcomes)


def main() -> None:
    """Run one benchmark split from the project root."""
    parser = argparse.ArgumentParser(description="Evaluate the JobFit job panel.")
    parser.add_argument("--split", choices=("dev", "heldout"), default="dev")
    args = parser.parse_args()
    run_benchmark(args.split, Settings())


if __name__ == "__main__":
    main()