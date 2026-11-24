"""Compare JobFit retrieval modes on the labeled evaluation cases."""

import argparse
from pathlib import Path

from app.config import Settings
from app.dataset.dataset_loader import load_jobs
from app.evaluation.fixtures import EvaluationCase, JobJudgment, load_evaluation
from app.evaluation.metrics import evaluate_ranking
from app.resumes.profile import load_skill_vocabulary, prepare_resume
from app.retrieval.hybrid import retrieve_candidates
from app.schemas import ParsedResume, PreparedResume, ResumePage, MatchResponse
from app.evaluation.audits import audit_matches
from app.matching.matcher import match_candidates


MODES = ("bm25", "dense", "hybrid")


def _prepare_case(case: EvaluationCase, vocabulary: list[str]) -> PreparedResume:
    """Convert the fixture pages into the same profile used by JobFit."""
    resume = ParsedResume(
        pages=[
            ResumePage(page=number, text=text)
            for number, text in enumerate(case.resume_pages, start=1)
        ]
    )
    return prepare_resume(resume, vocabulary, case.preferences)


def _format_metric(value: float | None) -> str:
    """Display an undefined metric as n/a."""
    return "n/a" if value is None else f"{value:.3f}"


def _print_metrics(
    case_id: str,
    mode: str,
    ranked_ids: list[str],
    judgments: dict[str, JobJudgment],
    top_k: int,
) -> None:
    """Print the shortlist and the scores supported by its labels."""
    metrics = evaluate_ranking(ranked_ids, judgments, top_k)
    print(f"\n{case_id} | {mode}")
    print(f"  Ranked jobs: {', '.join(ranked_ids) or '(none)'}")
    print(f"  Judged results: {metrics.judged_count}/{metrics.returned_count}")
    print(f"  Judged precision: {_format_metric(metrics.judged_precision)}")
    print(f"  Known relevant recall: {_format_metric(metrics.known_relevant_recall)}")
    print(f"  Judged nDCG: {_format_metric(metrics.judged_ndcg)}")


def _print_match_audit(
    response: MatchResponse,
    prepared: PreparedResume,
    case: EvaluationCase,
    judgments: dict[str, JobJudgment],
) -> None:
    """Print the matching outcome and flag issues the labels expose."""
    audit = audit_matches(response, prepared, judgments, case.expected_no_match)

    print(f"  Match status: {response.status}")
    print(f"  Expected outcome: {'PASS' if audit.no_match_correct else 'FAIL'}")
    print(f"  Recommended and suitable: {', '.join(audit.suitable_ids) or '(none)'}")
    print(f"  Recommended but unsuitable: {', '.join(audit.unsuitable_ids) or '(none)'}")
    print(f"  Recommended but unjudged: {', '.join(audit.unjudged_ids) or '(none)'}")
    print(f"  Evidence errors: {len(audit.evidence_errors)}")
    for message in audit.evidence_errors:
        print(f"    {message}")

def main() -> None:
    """Load the fixtures and compare the requested retrieval modes."""
    parser = argparse.ArgumentParser(description="Evaluate JobFit retrieval.")
    parser.add_argument("--mode", choices=("all", *MODES), default="all")
    args = parser.parse_args()
    settings = Settings()

    with settings.prepared_jobs_path.open(encoding="utf-8") as source:
        jobs, rejected = load_jobs(source)
    if rejected:
        raise ValueError("Prepare a valid job catalog before running evaluation.")

    jobs_by_id = {job.job_id: job for job in jobs}
    suite, judgments = load_evaluation(
        Path("evaluation/suite.json"),
        Path("evaluation/judgments.jsonl"),
        {job.job_id for job in jobs},
    )
    vocabulary = load_skill_vocabulary(
        settings.prepared_data_dir / "skill_evidence.jsonl"
    )
    modes = MODES if args.mode == "all" else (args.mode,)

    for case in suite.cases:
        prepared = _prepare_case(case, vocabulary)
        for mode in modes:
            hits = []
            if prepared.retrieval_query:
                _, hits = retrieve_candidates(
                    prepared.retrieval_query,
                    suite.top_k,
                    settings.prepared_jobs_path,
                    mode=mode,
                    preferences=prepared.preferences,
                )

            ranked_ids = [document.metadata["job_id"] for document, _ in hits]
            _print_metrics(case.case_id, mode, ranked_ids, judgments[case.case_id], suite.top_k)
            response = match_candidates(
                (jobs_by_id[job_id] for job_id in ranked_ids),
                prepared,
            )
            _print_match_audit(response, prepared, case, judgments[case.case_id])

if __name__ == "__main__":
    main()