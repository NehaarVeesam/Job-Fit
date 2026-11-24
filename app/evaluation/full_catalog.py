"""Compare JobFit retrieval modes on the full catalog with pooled judgments."""

import argparse
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory
from typing import Literal

import pymupdf
from pydantic import BaseModel, TypeAdapter

from app.config import Settings
from app.dataset.dataset_loader import load_jobs
from app.evaluation.audits import audit_matches
from app.evaluation.fixtures import EvaluationCase, EvaluationSuite
from app.evaluation.metrics import graded_ndcg_at_k
from app.matching.pipeline import ResumeMatchOptions, run_resume_match
from app.resumes.profile import load_skill_vocabulary, prepare_resume
from app.resumes.resume_parser import parse_resume
from app.schemas import JobRecord


TOP_K = 5
RESUMES = Path("evaluation/full_catalog_resumes.json")
VARIANTS = {
    "BM25": ResumeMatchOptions(mode="bm25", top_k=TOP_K),
    "Dense": ResumeMatchOptions(mode="dense", top_k=TOP_K),
    "Hybrid": ResumeMatchOptions(mode="hybrid", top_k=TOP_K),
    "Hybrid + reranker": ResumeMatchOptions(mode="hybrid", top_k=TOP_K, rerank=True),
}
DEFAULT_RUNS = Path("artifacts/evaluation/full_catalog_runs.json")
LABELS = Path("evaluation/full_catalog_judgments.json")


class Run(BaseModel):
    """The shortlist, final recommendations, and citation audit for one run."""

    case_id: str
    variant: str
    ranked_ids: list[str]
    recommended_ids: list[str]
    evidence_errors: list[str]


class Judgment(BaseModel):
    """An assistant-authored suitability judgment and its rationale."""

    grade: Literal[0, 1, 2]
    reason: str


def _load_cases() -> list[EvaluationCase]:
    """Load the six standalone full-catalog cases."""
    suite = EvaluationSuite.model_validate_json(RESUMES.read_bytes())
    if len(suite.cases) != 6 or len({case.case_id for case in suite.cases}) != 6:
        raise ValueError("Expected six distinct resume cases.")
    if suite.top_k != TOP_K:
        raise ValueError("Expected a five-job shortlist.")
    return suite.cases


def _catalog(settings: Settings) -> dict[str, JobRecord]:
    """Load the validated 250-job catalog used by the normal pipeline."""
    with settings.prepared_jobs_path.open(encoding="utf-8") as source:
        jobs, rejected = load_jobs(source)
    if len(jobs) != 250 or rejected:
        raise ValueError("Prepare the complete 250-job catalog before benchmarking.")
    return {job.job_id: job for job in jobs}


def _write_pdf(pdf: Path, case: EvaluationCase) -> None:
    """Convert the fixture pages into a PDF for the normal resume parser."""
    with pymupdf.open() as document:
        for text in case.resume_pages:
            page = document.new_page()
            if page.insert_textbox(page.rect + (48, 48, -48, -48), text, fontsize=11) < 0:
                raise ValueError(f"Fixture page overflows for {case.case_id}.")
        document.save(pdf)


def _run_case(
    case: EvaluationCase, pdf: Path, settings: Settings, vocabulary: list[str]
) -> list[Run]:
    """Run all variants and check citations against the exact PDF used."""
    _write_pdf(pdf, case)
    prepared = prepare_resume(parse_resume(pdf), vocabulary, case.preferences)
    runs = []
    for variant, options in VARIANTS.items():
        result = run_resume_match(pdf, case.preferences, options, settings=settings)
        if result.catalog_size != 250:
            raise ValueError(f"Unexpected catalog size in {case.case_id} / {variant}.")
        audit = audit_matches(result.match_response, prepared, {}, case.expected_no_match)
        runs.append(Run(
            case_id=case.case_id,
            variant=variant,
            ranked_ids=result.ranked_job_ids,
            recommended_ids=[item.job.job_id for item in result.match_response.recommendations],
            evidence_errors=audit.evidence_errors,
        ))
    return runs


def _returned_pairs(runs: list[Run]) -> set[tuple[str, str]]:
    """Deduplicate returned jobs per resume across the four variants."""
    return {(run.case_id, job_id) for run in runs for job_id in run.ranked_ids}


def _print_pool(
    runs: list[Run], judgments: dict[str, dict[str, Judgment]], catalog: dict[str, JobRecord]
) -> None:
    """Print unjudged case-job pairs without variant names or rank positions."""
    missing = {(case_id, job_id) for case_id, job_id in _returned_pairs(runs)
               if job_id not in judgments.get(case_id, {})}
    print(f"New case-job pairs to judge: {len(missing)}")
    for case_id, job_id in sorted(missing):
        job = catalog[job_id]
        skills = ", ".join(job.required_skills)
        constraints = ", ".join(job.hard_constraints) or "none"
        print(f"{case_id} | {job_id} | {job.title} | {job.work_mode} | "
              f"{job.experience_min_years} years | {skills} | {constraints}")


def collect(output: Path) -> None:
    """Run the normal pipeline and save all shortlists and recommendations."""
    settings = Settings()
    catalog = _catalog(settings)
    cases = _load_cases()
    vocabulary = load_skill_vocabulary(settings.prepared_data_dir / "skill_evidence.jsonl")
    runs = []
    with TemporaryDirectory() as folder:
        for number, case in enumerate(cases, start=1):
            pdf = Path(folder) / f"resume_{number}.pdf"
            runs.extend(_run_case(case, pdf, settings, vocabulary))
            print(f"Completed {case.case_id} ({number}/{len(cases)})", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(TypeAdapter(list[Run]).dump_json(runs, indent=2) + b"\n")
    judgments = _read_judgments() if LABELS.is_file() else {}
    _print_pool(runs, judgments, catalog)


def _read_judgments() -> dict[str, dict[str, Judgment]]:
    """Load the shared ground-truth judgments."""
    return TypeAdapter(dict[str, dict[str, Judgment]]).validate_json(LABELS.read_bytes())


def _pool_grades(runs: list[Run]) -> dict[str, dict[str, int]]:
    """Score only returned resume-job pairs and require a judgment for each."""
    judgments = _read_judgments()
    grades: dict[str, dict[str, int]] = {run.case_id: {} for run in runs}
    for case_id, job_id in sorted(_returned_pairs(runs)):
        judgment = judgments.get(case_id, {}).get(job_id)
        if judgment is None:
            raise ValueError(f"Missing judgment for {case_id} / {job_id}.")
        grades[case_id][job_id] = judgment.grade
    return grades


def _score_variant(runs: list[Run], grades: dict[str, dict[str, int]]) -> str:
    """Score one variant against the same pooled ground truth."""
    ndcg = mean(graded_ndcg_at_k(run.ranked_ids, grades[run.case_id], TOP_K) or 0
                for run in runs)
    known_suitable = sum(sum(grade == 2 for grade in grades[run.case_id].values())
                         for run in runs)
    found = sum(sum(grades[run.case_id][job_id] == 2 for job_id in run.ranked_ids)
                for run in runs)
    recommended = [grades[run.case_id][job_id] for run in runs
                   for job_id in run.recommended_ids]
    strict = recommended.count(2)
    precision = f"{strict}/{len(recommended)}" if recommended else "n/a"
    citation_errors = sum(len(run.evidence_errors) for run in runs)
    return f"{ndcg:.3f} | {found}/{known_suitable} | {precision} | {citation_errors}"


def score(path: Path) -> None:
    """Print aggregate scores and the ranked outputs for inspection."""
    runs = TypeAdapter(list[Run]).validate_json(path.read_bytes())
    expected = {(case.case_id, variant) for case in _load_cases() for variant in VARIANTS}
    if len(runs) != len(expected) or {(run.case_id, run.variant) for run in runs} != expected:
        raise ValueError("Expected exactly one run per case and variant.")
    for run in runs:
        ranked = set(run.ranked_ids)
        recommended = set(run.recommended_ids)
        if len(ranked) != len(run.ranked_ids) or len(ranked) > TOP_K:
            raise ValueError(f"Invalid shortlist for {run.case_id} / {run.variant}.")
        if len(recommended) != len(run.recommended_ids) or not recommended <= ranked:
            raise ValueError(f"Invalid recommendations for {run.case_id} / {run.variant}.")
    grades = _pool_grades(runs)
    print("| Variant | nDCG@5 | Known suitable retrieved | Strict recommendation precision "
          "| Citation errors |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for variant in VARIANTS:
        selected = [run for run in runs if run.variant == variant]
        print(f"| {variant} | {_score_variant(selected, grades)} |")
    print("\n| Resume | Variant | Top five | Final recommendations |")
    print("| --- | --- | --- | --- |")
    for run in runs:
        ranked = ", ".join(run.ranked_ids) or "None"
        recommended = ", ".join(run.recommended_ids) or "None"
        print(f"| {run.case_id} | {run.variant} | {ranked} | {recommended} |")


def main() -> None:
    """Collect model outputs or score an existing run against the judgments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("collect", "score"))
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    args = parser.parse_args()
    if args.action == "collect":
        collect(args.runs)
    else:
        score(args.runs)


if __name__ == "__main__":
    main()
