"""Run the JobFit workflow from a PDF resume to job recommendations."""

from pathlib import Path

from pydantic import BaseModel, Field

from app.config import Settings
from app.dataset.dataset_loader import load_jobs
from app.matching.generation import explain_recommendation
from app.matching.matcher import match_candidates
from app.matching.semantic_evidence import review_required_skill_gaps
from app.resumes.profile import load_skill_vocabulary, prepare_resume
from app.resumes.resume_parser import parse_resume
from app.retrieval.hybrid import SearchMode, retrieve_candidates
from app.retrieval.reranking import create_reranker, rerank_hits
from app.schemas import (
    JobRecord,
    Preferences,
    PreparedResume,
    ResumeMatchResult,
    SemanticEvidenceReport,
)


class ResumeMatchOptions(BaseModel):
    """Control retrieval and optional model steps for one run."""

    mode: SearchMode = "hybrid"
    top_k: int | None = Field(default=None, ge=1)
    rerank: bool = False
    use_qwen: bool = False
    review_semantic_evidence: bool = False


def _load_catalog(path: Path) -> dict[str, JobRecord]:
    """Load the prepared job records and make them searchable by ID."""
    with path.open(encoding="utf-8") as source:
        jobs, rejections = load_jobs(source)

    if rejections:
        first = rejections[0]
        raise ValueError(
            f"Prepared catalog has an invalid row at line "
            f"{first.line_number}: {first.message}"
        )

    return {job.job_id: job for job in jobs}


def _retrieve_jobs(
    prepared: PreparedResume,
    catalog: dict[str, JobRecord],
    settings: Settings,
    options: ResumeMatchOptions,
) -> list[JobRecord]:
    """Retrieve jobs, optionally rerank them, and resolve them by ID."""
    query = prepared.retrieval_query
    if query is None:
        return []

    top_k = options.top_k or settings.retrieval_top_k
    count, hits = retrieve_candidates(
        query,
        top_k,
        settings.prepared_jobs_path,
        mode=options.mode,
        preferences=prepared.preferences,
    )

    if count != len(catalog):
        raise ValueError("The retrieval count differs from the catalog. Check the index.")

    if options.rerank and hits:
        reranked = rerank_hits(query, hits, create_reranker())
        documents = [hit.document for hit in reranked]
    else:
        documents = [document for document, _ in hits]

    # Retrieve by ID, then use the prepared catalog as the source of job requirements.
    shortlist: list[JobRecord] = []
    for document in documents:
        job_id = document.metadata["job_id"]
        if job_id not in catalog:
            raise ValueError(f"Job {job_id!r} is missing from the catalog. Rebuild the index.")
        shortlist.append(catalog[job_id])

    return shortlist


def run_resume_match(
    pdf_path: Path,
    preferences: Preferences | None = None,
    options: ResumeMatchOptions | None = None,
    *,
    settings: Settings | None = None,
) -> ResumeMatchResult:
    """Prepare a resume, find and assess jobs, and explain recommendations."""
    settings = settings if settings is not None else Settings()
    options = options if options is not None else ResumeMatchOptions()

    parsed_resume = parse_resume(pdf_path)
    vocabulary = load_skill_vocabulary(
        settings.prepared_data_dir / "skill_evidence.jsonl"
    )
    prepared = prepare_resume(parsed_resume, vocabulary, preferences)

    catalog = _load_catalog(settings.prepared_jobs_path)
    shortlist = _retrieve_jobs(prepared, catalog, settings, options)
    matches = match_candidates(shortlist, prepared)

    explanations = {
        item.job.job_id: explain_recommendation(item, use_qwen=options.use_qwen)
        for item in matches.recommendations
    }

    # Run the advisory review after matching so it cannot affect the decisions.
    semantic_reviews: dict[str, SemanticEvidenceReport] = {}
    if options.review_semantic_evidence:
        for item in matches.recommendations + matches.excluded:
            semantic_reviews[item.job.job_id] = review_required_skill_gaps(item, prepared)

    warnings = []
    if prepared.retrieval_query is None:
        warnings.append("No search query could be built from the resume or target role.")

    return ResumeMatchResult(
        retrieval_query=prepared.retrieval_query,
        catalog_size=len(catalog),
        ranked_job_ids=[job.job_id for job in shortlist],
        match_response=matches,
        explanations=explanations,
        semantic_reviews=semantic_reviews,
        warnings=warnings,
    )
