"""HTTP response contracts; failed/zero-overlap jobs can still be inspected."""

from typing import Literal

from pydantic import Field

from app.schemas import (
    AssessmentStatus, ConstraintAssessment, Contract, CoverageScore, JobRecord,
    MatchReason, MatchResponse, RelevanceScore, RerankerScore, SkillAssessment,
)


class UploadMatchResponse(Contract):
    run_id: str
    expires_in_seconds: int
    result: MatchResponse
    preparation_warnings: list[str]


class JobDetailResponse(Contract):
    dataset_id: Literal["synthetic_india_ai_sw_v2"] = "synthetic_india_ai_sw_v2"
    is_synthetic: Literal[True] = True
    job: JobRecord
    relevance: RelevanceScore | None
    reranker_relevance: RerankerScore | None = None
    required_coverage: CoverageScore
    preferred_coverage: CoverageScore
    eligibility: AssessmentStatus
    experience_assessment: ConstraintAssessment
    education_assessment: ConstraintAssessment
    constraints: list[ConstraintAssessment]
    skills: list[SkillAssessment]
    reasons: list[MatchReason] = Field(default_factory=list)
    passes_policy: bool
    was_recommended: bool
    rejection_reasons: list[str]
    warnings: list[str]
    generation_status: Literal["deterministic"] = "deterministic"
