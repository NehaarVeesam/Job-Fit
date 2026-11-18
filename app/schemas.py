"""Data contracts for the fresh build, including every supplied dataset field."""

import math
from datetime import date
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, StringConstraints, field_validator, model_validator

Text = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]
WorkMode = Literal["hybrid", "onsite", "remote", "unknown"]
AssessmentStatus = Literal["met", "not_met", "unknown"]
DatasetId = Literal["synthetic_india_ai_sw_v2"]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class JobRecord(Contract):
    job_id: Annotated[str, StringConstraints(pattern=r"^syn_job_\d{3}$")]
    title: Text
    company: Text
    domain: Literal["AI/ML", "Software"]
    role_family: Text
    seniority: Literal["Intern", "Entry", "Mid", "Senior", "Lead"]
    employment_type: Literal["full_time", "internship", "contract"]
    location_city: Text | None
    location_country: Literal["India"]
    work_mode: WorkMode
    experience_min_years: float | None = Field(ge=0)
    experience_max_years: float | None = Field(ge=0)
    education: Text
    required_skills: list[Text]
    preferred_skills: list[Text]
    hard_constraints: list[Text]
    description: Text
    source_url: HttpUrl | None
    posted_at: date
    dataset_id: DatasetId
    is_synthetic: Literal[True]

    @field_validator("is_synthetic", mode="before")
    @classmethod
    def require_true_marker(cls, value):
        if value is not True:
            raise ValueError("The source must explicitly identify synthetic records")
        return value

    @model_validator(mode="after")
    def check_job_fields(self) -> Self:
        low, high = self.experience_min_years, self.experience_max_years
        if low is not None and high is not None and low > high:
            raise ValueError("Minimum experience exceeds maximum experience")
        required = [skill.strip().casefold() for skill in self.required_skills]
        preferred = [skill.strip().casefold() for skill in self.preferred_skills]
        if any(not skill for skill in required + preferred):
            raise ValueError("Skill names must not be blank")
        if len(set(required)) != len(required) or len(set(preferred)) != len(preferred):
            raise ValueError("Skill lists must not contain duplicates")
        if set(required) & set(preferred):
            raise ValueError("Required and preferred skills must be separate")
        if any(skill not in self.description for skill in self.required_skills + self.preferred_skills):
            raise ValueError("Each listed skill must appear verbatim in the source description")
        if self.source_url is not None:
            raise ValueError("This supplied synthetic corpus has no real application URLs")
        return self


class ResumeEvidence(Contract):
    quote: Text
    page: int = Field(ge=1)


class JobEvidence(Contract):
    job_id: Text
    quote: Text
    section: Text = "description"


class CandidateFact(Contract):
    text: Text
    evidence: list[ResumeEvidence] = Field(min_length=1)


class CandidateProfile(Contract):
    skills: list[CandidateFact] = Field(default_factory=list)
    projects: list[CandidateFact] = Field(default_factory=list)
    experience: list[CandidateFact] = Field(default_factory=list)
    education: list[CandidateFact] = Field(default_factory=list)
    experience_years: float | None = Field(default=None, ge=0)
    experience_years_evidence: list[ResumeEvidence] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_duration_evidence(self) -> Self:
        if self.experience_years is not None and not self.experience_years_evidence:
            raise ValueError("Professional experience duration needs resume evidence")
        return self


class ResumePage(Contract):
    page: int = Field(ge=1)
    text: str


class ParsedResume(Contract):
    pages: list[ResumePage] = Field(min_length=1)

    @model_validator(mode="after")
    def check_pages(self) -> Self:
        if [page.page for page in self.pages] != list(range(1, len(self.pages) + 1)):
            raise ValueError("Pages must be sequential and one-based")
        if not any(page.text.strip() for page in self.pages):
            raise ValueError("Resume must have some extracted text")
        return self


class Preferences(Contract):
    target_role: Text | None = None
    location_city: Text | None = None
    work_mode: Literal["hybrid", "onsite", "remote"] | None = None
    unknown_metadata_policy: Literal["include_with_warning", "exclude"] = "include_with_warning"
    # Examples: graduation_year=2027 or willing_to_travel=true. These are
    # explicitly supplied answers, not facts inferred from a resume address.
    eligibility_facts: dict[str, str | bool | int] = Field(default_factory=dict)


class PreparedResume(Contract):
    parsed_resume: ParsedResume
    candidate_profile: CandidateProfile
    preferences: Preferences
    retrieval_query: Text | None
    warnings: list[Text] = Field(default_factory=list)

    @model_validator(mode="after")
    def verify_resume_quotes(self) -> Self:
        pages = {page.page: page.text for page in self.parsed_resume.pages}
        profile = self.candidate_profile
        citations = list(profile.experience_years_evidence)
        for fact in profile.skills + profile.projects + profile.experience + profile.education:
            citations.extend(fact.evidence)
        if any(item.page not in pages or item.quote not in pages[item.page] for item in citations):
            raise ValueError("Every resume quote must exist on its cited extracted page")
        return self


class CoverageScore(Contract):
    evidenced_count: int = Field(ge=0)
    total_count: int = Field(ge=0)
    percentage: float | None = Field(default=None, ge=0, le=100)

    @model_validator(mode="after")
    def check_arithmetic(self) -> Self:
        if self.evidenced_count > self.total_count:
            raise ValueError("Evidenced count cannot exceed the requirement count")
        expected = round(100 * self.evidenced_count / self.total_count, 1) if self.total_count else None
        if self.percentage is not None and (expected is None or not math.isclose(self.percentage, expected, abs_tol=1e-6)):
            raise ValueError("Coverage must agree with counts, rounded to one decimal place")
        self.percentage = expected
        return self


class RelevanceScore(Contract):
    value: float
    score_type: Literal["cosine", "bm25", "rrf", "cross_encoder"]

    @model_validator(mode="after")
    def check_cosine_range(self) -> Self:
        if self.score_type == "cosine" and not -1 <= self.value <= 1:
            raise ValueError("Cosine similarity must be between -1 and 1")
        return self


class RerankerScore(RelevanceScore):
    score_type: Literal["cross_encoder"] = "cross_encoder"


class ConstraintAssessment(Contract):
    condition: Text
    status: AssessmentStatus
    resume_evidence: list[ResumeEvidence] = Field(default_factory=list)
    user_fact_keys: list[Text] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_support(self) -> Self:
        if self.status != "unknown" and not (self.resume_evidence or self.user_fact_keys):
            raise ValueError("A constraint decision requires evidence or an explicit user fact")
        return self


class SkillAssessment(Contract):
    skill: Text
    requirement_type: Literal["required", "preferred"]
    status: Literal["evidenced", "not_evidenced_in_resume"]
    job_evidence: JobEvidence
    resume_evidence: list[ResumeEvidence] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_evidence_status(self) -> Self:
        if (self.status == "evidenced") != bool(self.resume_evidence):
            raise ValueError("Skill status must agree with the presence of resume evidence")
        return self


class MatchReason(Contract):
    explanation: Text
    resume_evidence: ResumeEvidence
    job_evidence: JobEvidence


class JobMatch(Contract):
    job: JobRecord
    relevance: RelevanceScore | None = None
    reranker_relevance: RerankerScore | None = None
    required_coverage: CoverageScore | None = None
    preferred_coverage: CoverageScore | None = None
    eligibility: AssessmentStatus = "unknown"
    experience_assessment: ConstraintAssessment | None = None
    education_assessment: ConstraintAssessment | None = None
    generation_status: Literal["deterministic", "qwen_grounded_plan", "fallback"] = "deterministic"
    constraints: list[ConstraintAssessment] = Field(default_factory=list)
    skills: list[SkillAssessment] = Field(default_factory=list)
    reasons: list[MatchReason] = Field(min_length=1)
    warnings: list[Text] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_job_evidence_and_coverage(self) -> Self:
        citations = [reason.job_evidence for reason in self.reasons]
        citations.extend(skill.job_evidence for skill in self.skills)
        if any(item.job_id != self.job.job_id or item.quote not in self.job.description for item in citations):
            raise ValueError("Job evidence must reference this job and its actual text")
        seen = set()
        for skill in self.skills:
            allowed = self.job.required_skills if skill.requirement_type == "required" else self.job.preferred_skills
            key = (skill.requirement_type, skill.skill)
            if skill.skill not in allowed or key in seen:
                raise ValueError("Assess each actual job skill at most once in its correct category")
            seen.add(key)
        for kind, coverage, source in (
            ("required", self.required_coverage, self.job.required_skills),
            ("preferred", self.preferred_coverage, self.job.preferred_skills),
        ):
            if coverage is not None:
                assessments = [skill for skill in self.skills if skill.requirement_type == kind]
                evidenced = sum(skill.status == "evidenced" for skill in assessments)
                if len(assessments) != len(source) or coverage.total_count != len(source) or coverage.evidenced_count != evidenced:
                    raise ValueError("Coverage must account for all skills in its job requirement list")
        if any(item.condition not in self.job.hard_constraints for item in self.constraints):
            raise ValueError("Constraint assessments must reference actual job conditions")
        if any(item.status == "not_met" for item in self.constraints) and self.eligibility != "not_met":
            raise ValueError("A failed mandatory constraint cannot be labeled eligible")
        if any(item is not None and item.status == "not_met" for item in
               (self.experience_assessment, self.education_assessment)) and self.eligibility != "not_met":
            raise ValueError("Failed experience/education requirements must be reflected in eligibility")
        return self


class RankedCandidate(Contract):
    job_id: Text
    relevance: RelevanceScore | None = None
    reranker_relevance: RerankerScore | None = None
    eligibility: AssessmentStatus = "unknown"


class MatchResponse(Contract):
    dataset_id: DatasetId = "synthetic_india_ai_sw_v2"
    is_synthetic: Literal[True] = True
    corpus_size: int = Field(default=250, ge=1, le=250)
    corpus_subset: Literal["all", "first_100"] = "all"
    status: Literal["matches_found", "no_suitable_matches"]
    applied_preferences: Preferences = Field(default_factory=Preferences)
    applied_filters: dict[str, str | bool | int] = Field(default_factory=dict)
    ranked_candidates: list[RankedCandidate] = Field(default_factory=list, max_length=250)
    recommendations: list[JobMatch] = Field(default_factory=list, max_length=5)
    warnings: list[Text] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_outcome(self) -> Self:
        if (self.status == "matches_found") != bool(self.recommendations):
            raise ValueError("Outcome must agree with the presence of recommendations")
        ranked_ids = [item.job_id for item in self.ranked_candidates]
        accepted_ids = [item.job.job_id for item in self.recommendations]
        if len(ranked_ids) != len(set(ranked_ids)) or len(accepted_ids) != len(set(accepted_ids)):
            raise ValueError("Rankings and recommendations must contain unique job IDs")
        if len(ranked_ids) > self.corpus_size or not set(accepted_ids) <= set(ranked_ids):
            raise ValueError("Recommendations must belong to the ranked corpus")
        if self.corpus_subset == "first_100" and self.corpus_size > 100:
            raise ValueError("The first_100 subset cannot contain more than 100 jobs")
        if any(item.eligibility == "not_met" for item in self.recommendations):
            raise ValueError("A known-ineligible job cannot be an accepted recommendation")
        rankings = {item.job_id: item for item in self.ranked_candidates}
        for match in self.recommendations:
            rank = rankings[match.job.job_id]
            if rank.eligibility != match.eligibility or rank.relevance != match.relevance:
                raise ValueError("Recommendation scores/status must agree with the ranked entry")
            for condition in match.constraints + [item for item in (
                match.experience_assessment, match.education_assessment) if item is not None]:
                if not set(condition.user_fact_keys) <= self.applied_preferences.eligibility_facts.keys():
                    raise ValueError("Constraint evidence must reference an actual user-provided fact")
        return self
