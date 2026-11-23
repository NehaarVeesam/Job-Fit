"""Validation models for job records and candidate facts with resume evidence."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, StrictBool, model_validator


class JobRecord(BaseModel):
    """One job posting with its search metadata and hiring requirements."""

    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(pattern=r"^syn_job_[0-9]{3}$")
    title: str = Field(min_length=1)
    company: str = Field(min_length=1)
    domain: Literal["AI/ML", "Software"]
    role_family: str
    seniority: Literal["Intern", "Entry", "Mid", "Senior", "Lead"]
    employment_type: Literal["full_time", "internship", "contract"]

    location_city: str | None
    location_country: Literal["India"]
    work_mode: Literal["hybrid", "onsite", "remote", "unknown"]

    # A null bound means the source provides no value for that end of the range.
    experience_min_years: float | None = Field(ge=0)
    experience_max_years: float | None = Field(ge=0)
    education: str

    required_skills: list[str]
    preferred_skills: list[str]
    hard_constraints: list[str]
    description: str = Field(min_length=1)

    source_url: HttpUrl | None
    posted_at: date
    dataset_id: str
    is_synthetic: StrictBool

    @model_validator(mode="after")
    def validate_experience_range(self) -> "JobRecord":
        """Ensure known bounds form a valid experience range."""
        minimum = self.experience_min_years
        maximum = self.experience_max_years

        if minimum is None or maximum is None:
            return self

        if minimum > maximum:
            raise ValueError("Minimum experience cannot exceed maximum experience")

        return self


class ResumeEvidence(BaseModel):
    """An exact resume quote and its one-based page number."""

    quote: str = Field(pattern=r"\S")
    page: int = Field(ge=1)


class CandidateFact(BaseModel):
    """One extracted candidate fact backed by at least one resume quote."""

    text: str = Field(pattern=r"\S")
    evidence: list[ResumeEvidence] = Field(min_length=1)


class CandidateProfile(BaseModel):
    """Candidate facts grouped by category while retaining their source evidence.

    CandidateProfile
    └── skills
        └── CandidateFact
            ├── text = "FastAPI"
            └── evidence
                └── ResumeEvidence
                    ├── quote = "Built Python APIs with FastAPI."
                    └── page = 1
    """

    skills: list[CandidateFact] = Field(default_factory=list)
    projects: list[CandidateFact] = Field(default_factory=list)
    experience: list[CandidateFact] = Field(default_factory=list)
    education: list[CandidateFact] = Field(default_factory=list)
    experience_years: float | None = Field(default=None, ge=0)
    experience_years_evidence: list[ResumeEvidence] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_experience_evidence(self) -> "CandidateProfile":
        """Require a supporting quote whenever an experience duration is known."""
        # Zero is an explicit duration. None represents an unknown duration.
        if self.experience_years is not None and not self.experience_years_evidence:
            raise ValueError("Experience years require resume evidence")
        return self


class ResumePage(BaseModel):
    """Extracted text from one page of a resume."""

    page: int = Field(ge=1)
    text: str


class ParsedResume(BaseModel):
    """The original pages in order, including any blank pages."""

    pages: list[ResumePage] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_pages(self) -> "ParsedResume":
        """Check page numbering and require some extractable text."""
        for expected_number, page in enumerate(self.pages, start=1):
            if page.page != expected_number:
                raise ValueError("Resume pages must be numbered consecutively from 1")

        if not any(page.text.strip() for page in self.pages):
            raise ValueError("The resume contains no extractable text")

        return self


class Preferences(BaseModel):
    """Optional job preferences supplied by the candidate."""

    target_role: str | None = None
    location_city: str | None = None
    work_mode: Literal["hybrid", "onsite", "remote"] | None = None
    eligibility_answers: dict[str, StrictBool] = Field(default_factory=dict)

class PreparedResume(BaseModel):
    """Bundle the resume, its extracted facts, and the candidate's preferences.

    PreparedResume
    ├── parsed_resume → ParsedResume
    │   └── pages
    │       └── ResumePage
    │           ├── page = 1
    │           └── text = "Built RAG applications using Python, LangChain, and FastAPI."
    ├── candidate_profile → CandidateProfile with extracted facts and evidence
    ├── preferences → Preferences
    │   ├── target_role = "Generative AI Engineer"
    │   ├── location_city = "Bengaluru"
    │   ├── work_mode = "hybrid"
    │   └── eligibility_answers = {}
    └── retrieval_query = "Generative AI Engineer Python LangChain RAG FastAPI"
    """

    parsed_resume: ParsedResume
    candidate_profile: CandidateProfile
    preferences: Preferences = Field(default_factory=Preferences)
    retrieval_query: str | None = None

    @model_validator(mode="after")
    def validate_evidence_quotes(self) -> "PreparedResume":
        """Check that each evidence quote appears on its cited page."""
        page_text = {page.page: page.text for page in self.parsed_resume.pages}
        profile = self.candidate_profile

        facts = profile.skills + profile.projects + profile.experience + profile.education
        citations = list(profile.experience_years_evidence)
        for fact in facts:
            citations.extend(fact.evidence)

        for evidence in citations:
            text = page_text.get(evidence.page)
            if text is None or evidence.quote not in text:
                raise ValueError(f"Evidence quote was not found on page {evidence.page}")

        return self


AssessmentStatus = Literal["met", "not_met", "unknown"]

class ConstraintAssessment(BaseModel):
    """The result of checking one job requirement against candidate evidence."""

    requirement: str
    status: AssessmentStatus
    reason: str
    resume_evidence: list[ResumeEvidence] = Field(default_factory=list)
    candidate_answer: bool | None = None



class SkillAssessment(BaseModel):
    """One job skill and the evidence available on each side."""

    skill: str
    requirement_type: Literal["required", "preferred"]
    matched: bool
    job_quote: str
    resume_evidence: list[ResumeEvidence] = Field(default_factory=list)


class SkillCoverage(BaseModel):
    """How many skills in one requirement group the resume supports."""

    matched_count: int
    total_count: int
    percentage: float | None


class JobSkillScore(BaseModel):
    """The skill assessment for one job.

    JobSkillScore
    ├── required → SkillCoverage
    │   ├── matched_count = 4
    │   ├── total_count = 6
    │   └── percentage = 66.7
    ├── preferred → SkillCoverage
    │   ├── matched_count = 0
    │   ├── total_count = 3
    │   └── percentage = 0.0
    └── assessments
        └── SkillAssessment (one example)
            ├── skill = "FastAPI"
            ├── requirement_type = "required"
            ├── matched = True
            ├── job_quote → the original job description line listing FastAPI
            └── resume_evidence
                └── ResumeEvidence
                    ├── quote = "Built RAG applications using Python, LangChain, and FastAPI."
                    └── page = 1
    """

    required: SkillCoverage
    preferred: SkillCoverage
    assessments: list[SkillAssessment]


class JobAssessment(BaseModel):
    """The requirements, skill coverage, and decision reasons for one job.

    JobAssessment
    ├── job → JobRecord with the description and requirements
    ├── eligibility = "met"
    ├── experience → ConstraintAssessment for the minimum years
    ├── education → ConstraintAssessment for the degree requirement
    ├── hard_constraints = []
    ├── skill_score → JobSkillScore with required and preferred coverage
    ├── rejection_reasons = []
    └── warnings = []
    """

    job: JobRecord
    eligibility: AssessmentStatus
    experience: ConstraintAssessment | None
    education: ConstraintAssessment
    hard_constraints: list[ConstraintAssessment]
    skill_score: JobSkillScore
    rejection_reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class MatchResponse(BaseModel):
    """Recommendations and excluded jobs from one shortlist.

    MatchResponse
    ├── status = "matches_found"
    ├── recommendations
    │   └── JobAssessment
    │       ├── job.job_id = "syn_job_001"
    │       └── rejection_reasons = []
    └── excluded
        └── JobAssessment
            ├── job.job_id = "syn_job_006"
            └── rejection_reasons
                └── "Hard requirement needs confirmation: Must be able to join within 60 days"
    """

    status: Literal["matches_found", "no_suitable_matches"]
    recommendations: list[JobAssessment]
    excluded: list[JobAssessment]
