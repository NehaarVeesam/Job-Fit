from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, StrictBool, model_validator

class JobRecord(BaseModel):
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
        minimum = self.experience_min_years
        maximum = self.experience_max_years

        if minimum is None or maximum is None:
            return self

        if minimum > maximum:
            raise ValueError("Minimum experience cannot exceed maximum experience")

        return self


class ResumeEvidence(BaseModel):
    quote: str = Field(pattern=r"\S")
    page: int = Field(ge=1)


class CandidateFact(BaseModel):
    text: str = Field(pattern=r"\S")
    evidence: list[ResumeEvidence] = Field(min_length=1)


class CandidateProfile(BaseModel):
    skills: list[CandidateFact] = Field(default_factory=list)
    projects: list[CandidateFact] = Field(default_factory=list)
    experience: list[CandidateFact] = Field(default_factory=list)
    education: list[CandidateFact] = Field(default_factory=list)
    experience_years: float | None = Field(default=None, ge=0)
    experience_years_evidence: list[ResumeEvidence] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_experience_evidence(self) -> "CandidateProfile":
        if self.experience_years is not None and not self.experience_years_evidence:
            raise ValueError("Experience years require resume evidence")
        return self
'''
CandidateProfile
└── skills
    └── CandidateFact
        ├── text = "FastAPI"
        └── evidence
            └── ResumeEvidence
                ├── quote = "Built Python APIs with FastAPI."
                └── page = 1
'''