"""Build a candidate profile from parsed resume text using explicit rules."""

import re
from collections.abc import Iterable
from pathlib import Path

from app.dataset.dataset_loader import JobSkillEvidence
from app.schemas import (
    CandidateFact,
    CandidateProfile,
    ParsedResume,
    Preferences,
    PreparedResume,
    ResumeEvidence,
)

EXPERIENCE_YEARS = re.compile(
    r"(?:total\s+)?(?:professional|work)\s+experience\s*:\s*"
    r"(\d+(?:\.\d+)?)\s+years?\.?",
    re.IGNORECASE,
)

DEGREE = re.compile(
    r"\b(?:B\.?Tech|M\.?Tech|B\.E|M\.E|B\.?Sc|M\.?Sc|Ph\.?D|"
    r"Bachelor(?:'s| of)|Master(?:'s| of))\b",
    re.IGNORECASE,
)


def load_skill_vocabulary(evidence_path: Path) -> list[str]:
    """Read distinct skill names from the prepared job evidence."""
    names: dict[str, str] = {}

    with evidence_path.open(encoding="utf-8") as source:
        for line in source:
            evidence = JobSkillEvidence.model_validate_json(line)
            # Merge casing variants while keeping the first spelling for display.
            names.setdefault(evidence.skill.casefold(), evidence.skill)

    return list(names.values())


def extract_skills(
    lines: list[ResumeEvidence],
    vocabulary: Iterable[str],
) -> list[CandidateFact]:
    """Find vocabulary terms in the resume and collect their supporting lines."""
    skills: list[CandidateFact] = []

    for skill in vocabulary:
        # Escape names such as C++ and match them as complete terms.
        pattern = re.compile(
            rf"(?<![\w.]){re.escape(skill)}(?![\w+#]|\.\w)",
            re.IGNORECASE,
        )
        matches = [line for line in lines if pattern.search(line.quote)]

        if matches:
            skills.append(CandidateFact(text=skill, evidence=matches))

    return skills


def build_profile(
    resume: ParsedResume,
    vocabulary: Iterable[str],
) -> CandidateProfile:
    """Extract supported skills, explicit experience, and degree mentions."""
    lines = []
    for page in resume.pages:
        for raw_line in page.text.splitlines():
            quote = raw_line.strip()
            if quote:
                lines.append(ResumeEvidence(quote=quote, page=page.page))

    experience: list[CandidateFact] = []
    education: list[CandidateFact] = []
    duration_evidence: dict[float, list[ResumeEvidence]] = {}

    for line in lines:
        duration = EXPERIENCE_YEARS.fullmatch(line.quote)
        if duration:
            years = float(duration.group(1))
            experience.append(CandidateFact(text=line.quote, evidence=[line]))
            duration_evidence.setdefault(years, []).append(line)

        if DEGREE.search(line.quote):
            education.append(CandidateFact(text=line.quote, evidence=[line]))

    # A single explicit duration is usable. Conflicting values leave it unknown.
    experience_years = None
    experience_years_evidence = []
    if len(duration_evidence) == 1:
        experience_years, experience_years_evidence = next(
            iter(duration_evidence.items())
        )

    return CandidateProfile(
        skills=extract_skills(lines, vocabulary),
        experience=experience,
        education=education,
        experience_years=experience_years,
        experience_years_evidence=experience_years_evidence,
    )


def build_retrieval_query(
    profile: CandidateProfile, preferences: Preferences
) -> str | None:
    """Combine the target role and extracted skills into search text."""
    terms = [skill.text for skill in profile.skills]

    if preferences.target_role:
        terms.insert(0, preferences.target_role.strip())

    return " ".join(terms).strip() or None


def prepare_resume(
    resume: ParsedResume,
    vocabulary: Iterable[str],
    preferences: Preferences | None = None,
) -> PreparedResume:
    """Build a candidate profile and query from a parsed resume."""
    if preferences is None:
        preferences = Preferences()

    profile = build_profile(resume, vocabulary)

    return PreparedResume(
        parsed_resume=resume,
        candidate_profile=profile,
        preferences=preferences,
        retrieval_query=build_retrieval_query(profile, preferences),
    )
