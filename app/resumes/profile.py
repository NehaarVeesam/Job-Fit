"""Deterministic, corpus-vocabulary resume preparation; no model calls."""

import hashlib
import json
import re
from pathlib import Path

from app.schemas import (
    CandidateFact, CandidateProfile, JobRecord, ParsedResume, Preferences,
    PreparedResume, ResumeEvidence,
)

PROFILE_VERSION = "jobfit-profile-v1"
HEADINGS = {
    "skills": "skills", "technical skills": "skills", "technologies": "skills",
    "technical expertise": "skills", "core skills": "skills",
    "projects": "projects", "personal projects": "projects", "academic projects": "projects",
    "experience": "experience", "work experience": "experience",
    "professional experience": "experience", "employment history": "experience",
    "internship": "internships", "internships": "internships",
    "internship experience": "internships",
    "education": "education", "qualifications": "education",
    "summary": "summary", "professional summary": "summary", "profile": "summary",
    "objective": "summary", "career objective": "summary",
    "certifications": "other", "certificates": "other",
    "interests": "other", "hobbies": "other", "references": "other",
    "contact": "other", "contact details": "other", "personal details": "other",
}
CONTACT = re.compile(
    r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|https?://|www\.|"
    r"(?:\+?\d[\d ()-]{6,}\d)|"
    r"\b(?:email|phone|mobile|address|linkedin|github|location)\s*:", re.I,
)
UNCERTAIN = re.compile(
    r"\b(?:no|not|never|without|lack|lacking|learning|beginner|planned|aspiring)\b|"
    r"\b(?:plan to|want to|interested in|would like to)\b", re.I,
)
EXPERIENCE_PATTERNS = [
    re.compile(r"^(?:professional|full-time) (?:work )?experience\s*:\s*(\d+(?:\.\d+)?)\s+years?\.?$", re.I),
    re.compile(r"^(\d+(?:\.\d+)?)\s+years? of (?:professional|full-time)(?: work)? experience\.?$", re.I),
]


def load_prepared_jobs(directory: Path) -> tuple[list[JobRecord], dict]:
    """Require a successful Module 1 report and its matching artifact hashes."""
    report = json.loads((directory / "validation_report.json").read_text(encoding="utf-8"))
    if report.get("status") != "ok":
        raise ValueError("Prepared corpus requires review; resolve Module 1 errors first")
    for filename in ("jobs.jsonl", "skill_evidence.jsonl"):
        expected = report.get("outputs", {}).get(filename, {}).get("sha256")
        if hashlib.sha256((directory / filename).read_bytes()).hexdigest() != expected:
            raise ValueError("Prepared corpus checksum mismatch; rerun Module 1")
    jobs = [JobRecord.model_validate_json(line) for line in
            (directory / "jobs.jsonl").read_text(encoding="utf-8").splitlines()]
    if not jobs or len(jobs) != report["counts"]["output"] or len({job.job_id for job in jobs}) != len(jobs):
        raise ValueError("Prepared corpus is empty or inconsistent; rerun Module 1")
    return jobs, {
        "source": report["source"], "configuration": report["configuration"],
        "jobs_sha256": report["outputs"]["jobs.jsonl"]["sha256"],
        "job_count": len(jobs),
    }


def skill_vocabulary(jobs: list[JobRecord]) -> list[str]:
    # Stable spelling/order; no related-tool equivalences or inferred aliases.
    canonical = {}
    for job in jobs:
        for skill in job.required_skills + job.preferred_skills:
            canonical.setdefault(skill.casefold(), skill)
    return sorted(canonical.values(), key=str.casefold)


def build_profile(parsed: ParsedResume, vocabulary: list[str],
                  preferences: Preferences | None = None) -> PreparedResume:
    preferences = preferences or Preferences()
    patterns = [(skill, re.compile(r"(?<![\w.+#])" + re.escape(skill) + r"(?![\w+#])", re.I))
                for skill in sorted(set(vocabulary), key=str.casefold)]
    facts = {"projects": [], "experience": [], "education": []}
    found: dict[str, list[ResumeEvidence]] = {}
    durations = []
    warnings = [
        "Skills are literal vocabulary mentions, not verified proficiency; rule-based extraction may miss context or synonyms.",
        "Raw resume text and facts are untrusted source data; no embedded instructions are executed.",
    ]
    section = None
    excluded_uncertain = 0
    for page in parsed.pages:
        for original in page.text.splitlines():
            line = original.strip()
            if not line:
                continue
            # Accept both standalone headings and 'Skills: Python, SQL'.
            heading, separator, content = line.partition(":")
            key = heading.strip().casefold()
            if key in HEADINGS:
                section = HEADINGS[key]
                # Keep duration statements intact, including their field label.
                if not any(pattern.fullmatch(line) for pattern in EXPERIENCE_PATTERNS):
                    line = content.strip() if separator else ""
                    if not line:
                        continue
            if section not in ("skills", "projects", "experience", "internships", "education", "summary"):
                continue
            if CONTACT.search(line):
                continue
            evidence = ResumeEvidence(quote=original.strip(), page=page.page)
            text = line.lstrip("-•* ")
            category = "experience" if section == "internships" else section
            if category in facts:
                facts[category].append(CandidateFact(text=text, evidence=[evidence]))
            # An uncertain line is never counted as positive skill evidence.
            if UNCERTAIN.search(line):
                if any(pattern.search(line) for _, pattern in patterns):
                    excluded_uncertain += 1
                continue
            for skill, pattern in patterns:
                if pattern.search(line):
                    found.setdefault(skill, []).append(evidence)
            if section in ("summary", "experience"):
                for pattern in EXPERIENCE_PATTERNS:
                    match = pattern.fullmatch(text)
                    if match:
                        durations.append((float(match.group(1)), evidence))
                        break

    values = {duration for duration, _ in durations}
    years = next(iter(values)) if len(values) == 1 else None
    if years is None:
        warnings.append("Professional experience years remain unknown; no unique explicit duration statement was established.")
    if excluded_uncertain:
        warnings.append(f"Excluded {excluded_uncertain} uncertain/negative skill lines from positive skill evidence.")
    profile = CandidateProfile(
        skills=[CandidateFact(text=skill, evidence=evidence) for skill, evidence in found.items()],
        **facts, experience_years=years,
        experience_years_evidence=[evidence for _, evidence in durations] if years is not None else [],
    )
    # Build only from canonical skills and an explicit user role, never raw lines.
    query_parts = []
    if preferences.target_role:
        if CONTACT.search(preferences.target_role):
            raise ValueError("Target role must be a role description without contact information")
        query_parts.append("Target role: " + preferences.target_role)
    if profile.skills:
        query_parts.append("Skills: " + ", ".join(fact.text for fact in profile.skills))
    query = "\n".join(query_parts) or None
    if query is None:
        warnings.append("No search query could be built: no recognized corpus skills or explicit target role. Review the resume headings and vocabulary.")
    return PreparedResume(parsed_resume=parsed, candidate_profile=profile,
                          preferences=preferences, retrieval_query=query, warnings=warnings)
