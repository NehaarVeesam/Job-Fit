"""Conservative requirement decisions with explicit source support."""

import re

from app.profile import EXPERIENCE_PATTERNS
from app.schemas import ConstraintAssessment


def decision(condition, facts, key, predicate=lambda value: value is True, *, numeric=False):
    value = facts.get(key)
    valid = (type(value) is int and value >= 0) if numeric else type(value) is bool
    if numeric and valid:
        bounds = {"graduation_year": (1900, 2200), "max_travel_percent": (0, 100),
                  "notice_period_days": (0, 3650)}
        low, high = bounds.get(key, (0, 10000))
        valid = low <= value <= high
    return ConstraintAssessment(condition=condition,
                                status=("met" if predicate(value) else "not_met") if valid else "unknown",
                                user_fact_keys=[key] if valid else [])


def hard_constraint(condition, facts):
    # Every condition can be answered explicitly using its exact source wording.
    exact = "constraint:" + condition
    if type(facts.get(exact)) is bool:
        return decision(condition, facts, exact)
    boolean_keys = {
        "Final-year students only": "is_final_year",
        "Must be based in India": "based_in_india",
        "Must work from the Jaipur office": "willing_jaipur_office",
        "Rotational shifts including nights": "willing_rotational_night_shifts",
    }
    if condition in boolean_keys:
        return decision(condition, facts, boolean_keys[condition])
    if "on-call" in condition.lower():
        # Do not turn general availability into acceptance of a specific rotation.
        return decision(condition, facts, exact)
    if condition == "Must be able to join within 60 days":
        return decision(condition, facts, "notice_period_days", lambda value: value <= 60, numeric=True)
    if condition == "Open to 2025 and 2026 graduates only":
        return decision(condition, facts, "graduation_year", lambda value: value in (2025, 2026), numeric=True)
    if condition == "Final-year students graduating in 2027":
        parts = [decision(condition, facts, "is_final_year"),
                 decision(condition, facts, "graduation_year", lambda value: value == 2027, numeric=True)]
        return combine(condition, parts)
    travel = re.fullmatch(r"Up to (\d+)% travel to client sites", condition)
    if travel:
        return decision(condition, facts, "max_travel_percent", lambda value: value >= int(travel[1]), numeric=True)
    # Contract duration alone does not imply willingness to accept that contract.
    # Unrecognized or compound conditions require an exact explicit answer.
    return ConstraintAssessment(condition=condition, status="unknown")


def combine(condition, parts):
    status = ("not_met" if any(part.status == "not_met" for part in parts)
              else "met" if all(part.status == "met" for part in parts) else "unknown")
    return ConstraintAssessment(condition=condition, status=status,
                                resume_evidence=[evidence for part in parts for evidence in part.resume_evidence],
                                user_fact_keys=list(dict.fromkeys(key for part in parts for key in part.user_fact_keys)))


def experience_requirement(job, prepared):
    minimum = job.experience_min_years
    condition = f"Minimum professional experience: {minimum} years" if minimum is not None else "Professional experience requirement is unspecified"
    profile = prepared.candidate_profile
    if minimum is None or profile.experience_years is None:
        return ConstraintAssessment(condition=condition, status="unknown")
    citations = []
    for evidence in profile.experience_years_evidence:
        for pattern in EXPERIENCE_PATTERNS:
            match = pattern.fullmatch(evidence.quote.strip().lstrip("-•* "))
            if match and float(match[1]) == profile.experience_years:
                citations.append(evidence)
    if not citations:
        return ConstraintAssessment(condition=condition, status="unknown")
    # An advertised maximum is not treated as a strict rejection cap.
    return ConstraintAssessment(condition=condition,
                                status="met" if profile.experience_years >= minimum else "not_met",
                                resume_evidence=citations)


def education_requirement(job, prepared):
    condition = job.education
    facts = prepared.preferences.eligibility_facts
    explicit = "education_requirement_met:" + job.job_id
    if type(facts.get(explicit)) is bool:
        return decision(condition, facts, explicit)
    # Narrow positive rules. Different/missing degrees remain unknown because
    # related disciplines and practical-experience alternatives need review.
    degree = r"\b(?:B\.?\s*Tech|B\.?\s*E\.?|B\.?\s*Sc|Bachelor|M\.?\s*Tech|M\.?\s*Sc|Master|Ph\.?D)\b"
    fields = None
    if condition.startswith("B.E./B.Tech in Computer Science") or condition.startswith("Bachelor's degree in CS"):
        fields = r"\b(?:Computer Science|Computer Engineering|Information Technology|CS|IT)\b"
    elif condition.startswith("B.E./B.Tech or M.Tech in Electronics"):
        fields = r"\b(?:Electronics|Electrical)\b"
    elif condition.startswith("Degree in Statistics"):
        fields = r"\b(?:Statistics|Mathematics|Economics|Computer Science|CS)\b"
    elif condition.startswith("Any degree;"):
        fields = r"\S"
    if fields:
        evidence = [item for fact in prepared.candidate_profile.education for item in fact.evidence
                    if re.search(degree, item.quote, re.I) and re.search(fields, item.quote, re.I)
                    and not re.search(r"\b(?:pursuing|graduating|expected|incomplete|no degree)\b", item.quote, re.I)]
        if evidence:
            return ConstraintAssessment(condition=condition, status="met", resume_evidence=evidence)
    return ConstraintAssessment(condition=condition, status="unknown")
