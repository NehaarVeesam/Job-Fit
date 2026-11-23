"""Assess job requirements against the candidate profile."""

import re

from app.schemas import CandidateProfile, ConstraintAssessment, JobRecord, Preferences

COMMON_COMPUTING_REQUIREMENT = (
    "B.E./B.Tech in Computer Science, IT or a related field, "
    "or equivalent practical experience"
)

COMPLETED_COMPUTING_DEGREE = re.compile(
    r"(?:Completed|Earned)\s+(?:a\s+)?B\.?(?:E|Tech)\.?\s+in\s+"
    r"(?:Computer Science|Information Technology)\.?",
    re.IGNORECASE,
)

def assess_experience(
    job: JobRecord,
    profile: CandidateProfile,
) -> ConstraintAssessment | None:
    """Compare the stated experience duration with the job's minimum."""
    minimum = job.experience_min_years

    # A job with no positive minimum has no experience threshold to check.
    if minimum is None or minimum == 0:
        return None

    requirement = f"At least {minimum:g} years of experience"
    years = profile.experience_years

    if years is None:
        return ConstraintAssessment(
            requirement=requirement,
            status="unknown",
            reason="The resume has no clear total experience duration.",
        )

    return ConstraintAssessment(
        requirement=requirement,
        status="met" if years >= minimum else "not_met",
        reason=f"The resume states {years:g} years. The job requires {minimum:g}.",
        resume_evidence=profile.experience_years_evidence,
    )

def assess_education(
    job: JobRecord,
    profile: CandidateProfile,
) -> ConstraintAssessment:
    """Check for a completed computing degree supported by a resume quote."""
    requirement = job.education
    accepts_computing_degree = (
        requirement == COMMON_COMPUTING_REQUIREMENT
        or requirement.startswith("Any degree;")
    )

    if accepts_computing_degree:
        for fact in profile.education:
            for evidence in fact.evidence:
                if COMPLETED_COMPUTING_DEGREE.fullmatch(evidence.quote.strip()):
                    return ConstraintAssessment(
                        requirement=requirement,
                        status="met",
                        reason="The resume states that the candidate completed an accepted degree.",
                        resume_evidence=[evidence],
                    )

    return ConstraintAssessment(
        requirement=requirement,
        status="unknown",
        reason="The current education rule could not verify this requirement.",
    )


def assess_hard_constraints(
    job: JobRecord,
    preferences: Preferences,
) -> list[ConstraintAssessment]:
    """Assess each hard requirement using the candidate's direct answer."""
    assessments: list[ConstraintAssessment] = []

    for requirement in job.hard_constraints:
        answer = preferences.eligibility_answers.get(requirement)

        if answer is None:
            status = "unknown"
            reason = "The candidate has not answered this requirement."
        elif answer:
            status = "met"
            reason = "The candidate confirmed they meet this requirement."
        else:
            status = "not_met"
            reason = "The candidate confirmed they cannot meet this requirement."

        assessments.append(
            ConstraintAssessment(
                requirement=requirement,
                status=status,
                reason=reason,
                candidate_answer=answer,
            )
        )

    return assessments