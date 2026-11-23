"""Apply recommendation rules to shortlisted jobs."""

from collections.abc import Iterable

from app.matching.constraints import (
    assess_education,
    assess_experience,
    assess_hard_constraints,
)
from app.matching.scoring import score_job_skills
from app.schemas import (
    AssessmentStatus,
    JobAssessment,
    JobRecord,
    MatchResponse,
    PreparedResume,
)


MIN_REQUIRED_COVERAGE = 60.0


def assess_job(job: JobRecord, prepared: PreparedResume) -> JobAssessment:
    """Check one job's requirements and record any reasons to exclude it."""
    profile = prepared.candidate_profile
    experience = assess_experience(job, profile)
    education = assess_education(job, profile)
    hard_constraints = assess_hard_constraints(job, prepared.preferences)

    checks = [education, *hard_constraints]
    if experience is not None:
        checks.insert(0, experience)

    statuses = {check.status for check in checks}
    eligibility: AssessmentStatus = "met"
    if "not_met" in statuses:
        eligibility = "not_met"
    elif "unknown" in statuses:
        eligibility = "unknown"

    rejection_reasons = [
        f"{check.requirement}: {check.reason}"
        for check in checks
        if check.status == "not_met"
    ]
    warnings = [
        f"{check.requirement}: {check.reason}"
        for check in checks
        if check.status == "unknown"
    ]

    # An unanswered hard requirement blocks a recommendation pending confirmation.
    for check in hard_constraints:
        if check.status == "unknown":
            rejection_reasons.append(
                f"Hard requirement needs confirmation: {check.requirement}"
            )

    skill_score = score_job_skills(job, profile)
    coverage = skill_score.required
    if coverage.total_count:
        required_pct = 100 * coverage.matched_count / coverage.total_count
        if required_pct < MIN_REQUIRED_COVERAGE:
            rejection_reasons.append(
                f"Required skill coverage is {required_pct:.1f}%. "
                f"The recommendation minimum is {MIN_REQUIRED_COVERAGE:g}%."
            )

    return JobAssessment(
        job=job,
        eligibility=eligibility,
        experience=experience,
        education=education,
        hard_constraints=hard_constraints,
        skill_score=skill_score,
        rejection_reasons=rejection_reasons,
        warnings=warnings,
    )


def match_candidates(
    jobs: Iterable[JobRecord],
    prepared: PreparedResume,
) -> MatchResponse:
    """Assess each shortlisted job and keep their incoming order."""
    recommendations: list[JobAssessment] = []
    excluded: list[JobAssessment] = []

    for job in jobs:
        assessment = assess_job(job, prepared)

        if assessment.rejection_reasons:
            excluded.append(assessment)
        else:
            recommendations.append(assessment)

    return MatchResponse(
        status="matches_found" if recommendations else "no_suitable_matches",
        recommendations=recommendations,
        excluded=excluded,
    )