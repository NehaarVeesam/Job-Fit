"""Compare a job's skill requirements with the candidate's resume evidence."""

from app.dataset.dataset_loader import extract_skill_evidence
from app.schemas import (
    CandidateProfile,
    JobRecord,
    JobSkillScore,
    ResumeEvidence,
    SkillAssessment,
    SkillCoverage,
)


def _coverage(assessments: list[SkillAssessment]) -> SkillCoverage:
    """Calculate the evidence coverage for one group of skills."""
    total_count = len(assessments)
    matched_count = sum(item.matched for item in assessments)

    return SkillCoverage(
        matched_count=matched_count,
        total_count=total_count,
        percentage=round(100 * matched_count / total_count, 1) if total_count else None,
    )


def score_job_skills(job: JobRecord, profile: CandidateProfile) -> JobSkillScore:
    """Check each listed job skill for supporting resume evidence."""
    resume_by_skill: dict[str, list[ResumeEvidence]] = {}

    for fact in profile.skills:
        key = fact.text.strip().casefold()
        resume_by_skill.setdefault(key, []).extend(fact.evidence)

    assessments: list[SkillAssessment] = []

    # Reuse Module 02's evidence extractor for this shortlisted job.
    for job_evidence in extract_skill_evidence(job):
        key = job_evidence.skill.strip().casefold()
        resume_quotes = resume_by_skill.get(key, [])

        assessments.append(
            SkillAssessment(
                skill=job_evidence.skill,
                requirement_type=job_evidence.requirement_type,
                matched=bool(resume_quotes),
                job_quote=job_evidence.quote,
                resume_evidence=resume_quotes,
            )
        )

    required = [item for item in assessments if item.requirement_type == "required"]
    preferred = [item for item in assessments if item.requirement_type == "preferred"]

    return JobSkillScore(
        required=_coverage(required),
        preferred=_coverage(preferred),
        assessments=assessments,
    )