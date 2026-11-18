"""Evidence coverage and recommendation gates; no model calls."""

import re
from dataclasses import dataclass

from app.constraints import education_requirement, experience_requirement, hard_constraint
from app.filters import job_passes_preferences, preference_warnings
from app.profile import CONTACT, UNCERTAIN
from app.schemas import (
    ConstraintAssessment, CoverageScore, JobEvidence, JobMatch, JobRecord,
    MatchReason, RelevanceScore, SkillAssessment,
)


@dataclass
class Assessment:
    job: JobRecord
    relevance: RelevanceScore | None
    skills: list[SkillAssessment]
    required: CoverageScore
    preferred: CoverageScore
    experience: ConstraintAssessment
    education: ConstraintAssessment
    constraints: list[ConstraintAssessment]
    eligibility: str
    warnings: list[str]


def source_evidence(job, skill):
    position = job.description.index(skill)
    start = job.description.rfind("\n", 0, position) + 1
    end = job.description.find("\n", position)
    quote = job.description[start:end if end != -1 else len(job.description)]
    return JobEvidence(job_id=job.job_id, quote=quote)


def assess_job(job, prepared, relevance=None):
    skills = []
    for kind, names in (("required", job.required_skills), ("preferred", job.preferred_skills)):
        for name in names:
            pattern = re.compile(r"(?<![\w.+#])" + re.escape(name) + r"(?![\w+#])", re.I)
            evidence = [item for fact in prepared.candidate_profile.skills
                        if fact.text.strip().casefold() == name.strip().casefold() for item in fact.evidence
                        if pattern.search(item.quote) and not CONTACT.search(item.quote)
                        and not UNCERTAIN.search(item.quote)]
            skills.append(SkillAssessment(
                skill=name, requirement_type=kind,
                status="evidenced" if evidence else "not_evidenced_in_resume",
                job_evidence=source_evidence(job, name), resume_evidence=evidence,
            ))
    def coverage(kind):
        entries = [entry for entry in skills if entry.requirement_type == kind]
        return CoverageScore(total_count=len(entries), evidenced_count=sum(entry.status == "evidenced" for entry in entries))
    experience = experience_requirement(job, prepared)
    education = education_requirement(job, prepared)
    constraints = [hard_constraint(condition, prepared.preferences.eligibility_facts) for condition in job.hard_constraints]
    checks = [experience, education] + constraints
    eligibility = ("not_met" if any(item.status == "not_met" for item in checks)
                   else "met" if all(item.status == "met" for item in checks) else "unknown")
    warnings = preference_warnings(job, prepared.preferences)
    warnings += [f"Unresolved requirement: {item.condition}" for item in checks if item.status == "unknown"]
    return Assessment(job, relevance, skills, coverage("required"), coverage("preferred"),
                      experience, education, constraints, eligibility, warnings)


def rejection_reasons(assessment, prepared, settings, *, dense_relevance=None):
    reasons = []
    if not job_passes_preferences(assessment.job, prepared.preferences):
        reasons.append("Explicit preference mismatch")
    if assessment.eligibility == "not_met":
        reasons.append("Known requirement failure")
    if assessment.eligibility == "unknown" and prepared.preferences.unknown_metadata_policy == "exclude":
        reasons.append("Unresolved eligibility excluded by strict unknown policy")
    score = assessment.relevance
    if score is not None and score.score_type == "rrf":
        if score.value <= 0:
            reasons.append("No positive fusion score")
        score = dense_relevance
    if score is not None and score.score_type == "bm25":
        if score.value <= 0:
            reasons.append("No positive BM25 keyword evidence")
    elif score is None or score.score_type != "cosine" or score.value < settings.min_cosine_similarity:
        reasons.append("Cosine similarity below the baseline threshold or unavailable")
    if assessment.required.percentage is None or assessment.required.percentage < settings.min_required_coverage:
        reasons.append("Required-skill coverage below the baseline threshold or unavailable")
    minimum = min(settings.min_evidenced_required, assessment.required.total_count)
    if assessment.required.evidenced_count < minimum or assessment.required.evidenced_count == 0:
        reasons.append("Too few evidenced required skills")
    return reasons


def make_match(assessment, strength_order=None, generation_status="deterministic", extra_warnings=(), *, reranker_relevance=None):
    positive = {item.skill: item for item in assessment.skills if item.status == "evidenced"}
    order = strength_order or list(positive)[:3]
    reasons = []
    for name in order:
        item = positive[name]
        reasons.append(MatchReason(
            explanation=f"Your resume mentions {name}; this job lists it as a {item.requirement_type} skill.",
            resume_evidence=item.resume_evidence[0], job_evidence=item.job_evidence,
        ))
    missing = [item.skill for item in assessment.skills if item.status == "not_evidenced_in_resume"]
    warnings = assessment.warnings + list(extra_warnings)
    if missing:
        warnings.append("Not evidenced in resume: " + ", ".join(missing))
    return JobMatch(
        job=assessment.job, relevance=assessment.relevance, required_coverage=assessment.required,
        reranker_relevance=reranker_relevance,
        preferred_coverage=assessment.preferred, eligibility=assessment.eligibility,
        experience_assessment=assessment.experience, education_assessment=assessment.education,
        constraints=assessment.constraints, skills=assessment.skills, reasons=reasons,
        generation_status=generation_status, warnings=warnings,
    )
