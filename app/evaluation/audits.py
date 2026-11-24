"""Check matching outcomes and evidence against the evaluation fixtures."""

from collections.abc import Mapping
from dataclasses import dataclass

from app.evaluation.fixtures import JobJudgment
from app.schemas import MatchResponse, PreparedResume


@dataclass(frozen=True)
class MatchAudit:
    """The recommendation checks for one case and retrieval mode."""

    no_match_correct: bool
    suitable_ids: list[str]
    unsuitable_ids: list[str]
    unjudged_ids: list[str]
    evidence_errors: list[str]


def audit_matches(
    response: MatchResponse,
    prepared: PreparedResume,
    judgments: Mapping[str, JobJudgment],
    expected_no_match: bool,
) -> MatchAudit:
    """Compare recommendations with labels and verify their source quotes."""
    suitable_ids: list[str] = []
    unsuitable_ids: list[str] = []
    unjudged_ids: list[str] = []
    evidence_errors: list[str] = []

    pages = {page.page: page.text for page in prepared.parsed_resume.pages}

    for item in response.recommendations:
        job_id = item.job.job_id
        judgment = judgments.get(job_id)
        if judgment is None:
            unjudged_ids.append(job_id)
        elif judgment.suitable:
            suitable_ids.append(job_id)
        else:
            unsuitable_ids.append(job_id)

    for item in response.recommendations + response.excluded:
        job = item.job
        citations = []

        for skill in item.skill_score.assessments:
            if skill.job_quote not in job.description:
                evidence_errors.append(f"{job.job_id}: missing job quote for {skill.skill}")
            citations.extend(skill.resume_evidence)

        checks = [item.education, *item.hard_constraints]
        if item.experience is not None:
            checks.append(item.experience)
        for check in checks:
            citations.extend(check.resume_evidence)

        for evidence in citations:
            if evidence.quote not in pages.get(evidence.page, ""):
                evidence_errors.append(f"{job.job_id}: missing resume quote on page {evidence.page}")

    actual_no_match = not response.recommendations
    return MatchAudit(
        no_match_correct=actual_no_match == expected_no_match,
        suitable_ids=suitable_ids,
        unsuitable_ids=unsuitable_ids,
        unjudged_ids=unjudged_ids,
        evidence_errors=evidence_errors,
    )