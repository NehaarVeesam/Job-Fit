"""Review uncovered required skills against cited resume lines."""

import json

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from openai import APIError

from app.matching.generation import create_qwen_model
from app.schemas import (
    JobAssessment,
    JobRecord,
    PreparedResume,
    ResumeEvidence,
    SemanticEvidenceChoice,
    SemanticEvidenceReport,
    SemanticSkillFinding,
)


REVIEW_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Use the job description only to clarify what the requested skill means. "
            "Judge whether the resume demonstrates that skill in any project or domain. "
            "Choose supported:<line_id> when the candidate describes performing the "
            "core work of the skill, even if the skill name or acronym is absent. "
            "Choose related:<line_id> when the line demonstrates only part of the "
            "practice or experience with a different named tool. "
            "For a requirement that names a specific tool, look for that tool "
            "or an unambiguous alias. "
            "Choose unclear when no line gives relevant evidence. "
        ),
        ("human", "{review_data}"),
    ]
)


def _review_skill(
    job: JobRecord,
    skill: str,
    lines: dict[str, ResumeEvidence],
) -> SemanticSkillFinding:
    """Ask Qwen for one decision and resolve it to a resume quote."""
    choices = ["unclear"] + [
        f"{status}:{line_id}"
        for status in ("supported", "related")
        for line_id in lines
    ]

    # Build the allowed answers from this resume's actual lines.
    schema = SemanticEvidenceChoice.model_json_schema()
    schema["properties"]["choice"]["enum"] = choices

    model = create_qwen_model().bind(
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "semantic_evidence_choice",
                "schema": schema,
            },
        }
    )
    parser = PydanticOutputParser(pydantic_object=SemanticEvidenceChoice)

    payload = {
        "skill": skill,
        "job_title": job.title,
        "job_description": job.description,
        "resume_lines": {line_id: item.quote for line_id, item in lines.items()},
    }
    result = (REVIEW_PROMPT | model | parser).invoke(
        {"review_data": json.dumps(payload, ensure_ascii=False)}
    )

    # Check that the citation can be resolved before returning the model's judgment.
    if result.choice not in choices:
        raise ValueError("Qwen selected a decision outside the allowed list.")

    status, _, line_id = result.choice.partition(":")
    return SemanticSkillFinding(
        skill=skill,
        status=status,
        resume_evidence=[lines[line_id]] if line_id else [],
    )


def review_required_skill_gaps(
    assessment: JobAssessment,
    prepared: PreparedResume,
) -> SemanticEvidenceReport:
    """Review required skills that exact matching missed."""
    job = assessment.job
    missing_skills = [
        item.skill
        for item in assessment.skill_score.assessments
        if item.requirement_type == "required" and not item.matched
    ]
    if not missing_skills:
        return SemanticEvidenceReport(job_id=job.job_id, status="not_needed")

    lines: dict[str, ResumeEvidence] = {}
    for page in prepared.parsed_resume.pages:
        for number, raw_line in enumerate(page.text.splitlines(), start=1):
            quote = raw_line.strip()
            if quote:
                lines[f"p{page.page}_l{number}"] = ResumeEvidence(
                    quote=quote, page=page.page
                )

    findings: list[SemanticSkillFinding] = []
    warnings: list[str] = []

    for skill in missing_skills:
        try:
            findings.append(_review_skill(job, skill, lines))
        except APIError as error:
            # Preserve the completed findings if a later Ollama request fails.
            warnings.append(f"{skill}: Ollama request failed ({type(error).__name__}).")
            break
        except OutputParserException:
            warnings.append(f"{skill}: Qwen's response did not match the schema.")
        except ValueError as error:
            warnings.append(f"{skill}: {error}")

    if not warnings:
        status = "reviewed"
    elif findings:
        status = "partial"
    else:
        status = "unavailable"

    return SemanticEvidenceReport(
        job_id=job.job_id,
        status=status,
        findings=findings,
        warning=" ".join(warnings) or None,
    )
