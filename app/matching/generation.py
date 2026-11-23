"""Build and render evidence-backed explanations for job recommendations."""

import json

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from openai import APIError

from app.schemas import (
    ExplanationEvidence,
    ExplanationItem,
    ExplanationPlan,
    JobAssessment,
    JobExplanation,
)


def build_explanation_evidence(assessment: JobAssessment) -> ExplanationEvidence:
    """Group existing skill findings into strengths and gaps for one job."""
    strengths: list[ExplanationItem] = []
    gaps: list[ExplanationItem] = []

    for index, skill in enumerate(assessment.skill_score.assessments, start=1):
        item = ExplanationItem(evidence_id=f"skill_{index}", assessment=skill)

        if skill.matched:
            strengths.append(item)
        else:
            gaps.append(item)

    return ExplanationEvidence(
        job_id=assessment.job.job_id,
        job_title=assessment.job.title,
        strengths=strengths,
        gaps=gaps,
    )


def _skill_options(items: list[ExplanationItem]) -> list[dict[str, str]]:
    """Give Qwen the option labels and keep the source quotes in JobFit."""
    return [
        {
            "id": item.evidence_id,
            "skill": item.assessment.skill,
            "requirement_type": item.assessment.requirement_type,
        }
        for item in items
    ]


def create_qwen_model() -> ChatOpenAI:
    """Connect LangChain to the local Qwen model in Ollama."""
    return ChatOpenAI(
        model="qwen2.5:3b",
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        temperature=0,
        timeout=120,
        max_retries=0,
    )


def plan_explanation(evidence: ExplanationEvidence) -> ExplanationPlan:
    """Ask local Qwen to select strengths and gaps as structured data."""
    output_parser = PydanticOutputParser(pydantic_object=ExplanationPlan)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Select one or two evidence IDs from each group when options exist. "
                "Prioritize required skills. Use an empty list for an empty group. "
                "Copy IDs exactly from their own group and return only a JSON object.\n"
                "{format_instructions}",
            ),
            ("human", "Available skill options:\n{options}"),
        ]
    ).partial(format_instructions=output_parser.get_format_instructions())

    schema = ExplanationPlan.model_json_schema()
    properties = schema["properties"]

    # Restrict each list to the evidence available for this job.

    for field_name, items in (
        ("strength_ids", evidence.strengths),
        ("gap_ids", evidence.gaps),
    ):
        allowed_ids = [item.evidence_id for item in items]

        if not allowed_ids:
            properties[field_name] = {"const": []}
            continue

        field = properties[field_name]
        field["items"]["enum"] = allowed_ids
        field["minItems"] = 1
        field["maxItems"] = min(field["maxItems"], len(allowed_ids))

    model = create_qwen_model().bind(
        # Give Ollama the allowed IDs and list limits before it generates the response.
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "explanation_plan",
                "schema": schema,
            },
        },
    )

    options = {
        "strengths": _skill_options(evidence.strengths),
        "gaps": _skill_options(evidence.gaps),
    }

    chain = prompt | model | output_parser
    return chain.invoke({"options": json.dumps(options)})


def _resolve_group(
    label: str,
    options: list[ExplanationItem],
    selected_ids: list[str],
) -> list[ExplanationItem]:
    """Look up the selected IDs in the group they are meant to come from."""
    by_id = {item.evidence_id: item for item in options}

    if options and not selected_ids:
        raise ValueError(f"The plan left the {label} group empty.")
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError(f"The plan repeated an ID in the {label} group.")

    for evidence_id in selected_ids:
        if evidence_id not in by_id:
            raise ValueError(f"{evidence_id!r} is invalid for the {label} group.")

    return [by_id[evidence_id] for evidence_id in selected_ids]


def render_explanation(evidence: ExplanationEvidence, plan: ExplanationPlan) -> str:
    """Validate selected IDs and explain them using the saved quotes."""
    strengths = _resolve_group("strengths", evidence.strengths, plan.strength_ids)
    gaps = _resolve_group("gaps", evidence.gaps, plan.gap_ids)

    lines = [f"Skill evidence for {evidence.job_title}"]

    if strengths:
        lines.append("\nStrengths")
        for item in strengths:
            skill = item.assessment
            quote = skill.resume_evidence[0]
            lines.append(
                f'- {skill.skill} ({skill.requirement_type}): '
                f'page {quote.page} says "{quote.quote}"'
            )

    if gaps:
        lines.append("\nGaps")
        for item in gaps:
            skill = item.assessment
            lines.append(
                f"- {skill.skill} ({skill.requirement_type}): "
                "No resume evidence found."
            )

    # Several skills can cite the same job description line.
    job_quotes = dict.fromkeys(
        item.assessment.job_quote for item in strengths + gaps
    )
    if job_quotes:
        lines.append("\nJob posting quotes")
        lines.extend(f'"{quote}"' for quote in job_quotes)

    return "\n".join(lines)


def explain_recommendation(
    assessment: JobAssessment,
    *,
    use_qwen: bool = False,
) -> JobExplanation:
    """Explain one recommendation with an optional local Qwen plan."""
    evidence = build_explanation_evidence(assessment)

    if use_qwen:
        try:
            plan = plan_explanation(evidence)
            text = render_explanation(evidence, plan)
        except (APIError, ValueError):
            # Continue with the template when the request or plan fails.
            pass
        else:
            return JobExplanation(source="qwen_plan", plan=plan, text=text)

    plan = ExplanationPlan(
        strength_ids=[item.evidence_id for item in evidence.strengths[:2]],
        gap_ids=[item.evidence_id for item in evidence.gaps[:2]],
    )
    return JobExplanation(
        source="template",
        plan=plan,
        text=render_explanation(evidence, plan),
    )
