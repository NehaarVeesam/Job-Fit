"""Load and validate the labeled JobFit evaluation cases."""

from collections.abc import Collection
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, ValidationError

from app.schemas import Preferences


class EvaluationCase(BaseModel):
    """One synthetic resume and its expected matching outcome."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    resume_pages: list[str] = Field(min_length=1)
    preferences: Preferences = Field(default_factory=Preferences)
    expected_no_match: StrictBool


class EvaluationSuite(BaseModel):
    """The cases and shortlist size used for evaluation."""

    model_config = ConfigDict(extra="forbid")

    top_k: int = Field(ge=1)
    cases: list[EvaluationCase] = Field(min_length=1)


class JobJudgment(BaseModel):
    """A human relevance and suitability judgment for a case and job."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    job_id: str
    relevance: Literal[0, 1, 2]
    suitable: StrictBool
    reason: str


def load_evaluation(
    suite_path: Path,
    judgments_path: Path,
    catalog_ids: Collection[str],
) -> tuple[EvaluationSuite, dict[str, dict[str, JobJudgment]]]:
    """Load the suite and index its judgments by case ID and job ID."""
    suite = EvaluationSuite.model_validate_json(suite_path.read_text(encoding="utf-8"))
    judgments = {case.case_id: {} for case in suite.cases}

    if len(judgments) != len(suite.cases):
        raise ValueError("Evaluation case IDs must be unique.")

    with judgments_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                judgment = JobJudgment.model_validate_json(line)
            except ValidationError as error:
                raise ValueError(f"Invalid judgment on line {line_number}: {error}") from error

            if judgment.case_id not in judgments:
                raise ValueError(f"Unknown case ID on line {line_number}: {judgment.case_id}")
            if judgment.job_id not in catalog_ids:
                raise ValueError(f"Unknown job ID on line {line_number}: {judgment.job_id}")

            case_judgments = judgments[judgment.case_id]
            if judgment.job_id in case_judgments:
                raise ValueError(f"Duplicate case-job judgment on line {line_number}.")

            case_judgments[judgment.job_id] = judgment

    for case in suite.cases:
        # A positive case needs a labeled recommendation to evaluate against.
        has_suitable_job = any(item.suitable for item in judgments[case.case_id].values())
        if case.expected_no_match == has_suitable_job:
            raise ValueError(f"Suitability labels conflict with case {case.case_id!r}.")

    return suite, judgments