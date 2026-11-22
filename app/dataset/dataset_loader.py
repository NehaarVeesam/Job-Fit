"""Prepare the synthetic job catalog and write its skill evidence and validation report."""

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

from pydantic import BaseModel, ValidationError

from app.config import Settings
from app.schemas import JobRecord


def iter_job_rows(lines: Iterable[str]) -> Iterator[tuple[int, JobRecord | ValidationError]]:
    """Yield each source line number alongside its validated job or error."""
    for line_number, line in enumerate(lines, start=1):
        try:
            job = JobRecord.model_validate_json(line)
        except ValidationError as error:
            yield line_number, error
        else:
            yield line_number, job


class RejectedRow(BaseModel):
    """The reason a source row was rejected and where to find it."""

    line_number: int
    kind: Literal["validation_error", "duplicate_job_id"]
    message: str


def load_jobs(
    lines: Iterable[str],
) -> tuple[list[JobRecord], list[RejectedRow]]:
    """Keep the first valid row for each job ID and collect rejections."""
    accepted_jobs: list[JobRecord] = []
    rejected_rows: list[RejectedRow] = []
    # Remember the original line so a duplicate points to the accepted record.
    first_seen: dict[str, int] = {}

    for line_number, result in iter_job_rows(lines):
        if isinstance(result, ValidationError):
            rejected_rows.append(
                RejectedRow(
                    line_number=line_number,
                    kind="validation_error",
                    message=result.errors()[0]["msg"],
                )
            )
            continue

        job_id = result.job_id
        if job_id in first_seen:
            rejected_rows.append(
                RejectedRow(
                    line_number=line_number,
                    kind="duplicate_job_id",
                    message=f"{job_id} was already accepted on line {first_seen[job_id]}",
                )
            )
            continue

        first_seen[job_id] = line_number
        accepted_jobs.append(result)

    return accepted_jobs, rejected_rows


RequirementType = Literal["required", "preferred"]


class JobSkillEvidence(BaseModel):
    """One required or preferred skill and its exact quote from a job posting."""

    job_id: str
    skill: str
    requirement_type: RequirementType
    quote: str


def extract_skill_evidence(job: JobRecord) -> list[JobSkillEvidence]:
    """Find a source quote for each listed skill under its expected heading."""
    evidence: list[JobSkillEvidence] = []
    description_lines = job.description.splitlines()

    skill_groups: dict[RequirementType, tuple[str, list[str]]] = {
        "required": ("Must-have skills", job.required_skills),
        "preferred": ("Good to have", job.preferred_skills),
    }

    for requirement_type, (heading, skills) in skill_groups.items():
        if not skills:
            continue

        quote = ""
        skill_text = ""

        # The catalog puts required skills inline and preferred skills on the next line.
        for index, line in enumerate(description_lines):
            cleaned_line = line.strip().removeprefix("- ")

            if cleaned_line.startswith(f"{heading}:"):
                quote = line
                skill_text = cleaned_line.partition(":")[2]
                break

            if cleaned_line == heading and index + 1 < len(description_lines):
                quote = description_lines[index + 1]
                skill_text = quote.strip().removeprefix("- ")
                break

        # Match whole list entries so a skill like R cannot match RAG.
        listed_skills = {name.strip() for name in skill_text.split(",")}

        for skill in skills:
            if not quote or skill not in listed_skills:
                raise ValueError(
                    f"{job.job_id}: {requirement_type} skill {skill!r} "
                    f"was not found under {heading!r}"
                )

            evidence.append(
                JobSkillEvidence(
                    job_id=job.job_id,
                    skill=skill,
                    requirement_type=requirement_type,
                    quote=quote,
                )
            )

    return evidence


def write_jsonl(path: Path, records: Iterable[BaseModel]) -> None:
    """Serialize each model as one UTF-8 JSON line."""
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(record.model_dump_json() + "\n")


def main() -> None:
    """Load and validate the catalog, then publish the prepared files."""
    settings = Settings()
    output_dir = settings.prepared_data_dir

    jobs_path = settings.prepared_jobs_path
    evidence_path = output_dir / "skill_evidence.jsonl"
    report_path = output_dir / "validation_report.json"
    output_paths = (jobs_path, evidence_path, report_path)

    if settings.raw_jobs_path.resolve() in {path.resolve() for path in output_paths}:
        raise SystemExit("The source dataset must be separate from the output files.")

    with settings.raw_jobs_path.open(encoding="utf-8") as dataset:
        accepted_jobs, rejected_rows = load_jobs(dataset)

    skill_evidence = []
    for job in accepted_jobs:
        skill_evidence.extend(extract_skill_evidence(job))

    report = {
        "total_rows": len(accepted_jobs) + len(rejected_rows),
        "accepted_jobs": len(accepted_jobs),
        "rejected_rows": len(rejected_rows),
        "skill_evidence_records": len(skill_evidence),
        "rejections": [row.model_dump(mode="json") for row in rejected_rows],
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    # Stage complete files on the output filesystem before replacing existing copies.
    with TemporaryDirectory(dir=output_dir) as temporary_dir:
        staging_dir = Path(temporary_dir)

        write_jsonl(staging_dir / jobs_path.name, accepted_jobs)
        write_jsonl(staging_dir / evidence_path.name, skill_evidence)
        (staging_dir / report_path.name).write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        # Publish the validation report after the two data files.
        for path in output_paths:
            (staging_dir / path.name).replace(path)

    print(f"Total rows: {report['total_rows']}")
    print(f"Accepted jobs: {report['accepted_jobs']}")
    print(f"Rejected rows: {report['rejected_rows']}")
    print(f"Skill evidence records: {report['skill_evidence_records']}")
    print(f"Prepared dataset: {output_dir}")


if __name__ == "__main__":
    main()
