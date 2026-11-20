"""Validate and prepare the 250-job synthetic dataset locally.

Run: python -m app.dataset.dataset_loader
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from app.config import DATASET_JOB_IDS
from app.schemas import JobRecord

VALIDATION_VERSION = "jobfit-dataset-v2"


@dataclass
class PreparedDataset:
    jobs: list[JobRecord]
    evidence: list[dict]
    report: dict


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON object contains a duplicate field")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("Non-finite JSON numbers are not supported")


def _summary(jobs: list[JobRecord]) -> dict:
    lengths = [len(job.description) for job in jobs]
    words = [len(job.description.split()) for job in jobs]

    def lengths_summary(values):
        return {
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "mean": round(sum(values) / len(values), 1) if values else None,
        }

    return {
        "domains": dict(sorted(Counter(job.domain for job in jobs).items())),
        "seniority": dict(sorted(Counter(job.seniority for job in jobs).items())),
        "work_modes": dict(sorted(Counter(job.work_mode for job in jobs).items())),
        "company_count": len({job.company for job in jobs}),
        "jobs_with_hard_constraints": sum(bool(job.hard_constraints) for job in jobs),
        "unknowns": {
            "work_mode": [job.job_id for job in jobs if job.work_mode == "unknown"],
            "location_city": [job.job_id for job in jobs if job.location_city is None],
            "experience": [job.job_id for job in jobs if (
                job.experience_min_years is None or job.experience_max_years is None
            )],
        },
        "description_characters": lengths_summary(lengths),
        "description_whitespace_words": lengths_summary(words),
        "length_note": "Word counts are whitespace counts, not model token counts.",
    }


def _skill_evidence(job: JobRecord, source_line: int) -> dict:
    result = {"job_id": job.job_id, "source_line": source_line}
    for field in ("required_skills", "preferred_skills"):
        result[field] = []
        for skill in getattr(job, field):
            start = job.description.index(skill)
            # Use the complete source line containing this occurrence as context.
            line_start = job.description.rfind("\n", 0, start) + 1
            line_end = job.description.find("\n", start)
            if line_end == -1:
                line_end = len(job.description)
            result[field].append({
                "skill": skill, "job_id": job.job_id, "section": "description",
                "quote": job.description[line_start:line_end],
                "start": start, "end": start + len(skill),
            })
    return result


def prepare_dataset(source: Path) -> PreparedDataset:
    """Validate every input line and require the complete synthetic dataset.

    A repeated ID is rejected; the first valid occurrence wins. Identical content
    under different IDs is flagged and retained for explicit review.
    """
    source = source.resolve()
    raw = source.read_bytes()
    lines = raw.splitlines()
    accepted: dict[str, tuple[JobRecord, int]] = {}
    fingerprints: dict[str, str] = {}
    rejected, suspicious = [], []
    for number, line in enumerate(lines, 1):
        try:
            if not line.strip():
                raise ValueError("Blank JSONL line")
            data = json.loads(line.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys,
                              parse_constant=_reject_constant)
            job = JobRecord.model_validate(data)
        except (UnicodeDecodeError, ValueError, ValidationError) as exc:
            if isinstance(exc, ValidationError):
                details = [{"field": ".".join(map(str, error["loc"])),
                            "type": error["type"], "message": error["msg"]}
                           for error in exc.errors(include_input=False, include_url=False)]
            else:
                details = [{"type": type(exc).__name__, "message": str(exc)}]
            rejected.append({"line": number, "reason": "invalid_record", "details": details})
            continue
        if job.job_id in accepted:
            first_job, first_line = accepted[job.job_id]
            rejected.append({
                "line": number, "job_id": job.job_id, "first_line": first_line,
                "reason": "duplicate_record" if job == first_job else "conflicting_job_id",
            })
            continue
        content = job.model_dump(mode="json", exclude={"job_id"})
        fingerprint = hashlib.sha256(_json(content).encode()).hexdigest()
        if fingerprint in fingerprints:
            suspicious.append({"job_id": job.job_id, "line": number,
                               "same_content_as": fingerprints[fingerprint],
                               "reason": "identical_content_different_id_retained"})
        else:
            fingerprints[fingerprint] = job.job_id
        accepted[job.job_id] = (job, number)

    # Source IDs have fixed-width numeric suffixes, so lexical order is stable.
    jobs = [accepted[key][0] for key in sorted(accepted)]
    issues = [] if lines else ["Input file is empty"]
    if set(accepted) != DATASET_JOB_IDS:
        issues.append("Expected exactly syn_job_001 through syn_job_250")
    report = {
        "validation_version": VALIDATION_VERSION,
        "dataset_id": "synthetic_india_ai_sw_v2", "is_synthetic": True,
        "source": {"filename": source.name, "sha256": hashlib.sha256(raw).hexdigest(),
                   "bytes": len(raw)},
        "configuration": {"subset": "all"},
        "status": "requires_review" if rejected or suspicious or issues else "ok",
        "counts": {"input_lines": len(lines), "rejected": len(rejected), "output": len(jobs)},
        "rejected_records": rejected, "suspicious_records": suspicious, "issues": issues,
        "output_summary": _summary(jobs),
        "warnings": [
            "Fictional jobs: dates do not establish live vacancies.",
            "Descriptions are untrusted source data; embedded instructions are preserved, never executed.",
        ],
    }
    evidence = [_skill_evidence(job, accepted[job.job_id][1]) for job in jobs]
    return PreparedDataset(jobs, evidence, report)


def write_dataset(dataset: PreparedDataset, output_dir: Path, source: Path) -> None:
    """Write reproducible artifacts, replacing each file atomically."""
    output_dir, source = output_dir.resolve(), source.resolve()
    if output_dir == source.parent or source.parent in output_dir.parents:
        raise ValueError("Write derived output outside the input dataset folder")
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "jobs.jsonl": "".join(_json(job.model_dump(mode="json")) + "\n" for job in dataset.jobs),
        "skill_evidence.jsonl": "".join(_json(item) + "\n" for item in dataset.evidence),
    }
    report = dict(dataset.report)
    report["outputs"] = {name: {"sha256": hashlib.sha256(content.encode("utf-8")).hexdigest()}
                         for name, content in outputs.items()}
    outputs["validation_report.json"] = json.dumps(report, ensure_ascii=False, sort_keys=True,
                                                  indent=2, allow_nan=False) + "\n"
    for filename, content in outputs.items():
        # Reject symlink targets, including links into the source folder.
        target = output_dir / filename
        if target.is_symlink():
            raise ValueError("Output files must not be symlinks")
    for filename, content in outputs.items():
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                             dir=output_dir, delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(content)
            os.replace(temporary, output_dir / filename)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def main(argv=None) -> int:
    from app.config import get_settings

    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    try:
        settings = get_settings()
        dataset = prepare_dataset(settings.dataset_path)
        output = settings.artifacts_dir / "dataset" / "all"
        write_dataset(dataset, output, settings.dataset_path)
    except ValidationError as exc:
        fields = sorted({".".join(map(str, error["loc"])) for error in exc.errors()})
        print("Configuration validation failed. Check fields: " + ", ".join(fields), file=sys.stderr)
        return 1
    except (OSError, ValueError) as exc:
        print(f"Dataset preparation failed: {exc}", file=sys.stderr)
        return 1
    counts = dataset.report["counts"]
    print(f"Synthetic dataset: {counts['input_lines']} input lines, "
          f"{counts['rejected']} rejected, {counts['output']} prepared jobs.")
    print(f"Status: {dataset.report['status']}. Artifacts: {output}")
    return 0 if dataset.report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
