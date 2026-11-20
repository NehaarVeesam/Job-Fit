"""Prepare the supplied JSONL corpus without changing input or calling services.

Run: python -m app.dataset_loader
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from app.schemas import JobRecord

VALIDATION_VERSION = "jobfit-dataset-v1"


@dataclass(frozen=True)
class CorpusOptions:
    subset: Literal["all", "first_100"] = "all"
    recency_enabled: bool = False
    reference_date: date = date(2026, 11, 18)
    recency_window_days: int = 90

    def __post_init__(self):
        if self.subset not in ("all", "first_100"):
            raise ValueError("Subset must be all or first_100")
        if self.recency_window_days < 1:
            raise ValueError("Recency window must be positive")


@dataclass
class PreparedCorpus:
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


def prepare_corpus(source: Path, options: CorpusOptions | None = None) -> PreparedCorpus:
    """Validate every physical input line, then apply optional corpus filters.

    A repeated ID is rejected; the first valid occurrence wins. Identical content
    under different IDs is flagged and retained for explicit review.
    """
    options = options or CorpusOptions()
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
    valid = [accepted[key][0] for key in sorted(accepted)]
    cutoff = options.reference_date - timedelta(days=options.recency_window_days)
    stale = [job.job_id for job in valid if job.posted_at < cutoff]
    future = [job.job_id for job in valid if job.posted_at > options.reference_date]
    jobs, excluded = [], []
    for job in valid:
        if options.subset == "first_100" and not 1 <= int(job.job_id.removeprefix("syn_job_")) <= 100:
            reason = "outside_first_100_ids"
        elif options.recency_enabled and job.posted_at < cutoff:
            reason = "older_than_recency_window"
        elif options.recency_enabled and job.posted_at > options.reference_date:
            reason = "after_reference_date"
        else:
            jobs.append(job)
            continue
        excluded.append({"job_id": job.job_id, "reason": reason})

    issues = [] if lines else ["Input file is empty"]
    report = {
        "validation_version": VALIDATION_VERSION,
        "dataset_id": "synthetic_india_ai_sw_v2", "is_synthetic": True,
        "source": {"filename": source.name, "sha256": hashlib.sha256(raw).hexdigest(),
                   "bytes": len(raw)},
        "configuration": {
            "subset": options.subset, "recency_enabled": options.recency_enabled,
            "reference_date": options.reference_date.isoformat(),
            "recency_window_days": options.recency_window_days,
            "recency_cutoff_inclusive": cutoff.isoformat(),
        },
        "status": "requires_review" if rejected or suspicious or issues else "ok",
        "counts": {"input_lines": len(lines), "rejected": len(rejected),
                   "valid_unique": len(valid), "excluded": len(excluded), "output": len(jobs)},
        "rejected_records": rejected, "excluded_records": excluded,
        "suspicious_records": suspicious, "issues": issues,
        "date_flags": {"older_than_window": stale, "after_reference_date": future},
        "validated_summary": _summary(valid), "output_summary": _summary(jobs),
        "warnings": [
            "Fictional jobs: dates do not establish live vacancies.",
            "Descriptions are untrusted source data; embedded instructions are preserved, never executed.",
        ],
    }
    evidence = [_skill_evidence(job, accepted[job.job_id][1]) for job in jobs]
    return PreparedCorpus(jobs, evidence, report)


def write_corpus(corpus: PreparedCorpus, output_dir: Path, source: Path) -> None:
    """Write reproducible artifacts, replacing each file atomically."""
    output_dir, source = output_dir.resolve(), source.resolve()
    if output_dir == source.parent or source.parent in output_dir.parents:
        raise ValueError("Write derived output outside the input dataset folder")
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "jobs.jsonl": "".join(_json(job.model_dump(mode="json")) + "\n" for job in corpus.jobs),
        "skill_evidence.jsonl": "".join(_json(item) + "\n" for item in corpus.evidence),
    }
    report = dict(corpus.report)
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
    parser.add_argument("--subset", choices=("all", "first_100"))
    parser.add_argument("--recency", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--reference-date", type=date.fromisoformat)
    parser.add_argument("--window-days", type=int)
    args = parser.parse_args(argv)
    try:
        settings = get_settings()
        options = CorpusOptions(
            subset=args.subset or settings.corpus_subset,
            recency_enabled=settings.recency_filter_enabled if args.recency is None else args.recency,
            reference_date=args.reference_date or settings.reference_date,
            recency_window_days=(settings.recency_window_days if args.window_days is None else args.window_days),
        )
        corpus = prepare_corpus(settings.dataset_path, options)
        mode = options.subset
        if options.recency_enabled:
            mode += f"_recent_{options.reference_date.isoformat()}_{options.recency_window_days}d"
        output = settings.artifacts_dir / "dataset" / mode
        write_corpus(corpus, output, settings.dataset_path)
    except ValidationError as exc:
        fields = sorted({".".join(map(str, error["loc"])) for error in exc.errors()})
        print("Configuration validation failed. Check fields: " + ", ".join(fields), file=sys.stderr)
        return 1
    except (OSError, ValueError) as exc:
        print(f"Dataset preparation failed: {exc}", file=sys.stderr)
        return 1
    counts = corpus.report["counts"]
    print(f"Synthetic corpus: {counts['input_lines']} input lines; {counts['valid_unique']} valid unique; "
          f"{counts['rejected']} rejected; {counts['excluded']} excluded; {counts['output']} output.")
    print(f"Status: {corpus.report['status']}. Artifacts: {output}")
    return 0 if corpus.report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
