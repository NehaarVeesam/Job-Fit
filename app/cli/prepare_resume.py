"""Prepare a PDF resume and write its validated profile as JSON."""

import argparse
from pathlib import Path

from app.config import Settings
from app.resumes.profile import load_skill_vocabulary, prepare_resume
from app.resumes.resume_parser import parse_resume
from app.schemas import Preferences


def main() -> None:
    """Read command-line options and prepare the supplied resume."""
    parser = argparse.ArgumentParser(description="Prepare a JobFit resume.")
    parser.add_argument("pdf_path", type=Path)
    parser.add_argument("--target-role")
    parser.add_argument("--location-city")
    parser.add_argument("--work-mode", choices=("hybrid", "onsite", "remote"))
    parser.add_argument("--output", type=Path, help="Path for the prepared JSON.")
    args = parser.parse_args()

    settings = Settings()
    evidence_path = settings.prepared_data_dir / "skill_evidence.jsonl"

    if args.output is not None:
        # Keep the resume output separate from the PDF and the job catalog.
        catalog_paths = (
            settings.raw_jobs_path,
            settings.prepared_jobs_path,
            evidence_path,
            settings.prepared_data_dir / "validation_report.json",
        )
        protected_paths = {args.pdf_path.resolve()}
        protected_paths.update(path.resolve() for path in catalog_paths)
        if args.output.resolve() in protected_paths:
            parser.error("--output must be separate from the input PDF and job catalog.")

    preferences = Preferences(
        target_role=args.target_role,
        location_city=args.location_city,
        work_mode=args.work_mode,
    )

    vocabulary = load_skill_vocabulary(evidence_path)
    parsed_resume = parse_resume(args.pdf_path)
    prepared = prepare_resume(parsed_resume, vocabulary, preferences)
    payload = prepared.model_dump_json(indent=2)

    if args.output is None:
        print(payload)
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n", encoding="utf-8")
    print(f"Prepared resume saved to {args.output}")


if __name__ == "__main__":
    main()
