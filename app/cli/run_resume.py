"""Run the full JobFit matching pipeline from the command line."""

import argparse
import json
from pathlib import Path

from pymupdf import FileDataError

from app.matching.pipeline import ResumeMatchOptions, run_resume_match
from app.schemas import Preferences


def main() -> None:
    """Read command-line options and return the match result as JSON."""
    parser = argparse.ArgumentParser(description="Match a PDF resume against JobFit jobs.")
    parser.add_argument("pdf_path", type=Path, help="Path to a text-based PDF resume.")
    parser.add_argument("--mode", choices=("dense", "bm25", "hybrid"), default="hybrid")
    parser.add_argument("--top-k", type=int, help="Number of jobs to shortlist.")
    parser.add_argument("--target-role")
    parser.add_argument("--location-city")
    parser.add_argument("--work-mode", choices=("hybrid", "onsite", "remote"))
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--use-qwen", action="store_true", help="Use Qwen to plan explanations.")
    parser.add_argument("--review-semantic-evidence", action="store_true")
    parser.add_argument("--eligibility-answers", type=Path, help="JSON file of direct answers.")
    parser.add_argument("--output", type=Path, help="Path to a new result JSON file.")
    args = parser.parse_args()

    try:
        answers = {}
        if args.eligibility_answers is not None:
            answers = json.loads(args.eligibility_answers.read_text(encoding="utf-8-sig"))

        preferences = Preferences(
            target_role=args.target_role,
            location_city=args.location_city,
            work_mode=args.work_mode,
            eligibility_answers=answers,
        )
        options = ResumeMatchOptions(
            mode=args.mode,
            top_k=args.top_k,
            rerank=args.rerank,
            use_qwen=args.use_qwen,
            review_semantic_evidence=args.review_semantic_evidence,
        )
        result = run_resume_match(args.pdf_path, preferences, options)
        payload = result.model_dump_json(indent=2)

        if args.output is None:
            print(payload)
            return

        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation protects existing files from an accidental overwrite.
        with args.output.open("x", encoding="utf-8") as output:
            output.write(payload + "\n")
    except (OSError, ValueError, FileDataError) as error:
        parser.error(str(error))

    print(f"Status: {result.match_response.status}")
    print(f"Recommended jobs: {len(result.match_response.recommendations)}")
    print(f"Saved result to {args.output}")


if __name__ == "__main__":
    main()
