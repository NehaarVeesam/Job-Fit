"""Offline Module 0 check: configuration, one source record, and example contracts."""

import sys

from pydantic import ValidationError

from app.config import PROJECT_ROOT, get_settings
from app.schemas import CandidateProfile, JobRecord, MatchResponse, Preferences


def main() -> int:
    try:
        settings = get_settings()
        with settings.dataset_path.open(encoding="utf-8") as handle:
            JobRecord.model_validate_json(next(handle))
        for filename, schema in (
            ("candidate_profile.json", CandidateProfile), ("preferences.json", Preferences),
            ("match_response.json", MatchResponse), ("no_match_response.json", MatchResponse),
        ):
            schema.model_validate_json((PROJECT_ROOT / "examples" / filename).read_text(encoding="utf-8"))
    except ValidationError as error:
        fields = ", ".join(".".join(map(str, entry["loc"])) for entry in error.errors())
        print(f"Setup validation failed. Check fields: {fields}", file=sys.stderr)
        return 1
    except (OSError, UnicodeError, StopIteration):
        print("Setup failed: check dataset path and example files.", file=sys.stderr)
        return 1
    print("JobFit: Module 0 setup OK")
    print(f"Dataset: {settings.dataset_path.name}; 250 synthetic jobs")
    print(f"Embeddings configured: {settings.embedding_model}, 1024 dimensions, normalized")
    print(f"Generation configured: {settings.llm_provider} / {settings.llm_model}")
    print("Validated one source record and four example contracts. No external services contacted.")
    print("For full dataset preparation, run: python -m app.dataset.dataset_loader")
    print("For resume preparation, run: python -m app.cli.prepare_resume resume.pdf")
    print("For dense indexing/search, run: python -m app.cli.dense_cli --help")
    print("For matching/explanations, run: python -m app.cli.match_cli --help")
    print("For the API, run: uvicorn app.api.main:app --host 127.0.0.1 --workers 1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
