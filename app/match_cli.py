"""Run Module 4: python -m app.match_cli --prepared-resume PATH [--no-llm]."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from app.config import get_settings
from app.matching import match_candidates
from app.profile import load_prepared_jobs
from app.schemas import PreparedResume
from app.vector_index import save_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-resume", required=True, type=Path)
    parser.add_argument("--corpus-dir", type=Path)
    parser.add_argument("--retrieval-report", type=Path, help="Reuse a Module 3 result instead of rerunning dense search")
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args(argv)
    try:
        settings = get_settings()
        jobs, provenance = load_prepared_jobs(args.corpus_dir or settings.artifacts_dir / "dataset/all")
        prepared = PreparedResume.model_validate_json(args.prepared_resume.read_text(encoding="utf-8"))
        if args.retrieval_report:
            retrieval = json.loads(args.retrieval_report.read_text(encoding="utf-8"))
            index_id = retrieval.get("index_id", "")
            if not isinstance(index_id, str) or len(index_id) != 64 or any(c not in "0123456789abcdef" for c in index_id):
                raise ValueError("Invalid retrieval index identity")
            manifest = json.loads((settings.artifacts_dir / "indexes" / (index_id + ".json")).read_text())
            if manifest["identity"]["corpus"] != provenance or manifest["status"] != "ready":
                raise ValueError("Retrieval report corpus differs from prepared jobs")
            vector_config = manifest["identity"]["embeddings"]
            if vector_config["model"] != settings.embedding_model or vector_config["revision"] != settings.embedding_revision:
                raise ValueError("Retrieval model differs from the configured embedding model")
        else:
            from app.embeddings import load_embeddings
            from app.retrieval import retrieve
            print("Retrieving dense candidates...", flush=True)
            embeddings, info = load_embeddings(settings)
            retrieval = retrieve(settings, jobs, provenance, embeddings, info, prepared, k=settings.candidate_count)
        print("Assessing requirements" + (" and requesting grounded Qwen plans..." if not args.no_llm else " offline..."), flush=True)
        response, report = match_candidates(settings, jobs, provenance, prepared, retrieval, use_llm=not args.no_llm)
        key = hashlib.sha256((prepared.model_dump_json() + json.dumps(report, sort_keys=True)).encode()).hexdigest()
        output = settings.artifacts_dir / "matches" / key
        save_json(output / "response.json", response.model_dump(mode="json"))
        save_json(output / "assessment_report.json", report)
        print(f"{response.status}: {len(response.recommendations)} recommendations from {len(response.ranked_candidates)} candidates.")
        for match in response.recommendations:
            print(f"{match.job.job_id}: coverage={match.required_coverage.percentage}%; "
                  f"eligibility={match.eligibility}; explanation={match.generation_status}")
        print(f"Qwen calls attempted: {report['generation_calls']}. Output: {output}")
    except (OSError, ValueError, RuntimeError, KeyError):
        print("Matching failed: check prepared resume/corpus, retrieval provenance, model cache, and configuration.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
