"""Match a prepared resume using dense, BM25, or hybrid retrieval."""

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
    parser.add_argument("--retrieval-mode", choices=("dense", "bm25", "hybrid"))
    args = parser.parse_args(argv)
    try:
        settings = get_settings()
        if args.retrieval_mode:
            settings = settings.model_copy(update={"retrieval_mode": args.retrieval_mode})
        jobs, provenance = load_prepared_jobs(args.corpus_dir or settings.artifacts_dir / "dataset/all")
        prepared = PreparedResume.model_validate_json(args.prepared_resume.read_text(encoding="utf-8"))
        if args.retrieval_report:
            retrieval = json.loads(args.retrieval_report.read_text(encoding="utf-8"))
            if retrieval.get("retrieval_mode", "dense") != settings.retrieval_mode:
                raise ValueError("Cached retrieval mode differs from requested mode")
            if settings.retrieval_mode != "dense":
                from app.hybrid_retrieval import lexical_identity
                expected, _ = lexical_identity(settings, jobs, provenance, prepared.preferences)
                if retrieval.get("lexical_identity") != expected:
                    raise ValueError("Cached lexical corpus or parameters differ")
                if settings.retrieval_mode == "hybrid" and retrieval.get("rrf_constant") != settings.rrf_constant:
                    raise ValueError("Cached RRF parameters differ")
                if settings.retrieval_mode == "hybrid" and retrieval.get("branch_windows") != {
                        "dense": settings.candidate_count, "bm25": settings.bm25_candidate_count}:
                    raise ValueError("Cached retrieval windows differ")
            index_id = retrieval.get("dense_index_id", retrieval.get("index_id", ""))
            if settings.retrieval_mode != "bm25" and retrieval.get("status") not in ("no_query", "no_eligible_jobs"):
                validate_dense_manifest(settings, provenance, index_id)
        else:
            from app.hybrid_retrieval import retrieve_candidates
            embeddings, info = None, None
            if settings.retrieval_mode != "bm25" and prepared.retrieval_query is not None:
                from app.embeddings import load_embeddings
                print("Loading local embedding model...", flush=True)
                embeddings, info = load_embeddings(settings)
            print(f"Retrieving {settings.retrieval_mode} candidates...", flush=True)
            retrieval = retrieve_candidates(settings, jobs, provenance, prepared,
                                            embeddings=embeddings, embedding_info=info)
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


def validate_dense_manifest(settings, provenance, index_id):
    if not isinstance(index_id, str) or len(index_id) != 64 or any(c not in "0123456789abcdef" for c in index_id):
        raise ValueError("Invalid retrieval index identity")
    manifest = json.loads((settings.artifacts_dir / "indexes" / (index_id + ".json")).read_text())
    if manifest["identity"]["corpus"] != provenance or manifest["status"] != "ready":
        raise ValueError("Retrieval report corpus differs from prepared jobs")
    vector_config = manifest["identity"]["embeddings"]
    if vector_config["model"] != settings.embedding_model or vector_config["revision"] != settings.embedding_revision:
        raise ValueError("Retrieval model differs from the configured embedding model")

if __name__ == "__main__":
    raise SystemExit(main())
