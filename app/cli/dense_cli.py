"""Build/search a BGE-M3 Qdrant index: python -m app.dense_cli --help."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from app.config import get_settings
from app.retrieval.embeddings import load_embeddings
from app.resumes.profile import load_prepared_jobs
from app.retrieval.dense import retrieve
from app.schemas import PreparedResume
from app.retrieval.vector_index import build_index, manifest_path, save_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("index", "search"))
    parser.add_argument("--corpus-dir", type=Path)
    parser.add_argument("--prepared-resume", type=Path)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args(argv)
    if args.command == "search" and args.prepared_resume is None:
        parser.error("search requires --prepared-resume")
    try:
        settings = get_settings()
        directory = args.corpus_dir or settings.artifacts_dir / "dataset" / "all"
        jobs, provenance = load_prepared_jobs(directory)
        prepared = None
        if args.command == "search":
            prepared = PreparedResume.model_validate_json(args.prepared_resume.read_text(encoding="utf-8"))
        print("Loading cached BGE-M3 locally...", flush=True)
        model_start = time.perf_counter()
        embeddings, info = load_embeddings(settings)
        load_seconds = round(time.perf_counter() - model_start, 3)
        if args.command == "index":
            print(f"Checking tokens and indexing {len(jobs)} synthetic jobs...", flush=True)
            report = build_index(settings, jobs, provenance, embeddings, info)
            report["timing_seconds"]["model_load"] = load_seconds
            save_json(manifest_path(settings, report["identity"]), report)
            print(f"Ready: {report['vector_audit']['point_count']} points; "
                  f"job tokens {report['token_lengths']['min']}–{report['token_lengths']['max']}; "
                  f"reused={report['reused_existing_vectors']}.")
            print("Collection: " + report["collection_name"])
        else:
            report = retrieve(settings, jobs, provenance, embeddings, info, prepared, k=args.k)
            report["timing_seconds"]["model_load"] = load_seconds
            # Hash the prepared input so changing preferences/query cannot reuse a stale report name.
            input_hash = hashlib.sha256(prepared.model_dump_json().encode()).hexdigest()
            path = settings.artifacts_dir / "retrieval" / f"{input_hash}_{args.k}.json"
            save_json(path, report)
            print(f"{report['status']}: {len(report['candidates'])} candidates / {report['eligible_count']} eligible jobs.")
            for item in report["candidates"]:
                job = item["job"]
                print(f"{item['rank']}. {job['job_id']} | {job['title']} | cosine={item['relevance']['value']:.4f}")
            print(f"Result: {path}")
        print(f"Model load: {load_seconds}s. No Qwen generation calls.")
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"Dense retrieval failed ({type(exc).__name__}). Check corpus/index consistency, "
              "Qdrant availability, and cached model files. Download with: "
              "python -m app.cli.download_models embeddings", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
