"""Run from project root: python -m python_only.cli index|search --help."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from app.config import get_settings
from app.resumes.profile import load_prepared_jobs
from app.schemas import PreparedResume
from python_only.engine import Encoder, JobIndex


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("index", "search"))
    parser.add_argument("--corpus-dir", type=Path)
    parser.add_argument("--prepared-resume", type=Path)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args(argv)
    if args.command == "search" and args.prepared_resume is None:
        parser.error("search requires --prepared-resume")
    index = None
    try:
        settings = get_settings()
        jobs, provenance = load_prepared_jobs(args.corpus_dir or settings.artifacts_dir / "dataset/all")
        prepared = (PreparedResume.model_validate_json(args.prepared_resume.read_text(encoding="utf-8"))
                    if args.command == "search" else None)
        print("Loading cached BGE-M3 with Sentence Transformers directly...", flush=True)
        encoder = Encoder(settings)
        index = JobIndex(settings, jobs, provenance, encoder)
        started = time.perf_counter()
        if args.command == "index":
            print(f"Indexing/checking {len(jobs)} jobs using QdrantClient directly...", flush=True)
            report = index.build()
            key = "index"
        else:
            report = index.search(prepared, args.k)
            key = hashlib.sha256((prepared.model_dump_json() + str(args.k)).encode()).hexdigest()
            for item in report["candidates"]:
                print(f"{item['rank']}. {item['job_id']} | {item['title']} | cosine={item['relevance']['value']:.4f}")
        report["seconds_after_model_load"] = round(time.perf_counter() - started, 3)
        report["identity"] = index.identity
        path = settings.artifacts_dir / "python_only/reports" / (key + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():
            raise ValueError("Report output must not be a symlink")
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        if args.command == "index":
            print(f"{report['points']} points; reused={report['reused']}; tokens={report['tokens_min']}–{report['tokens_max']}")
        print(f"Report: {path}")
        print("No LangChain imports or Qwen calls.")
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"Python-only run failed ({type(exc).__name__}); check prepared inputs, cached model, and local index.", file=sys.stderr)
        return 1
    finally:
        if index is not None:
            index.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
