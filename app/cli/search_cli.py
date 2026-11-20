"""Compare retrieval modes: python -m app.cli.search_cli --help."""

import argparse
import hashlib
import sys
from pathlib import Path

from app.config import get_settings
from app.retrieval.hybrid import MODES, retrieve_candidates
from app.resumes.profile import load_prepared_jobs
from app.schemas import PreparedResume
from app.retrieval.vector_index import save_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-resume", type=Path, required=True)
    parser.add_argument("--mode", choices=(*MODES, "compare"), default="compare")
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args(argv)
    try:
        if not 1 <= args.k <= 250:
            raise ValueError("Invalid candidate count")
        settings = get_settings()
        if args.rerank is not None:
            settings = settings.model_copy(update={'rerank_enabled': args.rerank})
        jobs, provenance = load_prepared_jobs(settings.artifacts_dir / "dataset/all")
        prepared = PreparedResume.model_validate_json(args.prepared_resume.read_text(encoding="utf-8"))
        embeddings, info = None, None
        if args.mode != "bm25" and prepared.retrieval_query is not None:
            from app.retrieval.embeddings import load_embeddings
            print("Loading local embedding model...", flush=True)
            embeddings, info = load_embeddings(settings)
        modes = MODES if args.mode == "compare" else (args.mode,)
        reports = {}
        cached_reranker = [None]
        def get_reranker():
            from app.retrieval.reranking import load_reranker
            if cached_reranker[0] is None:
                cached_reranker[0] = load_reranker(settings)
            return cached_reranker[0]
        key = hashlib.sha256(prepared.model_dump_json().encode()).hexdigest()
        output = settings.artifacts_dir / "hybrid_retrieval" / key
        for mode in modes:
            if mode == "dense" and embeddings is None:
                # No-query results need no model or index in any mode.
                from app.retrieval.hybrid import empty_dense_report
                report = empty_dense_report(jobs, provenance, prepared, args.k)
            else:
                report = retrieve_candidates(settings, jobs, provenance, prepared, mode=mode, k=args.k,
                                             embeddings=embeddings, embedding_info=info)
            save_json(output / f"{mode}.json", report)
            if settings.rerank_enabled:
                from app.retrieval.reranking import rerank_candidates
                report = rerank_candidates(settings, prepared, report, reranker_loader=get_reranker)
                save_json(output / f'{mode}_reranked.json', report)
                print('Reranking: ' + report['reranking']['status'], flush=True)
            reports[mode] = report
            print(f"{mode}: {len(report['candidates'])} candidates / {report['eligible_count']} eligible")
            for item in report['candidates'][:5]:
                print(f"  {item['rank']}. {item['job_id']} | {item['job']['title']} | "
                      f"{item['relevance']['score_type']}={item['relevance']['value']:.6f}" +
                      (f" | cross_encoder={item['reranker_relevance']['value']:.6f}" if item.get('reranker_relevance') else ''))
        if args.mode == "compare":
            sets = {mode: {item['job_id'] for item in report['candidates']} for mode, report in reports.items()}
            summary = {"corpus": provenance, "query_sha256": reports['dense']['query_sha256'],
                       "applied_preferences": prepared.preferences.model_dump(mode="json"),
                       "k": args.k, "candidate_counts": {mode: len(ids) for mode, ids in sets.items()},
                       "dense_bm25_overlap": len(sets['dense'] & sets['bm25']),
                       "hybrid_new_vs_dense": sorted(sets['hybrid'] - sets['dense']),
                       "reranking": {mode: report.get('reranking', {'status': 'disabled'}) for mode, report in reports.items()},
                       "note": "Rank/overlap diagnostics only; no suitability judgments or quality improvement claim."}
            save_json(output / "comparison.json", summary)
        print(f"Reports: {output}. No Qwen calls.")
    except (OSError, ValueError, RuntimeError, KeyError):
        print("Search failed: check prepared inputs, candidate count, and model/index for dense or hybrid mode.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
