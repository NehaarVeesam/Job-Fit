"""Dense/BM25 retrieval and traceable reciprocal rank fusion (RRF)."""

import hashlib
import json
import time

from app.retrieval.documents import document_fingerprint, job_document
from app.retrieval.filters import eligible_ids, preference_warnings
from app.retrieval.keyword_search import ALIASES, BM25Index, TOKENIZER_VERSION
from app.retrieval import dense as dense_retrieval

MODES = ("dense", "bm25", "hybrid")


def empty_dense_report(jobs, provenance, prepared, k):
    return {"dataset_id": "synthetic_india_ai_sw_v2", "is_synthetic": True,
            "retrieval_mode": "dense", "corpus": provenance, "corpus_size": len(jobs),
            "query_sha256": hashlib.sha256(b"").hexdigest(),
            "applied_preferences": prepared.preferences.model_dump(mode="json"),
            "eligible_count": len(eligible_ids(jobs, prepared.preferences)),
            "requested_k": k, "candidates": [], "status": "no_query",
            "warnings": ["Resume preparation produced no query."],
            "timing_seconds": {"retrieval": 0}}


def reciprocal_rank_fusion(rankings, *, constant=60):
    if not isinstance(constant, int) or isinstance(constant, bool) or constant < 1:
        raise ValueError("RRF constant must be a positive integer")
    scores, traces = {}, {}
    for name, ranking in rankings.items():
        seen = set()
        for rank, job_id in enumerate(ranking, 1):
            if job_id in seen:
                raise ValueError("Duplicate ID in retrieval branch")
            seen.add(job_id)
            scores[job_id] = scores.get(job_id, 0) + 1 / (constant + rank)
            traces.setdefault(job_id, {})[name] = rank
    return [(job_id, scores[job_id], traces[job_id])
            for job_id in sorted(scores, key=lambda key: (-scores[key], key))]


def lexical_identity(settings, jobs, provenance, preferences):
    ids = set(eligible_ids(jobs, preferences))
    documents = [job_document(job) for job in jobs if job.job_id in ids]
    identity = {"version": "jobfit-bm25-v1", "corpus": provenance,
                "documents_sha256": document_fingerprint(documents),
                "eligible_ids": sorted(ids), "tokenizer": TOKENIZER_VERSION,
                "aliases": ALIASES, "k1": settings.bm25_k1, "b": settings.bm25_b}
    identity["index_id"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return identity, documents


def retrieve_candidates(settings, jobs, provenance, prepared, *, mode=None, k=None,
                        embeddings=None, embedding_info=None):
    mode = mode or settings.retrieval_mode
    k = settings.candidate_count if k is None else k
    if mode not in MODES or not 1 <= k <= 250:
        raise ValueError("Invalid retrieval mode or candidate count")
    if mode == "dense" and prepared.retrieval_query is None:
        return empty_dense_report(jobs, provenance, prepared, k)
    if mode == "dense":
        result = dense_retrieval.retrieve(settings, jobs, provenance, embeddings, embedding_info, prepared, k=k)
        result.update(retrieval_mode=mode, corpus=provenance)
        return result
    started = time.perf_counter()
    identity, documents = lexical_identity(settings, jobs, provenance, prepared.preferences)
    result = {"dataset_id": "synthetic_india_ai_sw_v2", "is_synthetic": True,
              "retrieval_mode": mode, "corpus": provenance, "index_id": identity["index_id"],
              "lexical_identity": identity, "corpus_size": len(jobs),
              "query_sha256": hashlib.sha256((prepared.retrieval_query or "").encode()).hexdigest(),
              "eligible_count": len(documents), "requested_k": k,
              "applied_preferences": prepared.preferences.model_dump(mode="json"),
              "candidates": [], "warnings": ["Retrieval scores are not suitability probabilities."]}
    if mode == "hybrid":
        result.update(rrf_constant=settings.rrf_constant,
                      branch_windows={"dense": settings.candidate_count, "bm25": settings.bm25_candidate_count})
    if prepared.retrieval_query is None or not documents:
        result.update(status="no_query" if prepared.retrieval_query is None else "no_eligible_jobs",
                      timing_seconds={"retrieval": round(time.perf_counter() - started, 3)})
        return result
    lexical = BM25Index([(doc.metadata["job_id"], doc.page_content) for doc in documents],
                        k1=settings.bm25_k1, b=settings.bm25_b)
    lexical_results = lexical.search(prepared.retrieval_query, k=k if mode == "bm25" else settings.bm25_candidate_count)
    bm25_scores = dict(lexical_results)
    dense_scores = {}
    if mode == "hybrid":
        # For this <=250-job teaching corpus, retain all eligible cosine scores.
        # Only the configured top dense window participates in RRF. Extra scores
        # preserve the baseline semantic gate for candidates found only by BM25.
        dense = dense_retrieval.retrieve(settings, jobs, provenance, embeddings, embedding_info, prepared, k=250)
        dense_scores = {item["job_id"]: item["relevance"] for item in dense["candidates"]}
        dense_ids = [item["job_id"] for item in dense["candidates"][:settings.candidate_count]]
        fused = reciprocal_rank_fusion({"dense": dense_ids, "bm25": list(bm25_scores)}, constant=settings.rrf_constant)
        ranking = fused[:k]
        result.update(dense_index_id=dense["index_id"], rrf_constant=settings.rrf_constant,
                      branch_windows={"dense": settings.candidate_count, "bm25": settings.bm25_candidate_count},
                      branch_rankings={"dense": dense_ids, "bm25": list(bm25_scores)})
    else:
        ranking = [(job_id, score, {"bm25": rank}) for rank, (job_id, score) in enumerate(lexical_results, 1)]
    by_id = {job.job_id: job for job in jobs}
    for rank, (job_id, score, ranks) in enumerate(ranking, 1):
        result["candidates"].append({"rank": rank, "job_id": job_id,
            "job": by_id[job_id].model_dump(mode="json"),
            "relevance": {"value": score, "score_type": "rrf" if mode == "hybrid" else "bm25"},
            "dense_relevance": dense_scores.get(job_id),
            "retrieval_sources": {name: {"rank": branch_rank, "contribution": 1 / (settings.rrf_constant + branch_rank)}
                                  for name, branch_rank in ranks.items()} if mode == "hybrid" else {"bm25": {"rank": ranks["bm25"]}},
            "bm25_score": bm25_scores.get(job_id),
            "warnings": preference_warnings(by_id[job_id], prepared.preferences)})
    result.update(status="candidates_found" if ranking else "no_candidates",
                  timing_seconds={"retrieval": round(time.perf_counter() - started, 3)})
    return result
