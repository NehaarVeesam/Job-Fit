"""Dense candidate retrieval, with source traceability and shared hard filters."""

import math
import hashlib
import time

from app.filters import eligible_ids, preference_warnings, qdrant_filter
from app.vector_index import (
    collection_name, manifest_path, open_client, prepare_documents, verify_collection, vector_store,
)


def retrieve(settings, jobs, provenance, embeddings, embedding_info, prepared, *, k=5) -> dict:
    if not 1 <= k <= 250:
        raise ValueError("k must be between 1 and 250")
    documents, identity = prepare_documents(jobs, provenance, embedding_info)
    ids = eligible_ids(jobs, prepared.preferences)
    result = {"dataset_id": "synthetic_india_ai_sw_v2", "is_synthetic": True,
              "index_id": identity["index_id"], "corpus_size": len(jobs),
              "query_sha256": hashlib.sha256((prepared.retrieval_query or "").encode()).hexdigest(),
              "eligible_count": len(ids), "requested_k": k,
              "applied_preferences": prepared.preferences.model_dump(mode="json"),
              "candidates": [], "warnings": [
                  "Nearest neighbors are retrieval candidates; suitability and skill coverage are not assessed yet."]}
    if prepared.retrieval_query is None:
        result.update(status="no_query", timing_seconds={"embedding_and_search": 0})
        result["warnings"].append("Resume preparation produced no query; review Module 2 warnings.")
        return result
    if not ids:
        result.update(status="no_eligible_jobs", timing_seconds={"embedding_and_search": 0})
        return result
    embeddings.token_counts([prepared.retrieval_query])
    if not manifest_path(settings, identity).exists():
        raise ValueError("No manifest for this corpus/model; build its index first")
    client = open_client(settings)
    name = collection_name(settings, identity)
    try:
        audit = verify_collection(client, name, identity, documents)
        if not audit["complete"]:
            raise ValueError("Index is incomplete; rebuild before retrieval")
        store = vector_store(client, name, embeddings)
        started = time.perf_counter()
        candidates = store.similarity_search_with_score(
            prepared.retrieval_query, k=min(k, len(ids)), filter=qdrant_filter(ids))
        elapsed = time.perf_counter() - started
    finally:
        client.close()
    by_id = {job.job_id: job for job in jobs}
    for rank, (doc, score) in enumerate(candidates, 1):
        job_id = doc.metadata["job_id"]
        if job_id not in ids or doc.metadata["index_id"] != identity["index_id"]:
            raise ValueError("Retrieved candidate does not belong to the filtered corpus")
        if not math.isfinite(score) or not -1.00001 <= score <= 1.00001:
            raise ValueError("Unexpected cosine score")
        job = by_id[job_id]
        result["candidates"].append({"rank": rank, "job_id": job_id,
                                      "relevance": {"value": max(-1, min(1, score)), "score_type": "cosine"},
                                      "job": job.model_dump(mode="json"),
                                      "warnings": preference_warnings(job, prepared.preferences)})
    result.update(status="candidates_found" if candidates else "no_candidates",
                  timing_seconds={"embedding_and_search": round(elapsed, 3)})
    return result
