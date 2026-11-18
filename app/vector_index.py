"""Persistent LangChain/Qdrant index with explicit provenance and stable IDs."""

import hashlib
import json
import os
import platform
import tempfile
import time
from importlib.metadata import version
from pathlib import Path

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from app.documents import TEXT_VERSION, document_fingerprint, job_document, point_id
from app.embeddings import validate_vectors


def open_client(settings) -> QdrantClient:
    if settings.qdrant_mode == "local":
        settings.qdrant_path.mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=str(settings.qdrant_path))
    return QdrantClient(url=str(settings.qdrant_url), timeout=60)


def index_identity(documents, provenance: dict, embedding_info: dict) -> dict:
    semantic = {key: embedding_info[key] for key in
                ("model", "revision", "dimension", "normalize_embeddings", "distance", "max_tokens")}
    identity = {"version": "jobfit-dense-index-v1", "text_version": TEXT_VERSION,
                "corpus": provenance, "documents_sha256": document_fingerprint(documents),
                "embeddings": semantic}
    identity["index_id"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return identity


def collection_name(settings, identity: dict) -> str:
    return settings.qdrant_collection + "_" + identity["index_id"][:16]


def manifest_path(settings, identity: dict) -> Path:
    return settings.artifacts_dir / "indexes" / (identity["index_id"] + ".json")


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("Report output must not be a symlink")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def verify_collection(client, name: str, identity: dict, documents, *, audit_vectors=False) -> dict:
    info = client.get_collection(name)
    parameters = info.config.params.vectors
    if not isinstance(parameters, models.VectorParams):
        raise ValueError("Expected a single unnamed dense vector configuration")
    if parameters.size != identity["embeddings"]["dimension"] or parameters.distance != models.Distance.COSINE:
        raise ValueError("Collection dimensions or distance do not match the embedding contract")
    if info.config.metadata != identity:
        raise ValueError("Collection provenance does not match this corpus/model")
    expected = {point_id(doc.metadata["job_id"]): doc for doc in documents}
    observed = set()
    norms = []
    offset = None
    while True:
        points, offset = client.scroll(name, limit=128, offset=offset,
                                       with_payload=True, with_vectors=audit_vectors)
        for point in points:
            point_key = str(point.id)
            if point_key not in expected:
                raise ValueError("Unexpected point in collection; use an isolated collection")
            doc = expected[point_key]
            payload = point.payload or {}
            if payload.get("page_content") != doc.page_content or payload.get("metadata") != doc.metadata:
                raise ValueError("Stored job payload differs from prepared corpus")
            observed.add(point_key)
            if audit_vectors:
                validate_vectors([point.vector], identity["embeddings"]["dimension"])
                norms.append(sum(value * value for value in point.vector) ** 0.5)
        if offset is None:
            break
    return {"point_count": len(observed), "complete": observed == set(expected),
            "min_norm": min(norms) if norms else None, "max_norm": max(norms) if norms else None}


def prepare_documents(jobs, provenance, embedding_info):
    documents = [job_document(job) for job in sorted(jobs, key=lambda job: job.job_id)]
    identity = index_identity(documents, provenance, embedding_info)
    for doc in documents:
        doc.metadata.update({"index_id": identity["index_id"],
                             "source_sha256": provenance["source"]["sha256"],
                             "corpus_subset": provenance["configuration"]["subset"]})
    return documents, identity


def vector_store(client, name, embeddings):
    # We verify size/distance/provenance ourselves; avoid the integration's extra
    # dummy-text embedding when validating an existing collection.
    return QdrantVectorStore(client=client, collection_name=name, embedding=embeddings,
                             distance=models.Distance.COSINE, validate_collection_config=False)


def build_index(settings, jobs, provenance, embeddings, embedding_info) -> dict:
    started = time.perf_counter()
    documents, identity = prepare_documents(jobs, provenance, embedding_info)
    counts = embeddings.token_counts([doc.page_content for doc in documents])
    name = collection_name(settings, identity)
    client = open_client(settings)
    try:
        exists = client.collection_exists(name)
        reused = False
        if exists:
            check = verify_collection(client, name, identity, documents, audit_vectors=True)
            reused = check["complete"]
        else:
            client.create_collection(name, vectors_config=models.VectorParams(
                size=settings.embedding_dimension, distance=models.Distance.COSINE), metadata=identity)
        store = vector_store(client, name, embeddings)
        embed_start = time.perf_counter()
        if not reused:
            store.add_documents(documents, ids=[point_id(doc.metadata["job_id"]) for doc in documents],
                                batch_size=settings.embedding_batch_size)
        embed_seconds = time.perf_counter() - embed_start
        audit = verify_collection(client, name, identity, documents, audit_vectors=True)
        if not audit["complete"]:
            raise ValueError("Index is incomplete; rerun index building before retrieval")
    finally:
        client.close()
    report = {
        "identity": identity, "collection_name": name, "qdrant_mode": settings.qdrant_mode,
        "status": "ready", "reused_existing_vectors": reused,
        "token_lengths": {"min": min(counts), "max": max(counts),
                          "mean": round(sum(counts) / len(counts), 1)},
        "vector_audit": audit, "runtime_embeddings": embedding_info,
        "timing_seconds": {"embed_and_upsert": round(embed_seconds, 3),
                           "total_after_model_load": round(time.perf_counter() - started, 3)},
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "logical_cpus": os.cpu_count(),
                    "packages": {package: version(package) for package in (
                        "langchain-core", "langchain-huggingface", "langchain-qdrant",
                        "qdrant-client", "sentence-transformers", "transformers", "torch")}},
        "warnings": ["Synthetic corpus; retrieval scores are not qualification or hiring probabilities."]}
    save_json(manifest_path(settings, identity), report)
    return report
