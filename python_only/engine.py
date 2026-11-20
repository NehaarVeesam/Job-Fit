"""Follow this file top to bottom: text -> encode -> upsert -> query.

No LangChain imports, including through the shared app modules used here.
"""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from app.retrieval.filters import eligible_ids, preference_warnings, qdrant_filter


@dataclass
class JobText:
    """An ordinary Python object instead of a LangChain Document."""

    text: str
    metadata: dict


def prepare_job(job) -> JobText:
    # Same text as the LangChain version so scores can be compared fairly.
    text = "\n".join([
        f"Title: {job.title}", f"Role family: {job.role_family}",
        "Required skills: " + ", ".join(job.required_skills),
        "Preferred skills: " + ", ".join(job.preferred_skills),
        "Description:\n" + job.description,
    ])
    return JobText(text=text, metadata=job.model_dump(mode="json"))


def point_id(job_id):
    return str(uuid5(NAMESPACE_URL, f"jobfit/synthetic_india_ai_sw_v2/{job_id}"))


def check_vectors(vectors):
    for vector in vectors:
        if len(vector) != 1024 or not all(math.isfinite(value) for value in vector):
            raise ValueError("Expected 1024 finite embedding values")
        if not math.isclose(sum(value * value for value in vector) ** 0.5, 1, abs_tol=1e-4):
            raise ValueError("Expected unit-normalized embedding")


class Encoder:
    """Call SentenceTransformer directly, with the same model and checks."""

    def __init__(self, settings):
        import torch
        from huggingface_hub import snapshot_download
        from sentence_transformers import SentenceTransformer

        if settings.embedding_device == "cpu":
            torch.set_num_threads(settings.embedding_cpu_threads)
        snapshot = snapshot_download(
            repo_id=settings.embedding_model, revision=settings.embedding_revision,
            cache_dir=settings.embedding_cache_dir, local_files_only=True,
        )
        self.model = SentenceTransformer(snapshot, device=settings.embedding_device,
                                         local_files_only=True, trust_remote_code=False)
        self.batch_size = settings.embedding_batch_size
        self.max_tokens = min(8192, self.model.max_seq_length)

    def token_counts(self, texts):
        if any(not text.strip() for text in texts):
            raise ValueError("Cannot encode empty text")
        tokens = self.model.tokenizer(texts, add_special_tokens=True, truncation=False,
                                      padding=False, return_attention_mask=False)
        counts = [len(ids) for ids in tokens["input_ids"]]
        if any(count > self.max_tokens for count in counts):
            raise ValueError("Input exceeds model token limit; explicit chunking is required")
        return counts

    def encode(self, texts):
        if not texts:
            return []
        self.token_counts(texts)
        # Match HuggingFaceEmbeddings' newline handling for this comparison.
        inputs = [text.replace("\n", " ") for text in texts]
        vectors = self.model.encode(inputs, batch_size=self.batch_size,
                                    normalize_embeddings=True, show_progress_bar=False).tolist()
        if len(vectors) != len(texts):
            raise ValueError("Embedding count differs from input count")
        check_vectors(vectors)
        return vectors


class JobIndex:
    """Use Qdrant directly; isolated from the main LangChain index."""

    def __init__(self, settings, jobs, provenance, encoder):
        self.jobs = jobs
        self.encoder = encoder
        self.records = [prepare_job(job) for job in sorted(jobs, key=lambda job: job.job_id)]
        self.identity = {
            "implementation": "python-only-v1", "corpus": provenance,
            "model": settings.embedding_model, "revision": settings.embedding_revision,
            "dimension": 1024, "normalized": True, "distance": "Cosine",
            "text_sha256": hashlib.sha256(json.dumps([
                [record.text, record.metadata] for record in self.records
            ], sort_keys=True).encode()).hexdigest(),
        }
        digest = hashlib.sha256(json.dumps(self.identity, sort_keys=True).encode()).hexdigest()
        self.name = "jobfit_python_only_synthetic_" + digest[:16]
        path = (settings.artifacts_dir / "python_only/qdrant").resolve()
        protected = [settings.dataset_path.parent.resolve(),
                     (Path(__file__).resolve().parents[1] / "dataset").resolve()]
        if any(path == folder or folder in path.parents for folder in protected):
            raise ValueError("Keep vector output outside the input dataset")
        path.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(path))

    def close(self):
        self.client.close()

    def audit(self):
        info = self.client.get_collection(self.name)
        parameters = info.config.params.vectors
        if (not isinstance(parameters, models.VectorParams) or parameters.size != 1024
                or parameters.distance != models.Distance.COSINE or info.config.metadata != self.identity):
            raise ValueError("Stored index configuration/provenance mismatch")
        expected = {point_id(record.metadata["job_id"]): record for record in self.records}
        observed = set()
        offset = None
        while True:
            points, offset = self.client.scroll(self.name, limit=128, offset=offset,
                                                with_payload=True, with_vectors=True)
            for point in points:
                key = str(point.id)
                if key not in expected:
                    raise ValueError("Unexpected point in learning index")
                record = expected[key]
                if point.payload != {"text": record.text, "metadata": record.metadata}:
                    raise ValueError("Stored job payload differs from input")
                check_vectors([point.vector])
                observed.add(key)
            if offset is None:
                break
        return observed == set(expected)

    def build(self):
        # Preflight every text before writing any vectors.
        counts = self.encoder.token_counts([record.text for record in self.records])
        exists = self.client.collection_exists(self.name)
        if exists and self.audit():
            return {"collection": self.name, "points": len(self.records), "reused": True,
                    "tokens_min": min(counts), "tokens_max": max(counts)}
        if not exists:
            self.client.create_collection(
                self.name, vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
                metadata=self.identity,
            )
        for start in range(0, len(self.records), self.encoder.batch_size):
            batch = self.records[start:start + self.encoder.batch_size]
            # These two explicit steps are handled by add_documents in LangChain.
            vectors = self.encoder.encode([record.text for record in batch])
            points = [models.PointStruct(
                id=point_id(record.metadata["job_id"]), vector=vector,
                payload={"text": record.text, "metadata": record.metadata},
            ) for record, vector in zip(batch, vectors, strict=True)]
            self.client.upsert(self.name, points=points, wait=True)
        if not self.audit():
            raise ValueError("Index is incomplete")
        return {"collection": self.name, "points": len(self.records), "reused": False,
                "tokens_min": min(counts), "tokens_max": max(counts)}

    def search(self, prepared, k=5):
        if not 1 <= k <= 250:
            raise ValueError("k must be between 1 and 250")
        ids = eligible_ids(self.jobs, prepared.preferences)
        result = {"implementation": "python_only", "dataset_id": "synthetic_india_ai_sw_v2",
                  "is_synthetic": True, "eligible_count": len(ids), "candidates": [],
                  "warnings": ["Retrieval candidates; suitability and skill coverage are not assessed."]}
        if prepared.retrieval_query is None:
            return dict(result, status="no_query")
        if not ids:
            return dict(result, status="no_eligible_jobs")
        if not self.client.collection_exists(self.name) or not self.audit():
            raise ValueError("Build a complete index before searching")
        # These explicit steps are handled by similarity_search_with_score in LangChain.
        query_vector = self.encoder.encode([prepared.retrieval_query])[0]
        response = self.client.query_points(
            collection_name=self.name, query=query_vector,
            query_filter=qdrant_filter(ids), limit=min(k, len(ids)), with_payload=True,
        )
        by_id = {job.job_id: job for job in self.jobs}
        for rank, point in enumerate(response.points, 1):
            job_id = point.payload["metadata"]["job_id"]
            if job_id not in ids or not math.isfinite(point.score) or not -1.00001 <= point.score <= 1.00001:
                raise ValueError("Invalid retrieved point or score")
            job = by_id[job_id]
            result["candidates"].append({
                "rank": rank, "job_id": job_id, "title": job.title,
                "relevance": {"value": max(-1, min(1, point.score)), "score_type": "cosine"},
                "job": job.model_dump(mode="json"),
                "warnings": preference_warnings(job, prepared.preferences),
            })
        return dict(result, status="candidates_found" if response.points else "no_candidates")
