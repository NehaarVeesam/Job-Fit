"""LangChain BGE-M3 adapter with token and vector checks on both input paths."""

import math

from langchain_core.embeddings import Embeddings


def validate_vectors(vectors: list[list[float]], dimension: int = 1024) -> None:
    for vector in vectors:
        if len(vector) != dimension or not all(math.isfinite(x) for x in vector):
            raise ValueError(f"Expected finite {dimension}-dimensional embedding")
        norm = math.sqrt(sum(x * x for x in vector))
        if not math.isclose(norm, 1, abs_tol=1e-4):
            raise ValueError("Embedding must have unit L2 norm")


class CheckedEmbeddings(Embeddings):
    """An Embeddings interface wrapping LangChain's HuggingFaceEmbeddings."""

    def __init__(self, delegate, tokenizer, max_tokens: int, dimension: int = 1024):
        self.delegate = delegate
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.dimension = dimension

    def token_counts(self, texts: list[str]) -> list[int]:
        if any(not text.strip() for text in texts):
            raise ValueError("Cannot embed empty text")
        encoded = self.tokenizer(texts, add_special_tokens=True, truncation=False,
                                 padding=False, return_attention_mask=False)
        counts = [len(tokens) for tokens in encoded["input_ids"]]
        if any(count > self.max_tokens for count in counts):
            raise ValueError(f"Input exceeds {self.max_tokens} model tokens; explicit chunking is required")
        return counts

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self.token_counts(texts)
        vectors = self.delegate.embed_documents(texts)
        if len(vectors) != len(texts):
            raise ValueError("Embedding count does not match document count")
        validate_vectors(vectors, self.dimension)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        self.token_counts([text])
        vector = self.delegate.embed_query(text)
        validate_vectors([vector], self.dimension)
        return vector


def load_embeddings(settings) -> tuple[CheckedEmbeddings, dict]:
    # Heavy imports/model load happen explicitly, never when app.config is imported.
    import torch
    from huggingface_hub import snapshot_download
    from langchain_huggingface import HuggingFaceEmbeddings

    if settings.embedding_device == "cpu":
        torch.set_num_threads(settings.embedding_cpu_threads)
    snapshot = snapshot_download(repo_id=settings.embedding_model,
                                 revision=settings.embedding_revision,
                                 cache_dir=settings.embedding_cache_dir, local_files_only=True)
    kwargs = {"device": settings.embedding_device, "trust_remote_code": False,
              "local_files_only": True}
    delegate = HuggingFaceEmbeddings(
        # A local path also prevents Transformers' tokenizer compatibility check
        # from querying remote model metadata despite local_files_only=True.
        model_name=snapshot,
        cache_folder=str(settings.embedding_cache_dir),
        model_kwargs=kwargs,
        encode_kwargs={"normalize_embeddings": True, "batch_size": settings.embedding_batch_size},
        show_progress=False,
    )
    client = delegate._client
    limit = min(8192, client.max_seq_length)
    from pathlib import Path
    revision = Path(snapshot).name
    if revision != settings.embedding_revision:
        raise ValueError("Cannot identify downloaded embedding model revision")
    checked = CheckedEmbeddings(delegate, client.tokenizer, limit, settings.embedding_dimension)
    return checked, {
        "model": settings.embedding_model, "revision": revision,
        "dimension": settings.embedding_dimension, "normalize_embeddings": True,
        "distance": "Cosine", "max_tokens": limit,
        "device": settings.embedding_device, "batch_size": settings.embedding_batch_size,
        "cpu_threads": settings.embedding_cpu_threads,
    }
