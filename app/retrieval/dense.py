"""Search the saved JobFit catalog with dense vectors."""

from pathlib import Path

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from app.retrieval.embeddings import create_embeddings
from app.retrieval.filters import matches_preferences
from app.retrieval.vector_index import COLLECTION_NAME, INDEX_PATH
from app.schemas import Preferences


def search_dense(
    query: str,
    top_k: int,
    index_path: Path = INDEX_PATH,
    *,
    preferences: Preferences | None = None,
) -> tuple[int, list[tuple[Document, float]]]:
    """Return the number of indexed jobs and the ranked document-score pairs."""
    query = query.strip()
    if not query:
        raise ValueError("The query must contain some text.")
    if top_k < 1:
        raise ValueError("--top-k must be at least 1.")
    if not index_path.is_dir():
        raise FileNotFoundError(
            "Build the index first with python -m app.retrieval.vector_index."
        )

    client = QdrantClient(path=str(index_path))
    try:
        if not client.collection_exists(collection_name=COLLECTION_NAME):
            raise FileNotFoundError(
                f"Collection {COLLECTION_NAME!r} is missing. Build the index first."
            )

        indexed_jobs = client.count(collection_name=COLLECTION_NAME, exact=True).count
        if indexed_jobs == 0:
            return indexed_jobs, []

        store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=create_embeddings(),
        )

        # With filters, read the full ranking so eligible jobs below top_k can move up.
        filters_active = preferences is not None and bool(
            (preferences.location_city or "").strip() or preferences.work_mode
        )
        limit = indexed_jobs if filters_active else top_k

        hits = store.similarity_search_with_score(query, k=limit)
        filtered_hits = [
            hit for hit in hits if matches_preferences(hit[0], preferences)
        ]
        return indexed_jobs, filtered_hits[:top_k]
    finally:
        client.close()
