"""Build a persistent Qdrant index from the prepared job catalog."""

from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from app.config import Settings
from app.retrieval.documents import load_job_documents
from app.retrieval.embeddings import EMBEDDING_DIMENSION, create_embeddings


INDEX_PATH = Path("artifacts/qdrant")
COLLECTION_NAME = "jobfit_jobs"


def build_index(jobs_path: Path, index_path: Path = INDEX_PATH) -> int:
    """Index the prepared jobs and return the number of stored points."""
    documents = load_job_documents(jobs_path)
    if not documents:
        raise ValueError("The prepared catalog is empty.")

    embeddings = create_embeddings()

    # The same job ID produces the same Qdrant point ID on every run.
    point_ids = [
        str(uuid5(NAMESPACE_URL, f"jobfit/jobs/{document.metadata['job_id']}"))
        for document in documents
    ]

    client = QdrantClient(path=str(index_path))
    try:
        if not client.collection_exists(collection_name=COLLECTION_NAME):
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=EMBEDDING_DIMENSION,
                    distance=models.Distance.COSINE,
                ),
            )

        store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=embeddings,
        )

        # Send one write to local Qdrant after encoding the small catalog.
        store.add_documents(
            documents=documents,
            ids=point_ids,
            batch_size=len(documents),
        )

        # Remove points for jobs that are no longer in the prepared catalog.
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.Filter(
                must_not=[models.HasIdCondition(has_id=point_ids)]
            ),
            wait=True,
        )

        count = client.count(collection_name=COLLECTION_NAME, exact=True).count
        if count != len(documents):
            raise RuntimeError(f"Expected {len(documents)} points, found {count}.")

        return count
    finally:
        # Release the local database lock so the search command can reopen it.
        client.close()


def main() -> None:
    """Build the local index using the configured prepared catalog."""
    count = build_index(Settings().prepared_jobs_path)
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Indexed jobs: {count}")
    print(f"Index path: {INDEX_PATH}")


if __name__ == "__main__":
    main()
