"""Run a sample similarity search against the saved JobFit index."""

import argparse

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from app.config import Settings
from app.retrieval.embeddings import create_embeddings
from app.retrieval.vector_index import COLLECTION_NAME, INDEX_PATH


def main() -> None:
    """Search for a query and print the highest-scoring jobs."""
    parser = argparse.ArgumentParser(description="Search the JobFit vector index.")
    parser.add_argument("query", help="Search text, such as 'Python FastAPI'.")
    parser.add_argument("--top-k", type=int, default=Settings().retrieval_top_k)
    args = parser.parse_args()

    query = args.query.strip()
    if not query:
        parser.error("The query must contain some text.")
    if args.top_k < 1:
        parser.error("--top-k must be at least 1.")
    if not INDEX_PATH.is_dir():
        parser.error("Build the index first with python -m app.retrieval.vector_index.")

    client = QdrantClient(path=str(INDEX_PATH))
    try:
        if not client.collection_exists(collection_name=COLLECTION_NAME):
            parser.error(f"Collection {COLLECTION_NAME!r} is missing. Build the index first.")

        count = client.count(collection_name=COLLECTION_NAME, exact=True).count

        # Reuse the same model and collection configuration as the index build.
        store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=create_embeddings(),
        )
        # LangChain embeds the query before Qdrant compares it with stored vectors.
        results = store.similarity_search_with_score(query, k=args.top_k)
    finally:
        client.close()

    print(f"Indexed jobs: {count}")
    print(f"Query: {query}")

    for rank, (document, score) in enumerate(results, start=1):
        job_id = document.metadata["job_id"]
        title = document.metadata["title"]
        print(f"{rank}. {score:.4f} | {job_id} | {title}")


if __name__ == "__main__":
    main()
