"""Run a dense job search from the command line."""

import argparse

from app.config import Settings
from app.retrieval.dense import search_dense


def main() -> None:
    """Read the query and display the highest-scoring jobs."""
    parser = argparse.ArgumentParser(description="Search the JobFit vector index.")
    parser.add_argument("query", help="Search text, such as 'Python FastAPI'.")
    parser.add_argument("--top-k", type=int, default=Settings().retrieval_top_k)
    args = parser.parse_args()

    try:
        indexed_jobs, hits = search_dense(args.query, args.top_k)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))

    print(f"Indexed jobs: {indexed_jobs}")
    print(f"Query: {args.query.strip()}")

    for rank, (document, score) in enumerate(hits, start=1):
        job_id = document.metadata["job_id"]
        title = document.metadata["title"]
        print(f"{rank}. {score:.4f} | {job_id} | {title}")


if __name__ == "__main__":
    main()
