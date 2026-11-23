"""Compare the original JobFit ranking with cross-encoder reranking."""

import argparse

from app.config import Settings
from app.retrieval.hybrid import retrieve_candidates
from app.retrieval.reranking import create_reranker, rerank_hits
from app.schemas import Preferences


def main() -> None:
    """Retrieve jobs, rerank the shortlist, and print both orders."""
    settings = Settings()
    parser = argparse.ArgumentParser(description="Retrieve and rerank JobFit jobs.")
    parser.add_argument("query", help="Search text, such as 'Python FastAPI'.")
    parser.add_argument("--top-k", type=int, default=settings.retrieval_top_k)
    parser.add_argument("--mode", choices=("dense", "bm25", "hybrid"), default="hybrid")
    parser.add_argument("--location-city")
    parser.add_argument("--work-mode", choices=("hybrid", "onsite", "remote"))
    args = parser.parse_args()

    preferences = Preferences(
        location_city=args.location_city,
        work_mode=args.work_mode,
    )

    try:
        catalog_size, hits = retrieve_candidates(
            args.query,
            args.top_k,
            settings.prepared_jobs_path,
            mode=args.mode,
            preferences=preferences,
        )
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))

    print(f"Search mode: {args.mode}")
    print(f"Catalog jobs: {catalog_size}")

    if not hits:
        print("No jobs matched the query and filters.")
        return

    print("\nBefore reranking")
    for rank, (document, score) in enumerate(hits, start=1):
        metadata = document.metadata
        print(f"{rank}. {metadata['job_id']} | {metadata['title']} | Score={score:.5f}")

    # Load the cross-encoder only when there are jobs to rerank.
    reranked = rerank_hits(args.query, hits, create_reranker())

    print("\nAfter reranking")
    for new_rank, hit in enumerate(reranked, start=1):
        metadata = hit.document.metadata
        print(
            f"{hit.retrieval_rank} -> {new_rank} | {metadata['job_id']} "
            f"| {metadata['title']} | Rerank={hit.rerank_score:.4f}"
        )


if __name__ == "__main__":
    main()
