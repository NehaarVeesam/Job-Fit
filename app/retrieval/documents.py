"""Convert the prepared job catalog into LangChain documents."""

from pathlib import Path

from langchain_core.documents import Document

from app.dataset.dataset_loader import load_jobs
from app.schemas import JobRecord


def job_to_document(job: JobRecord) -> Document:
    """Use the job description as searchable text and retain its other fields."""
    # Keep dates and other values in JSON-friendly form for the vector store.
    metadata = job.model_dump(mode="json", exclude={"description"})
    return Document(page_content=job.description, metadata=metadata)


def load_job_documents(jobs_path: Path) -> list[Document]:
    """Read the prepared catalog and return one document per validated job."""
    with jobs_path.open(encoding="utf-8") as source:
        jobs, rejected_rows = load_jobs(source)

    # Index the whole prepared catalog or surface the first bad row.
    if rejected_rows:
        first = rejected_rows[0]
        raise ValueError(
            f"Prepared catalog has {len(rejected_rows)} rejected rows. "
            f"First issue on line {first.line_number}: {first.message}"
        )

    return [job_to_document(job) for job in jobs]
