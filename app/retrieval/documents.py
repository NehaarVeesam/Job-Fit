"""Turn validated source jobs into LangChain Documents without losing identity."""

import hashlib
import json
from uuid import NAMESPACE_URL, uuid5

from langchain_core.documents import Document

from app.schemas import JobRecord

TEXT_VERSION = "jobfit-job-text-v1"


def job_document(job: JobRecord) -> Document:
    text = "\n".join([
        f"Title: {job.title}",
        f"Role family: {job.role_family}",
        "Required skills: " + ", ".join(job.required_skills),
        "Preferred skills: " + ", ".join(job.preferred_skills),
        "Description:\n" + job.description,
    ])
    return Document(page_content=text, metadata={
        **job.model_dump(mode="json"),
        "text_version": TEXT_VERSION,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
    })


def point_id(job_id: str, dataset_id: str = "synthetic_india_ai_sw_v2") -> str:
    return str(uuid5(NAMESPACE_URL, f"jobfit/{dataset_id}/{job_id}"))


def document_fingerprint(documents: list[Document]) -> str:
    payload = [(doc.metadata["job_id"], doc.metadata["text_sha256"]) for doc in documents]
    return hashlib.sha256(json.dumps(sorted(payload)).encode()).hexdigest()
