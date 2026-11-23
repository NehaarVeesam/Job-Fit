"""Expose the JobFit matching pipeline through FastAPI."""

from contextlib import nullcontext
from pathlib import Path
from shutil import copyfileobj
from tempfile import TemporaryDirectory
from threading import Lock
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pymupdf import FileDataError

from app.config import Settings
from app.matching.pipeline import ResumeMatchOptions, run_resume_match
from app.retrieval.hybrid import SearchMode
from app.retrieval.vector_index import INDEX_PATH
from app.schemas import Preferences, ResumeMatchResult


app = FastAPI(title="JobFit API")
DENSE_MATCH_LOCK = Lock()


class MatchRequest(BaseModel):
    """Preferences and pipeline options sent alongside the resume PDF."""

    model_config = ConfigDict(extra="forbid")

    preferences: Preferences = Field(default_factory=Preferences)
    options: ResumeMatchOptions = Field(default_factory=ResumeMatchOptions)


@app.get("/health")
def health() -> dict[str, str]:
    """Confirm that the API is responding."""
    return {"status": "ok"}


@app.get("/ready")
def ready(mode: SearchMode = "hybrid") -> dict[str, str]:
    """Check that the local files needed by the requested mode are present."""
    settings = Settings()
    required_files = (
        settings.prepared_jobs_path,
        settings.prepared_data_dir / "skill_evidence.jsonl",
    )
    missing = [str(path) for path in required_files if not path.is_file()]

    if mode in ("dense", "hybrid") and not INDEX_PATH.is_dir():
        missing.append(str(INDEX_PATH))

    if missing:
        raise HTTPException(status_code=503, detail={"missing": missing})

    return {"status": "ready", "mode": mode}


@app.post("/match")
def match_resume(
    resume: Annotated[UploadFile, File()],
    match_request: Annotated[str, Form()] = "{}",
) -> ResumeMatchResult:
    """Match an uploaded PDF and return the recommendations as JSON."""
    try:
        request = MatchRequest.model_validate_json(match_request)
    except ValidationError as error:
        raise HTTPException(
            status_code=422,
            detail=error.errors(
                include_url=False, include_context=False, include_input=False
            ),
        ) from error

    try:
        with TemporaryDirectory() as folder:
            # Use a fixed name for the path-based pipeline within this request.
            pdf_path = Path(folder) / "resume.pdf"
            with pdf_path.open("wb") as target:
                copyfileobj(resume.file, target)

            # Serialize dense requests because they open the same local Qdrant folder.
            guard = (
                DENSE_MATCH_LOCK
                if request.options.mode in ("dense", "hybrid")
                else nullcontext()
            )
            with guard:
                return run_resume_match(pdf_path, request.preferences, request.options)

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail="Prepare the job catalog and build the index for the requested mode.",
        ) from error
    except FileDataError as error:
        raise HTTPException(
            status_code=422,
            detail="The uploaded file could not be read as a PDF.",
        ) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
