"""Bounded, local PDF text extraction with one-based page evidence."""

from dataclasses import dataclass
import hashlib
from pathlib import Path

import pymupdf

from app.schemas import ParsedResume, ResumePage


class ResumeError(ValueError):
    """An actionable input error safe to display without raw resume content."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


@dataclass
class ExtractionResult:
    parsed: ParsedResume
    warnings: list[str]
    source_sha256: str


def parse_resume(path: Path, *, max_upload_mb: int = 5,
                 max_pages: int = 10, max_text_characters: int = 300_000) -> ExtractionResult:
    if min(max_upload_mb, max_pages, max_text_characters) < 1:
        raise ValueError("Resume limits must be positive")
    try:
        # Bounded read also handles a file growing after an initial stat check.
        with path.open("rb") as handle:
            raw = handle.read(max_upload_mb * 1024 * 1024 + 1)
    except OSError as exc:
        raise ResumeError("unreadable_file", "Cannot read resume; check the file path and permissions.") from exc
    if not raw:
        raise ResumeError("empty_file", "Resume file is empty; supply a text-based PDF.")
    if len(raw) > max_upload_mb * 1024 * 1024:
        raise ResumeError("file_too_large", f"Resume exceeds the {max_upload_mb} MiB limit.")
    if b"%PDF-" not in raw[:1024]:
        raise ResumeError("not_pdf", "Resume must be a PDF; renaming another file is insufficient.")
    warnings = []
    try:
        with pymupdf.open(stream=raw, filetype="pdf") as document:
            if document.needs_pass or document.is_encrypted:
                raise ResumeError("locked_pdf", "PDF is password protected; supply an unlocked copy.")
            if document.page_count == 0:
                raise ResumeError("empty_pdf", "PDF has no pages.")
            if document.page_count > max_pages:
                raise ResumeError("too_many_pages", f"Resume exceeds the {max_pages}-page limit.")
            if document.is_repaired:
                raise ResumeError("damaged_pdf", "PDF required structural repair; export a fresh PDF and retry.")
            pages = []
            length = 0
            for index, page in enumerate(document, 1):
                text = page.get_text("text", sort=True)
                length += len(text)
                if length > max_text_characters:
                    raise ResumeError("too_much_text", "Resume contains too much extracted text; supply a shorter PDF.")
                pages.append(ResumePage(page=index, text=text))
                if not text.strip():
                    warnings.append(f"Page {index} has no extractable text; it may be blank or need OCR.")
            if not any(page.text.strip() for page in pages):
                raise ResumeError("no_extractable_text", "PDF is blank or image-only; supply a searchable PDF. OCR is deferred.")
    except ResumeError:
        raise
    except (RuntimeError, ValueError) as exc:
        raise ResumeError("invalid_pdf", "Cannot parse this PDF; it may be corrupt. Export a fresh PDF and retry.") from exc
    return ExtractionResult(ParsedResume(pages=pages), warnings, hashlib.sha256(raw).hexdigest())
