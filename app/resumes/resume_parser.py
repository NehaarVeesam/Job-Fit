"""Extract page text from PDF resumes."""

from pathlib import Path

import pymupdf

from app.schemas import ParsedResume, ResumePage


def extract_pdf_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Return the original page numbers and extracted text from a PDF."""
    # Use a byte stream so a malformed upload cannot leave its file locked.
    with pymupdf.open(stream=pdf_path.read_bytes(), filetype="pdf") as document:
        if not document.is_pdf:
            raise ValueError("Expected a PDF resume.")

        if document.needs_pass:
            raise ValueError("The PDF requires a password.")

        # Keep blank pages so later evidence still cites the original page number.
        pages = [
            (page_number, page.get_text("text"))
            for page_number, page in enumerate(document, start=1)
        ]

    if not any(text.strip() for _, text in pages):
        raise ValueError("The PDF contains no extractable text.")

    return pages


def parse_resume(pdf_path: Path) -> ParsedResume:
    """Read a PDF and return its pages as a validated resume."""
    pages = [
        ResumePage(page=page_number, text=text)
        for page_number, text in extract_pdf_pages(pdf_path)
    ]
    return ParsedResume(pages=pages)
