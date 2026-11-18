"""Create text-based PDFs from public synthetic examples; no real personal data."""

from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent


def main():
    output = ROOT / "artifacts" / "demo_resumes"
    output.mkdir(parents=True, exist_ok=True)
    for source in sorted((ROOT / "examples" / "resumes").glob("*.txt")):
        target = output / (source.stem + ".pdf")
        if target.is_symlink():
            raise ValueError("Demo output must not be a symlink")
        with pymupdf.open() as document:
            page = document.new_page()
            remaining = page.insert_textbox(pymupdf.Rect(40, 40, 555, 802),
                                             source.read_text(encoding="utf-8"), fontsize=11)
            if remaining < 0:
                raise ValueError("Demo resume text does not fit; add explicit pagination")
            document.save(target)
        print(f"Created synthetic demo: {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
