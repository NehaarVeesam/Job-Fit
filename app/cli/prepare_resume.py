"""CLI: python -m app.prepare_resume resume.pdf [--preferences preferences.json]."""

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

from pydantic import ValidationError

from app.config import PROJECT_ROOT, get_settings
from app.resumes.profile import PROFILE_VERSION, build_profile, load_prepared_jobs, skill_vocabulary
from app.resumes.resume_parser import ResumeError, parse_resume
from app.schemas import Preferences


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("resume", type=Path)
    parser.add_argument("--preferences", type=Path)
    parser.add_argument("--corpus-dir", type=Path, help="Prepared Module 1 directory; defaults to artifacts/dataset/all")
    args = parser.parse_args(argv)
    try:
        settings = get_settings()
        directory = args.corpus_dir or settings.artifacts_dir / "dataset" / "all"
        jobs, provenance = load_prepared_jobs(directory)
        vocabulary = skill_vocabulary(jobs)
        preferences = (Preferences.model_validate_json(args.preferences.read_text(encoding="utf-8"))
                       if args.preferences else Preferences())
        result = parse_resume(args.resume, max_upload_mb=settings.max_upload_mb,
                              max_pages=settings.max_resume_pages)
        prepared = build_profile(result.parsed, vocabulary, preferences)
        prepared.warnings = result.warnings + prepared.warnings
        source_hash = result.source_sha256
        report = {
            "profile_version": PROFILE_VERSION, "resume_sha256": source_hash,
            "parser": "PyMuPDF", "parser_version": __import__("pymupdf").VersionBind,
            "corpus": provenance, "vocabulary_size": len(vocabulary),
            "vocabulary_sha256": hashlib.sha256(json.dumps(vocabulary).encode()).hexdigest(),
            "page_count": len(result.parsed.pages),
            "skill_count": len(prepared.candidate_profile.skills),
            "query_available": prepared.retrieval_query is not None,
            "warnings": prepared.warnings,
        }
        output = settings.artifacts_dir / "resumes" / source_hash
        # Do not let a user-configured output path overwrite source/corpus data.
        protected = [settings.dataset_path.parent.resolve(), (PROJECT_ROOT / "dataset").resolve(),
                     directory.resolve()]
        resolved = output.resolve()
        if any(resolved == folder or folder in resolved.parents for folder in protected):
            raise ValueError("Resume output must be outside the input and prepared-corpus folders")
        output.mkdir(parents=True, exist_ok=True)
        payloads = {
            "parsed_resume.json": prepared.parsed_resume.model_dump(mode="json"),
            "candidate_profile.json": prepared.candidate_profile.model_dump(mode="json"),
            "preferences.json": prepared.preferences.model_dump(mode="json"),
            "prepared_resume.json": prepared.model_dump(mode="json"),
            "preparation_report.json": report,
        }
        for name in payloads:
            target = output / name
            if target.is_symlink() or target.resolve() == args.resume.resolve():
                raise ValueError("Resume output files must not overwrite inputs or symlinks")
        for name, payload in payloads.items():
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output, delete=False) as handle:
                    temporary = Path(handle.name)
                    json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
                    handle.write("\n")
                os.replace(temporary, output / name)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
    except ResumeError as exc:
        print(f"Resume preparation failed [{exc.code}]: {exc}", file=sys.stderr)
        return 1
    except ValidationError:
        print("Resume preparation failed: invalid settings, preferences, or evidence contract. Check field types and source quotes.", file=sys.stderr)
        return 1
    except (OSError, ValueError, KeyError):
        print("Resume preparation failed: check input paths, prepared-corpus checksums, preferences, and output permissions. Run Module 1 first.", file=sys.stderr)
        return 1
    print(f"Prepared {report['page_count']} pages; {report['skill_count']} evidenced skill mentions; "
          f"search query {'available' if report['query_available'] else 'unavailable'}.")
    print(f"Local outputs: {output}")
    print("No resume content sent to services. Review warnings in preparation_report.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
