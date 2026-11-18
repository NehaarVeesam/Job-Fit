"""Real generated PDFs and deterministic profile/evidence failure cases."""

import json
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pymupdf
from pydantic import ValidationError

from app.config import PROJECT_ROOT, Settings
from app.dataset_loader import prepare_corpus, write_corpus
from app.prepare_resume import main
from app.profile import build_profile, load_prepared_jobs, skill_vocabulary
from app.resume_parser import ResumeError, parse_resume
from app.schemas import ParsedResume, Preferences, PreparedResume


def make_pdf(path, pages, **save_options):
    with pymupdf.open() as document:
        for text in pages:
            page = document.new_page()
            if text:
                page.insert_text((40, 40), text, fontsize=10)
        document.save(path, **save_options)


def parsed(text):
    return ParsedResume(pages=[{"page": 1, "text": text}])


class ResumeParserTests(unittest.TestCase):
    def test_extracts_pages_keeps_blank_pages_and_exact_quotes(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "resume.pdf"
            make_pdf(source, ["Skills\nPython, C++, C#, .NET, Node.js, CI/CD", ""])
            result = parse_resume(source)
        self.assertEqual([p.page for p in result.parsed.pages], [1, 2])
        self.assertEqual(result.parsed.pages[1].text, "")
        self.assertIn("C++, C#, .NET, Node.js, CI/CD", result.parsed.pages[0].text)
        self.assertTrue(any("Page 2" in warning for warning in result.warnings))

    def test_unreadable_empty_non_pdf_and_corrupt_have_clear_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "resume.pdf"
            for data, expected in ((None, "unreadable_file"), (b"", "empty_file"),
                                   (b"Skills: Python", "not_pdf"), (b"%PDF-1.7\ninvalid", "invalid_pdf")):
                if data is not None:
                    source.write_bytes(data)
                with self.subTest(expected=expected), self.assertRaises(ResumeError) as error:
                    parse_resume(source)
                self.assertEqual(error.exception.code, expected)

    def test_locked_pdf_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "locked.pdf"
            make_pdf(source, ["Skills\nPython"], encryption=pymupdf.PDF_ENCRYPT_AES_256,
                     owner_pw="owner", user_pw="secret")
            with self.assertRaises(ResumeError) as error:
                parse_resume(source)
        self.assertEqual(error.exception.code, "locked_pdf")

    def test_blank_and_image_only_pdfs_require_searchable_text(self):
        with tempfile.TemporaryDirectory() as directory:
            for image_only in (False, True):
                source = Path(directory) / f"blank{image_only}.pdf"
                with pymupdf.open() as document:
                    page = document.new_page()
                    if image_only:
                        image = pymupdf.Pixmap(pymupdf.csRGB, (0, 0, 50, 50))
                        image.clear_with(255)
                        page.insert_image(pymupdf.Rect(0, 0, 100, 100), pixmap=image)
                    document.save(source)
                with self.assertRaises(ResumeError) as error:
                    parse_resume(source)
                self.assertEqual(error.exception.code, "no_extractable_text")

    def test_file_page_and_text_limits_fail_without_silent_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "resume.pdf"
            make_pdf(source, ["Skills\nPython", "Education\nB.Tech"])
            for limits, expected in (({"max_pages": 1}, "too_many_pages"),
                                     ({"max_text_characters": 5}, "too_much_text")):
                with self.assertRaises(ResumeError) as error:
                    parse_resume(source, **limits)
                self.assertEqual(error.exception.code, expected)
            source.write_bytes(b"%PDF-" + b"0" * (1024 * 1024))
            with self.assertRaises(ResumeError) as error:
                parse_resume(source, max_upload_mb=1)
            self.assertEqual(error.exception.code, "file_too_large")


class ProfileTests(unittest.TestCase):
    def test_symbol_skills_boundaries_case_and_related_tools(self):
        result = build_profile(parsed("Skills\npython, C++, C#, .NET, Node.js, CI/CD, FastAPI, JavaScript"),
                               ["Python", "C", "C++", "C#", ".NET", "Node.js", "CI/CD", "FastAPI", "Flask", "Java"])
        found = {fact.text for fact in result.candidate_profile.skills}
        self.assertEqual(found, {"Python", "C++", "C#", ".NET", "Node.js", "CI/CD", "FastAPI"})

    def test_contacts_and_name_do_not_enter_query_or_automatic_preferences(self):
        text = "Python Sharma\nEmail: java@example.invalid\nAddress: Bengaluru\nSkills\nSQL\nGitHub: https://github.com/python\nProjects\nBuilt a SQL report."
        result = build_profile(parsed(text), ["Python", "Java", "SQL"])
        self.assertEqual(result.retrieval_query, "Skills: SQL")
        self.assertIsNone(result.preferences.location_city)
        self.assertIsNone(result.preferences.target_role)
        # Raw extracted pages remain local for evidence; only the query excludes PII.
        self.assertIn("java@example.invalid", result.parsed_resume.pages[0].text)

    def test_preferences_remain_explicit_and_contact_role_is_rejected(self):
        preferences = Preferences(target_role="Backend Engineer", location_city="Hyderabad", work_mode="remote")
        result = build_profile(parsed("Skills: Python"), ["Python"], preferences)
        self.assertEqual(result.preferences, preferences)
        self.assertIn("Target role: Backend Engineer", result.retrieval_query)
        with self.assertRaises(ValueError):
            build_profile(parsed("Skills: Python"), ["Python"], Preferences(target_role="person@example.invalid"))

    def test_negative_learning_and_related_skill_claims_are_not_positive_evidence(self):
        result = build_profile(parsed("Skills\nPython\nNo experience with Java\nLearning Flask\nProjects\nBuilt a FastAPI service"),
                               ["Python", "Java", "Flask", "FastAPI"])
        self.assertEqual({fact.text for fact in result.candidate_profile.skills}, {"Python", "FastAPI"})

    def test_only_explicit_professional_duration_is_accepted(self):
        for text, expected in (
            ("Summary\nProfessional experience: 3 years", 3),
            ("Summary\n2.5 years of full-time experience", 2.5),
            ("Projects\n3 years of professional experience", None),
            ("Internships\n3 years of professional experience", None),
            ("Experience\nWorked from 2020 to 2024", None),
            ("Summary\nOver 3 years of professional experience", None),
            ("Summary\n3 years of professional experience\n4 years of professional experience", None),
        ):
            with self.subTest(text=text):
                self.assertEqual(build_profile(parsed(text), []).candidate_profile.experience_years, expected)

    def test_fabricated_quote_or_wrong_page_fails_prepared_contract(self):
        result = build_profile(parsed("Skills: Python"), ["Python"])
        for updates in ({"quote": "Invented Python evidence"}, {"page": 2}):
            data = result.model_dump(mode="json")
            data["candidate_profile"]["skills"][0]["evidence"][0].update(updates)
            with self.assertRaises(ValidationError):
                PreparedResume.model_validate(data)

    def test_five_synthetic_profiles_use_real_pdfs_and_unrelated_stays_without_query(self):
        corpus = prepare_corpus(PROJECT_ROOT / "dataset" / "jobs_synthetic_250.jsonl")
        vocabulary = skill_vocabulary(corpus.jobs)
        with tempfile.TemporaryDirectory() as directory:
            for name in ("ai", "backend", "analyst", "fresher", "unrelated"):
                source = Path(directory) / f"{name}.pdf"
                make_pdf(source, [(PROJECT_ROOT / "examples" / "resumes" / f"{name}.txt").read_text()])
                extracted = parse_resume(source)
                result = build_profile(extracted.parsed, vocabulary)
                with self.subTest(name=name):
                    self.assertEqual(PreparedResume.model_validate_json(result.model_dump_json()), result)
                    if name == "unrelated":
                        self.assertIsNone(result.retrieval_query)
                    else:
                        self.assertIsNotNone(result.retrieval_query)
                    if name == "fresher":
                        self.assertIsNone(result.candidate_profile.experience_years)
                    if name == "ai":
                        self.assertEqual(result.candidate_profile.experience_years, 3)


class ResumePipelineTests(unittest.TestCase):
    def test_cli_writes_valid_artifacts_and_detects_corpus_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus_dir = root / "artifacts" / "dataset" / "all"
            source = PROJECT_ROOT / "dataset" / "jobs_synthetic_250.jsonl"
            write_corpus(prepare_corpus(source), corpus_dir, source)
            settings = Settings(_env_file=None, llm_provider="ollama", artifacts_dir=root / "artifacts")
            resume = root / "resume.pdf"
            make_pdf(resume, ["Skills\nPython, SQL\nProjects\nBuilt a SQL report."])
            preferences = root / "preferences.json"
            preferences.write_text(json.dumps({"target_role": "Backend Engineer"}))
            with patch("app.prepare_resume.get_settings", return_value=settings), redirect_stdout(StringIO()):
                self.assertEqual(main([str(resume), "--preferences", str(preferences)]), 0)
            results = list((root / "artifacts" / "resumes").glob("*/prepared_resume.json"))
            self.assertEqual(len(results), 1)
            PreparedResume.model_validate_json(results[0].read_text())
            (corpus_dir / "jobs.jsonl").write_text("tampered")
            with self.assertRaises(ValueError):
                load_prepared_jobs(corpus_dir)
            with patch("app.prepare_resume.get_settings", return_value=settings), redirect_stderr(StringIO()):
                self.assertEqual(main([str(resume)]), 1)


if __name__ == "__main__":
    unittest.main()
