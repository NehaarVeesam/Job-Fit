"""Meaningful contract checks; no model downloads, services, or PDF parsing."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from app.config import PROJECT_ROOT, Settings
from app.schemas import (
    CandidateProfile, ConstraintAssessment, CoverageScore, JobRecord, MatchResponse,
    ParsedResume, Preferences, RelevanceScore,
)


def example(filename):
    return json.loads((PROJECT_ROOT / "examples" / filename).read_text())


class DatasetContractTests(unittest.TestCase):
    def test_all_250_supplied_records_fit_schema_without_field_loss(self):
        lines = (PROJECT_ROOT / "dataset" / "jobs_synthetic_250.jsonl").read_text().splitlines()
        self.assertEqual(len(lines), 250)
        for number, line in enumerate(lines, 1):
            with self.subTest(record=number):
                source = json.loads(line)
                result = JobRecord.model_validate(source)
                self.assertEqual(result.model_dump(mode="json"), source)

    def test_missing_values_and_technical_tokens_are_preserved(self):
        jobs = [JobRecord.model_validate_json(line) for line in (
            PROJECT_ROOT / "dataset" / "jobs_synthetic_250.jsonl"
        ).read_text().splitlines()]
        self.assertEqual(sum(job.experience_min_years is None for job in jobs), 4)
        self.assertEqual(sum(job.location_city is None for job in jobs), 4)
        skills = {skill for job in jobs for skill in job.required_skills + job.preferred_skills}
        self.assertTrue({"C++", "C#", ".NET", "Node.js", "CI/CD"} <= skills)

    def test_false_synthetic_marker_and_fake_application_link_rejected(self):
        source = example("match_response.json")["recommendations"][0]["job"]
        for overrides in ({"is_synthetic": False}, {"is_synthetic": 1}, {"source_url": "https://example.com/apply"}):
            with self.subTest(overrides=overrides), self.assertRaises(ValidationError):
                JobRecord.model_validate(dict(source, **overrides))

    def test_invalid_experience_overlap_and_unsubstantiated_skill_rejected(self):
        source = example("match_response.json")["recommendations"][0]["job"]
        for overrides in (
            {"experience_min_years": 10, "experience_max_years": 2},
            {"preferred_skills": [source["required_skills"][0]]},
            {"required_skills": ["An invented skill not in the source"]},
            {"required_skills": [" "]},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValidationError):
                JobRecord.model_validate(dict(source, **overrides))


class ResponseContractTests(unittest.TestCase):
    def test_examples_validate_and_round_trip(self):
        for filename, schema in (
            ("candidate_profile.json", CandidateProfile), ("preferences.json", Preferences),
            ("match_response.json", MatchResponse), ("no_match_response.json", MatchResponse),
        ):
            with self.subTest(filename=filename):
                model = schema.model_validate(example(filename))
                self.assertEqual(schema.model_validate_json(model.model_dump_json()), model)

    def test_unknown_experience_and_missing_constraint_answers_remain_unknown(self):
        self.assertIsNone(CandidateProfile().experience_years)
        with self.assertRaises(ValidationError):
            CandidateProfile(experience_years=3)
        condition = ConstraintAssessment(condition="Able to travel", status="unknown")
        self.assertEqual(condition.status, "unknown")
        with self.assertRaises(ValidationError):
            ConstraintAssessment(condition="Able to travel", status="met")

    def test_coverage_is_count_based_and_empty_list_is_not_perfect(self):
        self.assertEqual(CoverageScore(evidenced_count=4, total_count=6).percentage, 66.7)
        self.assertIsNone(CoverageScore(evidenced_count=0, total_count=0).percentage)
        for values in (
            dict(evidenced_count=4, total_count=6, percentage=99),
            dict(evidenced_count=7, total_count=6),
            dict(evidenced_count=0, total_count=0, percentage=100),
        ):
            with self.subTest(values=values), self.assertRaises(ValidationError):
                CoverageScore(**values)

    def test_status_must_agree_with_recommendations(self):
        value = example("match_response.json")
        value["status"] = "no_suitable_matches"
        with self.assertRaises(ValidationError):
            MatchResponse.model_validate(value)
        with self.assertRaises(ValidationError):
            MatchResponse(status="matches_found")

    def test_duplicate_or_more_than_five_recommendations_rejected(self):
        for count in (2, 6):
            value = example("match_response.json")
            value["recommendations"] *= count
            with self.subTest(count=count), self.assertRaises(ValidationError):
                MatchResponse.model_validate(value)

    def test_fabricated_quote_and_wrong_job_reference_rejected(self):
        for field, changed in (("quote", "Fabricated quote"), ("job_id", "another_job")):
            value = example("match_response.json")
            value["recommendations"][0]["reasons"][0]["job_evidence"][field] = changed
            with self.subTest(field=field), self.assertRaises(ValidationError):
                MatchResponse.model_validate(value)

    def test_coverage_and_rank_must_agree_with_actual_assessments(self):
        value = example("match_response.json")
        value["recommendations"][0]["required_coverage"] = dict(evidenced_count=6, total_count=6)
        with self.assertRaises(ValidationError):
            MatchResponse.model_validate(value)
        value = example("match_response.json")
        value["ranked_candidates"][0]["eligibility"] = "not_met"
        with self.assertRaises(ValidationError):
            MatchResponse.model_validate(value)

    def test_nonfinite_and_out_of_range_similarity_rejected(self):
        for value in (float("nan"), float("inf"), 85):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                RelevanceScore(value=value, score_type="cosine")

    def test_blank_pdf_page_can_be_retained_but_empty_resume_is_invalid(self):
        result = ParsedResume(pages=[dict(page=1, text=""), dict(page=2, text="Skills: Python")])
        self.assertEqual(result.pages[0].text, "")
        with self.assertRaises(ValidationError):
            ParsedResume(pages=[dict(page=1, text="")])


class ConfigurationTests(unittest.TestCase):
    def test_remote_qwen_settings_and_secret_redaction(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(
                _env_file=None,
                QWEN_BASE_URL="https://model.example.com/v1",
                QWEN_MODEL="Qwen/Qwen3.8-27B",
                QWEN_API_KEY="test-private-token",
                QWEN_ENABLE_THINKING="false",
                QWEN_JSON_MODE="json_schema",
                QWEN_TEMPERATURE="",
                QWEN_TOP_P="",
                QWEN_MAX_MODEL_LEN="",
            )
        self.assertEqual(settings.llm_provider, "openai_compatible")
        self.assertEqual(settings.llm_model, "Qwen/Qwen3.8-27B")
        self.assertEqual(str(settings.llm_base_url), "https://model.example.com/v1")
        self.assertFalse(settings.llm_enable_thinking)
        self.assertEqual(settings.llm_json_mode, "json_schema")
        self.assertIsNone(settings.llm_temperature)
        self.assertIsNone(settings.llm_top_p)
        self.assertIsNone(settings.llm_max_model_len)
        self.assertNotIn("test-private-token", repr(settings))
        self.assertNotIn("test-private-token", settings.model_dump_json())

    def test_remote_provider_requires_endpoint_and_ollama_remains_available(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValidationError):
                Settings(_env_file=None, llm_provider="openai_compatible")
            settings = Settings(_env_file=None, llm_provider="ollama", llm_model="qwen3.8:27b")
        self.assertEqual(settings.llm_provider, "ollama")

    def test_example_environment_loads_with_frozen_reference_date(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(_env_file=PROJECT_ROOT / ".env.example")
        self.assertEqual(settings.embedding_dimension, 1024)
        self.assertTrue(settings.normalize_embeddings)
        self.assertFalse(settings.recency_filter_enabled)
        self.assertEqual(str(settings.reference_date), "2026-11-18")
        self.assertEqual(settings.corpus_subset, "all")
        self.assertEqual(settings.dataset_path, PROJECT_ROOT / "dataset" / "jobs_synthetic_250.jsonl")

    def test_environment_overrides_dotenv(self):
        with patch.dict(os.environ, {"JOBFIT_LLM_MODEL": "qwen2.5:7b"}, clear=True):
            settings = Settings(_env_file=PROJECT_ROOT / ".env.example")
        self.assertEqual(settings.llm_model, "qwen2.5:7b")

    def test_wrong_dimension_normalization_or_output_location_rejected(self):
        for overrides in (
            {"embedding_dimension": 384}, {"normalize_embeddings": False},
            {"artifacts_dir": "dataset/generated"}, {"qdrant_collection": "real_jobs"},
        ):
            with self.subTest(overrides=overrides), patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(ValidationError):
                    Settings(_env_file=None, **overrides)

    def test_misspelled_dotenv_field_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("JOBFIT_LLM_MODLE=typo\n")
            with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValidationError):
                Settings(_env_file=path)


if __name__ == "__main__":
    unittest.main()
