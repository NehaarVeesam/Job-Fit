"""Corpus accounting, preservation, filtering, provenance, and failure cases."""

import hashlib
import json
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

from app.config import PROJECT_ROOT
from app.dataset_loader import CorpusOptions, prepare_corpus, write_corpus

SOURCE = PROJECT_ROOT / "dataset" / "jobs_synthetic_250.jsonl"


class DatasetPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = SOURCE.read_bytes()
        cls.records = [json.loads(line) for line in cls.raw.splitlines()]
        cls.full = prepare_corpus(SOURCE)

    def fixture(self, directory, records):
        source = Path(directory) / "input" / "jobs.jsonl"
        source.parent.mkdir(exist_ok=True)
        source.write_text("".join(json.dumps(job) + "\n" for job in records))
        return source

    def test_full_corpus_accounted_for_and_source_fields_preserved(self):
        self.assertEqual(self.full.report["counts"], dict(
            input_lines=250, rejected=0, valid_unique=250, excluded=0, output=250,
        ))
        self.assertEqual(self.full.report["status"], "ok")
        self.assertEqual([job.model_dump(mode="json") for job in self.full.jobs], self.records)
        self.assertEqual(self.full.report["output_summary"]["domains"], {"AI/ML": 125, "Software": 125})
        self.assertEqual(self.full.report["source"]["sha256"], hashlib.sha256(self.raw).hexdigest())
        self.assertEqual(SOURCE.read_bytes(), self.raw)

    def test_unknowns_constraints_injection_and_role_variants_retained(self):
        jobs = {job.job_id: job for job in self.full.jobs}
        unknowns = self.full.report["output_summary"]["unknowns"]
        self.assertEqual(unknowns["work_mode"], ["syn_job_021", "syn_job_076", "syn_job_184"])
        self.assertEqual(len(unknowns["location_city"]), 4)
        self.assertEqual(len(unknowns["experience"]), 4)
        self.assertEqual(self.full.report["output_summary"]["jobs_with_hard_constraints"], 34)
        self.assertEqual(jobs["syn_job_213"].description, self.records[212]["description"])
        self.assertTrue({"syn_job_001", "syn_job_109", "syn_job_245", "syn_job_246", "syn_job_247"} <= jobs.keys())

    def test_every_skill_has_exact_source_quote_and_offsets(self):
        jobs = {job.job_id: job for job in self.full.jobs}
        skills = set()
        for record in self.full.evidence:
            job = jobs[record["job_id"]]
            self.assertEqual(self.records[record["source_line"] - 1]["job_id"], job.job_id)
            for field in ("required_skills", "preferred_skills"):
                self.assertEqual([item["skill"] for item in record[field]], getattr(job, field))
                for item in record[field]:
                    self.assertIn(item["quote"], job.description)
                    self.assertIn(item["skill"], item["quote"])
                    self.assertEqual(job.description[item["start"]:item["end"]], item["skill"])
                    skills.add(item["skill"])
        self.assertTrue({"C++", "C#", ".NET", "Node.js", "CI/CD"} <= skills)

    def test_fixed_date_recency_removes_exactly_four_known_ids(self):
        result = prepare_corpus(SOURCE, CorpusOptions(recency_enabled=True))
        stale = ["syn_job_128", "syn_job_157", "syn_job_196", "syn_job_238"]
        self.assertEqual(result.report["date_flags"]["older_than_window"], stale)
        self.assertEqual([item["job_id"] for item in result.report["excluded_records"]], stale)
        self.assertEqual(len(result.jobs), 246)
        self.assertEqual(len(self.full.jobs), 250)

    def test_subset_uses_ids_not_file_position(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.fixture(directory, list(reversed(self.records)))
            result = prepare_corpus(source, CorpusOptions(subset="first_100"))
        self.assertEqual([job.job_id for job in result.jobs], [f"syn_job_{i:03}" for i in range(1, 101)])
        self.assertEqual(result.report["counts"]["excluded"], 150)

    def test_bad_lines_and_duplicate_ids_have_reasons_and_accounting(self):
        with tempfile.TemporaryDirectory() as directory:
            changed = dict(self.records[0], title="Conflicting title")
            invalid = dict(self.records[1], is_synthetic=False)
            source = self.fixture(directory, [self.records[0], self.records[0], changed, invalid])
            with source.open("ab") as handle:
                handle.write(b'{bad json}\n\n{"job_id":"a","job_id":"b"}\n\xff\n')
            result = prepare_corpus(source)
        self.assertEqual(result.report["counts"], dict(
            input_lines=8, rejected=7, valid_unique=1, excluded=0, output=1,
        ))
        reasons = [item["reason"] for item in result.report["rejected_records"]]
        self.assertEqual(reasons[:2], ["duplicate_record", "conflicting_job_id"])
        self.assertEqual(result.report["status"], "requires_review")

    def test_same_content_under_different_id_is_flagged_and_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            clone = dict(self.records[0], job_id="syn_job_999")
            source = self.fixture(directory, [self.records[0], clone])
            result = prepare_corpus(source)
        self.assertEqual(len(result.jobs), 2)
        self.assertEqual(result.report["suspicious_records"][0]["same_content_as"], "syn_job_001")
        self.assertEqual(result.report["status"], "requires_review")

    def test_recency_boundary_is_inclusive_and_future_is_explicit(self):
        reference = date(2026, 11, 18)
        records = [dict(self.records[i], posted_at=day.isoformat()) for i, day in enumerate([
            reference - timedelta(days=90), reference - timedelta(days=91),
            reference, reference + timedelta(days=1),
        ])]
        with tempfile.TemporaryDirectory() as directory:
            source = self.fixture(directory, records)
            result = prepare_corpus(source, CorpusOptions(recency_enabled=True))
        self.assertEqual([job.job_id for job in result.jobs], ["syn_job_001", "syn_job_003"])
        self.assertEqual(result.report["excluded_records"][1]["reason"], "after_reference_date")

    def test_outputs_are_reproducible_hashed_and_cannot_overwrite_input(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.fixture(directory, self.records[:2])
            corpus = prepare_corpus(source)
            output = Path(directory) / "output"
            write_corpus(corpus, output, source)
            first = {p.name: p.read_bytes() for p in output.iterdir()}
            write_corpus(prepare_corpus(source), output, source)
            self.assertEqual(first, {p.name: p.read_bytes() for p in output.iterdir()})
            report = json.loads(first["validation_report.json"])
            for filename, metadata in report["outputs"].items():
                self.assertEqual(hashlib.sha256(first[filename]).hexdigest(), metadata["sha256"])
            with self.assertRaises(ValueError):
                write_corpus(corpus, source.parent / "generated", source)
            link = output / "jobs.jsonl"
            link.unlink()
            link.symlink_to(source)
            before = source.read_bytes()
            with self.assertRaises(ValueError):
                write_corpus(corpus, output, source)
            self.assertEqual(source.read_bytes(), before)

    def test_empty_input_and_empty_filtered_corpus_are_distinguished(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.fixture(directory, [])
            empty = prepare_corpus(source)
            self.assertEqual(empty.report["status"], "requires_review")
            self.assertEqual(empty.report["output_summary"]["description_characters"]["min"], None)
            source = self.fixture(directory, [dict(self.records[0], job_id="syn_job_200")])
            filtered = prepare_corpus(source, CorpusOptions(subset="first_100"))
            self.assertEqual(filtered.report["status"], "ok")
            self.assertEqual(filtered.report["counts"]["output"], 0)
            self.assertEqual(filtered.report["counts"]["excluded"], 1)


if __name__ == "__main__":
    unittest.main()
