"""Small direct-client tests; runnable without installing LangChain."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from app.config import PROJECT_ROOT, Settings
from app.resumes.profile import load_prepared_jobs
from app.corpus.dataset_loader import prepare_corpus, write_corpus
from app.schemas import CandidateProfile, ParsedResume, Preferences, PreparedResume
from python_only.engine import JobIndex, check_vectors, prepare_job


class TestEncoder:
    batch_size = 2

    def __init__(self):
        self.calls = 0

    def token_counts(self, texts):
        return [len(text.split()) + 2 for text in texts]

    def encode(self, texts):
        self.calls += len(texts)
        vectors = []
        for text in texts:
            vector = [0.0] * 1024
            vector[0 if "Python" in text else 1] = 1.0
            vectors.append(vector)
        return vectors


def resume(preferences=None, query="Python"):
    return PreparedResume(parsed_resume=ParsedResume(pages=[{"page": 1, "text": "Skills: Python"}]),
                          candidate_profile=CandidateProfile(), preferences=preferences or Preferences(),
                          retrieval_query=query)


class PythonOnlyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        source = PROJECT_ROOT / "dataset/jobs_synthetic_250.jsonl"
        corpus = root / "prepared"
        write_corpus(prepare_corpus(source), corpus, source)
        jobs, provenance = load_prepared_jobs(corpus)
        self.jobs = [jobs[0], jobs[20], jobs[74], jobs[108]]
        self.settings = Settings(_env_file=None, llm_provider="ollama", artifacts_dir=root / "artifacts")
        self.encoder = TestEncoder()
        self.index = JobIndex(self.settings, self.jobs, provenance, self.encoder)
        self.addCleanup(self.index.close)

    def test_index_reuse_and_persisted_search(self):
        first = self.index.build()
        self.assertEqual(first["points"], 4)
        calls = self.encoder.calls
        self.assertTrue(self.index.build()["reused"])
        self.assertEqual(self.encoder.calls, calls)
        self.index.close()
        self.index = JobIndex(self.settings, self.jobs, self.index.identity["corpus"], self.encoder)
        self.addCleanup(self.index.close)
        result = self.index.search(resume())
        self.assertEqual(len(result["candidates"]), 4)

    def test_filters_no_query_no_eligible_and_k_validation(self):
        self.index.build()
        result = self.index.search(resume(Preferences(work_mode="remote", unknown_metadata_policy="exclude")))
        self.assertEqual([item["job_id"] for item in result["candidates"]], ["syn_job_075"])
        self.assertEqual(self.index.search(resume(query=None))["status"], "no_query")
        self.assertEqual(self.index.search(resume(Preferences(location_city="Nowhere", work_mode="hybrid", unknown_metadata_policy="exclude")))["status"], "no_eligible_jobs")
        with self.assertRaises(ValueError):
            self.index.search(resume(), k=0)

    def test_missing_index_and_bad_vectors_fail_clearly(self):
        with self.assertRaises(ValueError):
            self.index.search(resume())
        for vector in ([0.0] * 1024, [float("nan")] * 1024, [1.0]):
            with self.assertRaises(ValueError):
                check_vectors([vector])

    def test_source_text_preserved(self):
        record = prepare_job(self.jobs[0])
        self.assertIn(self.jobs[0].description, record.text)
        self.assertEqual(record.metadata, self.jobs[0].model_dump(mode="json"))

    def test_entry_point_has_no_transitive_langchain_import(self):
        code = ("import sys; import python_only.cli; "
                "assert not [name for name in sys.modules if name.startswith(('langchain', 'langsmith'))]")
        subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, check=True,
                       capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
