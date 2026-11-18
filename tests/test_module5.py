"""HTTP integration tests: real PDFs/pipeline/local Qdrant, controlled vectors."""

import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import pymupdf
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings

from app.api_schemas import JobDetailResponse, UploadMatchResponse
from app.api_service import MatchingService
from app.config import PROJECT_ROOT, Settings
from app.dataset_loader import prepare_corpus, write_corpus
from app.embeddings import CheckedEmbeddings
from app.generation import GroundedExplainer
from app.main import create_app
from app.profile import load_prepared_jobs
from app.vector_index import build_index


class Tokenizer:
    def __call__(self, texts, **kwargs):
        return {"input_ids": [[0] * (len(text.split()) + 2) for text in texts]}


class Vectors(Embeddings):
    @staticmethod
    def vector(text):
        vector = [0.0] * 1024
        vector[0 if "LangChain" in text and "RAG" in text else 1] = 1.0
        return vector

    def embed_documents(self, texts):
        return [self.vector(text) for text in texts]

    def embed_query(self, text):
        return self.vector(text)


def pdf(text):
    with pymupdf.open() as document:
        page = document.new_page()
        if text:
            page.insert_text((40, 40), text, fontsize=10)
        return document.tobytes()


class APITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        root = Path(cls.temporary.name)
        cls.settings = Settings(_env_file=None, llm_provider="ollama", artifacts_dir=root / "artifacts",
                                qdrant_path=root / "qdrant", api_timeout_seconds=5)
        source = PROJECT_ROOT / "dataset/jobs_synthetic_250.jsonl"
        corpus = cls.settings.artifacts_dir / "dataset/all"
        write_corpus(prepare_corpus(source), corpus, source)
        jobs, provenance = load_prepared_jobs(corpus)
        cls.encoder = CheckedEmbeddings(Vectors(), Tokenizer(), 8192)
        cls.info = {"model": "BAAI/bge-m3", "revision": "test", "dimension": 1024,
                    "normalize_embeddings": True, "distance": "Cosine", "max_tokens": 8192,
                    "device": "test", "batch_size": 4}
        build_index(cls.settings, jobs, provenance, cls.encoder, cls.info)
        cls.ai_pdf = pdf((PROJECT_ROOT / "examples/resumes/ai.txt").read_text())

    def setUp(self):
        self.service = MatchingService(self.settings)
        self.service.embeddings, self.service.embedding_info = self.encoder, self.info
        self.app = create_app(self.settings, self.service)
        self.client = self.enterContext(TestClient(self.app))

    def upload(self, content=None, **fields):
        return self.client.post("/match", files={"resume": ("resume.pdf", content if content is not None else self.ai_pdf, "application/pdf")},
                                data={"use_llm": "false", **fields})

    def test_health_readiness_and_openapi_do_not_load_models(self):
        with patch("app.embeddings.load_embeddings", side_effect=AssertionError("Unexpected model load")):
            self.assertEqual(self.client.get("/health").json()["status"], "ok")
            self.assertEqual(self.client.get("/ready").status_code, 200)
            self.assertIn("/match", self.client.get("/openapi.json").json()["paths"])

    def test_upload_full_pipeline_and_non_recommended_job_detail(self):
        response = self.upload()
        self.assertEqual(response.status_code, 200, response.text)
        result = UploadMatchResponse.model_validate(response.json())
        self.assertTrue(result.result.is_synthetic)
        self.assertTrue(result.result.recommendations)
        run = self.app.state.runs[result.run_id][1]
        run.retrieval["candidates"] = [item for item in run.retrieval["candidates"] if item["job_id"] != "syn_job_234"]
        detail = self.client.get(f"/runs/{result.run_id}/jobs/syn_job_234")
        self.assertEqual(detail.status_code, 200, detail.text)
        model = JobDetailResponse.model_validate(detail.json())
        self.assertEqual(model.job.job_id, "syn_job_234")
        self.assertFalse(model.was_recommended)
        self.assertEqual(model.generation_status, "deterministic")
        self.assertEqual(list((self.settings.artifacts_dir / "uploads").glob("*.pdf")), [])

    def test_zero_overlap_detail_does_not_require_fake_positive_reason(self):
        result = self.upload(pdf("Summary\nPastry chef\nSkills\nCake decoration")).json()
        self.assertEqual(result["result"]["status"], "no_suitable_matches")
        response = self.client.get(f"/runs/{result['run_id']}/jobs/syn_job_001")
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["passes_policy"])
        self.assertEqual(response.json()["reasons"], [])
        self.assertEqual(response.json()["required_coverage"]["evidenced_count"], 0)

    def test_invalid_preferences_and_missing_upload_are_safe_422_errors(self):
        for value in ("not-json-private", '{"work_mode":"invalid-private"}'):
            result = self.upload(preferences=value)
            self.assertEqual(result.status_code, 422)
            self.assertNotIn("private", result.text)
        self.assertEqual(self.client.post("/match").status_code, 422)

    def test_invalid_empty_and_image_only_pdf_have_clear_errors_and_cleanup(self):
        for content, code in ((b"", "empty_file"), (b"private resume text", "not_pdf"),
                              (pdf(""), "no_extractable_text")):
            result = self.upload(content)
            self.assertEqual(result.status_code, 422)
            self.assertEqual(result.json()["detail"]["code"], code)
            self.assertNotIn("private resume text", result.text)
        self.assertEqual(list((self.settings.artifacts_dir / "uploads").glob("*.pdf")), [])

    def test_upload_limit_with_and_without_content_length(self):
        big = b"x" * (self.settings.max_upload_mb * 1024 * 1024 + 65537)
        self.assertEqual(self.upload(big).status_code, 413)
        self.assertEqual(self.upload(big[:self.settings.max_upload_mb * 1024 * 1024 + 1]).status_code, 413)
        def chunks():
            yield big
        response = self.client.post("/match", content=chunks(), headers={"content-type": "multipart/form-data; boundary=demo"})
        self.assertEqual(response.status_code, 413)

    def test_missing_corpus_and_index_return_503(self):
        self.service.jobs = []
        self.assertEqual(self.client.get("/ready").status_code, 503)
        self.assertEqual(self.upload().status_code, 503)
        self.service.jobs, _ = load_prepared_jobs(self.settings.artifacts_dir / "dataset/all")
        with patch("app.retrieval.retrieve", side_effect=ValueError("private provider body")):
            response = self.upload()
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("private", response.text)

    def test_missing_model_returns_503_and_cleans_upload(self):
        self.service.embeddings = None
        with patch("app.embeddings.load_embeddings", side_effect=RuntimeError("private model path")):
            response = self.upload()
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("private", response.text)
        self.assertEqual(list((self.settings.artifacts_dir / "uploads").glob("*.pdf")), [])

    def test_unavailable_qwen_is_a_visible_fallback_not_a_false_success_label(self):
        class Broken:
            def invoke(self, data):
                raise ConnectionError("private API key")
        def factory(settings, deadline=None):
            return GroundedExplainer(settings, chain=Broken(), deadline=deadline)
        with patch("app.api_service.GroundedExplainer", side_effect=factory):
            response = self.upload(use_llm="true")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(all(m["generation_status"] == "fallback" for m in response.json()["result"]["recommendations"]))
        self.assertNotIn("private API key", response.text)

    def test_unknown_expired_and_deleted_runs_and_unknown_jobs(self):
        self.assertEqual(self.client.get("/runs/unknown/jobs/syn_job_001").status_code, 404)
        run_id = self.upload().json()["run_id"]
        self.assertEqual(self.client.get(f"/runs/{run_id}/jobs/not-a-job").status_code, 404)
        deleted = self.client.delete(f"/runs/{run_id}")
        self.assertEqual(deleted.status_code, 204)
        self.assertEqual(deleted.content, b"")
        self.assertEqual(self.client.get(f"/runs/{run_id}/jobs/syn_job_001").status_code, 404)
        run_id = self.upload().json()["run_id"]
        _, run = self.app.state.runs[run_id]
        self.app.state.runs[run_id] = (time.monotonic() - 1, run)
        self.assertEqual(self.client.get(f"/runs/{run_id}/jobs/syn_job_001").status_code, 404)

    def test_session_capacity_evicts_oldest(self):
        old = self.app.state.settings
        self.app.state.settings = old.model_copy(update={"api_max_sessions": 1})
        first = self.upload().json()["run_id"]
        second = self.upload().json()["run_id"]
        self.assertEqual(len(self.app.state.runs), 1)
        self.assertEqual(self.client.get(f"/runs/{first}/jobs/syn_job_001").status_code, 404)
        self.assertEqual(self.client.get(f"/runs/{second}/jobs/syn_job_001").status_code, 200)

    def test_busy_request_is_rejected_without_queueing(self):
        entered, release = threading.Event(), threading.Event()
        original = self.service.match
        def slow(*args):
            entered.set()
            release.wait(2)
            return original(*args)
        with patch.object(self.service, "match", side_effect=slow), ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self.upload)
            try:
                self.assertTrue(entered.wait(2))
                self.assertEqual(self.upload().status_code, 429)
                self.assertTrue(self.client.get("/health").json()["busy"])
            finally:
                release.set()
            self.assertEqual(future.result().status_code, 200)

    def test_timeout_does_not_release_processing_slot_early_or_store_run(self):
        self.app.state.settings = self.settings.model_copy(update={"api_timeout_seconds": 0.02})
        entered, release = threading.Event(), threading.Event()
        def slow(*args):
            entered.set()
            release.wait(1)
            raise TimeoutError()
        with patch.object(self.service, "match", side_effect=slow):
            try:
                response = self.upload()
                self.assertEqual(response.status_code, 504)
                self.assertTrue(entered.is_set())
                self.assertEqual(self.upload().status_code, 429)
                self.assertEqual(len(self.app.state.runs), 0)
            finally:
                release.set()
        # Lifespan shutdown also waits for worker cleanup; allow its callback here.
        for _ in range(50):
            if not self.client.get("/health").json()["busy"]:
                break
            time.sleep(0.01)
        self.assertFalse(self.client.get("/health").json()["busy"])
        self.assertEqual(list((self.settings.artifacts_dir / "uploads").glob("*.pdf")), [])


if __name__ == "__main__":
    unittest.main()
