"""Offline adapter/contract tests using real persistent Qdrant and test vectors.

These tests do not claim BGE-M3 retrieval quality. Real-model runs are recorded
separately by the Module 3 CLI.
"""

import copy
from contextlib import closing
from functools import lru_cache
import tempfile
import unittest
from pathlib import Path

from langchain_core.embeddings import Embeddings
from pydantic import ValidationError
from qdrant_client import QdrantClient, models

from app.config import PROJECT_ROOT, Settings
from app.dataset_loader import prepare_corpus, write_corpus
from app.documents import job_document, point_id
from app.embeddings import CheckedEmbeddings, validate_vectors
from app.filters import eligible_ids, job_passes_preferences, preference_warnings
from app.profile import load_prepared_jobs
from app.retrieval import retrieve
from app.schemas import CandidateProfile, ParsedResume, Preferences, PreparedResume
from app.vector_index import build_index, collection_name, index_identity, prepare_documents


class ToyTokenizer:
    def __call__(self, texts, **kwargs):
        return {"input_ids": [[0] * (len(text.split()) + 2) for text in texts]}


class TestVectors(Embeddings):
    def __init__(self):
        self.document_calls = 0
        self.query_calls = 0

    def vector(self, text):
        # Deliberately tiny semantic test signal, not a substitute for BGE-M3.
        result = [0.0] * 1024
        result[0 if "Python" in text else 1] = 1
        return result

    def embed_documents(self, texts):
        self.document_calls += len(texts)
        return [self.vector(text) for text in texts]

    def embed_query(self, text):
        self.query_calls += 1
        return self.vector(text)


def checked(limit=8192):
    return CheckedEmbeddings(TestVectors(), ToyTokenizer(), limit)


INFO = {"model": "BAAI/bge-m3", "revision": "test-revision", "dimension": 1024,
        "normalize_embeddings": True, "distance": "Cosine", "max_tokens": 8192,
        "device": "test-double", "batch_size": 4}


@lru_cache(maxsize=1)
def source_corpus():
    # Tests work in a fresh checkout without ignored Module 1 artifacts.
    with tempfile.TemporaryDirectory() as directory:
        source = PROJECT_ROOT / "dataset/jobs_synthetic_250.jsonl"
        output = Path(directory) / "prepared"
        write_corpus(prepare_corpus(source), output, source)
        return load_prepared_jobs(output)


def prepared(preferences=None, query="Python"):
    return PreparedResume(parsed_resume=ParsedResume(pages=[{"page": 1, "text": "Skills: Python"}]),
                          candidate_profile=CandidateProfile(), preferences=preferences or Preferences(),
                          retrieval_query=query)


class DocumentAndFilterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.jobs, cls.provenance = source_corpus()

    def test_document_preserves_description_metadata_tokens_and_stable_uuid(self):
        job = self.jobs[0]
        document = job_document(job)
        self.assertIn(job.description, document.page_content)
        for key, value in job.model_dump(mode="json").items():
            self.assertEqual(document.metadata[key], value)
        self.assertEqual(point_id(job.job_id), point_id(job.job_id))
        self.assertNotEqual(point_id(job.job_id), point_id(self.jobs[1].job_id))

    def test_index_identity_changes_with_corpus_text_or_model(self):
        documents = [job_document(self.jobs[0])]
        first = index_identity(documents, self.provenance, INFO)
        self.assertNotEqual(first, index_identity(documents, self.provenance, dict(INFO, revision="other")))
        other = self.jobs[0].model_copy(update={"description": self.jobs[0].description + " More source text."})
        self.assertNotEqual(first, index_identity([job_document(other)], self.provenance, INFO))
        provenance = copy.deepcopy(self.provenance)
        provenance["configuration"]["subset"] = "first_100"
        self.assertNotEqual(first, index_identity(documents, provenance, INFO))

    def test_work_mode_unknown_policy_is_explicit(self):
        job = self.jobs[20]
        self.assertEqual(job.work_mode, "unknown")
        self.assertTrue(job_passes_preferences(job, Preferences(work_mode="remote")))
        self.assertFalse(job_passes_preferences(job, Preferences(work_mode="remote", unknown_metadata_policy="exclude")))
        self.assertTrue(preference_warnings(job, Preferences(work_mode="remote")))

    def test_city_unknown_and_india_wide_remote_are_distinct(self):
        remote = self.jobs[74]
        strict = Preferences(location_city="Bengaluru", unknown_metadata_policy="exclude")
        self.assertIsNone(remote.location_city)
        self.assertEqual(remote.work_mode, "remote")
        self.assertTrue(job_passes_preferences(remote, strict))
        unknown = remote.model_copy(update={"work_mode": "unknown"})
        self.assertFalse(job_passes_preferences(unknown, strict))
        self.assertTrue(job_passes_preferences(unknown, Preferences(location_city="Bengaluru")))
        other_city = remote.model_copy(update={"location_city": "Delhi"})
        self.assertFalse(job_passes_preferences(other_city, strict))


class EmbeddingGuardTests(unittest.TestCase):
    def test_document_and_query_vectors_use_same_checked_interface(self):
        embedding = checked()
        self.assertEqual(embedding.embed_documents(["Python"])[0], embedding.embed_query("Python"))
        validate_vectors(embedding.embed_documents(["Python", "Java"]))

    def test_oversized_and_empty_input_rejected_before_embedding(self):
        embedding = checked(limit=4)
        for method, value in ((embedding.embed_query, "a b c"),
                              (embedding.embed_documents, ["a b c"]),
                              (embedding.embed_query, "")):
            with self.assertRaises(ValueError):
                method(value)
        self.assertEqual(embedding.delegate.document_calls, 0)
        self.assertEqual(embedding.delegate.query_calls, 0)

    def test_wrong_dimension_nonfinite_and_unnormalized_vectors_rejected(self):
        for vector in ([0.0] * 1024, [1.0] * 1024, [1.0], [float("nan")] * 1024):
            with self.subTest(length=len(vector)), self.assertRaises(ValueError):
                validate_vectors([vector])

    def test_embedding_count_must_match_input(self):
        embedding = checked()
        embedding.delegate.embed_documents = lambda texts: []
        with self.assertRaises(ValueError):
            embedding.embed_documents(["Python"])


class PersistentRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.settings = Settings(_env_file=None, llm_provider="ollama", artifacts_dir=root / "artifacts",
                                 qdrant_path=root / "qdrant")
        jobs, self.provenance = source_corpus()
        self.jobs = [jobs[0], jobs[20], jobs[74], jobs[108]]
        self.provenance = dict(self.provenance, job_count=4)
        self.embedding = checked()

    def test_build_persist_reopen_and_repeat_without_duplicates_or_reembedding(self):
        first = build_index(self.settings, self.jobs, self.provenance, self.embedding, INFO)
        self.assertEqual(first["vector_audit"]["point_count"], 4)
        self.assertFalse(first["reused_existing_vectors"])
        calls = self.embedding.delegate.document_calls
        second = build_index(self.settings, self.jobs, self.provenance, self.embedding, INFO)
        self.assertTrue(second["reused_existing_vectors"])
        self.assertEqual(second["vector_audit"]["point_count"], 4)
        self.assertEqual(self.embedding.delegate.document_calls, calls)
        result = retrieve(self.settings, self.jobs, self.provenance, self.embedding, INFO, prepared(), k=3)
        self.assertEqual(len(result["candidates"]), 3)
        self.assertEqual(len({item["job_id"] for item in result["candidates"]}), 3)
        self.assertEqual(result["candidates"][0]["relevance"]["score_type"], "cosine")

    def test_qdrant_filters_match_shared_python_eligible_ids(self):
        build_index(self.settings, self.jobs, self.provenance, self.embedding, INFO)
        for prefs in (Preferences(location_city="Bengaluru"), Preferences(work_mode="remote"),
                      Preferences(work_mode="remote", unknown_metadata_policy="exclude")):
            result = retrieve(self.settings, self.jobs, self.provenance, self.embedding, INFO, prepared(prefs), k=250)
            self.assertEqual({item["job_id"] for item in result["candidates"]}, set(eligible_ids(self.jobs, prefs)))

    def test_no_query_no_eligible_jobs_and_invalid_k_are_explicit(self):
        no_query = retrieve(self.settings, self.jobs, self.provenance, self.embedding, INFO, prepared(query=None))
        self.assertEqual(no_query["status"], "no_query")
        no_jobs = retrieve(self.settings, self.jobs, self.provenance, self.embedding, INFO,
                           prepared(Preferences(location_city="NoSuchCity", work_mode="hybrid", unknown_metadata_policy="exclude")))
        self.assertEqual(no_jobs["status"], "no_eligible_jobs")
        self.assertEqual(self.embedding.delegate.query_calls, 0)
        with self.assertRaises(ValueError):
            retrieve(self.settings, self.jobs, self.provenance, self.embedding, INFO, prepared(), k=0)

    def test_missing_or_changed_model_index_cannot_be_searched(self):
        with self.assertRaises(ValueError):
            retrieve(self.settings, self.jobs, self.provenance, self.embedding, INFO, prepared())
        build_index(self.settings, self.jobs, self.provenance, self.embedding, INFO)
        with self.assertRaises(ValueError):
            retrieve(self.settings, self.jobs, self.provenance, self.embedding,
                     dict(INFO, revision="other"), prepared())

    def test_incomplete_collection_can_be_repaired_by_upserting_stable_ids(self):
        report = build_index(self.settings, self.jobs, self.provenance, self.embedding, INFO)
        with closing(QdrantClient(path=str(self.settings.qdrant_path))) as client:
            client.delete(report["collection_name"], points_selector=models.PointIdsList(
                points=[point_id(self.jobs[0].job_id)]))
        with self.assertRaises(ValueError):
            retrieve(self.settings, self.jobs, self.provenance, self.embedding, INFO, prepared())
        restored = build_index(self.settings, self.jobs, self.provenance, self.embedding, INFO)
        self.assertEqual(restored["vector_audit"]["point_count"], 4)
        self.assertFalse(restored["reused_existing_vectors"])

    def test_collection_provenance_mismatch_is_not_silently_overwritten(self):
        docs, identity = prepare_documents(self.jobs, self.provenance, INFO)
        with closing(QdrantClient(path=str(self.settings.qdrant_path))) as client:
            client.create_collection(collection_name(self.settings, identity),
                                     vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
                                     metadata={"unrelated": True})
        with self.assertRaises(ValueError):
            build_index(self.settings, self.jobs, self.provenance, self.embedding, INFO)


if __name__ == "__main__":
    unittest.main()
