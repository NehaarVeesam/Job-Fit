"""BM25 arithmetic, token boundaries, RRF provenance, and pipeline integration."""

import copy
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api_service import MatchingService
from app.config import PROJECT_ROOT, Settings
from app.hybrid_retrieval import lexical_identity, reciprocal_rank_fusion, retrieve_candidates
from app.keyword_search import BM25Index, tokenize
from app.main import create_app
from app.matching import match_candidates
from app.profile import build_profile, skill_vocabulary
from app.schemas import ParsedResume, Preferences
from app.vector_index import build_index
from test_module3 import INFO, checked, prepared as empty_profile, source_corpus
from test_module5 import pdf


def configuration(**kwargs):
    return Settings(_env_file=None, llm_provider="ollama", **kwargs)


class KeywordTests(unittest.TestCase):
    def test_technical_tokens_keep_distinct_meanings(self):
        self.assertEqual(tokenize("C++ C# C .NET Node.js ASP.NET Core CI/CD"),
                         ["c++", "c#", "c", ".net", "node.js", "asp.net", "core", "ci/cd"])
        self.assertNotIn("java", tokenize("JavaScript"))
        self.assertNotIn("flask", tokenize("FastAPI"))

    def test_explicit_aliases_and_case(self):
        self.assertEqual(tokenize("NODEJS dotnet k8s postgres sklearn"),
                         tokenize("Node.js .NET Kubernetes PostgreSQL scikit-learn"))
        self.assertNotEqual(tokenize("Flask"), tokenize("FastAPI"))

    def test_score_matches_hand_calculation(self):
        index = BM25Index([("a", "python python"), ("b", "java")])
        result = dict(index.search("python"))
        expected = math.log(2) * (2 * 2.5) / (2 + 1.5 * (0.25 + 0.75 * 2 / 1.5))
        self.assertAlmostEqual(result['a'], expected)
        self.assertNotIn('b', result)
        self.assertEqual(index.search("python python"), index.search("python"))

    def test_zero_overlap_empty_and_stable_ties(self):
        index = BM25Index([("b", "C++"), ("a", "C++"), ("c", "C#")])
        self.assertEqual([key for key, _ in index.search("C++")], ["a", "b"])
        self.assertEqual(index.search("C"), [])
        self.assertEqual(BM25Index([]).search("Python"), [])
        self.assertEqual(index.search(""), [])

    def test_invalid_parameters_and_duplicate_ids(self):
        for kwargs in ({"k1": 0}, {"b": 2}, {"k1": float('nan')}):
            with self.assertRaises(ValueError):
                BM25Index([], **kwargs)
        with self.assertRaises(ValueError):
            BM25Index([("a", "Python"), ("a", "Java")])
        with self.assertRaises(ValidationError):
            configuration(bm25_b=2)

    def test_rrf_arithmetic_unique_ids_and_rank_trace(self):
        result = reciprocal_rank_fusion({"dense": ["a", "b"], "bm25": ["b", "c"]})
        self.assertEqual(result[0][0], "b")
        self.assertAlmostEqual(result[0][1], 1 / 62 + 1 / 61)
        self.assertEqual(result[0][2], {"dense": 2, "bm25": 1})
        self.assertEqual(len(result), 3)
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion({"dense": ["a", "a"]})
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion({}, constant=0)


class HybridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.jobs, cls.provenance = source_corpus()
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        root = Path(cls.temp.name)
        cls.settings = configuration(artifacts_dir=root / "artifacts", qdrant_path=root / "qdrant")
        cls.encoder = checked()
        build_index(cls.settings, cls.jobs, cls.provenance, cls.encoder, INFO)
        cls.prepared = build_profile(ParsedResume(pages=[{"page": 1, "text":
            (PROJECT_ROOT / "examples/resumes/ai.txt").read_text()}]), skill_vocabulary(cls.jobs),
            Preferences.model_validate_json((PROJECT_ROOT / "examples/preferences.json").read_text()))

    def search(self, mode, prepared=None, **kwargs):
        return retrieve_candidates(self.settings, self.jobs, self.provenance, prepared or self.prepared,
                                   mode=mode, embeddings=self.encoder, embedding_info=INFO, **kwargs)

    def test_modes_share_corpus_query_preferences_and_eligible_set(self):
        reports = [self.search(mode) for mode in ("dense", "bm25", "hybrid")]
        for report in reports:
            self.assertEqual(report['corpus'], self.provenance)
            self.assertEqual(report['query_sha256'], reports[0]['query_sha256'])
            self.assertEqual(report['eligible_count'], reports[0]['eligible_count'])
            self.assertEqual(report['applied_preferences'], reports[0]['applied_preferences'])
            self.assertLessEqual(len(report['candidates']), 30)
            for item in report['candidates']:
                self.assertEqual(item['job']['location_city'], 'Bengaluru')
                self.assertIn(item['job']['work_mode'], ('hybrid', 'unknown'))

    def test_hybrid_components_reconstruct_every_fused_score(self):
        report = self.search("hybrid")
        seen = set()
        for item in report['candidates']:
            self.assertNotIn(item['job_id'], seen)
            seen.add(item['job_id'])
            self.assertAlmostEqual(item['relevance']['value'],
                                   sum(source['contribution'] for source in item['retrieval_sources'].values()))
            for branch, source in item['retrieval_sources'].items():
                self.assertEqual(report['branch_rankings'][branch][source['rank'] - 1], item['job_id'])
            self.assertEqual(item['dense_relevance']['score_type'], 'cosine')
        self.assertEqual(report, self.search("hybrid") | {'timing_seconds': report['timing_seconds']})

    def test_bm25_requires_no_vector_model_or_index(self):
        with patch('app.retrieval.retrieve', side_effect=AssertionError('Unexpected dense search')):
            result = retrieve_candidates(configuration(), self.jobs, self.provenance, self.prepared, mode="bm25")
        self.assertTrue(result['candidates'])
        self.assertTrue(all(item['relevance']['score_type'] == 'bm25' for item in result['candidates']))

    def test_no_query_and_no_eligible_jobs(self):
        for mode in ("dense", "bm25", "hybrid"):
            with patch('app.retrieval.retrieve', side_effect=AssertionError('Unexpected dense search')):
                self.assertEqual(self.search(mode, empty_profile(query=None))['status'], 'no_query')
        absent = self.prepared.model_copy(deep=True)
        absent.preferences.location_city = "NoSuchCity"
        for mode in ("dense", "bm25", "hybrid"):
            self.assertEqual(self.search(mode, absent)['status'], 'no_eligible_jobs')

    def test_zero_keyword_overlap_returns_no_fabricated_candidates(self):
        self.assertEqual(self.search('bm25', empty_profile(query='zzzznonexistentskill'))['candidates'], [])

    def test_identity_changes_for_parameters_subset_and_preferences(self):
        original, _ = lexical_identity(self.settings, self.jobs, self.provenance, self.prepared.preferences)
        variants = [lexical_identity(self.settings.model_copy(update={'bm25_k1': 2}), self.jobs,
                                    self.provenance, self.prepared.preferences)[0],
                    lexical_identity(self.settings, self.jobs[:100], self.provenance, self.prepared.preferences)[0],
                    lexical_identity(self.settings, self.jobs, self.provenance, Preferences())[0]]
        self.assertTrue(all(item['index_id'] != original['index_id'] for item in variants))

    def test_hybrid_keeps_coverage_and_eligibility_and_cosine_gate(self):
        dense, _ = match_candidates(self.settings, self.jobs, self.provenance, self.prepared,
                                    self.search('dense'), use_llm=False)
        hybrid, report = match_candidates(self.settings, self.jobs, self.provenance, self.prepared,
                                         self.search('hybrid'), use_llm=False)
        original = {item.job.job_id: item for item in dense.recommendations}
        self.assertTrue(hybrid.recommendations)
        for match in hybrid.recommendations:
            if match.job.job_id in original:
                self.assertEqual(match.required_coverage, original[match.job.job_id].required_coverage)
                self.assertEqual(match.eligibility, original[match.job.job_id].eligibility)
        low = copy.deepcopy(self.search('hybrid'))
        for item in low['candidates']:
            item['dense_relevance']['value'] = 0.1
        response, _ = match_candidates(self.settings, self.jobs, self.provenance, self.prepared, low, use_llm=False)
        self.assertFalse(response.recommendations)
        self.assertEqual(report['generation_calls'], 0)

    def test_malformed_mode_and_missing_hybrid_cosine_are_rejected(self):
        report = self.search('hybrid')
        report['candidates'][0]['dense_relevance'] = None
        with self.assertRaises(ValueError):
            match_candidates(self.settings, self.jobs, self.provenance, self.prepared, report, use_llm=False)
        report = self.search('bm25')
        report['retrieval_mode'] = 'dense'
        with self.assertRaises(ValueError):
            match_candidates(self.settings, self.jobs, self.provenance, self.prepared, report, use_llm=False)

    def test_bm25_retains_known_requirement_failure(self):
        junior = build_profile(ParsedResume(pages=[{'page': 1, 'text':
            'Summary\n1 year of professional experience\nSkills\nPython, LangChain, RAG, FastAPI, Docker'}]),
            skill_vocabulary(self.jobs), self.prepared.preferences)
        result, report = match_candidates(self.settings, self.jobs, self.provenance, junior,
                                         self.search('bm25', junior), use_llm=False)
        ai = next(item for item in report['assessments'] if item['job_id'] == 'syn_job_001')
        self.assertIn('Known requirement failure', ai['rejection_reasons'])
        self.assertNotIn('syn_job_001', [item.job.job_id for item in result.recommendations])

    def test_bm25_and_hybrid_api_upload_and_detail(self):
        for mode in ('bm25', 'hybrid'):
            settings = self.settings.model_copy(update={'retrieval_mode': mode})
            service = MatchingService(settings)
            service.jobs, service.provenance = self.jobs, self.provenance
            if mode == 'hybrid':
                service.embeddings, service.embedding_info = self.encoder, INFO
            with TestClient(create_app(settings, service)) as client:
                response = client.post('/match', files={'resume': ('demo.pdf', pdf(
                    (PROJECT_ROOT / 'examples/resumes/ai.txt').read_text()), 'application/pdf')},
                    data={'use_llm': 'false', 'preferences': self.prepared.preferences.model_dump_json()})
                self.assertEqual(response.status_code, 200, response.text)
                result = response.json()
                self.assertEqual(result['result']['applied_filters']['retrieval_mode'], mode)
                for job_id in ('syn_job_001', 'syn_job_234'):
                    detail = client.get(f"/runs/{result['run_id']}/jobs/{job_id}")
                    self.assertEqual(detail.status_code, 200, detail.text)
                self.assertEqual(client.delete(f"/runs/{result['run_id']}").status_code, 204)


if __name__ == '__main__':
    unittest.main()
