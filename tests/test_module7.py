"""Cross-encoder budgets, ordering, evidence invariants, fallback, and API wiring."""

import copy
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api_service import MatchingService
from app.config import PROJECT_ROOT, Settings
from app.hybrid_retrieval import retrieve_candidates
from app.main import create_app
from app.matching import match_candidates
from app.reranking import CheckedCrossEncoder, PairTooLong, rerank_candidates
from app.schemas import RerankerScore
from test_module3 import source_corpus
from test_module4 import profile, retrieval
from test_module5 import pdf


def settings(**kwargs):
    return Settings(_env_file=None, llm_provider='ollama', **kwargs)


class PairTokenizer:
    def __init__(self, fixed=None):
        self.fixed = fixed

    def __call__(self, queries, texts, **kwargs):
        assert kwargs['truncation'] is False
        return {'input_ids': [[0] * (self.fixed or (len(query.split()) + len(text.split()) + 3))
                              for query, text in zip(queries, texts)]}


class Predictor:
    def __init__(self, constant=None):
        self.constant = constant
        self.calls = 0

    def predict(self, pairs, **kwargs):
        self.calls += 1
        return [self.constant if self.constant is not None else float(len(text)) for _, text in pairs]


def checked(*, constant=None, token_count=None, batch=8):
    return CheckedCrossEncoder(Predictor(constant), PairTokenizer(token_count), batch_size=batch)


class CrossEncoderTests(unittest.TestCase):
    def test_bge_defaults_and_supported_budget(self):
        configuration = settings()
        self.assertEqual(configuration.rerank_model, 'BAAI/bge-reranker-v2-m3')
        self.assertEqual(configuration.rerank_revision, '953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e')
        self.assertEqual(configuration.rerank_batch_size, 2)
        self.assertEqual(settings(rerank_max_tokens=8192).rerank_max_tokens, 8192)
        with self.assertRaises(ValidationError):
            settings(rerank_model='cross-encoder/ms-marco-MiniLM-L6-v2')

    def test_token_budget_includes_both_texts_and_special_tokens(self):
        scorer = checked(token_count=512)
        scores, audit = scorer.score([('query', 'document')])
        self.assertEqual(len(scores), 1)
        self.assertEqual(audit['pair_token_counts'], [512])
        scorer = checked(token_count=513)
        with self.assertRaises(PairTooLong) as context:
            scorer.score([('query', 'document')])
        self.assertEqual(context.exception.audit['overflow_count'], 1)
        self.assertEqual(scorer.delegate.calls, 0)

    def test_batches_and_finite_scores(self):
        scorer = checked(batch=2)
        scores, _ = scorer.score([('q', 'x')] * 5)
        self.assertEqual(len(scores), 5)
        self.assertEqual(scorer.delegate.calls, 3)
        with self.assertRaises(ValueError):
            checked(constant=float('nan')).score([('q', 'd')])

    def test_wrong_score_count_is_rejected(self):
        scorer = checked()
        with patch.object(scorer.delegate, 'predict', return_value=[]):
            with self.assertRaises(ValueError):
                scorer.score([('q', 'd')])

    def test_deadline_stops_before_inference(self):
        scorer = checked()
        with self.assertRaises(TimeoutError):
            scorer.score([('q', 'd')], deadline=time.perf_counter() - 1)
        self.assertEqual(scorer.delegate.calls, 0)

    def test_reranker_score_type_is_explicit_and_not_probability_bounded(self):
        self.assertEqual(RerankerScore(value=-11).score_type, 'cross_encoder')
        with self.assertRaises(ValidationError):
            RerankerScore(value=0.5, score_type='cosine')
        with self.assertRaises(ValidationError):
            settings(rerank_max_tokens=8193)
        with self.assertRaises(ValidationError):
            settings(rerank_cache_dir=PROJECT_ROOT / 'dataset/model_cache')


class RerankingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.jobs, cls.provenance = source_corpus()
        cls.prepared = profile()
        cls.configuration = settings(rerank_enabled=True, retrieval_mode='bm25')
        cls.base = retrieve_candidates(cls.configuration, cls.jobs, cls.provenance, cls.prepared)

    def test_disabled_and_empty_do_not_load_model(self):
        with patch('app.reranking.load_reranker', side_effect=AssertionError('Unexpected model load')):
            result = rerank_candidates(settings(), self.prepared, self.base)
            self.assertEqual(result['reranking']['status'], 'disabled')
            self.assertEqual(result['candidates'], self.base['candidates'])
            empty = copy.deepcopy(self.base)
            empty['candidates'] = []
            self.assertEqual(rerank_candidates(self.configuration, self.prepared, empty)['reranking']['status'], 'skipped_empty')

    def test_reorders_without_adding_dropping_or_rewriting_original_evidence(self):
        snapshot = copy.deepcopy(self.base)
        result = rerank_candidates(self.configuration, self.prepared, self.base, reranker=checked())
        self.assertEqual(self.base, snapshot)
        self.assertNotEqual([item['job_id'] for item in result['candidates']], [item['job_id'] for item in self.base['candidates']])
        original = {item['job_id']: item for item in self.base['candidates']}
        self.assertEqual(set(original), {item['job_id'] for item in result['candidates']})
        for item in result['candidates']:
            for key in ('job', 'relevance', 'dense_relevance', 'retrieval_sources', 'bm25_score'):
                self.assertEqual(item[key], original[item['job_id']][key])

    def test_ties_keep_original_order(self):
        result = rerank_candidates(self.configuration, self.prepared, self.base, reranker=checked(constant=-3))
        self.assertEqual([item['job_id'] for item in result['candidates']], [item['job_id'] for item in self.base['candidates']])

    def test_budget_overflow_preserves_whole_original_list(self):
        result = rerank_candidates(self.configuration, self.prepared, self.base, reranker=checked(token_count=513))
        self.assertEqual(result['reranking']['status'], 'fallback')
        self.assertEqual(result['candidates'], self.base['candidates'])
        self.assertEqual(result['reranking']['token_audit']['overflow_count'], len(self.base['candidates']))

    def test_missing_model_and_invalid_scores_are_labeled_without_error_body(self):
        with patch('app.reranking.load_reranker', side_effect=RuntimeError('private-provider-body')):
            result = rerank_candidates(self.configuration, self.prepared, self.base)
        self.assertEqual(result['reranking']['error_type'], 'RuntimeError')
        self.assertNotIn('private-provider-body', str(result))
        result = rerank_candidates(self.configuration, self.prepared, self.base, reranker=checked(constant=float('inf')))
        self.assertEqual(result['reranking']['status'], 'fallback')

    def test_deadline_is_not_hidden_by_fallback(self):
        with self.assertRaises(TimeoutError):
            rerank_candidates(self.configuration, self.prepared, self.base, reranker=checked(), deadline=0)

    def test_wrong_query_duplicate_ids_and_double_reranking_are_rejected(self):
        wrong = copy.deepcopy(self.base)
        wrong['query_sha256'] = 'wrong'
        with self.assertRaises(ValueError):
            rerank_candidates(self.configuration, self.prepared, wrong, reranker=checked())
        wrong = copy.deepcopy(self.base)
        wrong['candidates'].append(wrong['candidates'][0])
        with self.assertRaises(ValueError):
            rerank_candidates(self.configuration, self.prepared, wrong, reranker=checked())
        result = rerank_candidates(self.configuration, self.prepared, self.base, reranker=checked())
        with self.assertRaises(ValueError):
            rerank_candidates(self.configuration, self.prepared, result, reranker=checked())

    def test_all_coverage_and_requirement_outcomes_stay_the_same(self):
        before, before_report = match_candidates(self.configuration, self.jobs, self.provenance,
                                                 self.prepared, self.base, use_llm=False)
        report = rerank_candidates(self.configuration, self.prepared, self.base, reranker=checked())
        response, audit = match_candidates(self.configuration, self.jobs, self.provenance,
                                           self.prepared, report, use_llm=False)
        original = {item['job_id']: item for item in before_report['assessments']}
        for item in audit['assessments']:
            for key in ('required_coverage', 'preferred_coverage', 'eligibility', 'experience', 'education', 'constraints', 'rejection_reasons'):
                self.assertEqual(item[key], original[item['job_id']][key])
        self.assertTrue(response.recommendations)
        self.assertTrue(all(item.reranker_relevance is not None for item in response.recommendations))
        self.assertEqual(audit['generation_calls'], 0)

    def test_high_reranker_score_cannot_override_weak_cosine_or_known_failure(self):
        weak = retrieval(self.prepared, score=0.1)
        ranked = rerank_candidates(self.configuration, self.prepared, weak, reranker=checked(constant=100))
        response, _ = match_candidates(self.configuration, self.jobs, self.provenance, self.prepared, ranked, use_llm=False)
        self.assertFalse(response.recommendations)
        junior = profile('Summary\n1 year of professional experience\nSkills\nPython, LangChain, RAG, FastAPI')
        ranked = rerank_candidates(self.configuration, junior, retrieval(junior), reranker=checked(constant=100))
        response, _ = match_candidates(self.configuration, self.jobs, self.provenance, junior, ranked, use_llm=False)
        self.assertFalse(response.recommendations)

    def test_fallback_is_visible_in_match_response(self):
        result = rerank_candidates(self.configuration, self.prepared, self.base, reranker=checked(token_count=513))
        response, _ = match_candidates(self.configuration, self.jobs, self.provenance, self.prepared, result, use_llm=False)
        self.assertEqual(response.applied_filters['reranking_status'], 'fallback')
        self.assertTrue(any('original retrieval order' in warning for warning in response.warnings))

    def test_api_enabled_reranking_reuses_model_and_detail_score(self):
        service = MatchingService(self.configuration)
        service.jobs, service.provenance = self.jobs, self.provenance
        service.reranker = checked()
        with patch('app.reranking.load_reranker', side_effect=AssertionError('Unexpected reload')):
            with TestClient(create_app(self.configuration, service)) as client:
                for _ in range(2):
                    result = client.post('/match', files={'resume': ('demo.pdf', pdf(
                        (PROJECT_ROOT / 'examples/resumes/ai.txt').read_text()), 'application/pdf')},
                        data={'use_llm': 'false'})
                    self.assertEqual(result.status_code, 200, result.text)
                    body = result.json()
                    self.assertEqual(body['result']['applied_filters']['reranking_status'], 'applied')
                    item = body['result']['ranked_candidates'][0]
                    detail = client.get(f"/runs/{body['run_id']}/jobs/{item['job_id']}")
                    self.assertEqual(detail.status_code, 200)
                    self.assertEqual(detail.json()['reranker_relevance'], item['reranker_relevance'])
                    other = client.get(f"/runs/{body['run_id']}/jobs/syn_job_234")
                    self.assertEqual(other.status_code, 200)
                    self.assertEqual(client.delete(f"/runs/{body['run_id']}").status_code, 204)


if __name__ == '__main__':
    unittest.main()
