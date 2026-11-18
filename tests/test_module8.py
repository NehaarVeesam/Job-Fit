"""Judgment-aware evaluation math, fixture isolation, and real pipeline audits."""

import copy
import json
import math
import io
from contextlib import redirect_stdout
import shutil
import tempfile
import unittest
from pathlib import Path

from app.config import PROJECT_ROOT, Settings
from app.evaluation.audits import edge_audit, evidence_audit, expectation_audit
from app.evaluation.fixtures import load_suite
from app.evaluation.metrics import aggregate, percentile, ranking_metrics
from app.evaluation.reporting import render_failures, render_summary
from app.evaluation.runner import panel_corpus
from app.evaluation.runner import run
from app.evaluation.summarize import rebuild
from app.dataset_loader import prepare_corpus, write_corpus
from app.matching import match_candidates
from app.profile import build_profile, skill_vocabulary
from app.schemas import ParsedResume, Preferences
from test_module3 import source_corpus
from test_module4 import profile, retrieval


def configuration():
    return Settings(_env_file=None, llm_provider='ollama')


def row(expected_no_match, predicted_no_match, metrics, elapsed=1):
    return {'metrics': metrics, 'expected_no_match': expected_no_match, 'predicted_no_match': predicted_no_match,
            'timing_seconds': {'end_to_end': elapsed, 'preparation': .01, 'retrieval': .1, 'reranking': 0, 'assessment': .02}}


class MetricTests(unittest.TestCase):
    def test_missing_positions_are_negative_and_partial_is_not_strict_relevance(self):
        result = ranking_metrics(['a','b'], ['a','b'], {'a':2,'b':1}, corpus_ids=['a','b'])
        self.assertEqual(result['precision_at_5'], .2)
        self.assertEqual(result['accepted_precision'], .5)
        self.assertEqual(result['accepted_partial'], 1)
        self.assertEqual(result['recall_at_5'], 1)

    def test_unjudged_results_have_bounds_not_implicit_negative_labels(self):
        result = ranking_metrics(['a','unknown'], ['unknown'], {'a':2}, corpus_ids=['a','unknown'])
        self.assertIsNone(result['precision_at_5'])
        self.assertEqual(result['precision_at_5_lower'], .2)
        self.assertEqual(result['precision_at_5_upper'], .4)
        self.assertIsNone(result['accepted_precision'])
        self.assertEqual(result['accepted_precision_lower'], 0)
        self.assertEqual(result['accepted_precision_upper'], 1)
        self.assertIsNone(result['recall_at_5'])
        self.assertIsNone(result['ndcg_at_5'])

    def test_empty_returns_do_not_get_perfect_precision_or_ndcg(self):
        result = ranking_metrics([], [], {'a':0}, corpus_ids=['a'])
        self.assertEqual(result['precision_at_5'], 0)
        self.assertIsNone(result['accepted_precision'])
        self.assertIsNone(result['ndcg_at_5'])
        self.assertIsNone(result['recall_at_5'])

    def test_graded_ndcg_matches_hand_calculation(self):
        result = ranking_metrics(['b','a'], [], {'a':2,'b':1}, corpus_ids=['a','b'])
        self.assertAlmostEqual(result['ndcg_at_5'], (1 + 3 / math.log2(3)) / (3 + 1 / math.log2(3)))

    def test_always_returning_none_is_penalized_by_no_match_confusion(self):
        empty = ranking_metrics([], [], {'a':0}, corpus_ids=['a'])
        summary = aggregate([row(True,True,empty),row(False,True,empty)], exhaustive_judgments=True)
        self.assertIsNone(summary['micro_accepted_precision'])
        self.assertEqual(summary['no_match']['precision'], .5)
        self.assertEqual(summary['no_match']['accuracy'], .5)

    def test_full_corpus_no_match_ground_truth_is_not_inferred_from_panel(self):
        empty = ranking_metrics([], [], {'a':0}, corpus_ids=['a','b'])
        self.assertIsNone(aggregate([row(True,True,empty)], exhaustive_judgments=False)['no_match'])

    def test_accepted_precision_is_micro_aggregated_and_latency_interpolated(self):
        one = ranking_metrics(['a'], ['a'], {'a':2}, corpus_ids=['a'])
        two = ranking_metrics(['a','b','c'], ['a','b','c'], {'a':2,'b':0,'c':1}, corpus_ids=['a','b','c'])
        result = aggregate([row(False,False,one,1),row(False,False,two,3)], exhaustive_judgments=True)
        self.assertEqual(result['micro_accepted_precision'], .5)
        self.assertEqual(result['latency_seconds']['median'], 2)
        self.assertAlmostEqual(result['latency_seconds']['p95'], 2.9)
        self.assertIsNone(result['no_match']['precision'])
        self.assertIsNone(percentile([], .95))

    def test_duplicate_ids_foreign_ids_and_invalid_grades_are_rejected(self):
        for args in ((['a','a'],[],{'a':2}), (['a'],['b'],{'a':2}), (['a'],[],{'a':True})):
            with self.assertRaises(ValueError): ranking_metrics(*args, corpus_ids=['a'])


class FixtureAndAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.jobs, cls.provenance = source_corpus()
        cls.suite, cls.cases, cls.labels, cls.hash = load_suite(PROJECT_ROOT/'evaluation', PROJECT_ROOT/'dataset/jobs_synthetic_250.jsonl')
        vocabulary = skill_vocabulary(cls.jobs)
        cls.prepared = {case['case_id']:build_profile(ParsedResume(pages=[{'page':1,'text':case['text']}]),vocabulary,
                                                    Preferences.model_validate(case['preferences'])) for case in cls.cases}

    def test_panel_is_exhaustively_judged_and_splits_have_no_exact_duplicates(self):
        self.assertEqual(len(self.cases), 12)
        self.assertEqual(len(self.labels), 12)
        self.assertEqual(sum(map(len,self.labels.values())), 204)
        self.assertEqual(len({case['resume_sha256'] for case in self.cases}), 12)
        self.assertEqual({case['split'] for case in self.cases}, {'dev','heldout'})

    def copied_suite(self):
        temp=self.enterContext(tempfile.TemporaryDirectory())
        target=Path(temp)/'evaluation'
        shutil.copytree(PROJECT_ROOT/'evaluation', target)
        return target

    def test_missing_and_duplicate_judgments_are_rejected(self):
        target=self.copied_suite(); path=target/'judgments.jsonl';lines=path.read_text().splitlines()
        path.write_text('\n'.join(lines[:-1])+'\n')
        with self.assertRaises(ValueError):load_suite(target, PROJECT_ROOT/'dataset/jobs_synthetic_250.jsonl')
        path.write_text('\n'.join(lines+[lines[0]])+'\n')
        with self.assertRaises(ValueError):load_suite(target, PROJECT_ROOT/'dataset/jobs_synthetic_250.jsonl')

    def test_resume_tampering_and_path_escape_are_rejected(self):
        target=self.copied_suite(); path=target/'resumes/dev_ai.txt';path.write_text('changed')
        with self.assertRaises(ValueError):load_suite(target, PROJECT_ROOT/'dataset/jobs_synthetic_250.jsonl')

    def test_pipeline_fingerprint_changes_are_rejected(self):
        target=self.copied_suite();path=target/'suite.json';suite=json.loads(path.read_text())
        suite['implementation_sha256']['app/scoring.py']='changed'
        path.write_text(json.dumps(suite))
        with self.assertRaises(ValueError):load_suite(target, PROJECT_ROOT/'dataset/jobs_synthetic_250.jsonl')
        suite=json.loads((target/'suite.json').read_text());suite['cases'][0]['resume_file']='../../outside.txt'
        (target/'suite.json').write_text(json.dumps(suite))
        with self.assertRaises(ValueError):load_suite(target, PROJECT_ROOT/'dataset/jobs_synthetic_250.jsonl')

    def test_leakage_and_contradictory_no_match_labels_are_rejected(self):
        target=self.copied_suite(); path=target/'suite.json';suite=json.loads(path.read_text())
        suite['cases'][4]['resume_file']=suite['cases'][0]['resume_file']
        suite['cases'][4]['resume_sha256']=suite['cases'][0]['resume_sha256'];path.write_text(json.dumps(suite))
        with self.assertRaises(ValueError):load_suite(target, PROJECT_ROOT/'dataset/jobs_synthetic_250.jsonl')
        suite=copy.deepcopy(self.suite);suite['cases'][0]['expected_no_match']=True;path.write_text(json.dumps(suite))
        with self.assertRaises(ValueError):load_suite(target, PROJECT_ROOT/'dataset/jobs_synthetic_250.jsonl')

    def test_panel_identity_preserves_source_and_does_not_reuse_full_index(self):
        snapshot=copy.deepcopy(self.provenance)
        panel, provenance=panel_corpus(self.jobs,self.provenance,self.suite['panel_job_ids'])
        self.assertEqual(len(panel),17)
        self.assertEqual(self.provenance,snapshot)
        self.assertNotEqual(provenance['jobs_sha256'],snapshot['jobs_sha256'])
        self.assertEqual(provenance['source'],snapshot['source'])
        with self.assertRaises(ValueError):panel_corpus(self.jobs,self.provenance,['foreign'])

    def test_frozen_independent_expectations_and_edge_audits_pass(self):
        checks=expectation_audit(self.suite,self.prepared,self.jobs)+edge_audit(configuration(),self.jobs,self.prepared)
        self.assertEqual(len(checks),37)
        self.assertTrue(all(check['passed'] for check in checks),checks)

    def test_explanation_support_audit_detects_invented_claim_and_bad_quote(self):
        prepared=profile(); source=retrieval(prepared)
        response,_=match_candidates(configuration(),self.jobs,self.provenance,prepared,source,use_llm=False)
        self.assertTrue(all(check['passed'] for check in evidence_audit(response,prepared)))
        response.recommendations[0].reasons[0].explanation='You are an expert and guaranteed a job.'
        self.assertTrue(any(not check['passed'] for check in evidence_audit(response,prepared)))
        response.recommendations[0].skills[0].resume_evidence[0].quote='invented source quote'
        self.assertTrue(any(not check['passed'] for check in evidence_audit(response,prepared)))

    def test_reporting_keeps_undefined_metrics_visible(self):
        metrics=aggregate([row(True,True,ranking_metrics([],[],{'a':0},corpus_ids=['a']))], exhaustive_judgments=True)
        summary={'scope':'panel','suite_sha256':self.hash,'dataset_sha256':self.suite['dataset_sha256'],
                 'metrics':{'dev':{'dense':metrics}},'checks':{'passed':0,'total':0},'reranker_status_counts':{}}
        self.assertIn('N/A',render_summary(summary))
        self.assertIn('not automatically application bugs',render_failures([]))

    def test_real_keyword_runner_panel_and_full_scope_without_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            settings=Settings(_env_file=None,llm_provider='ollama',artifacts_dir=Path(temporary)/'artifacts')
            source=PROJECT_ROOT/'dataset/jobs_synthetic_250.jsonl'
            write_corpus(prepare_corpus(source),settings.artifacts_dir/'dataset/all',source)
            with redirect_stdout(io.StringIO()):
                output,panel=run(split='dev',scope='panel',variants=('bm25',),base_settings=settings)
                _,full=run(split='heldout',scope='full',variants=('bm25',),base_settings=settings)
            self.assertTrue((output/'summary.json').exists())
            self.assertEqual(panel['corpus_size'],17)
            self.assertEqual(panel['checks']['passed'],panel['checks']['total'])
            self.assertEqual(panel['metrics']['dev']['bm25']['unjudged_top_5_count'],0)
            self.assertEqual(full['corpus_size'],250)
            self.assertGreater(full['metrics']['heldout']['bm25']['unjudged_top_5_count'],0)
            self.assertIsNone(full['metrics']['heldout']['bm25']['no_match'])
            self.assertEqual(panel['qwen_probe']['calls'],0)
            exported=Path(temporary)/'public'
            rebuilt=rebuild(output,export=exported)
            self.assertEqual(rebuilt['metrics'],panel['metrics'])
            self.assertTrue((exported/'results_summary.json').exists())
            saved=json.loads((output/'case_metrics.json').read_text())
            saved['cases'][0]['accepted_ids']=['changed_prediction']
            (output/'case_metrics.json').write_text(json.dumps(saved))
            with self.assertRaises(ValueError):rebuild(output)


if __name__=='__main__':unittest.main()
