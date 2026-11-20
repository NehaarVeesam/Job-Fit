"""python -m app.evaluation.runner: frozen synthetic suitability benchmark."""

import argparse
import copy
import hashlib
import json
import platform
import sys
import time
from collections import Counter
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from app.config import PROJECT_ROOT, Settings, get_settings
from app.retrieval.embeddings import load_embeddings
from app.evaluation.audits import edge_audit, evidence_audit, expectation_audit
from app.evaluation.fixtures import load_suite
from app.evaluation.metrics import aggregate, ranking_metrics
from app.evaluation.reporting import render_failures, render_summary
from app.retrieval.hybrid import retrieve_candidates
from app.matching.matcher import match_candidates
from app.resumes.profile import build_profile, load_prepared_jobs, skill_vocabulary
from app.retrieval.reranking import load_reranker, rerank_candidates
from app.schemas import ParsedResume, Preferences
from app.retrieval.vector_index import build_index, save_json

VARIANTS = ('dense', 'bm25', 'hybrid', 'hybrid_reranked')
FROZEN_KEYS = {'candidate_count', 'bm25_candidate_count', 'bm25_k1', 'bm25_b', 'rrf_constant',
               'min_cosine_similarity', 'min_required_coverage', 'min_evidenced_required',
               'max_recommendations', 'embedding_model', 'embedding_revision', 'rerank_model',
               'rerank_revision', 'rerank_batch_size', 'rerank_max_tokens', 'rerank_cpu_threads',
               'embedding_cpu_threads', 'embedding_batch_size', 'embedding_device', 'rerank_device', 'recency_filter_enabled'}


def panel_corpus(jobs, provenance, panel_ids):
    by_id = {job.job_id: job for job in jobs}
    if not set(panel_ids) <= set(by_id):
        raise ValueError('Panel contains a foreign job')
    selected = [by_id[key] for key in sorted(panel_ids)]
    derived = copy.deepcopy(provenance)
    derived['configuration']['evaluation_panel_ids'] = sorted(panel_ids)
    derived['scope'] = 'controlled_evaluation_panel'
    derived['jobs_sha256'] = hashlib.sha256(json.dumps([job.model_dump(mode='json') for job in selected], sort_keys=True).encode()).hexdigest()
    derived['job_count'] = len(selected)
    return selected, derived


def qwen_probe(settings, prepared, response):
    from app.matching.generation import GroundedExplainer
    from app.matching.scoring import assess_job, make_match
    if not response.recommendations:
        return {'status': 'skipped_no_accepted_jobs', 'calls': 0}
    original = response.recommendations[0]
    assessment = assess_job(original.job, prepared, original.relevance)
    explainer = GroundedExplainer(settings)
    plan = explainer.explain(assessment)
    rendered = make_match(assessment, plan.strengths if plan else None,
                          'qwen_grounded_plan' if plan else 'fallback')
    checks = evidence_audit(type(response)(**(response.model_dump() | {'recommendations': [rendered]})), prepared)
    return {'status': 'qwen_grounded_plan' if plan else 'fallback', 'calls': explainer.calls,
            'model': settings.llm_model, 'error_types': explainer.failures,
            'job_id': original.job.job_id, 'closed_template_support_checks': len(checks),
            'checks_passed': sum(item['passed'] for item in checks),
            'scope': 'One accepted synthetic pair; bounded structured-plan probe, not general LLM faithfulness evaluation.'}


def _run(*, directory=PROJECT_ROOT / 'evaluation', split='all', scope='panel', variants=VARIANTS,
        with_qwen=False, base_settings=None):
    if split not in ('all', 'dev', 'heldout') or scope not in ('panel', 'full') or not variants or not set(variants) <= set(VARIANTS) or len(set(variants)) != len(variants):
        raise ValueError('Invalid evaluation selection')
    original_settings = base_settings or get_settings()
    suite, cases, judgments, suite_hash = load_suite(directory, original_settings.dataset_path)
    if set(suite['frozen_settings']) != FROZEN_KEYS:
        raise ValueError('Frozen evaluation settings are incomplete or contain private/unapproved keys')
    settings = Settings(_env_file=None, **(original_settings.model_dump() | suite['frozen_settings']))
    if settings.qdrant_mode != 'local':
        raise ValueError('This benchmark verifies local Qdrant only')
    full_jobs, full_provenance = load_prepared_jobs(settings.artifacts_dir / 'dataset/all')
    if full_provenance['source']['sha256'] != suite['dataset_sha256'] or len(full_jobs) != 250:
        raise ValueError('Evaluation requires the complete, frozen 250-job prepared source')
    jobs, provenance = (panel_corpus(full_jobs, full_provenance, suite['panel_job_ids']) if scope == 'panel'
                        else (full_jobs, full_provenance))
    if scope == 'panel':
        settings = settings.model_copy(update={'qdrant_path': settings.artifacts_dir / 'evaluation/qdrant',
                                               'qdrant_collection': settings.qdrant_collection + '_evaluation'})
    selected = [case for case in cases if split == 'all' or case['split'] == split]
    vocabulary = skill_vocabulary(full_jobs)
    prepared_by_case, preparation_seconds = {}, {}
    for case in cases:
        start = time.perf_counter()
        prepared_by_case[case['case_id']] = build_profile(ParsedResume(pages=[{'page': 1, 'text': case['text']}]),
                                                         vocabulary, Preferences.model_validate(case['preferences']))
        preparation_seconds[case['case_id']] = time.perf_counter() - start
    started = time.perf_counter()
    embeddings, embedding_info = (load_embeddings(settings) if set(variants) != {'bm25'} else (None, None))
    embedding_load = time.perf_counter() - started
    index_report = None
    if embeddings is not None:
        print(f'Checking/building isolated {len(jobs)}-job {scope} index...', flush=True)
        index_report = build_index(settings, jobs, provenance, embeddings, embedding_info)
    started = time.perf_counter()
    reranker = load_reranker(settings) if 'hybrid_reranked' in variants else None
    reranker_load = time.perf_counter() - started
    checks = expectation_audit(suite, prepared_by_case, full_jobs)
    checks += edge_audit(settings, full_jobs, prepared_by_case)
    rows, failures, statuses = [], [], Counter()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = settings.artifacts_dir / 'evaluation' / f'{scope}_{split}_{suite_hash[:12]}_{stamp}'
    probe_input = None
    for case in selected:
        case_id = case['case_id']
        prepared = prepared_by_case[case_id]
        bases, base_timings = {}, {}
        for mode in dict.fromkeys('hybrid' if variant == 'hybrid_reranked' else variant for variant in variants):
            started = time.perf_counter()
            bases[mode] = retrieve_candidates(settings, jobs, provenance, prepared, mode=mode,
                                              embeddings=embeddings, embedding_info=embedding_info)
            base_timings[mode] = time.perf_counter() - started
        for variant in variants:
            mode = 'hybrid' if variant == 'hybrid_reranked' else variant
            retrieval = bases[mode]
            started = time.perf_counter()
            if variant == 'hybrid_reranked':
                retrieval = rerank_candidates(settings.model_copy(update={'rerank_enabled': True}), prepared,
                                               retrieval, reranker=reranker)
                statuses[retrieval['reranking']['status']] += 1
                if retrieval['candidates'] and retrieval['reranking']['status'] != 'applied':
                    raise ValueError('Real reranking fell back; evaluation cannot silently use the baseline')
            reranking_seconds = time.perf_counter() - started
            started = time.perf_counter()
            response, audit = match_candidates(settings, jobs, provenance, prepared, retrieval, use_llm=False)
            assessment_seconds = time.perf_counter() - started
            ranked = [item['job_id'] for item in retrieval['candidates']]
            accepted = [item.job.job_id for item in response.recommendations]
            metrics = ranking_metrics(ranked, accepted, judgments[case_id], corpus_ids=[job.job_id for job in jobs])
            timings = {'preparation': preparation_seconds[case_id], 'retrieval': base_timings[mode],
                       'reranking': reranking_seconds, 'assessment': assessment_seconds}
            timings['end_to_end'] = sum(timings.values())
            row = {'case_id': case_id, 'split': case['split'], 'variant': variant,
                   'expected_no_match': case['expected_no_match'], 'predicted_no_match': not accepted,
                   'metrics': metrics, 'timing_seconds': timings, 'accepted_ids': accepted,
                   'top_5': ranked[:5]}
            rows.append(row)
            local_checks = evidence_audit(response, prepared)
            checks.extend({'case_id': case_id, 'variant': variant, **check} for check in local_checks)
            for job_id in accepted:
                grade = judgments[case_id].get(job_id)
                if grade is not None and grade < 2:
                    failures.append({'case_id': case_id, 'variant': variant, 'job_id': job_id,
                                     'kind': 'accepted_partial' if grade == 1 else 'accepted_irrelevant', 'grade': grade})
            for job_id in ranked[:5]:
                if judgments[case_id].get(job_id) == 0:
                    failures.append({'case_id': case_id, 'variant': variant, 'job_id': job_id, 'kind': 'unsuitable_in_retrieval_top_5'})
            for job_id, grade in judgments[case_id].items():
                if grade == 2 and job_id not in accepted:
                    failures.append({'case_id': case_id, 'variant': variant, 'job_id': job_id, 'kind': 'suitable_not_recommended'})
            if scope == 'panel' and bool(accepted) == case['expected_no_match']:
                failures.append({'case_id': case_id, 'variant': variant, 'kind': 'wrong_no_match_decision'})
            if probe_input is None and response.recommendations:
                probe_input = prepared, response
            for name, payload in (('retrieval', retrieval), ('response', response.model_dump(mode='json')), ('assessment', audit)):
                save_json(output / 'cases' / f'{case_id}_{variant}_{name}.json', payload)
            print(f"{case_id}/{variant}: P@5={metrics['precision_at_5']}; accepted={','.join(accepted) or 'none'}", flush=True)
    failures += [{'kind': 'audit_failure', **check} for check in checks if not check['passed']]
    summary = {'suite_version': suite['suite_version'], 'suite_sha256': suite_hash,
               'dataset_sha256': suite['dataset_sha256'], 'scope': suite['scope'] if scope == 'panel' else
               'Full 250-job retrieval; only panel pairs are labeled. Unjudged outputs have bounds, not assumed negative labels.',
               'annotation_author': suite['annotation_author'], 'split_policy': suite['split_policy'],
               'implementation_sha256': suite['implementation_sha256'],
               'requirements_lock_sha256': suite['requirements_lock_sha256'],
               'corpus_size': len(jobs), 'selected_split': split, 'variants': list(variants),
               'frozen_settings': suite['frozen_settings'], 'model_load_seconds': {'embeddings': embedding_load, 'reranker': reranker_load},
               'index_reused': index_report['reused_existing_vectors'] if index_report else None,
               'hardware': {'platform': platform.platform(), 'machine': platform.machine(), 'cpu_threads': settings.rerank_cpu_threads},
               'runtime_versions': {name: __import__('importlib.metadata', fromlist=['version']).version(name)
                                    for name in ('sentence-transformers','torch','langchain-core','qdrant-client')},
               'qwen_probe': qwen_probe(settings, *probe_input) if with_qwen and probe_input else {'status': 'not_evaluated', 'calls': 0},
               'network_socket_connections_blocked': not with_qwen,
               'checks': {'passed': sum(check['passed'] for check in checks), 'total': len(checks),
                          'scope': 'Fixture expectations, arithmetic, literal skill support, closed-template explanations, deterministic edge cases; not general entailment.'},
               'reranker_status_counts': dict(statuses), 'metrics': {},
               'latency_method': 'Warm stage sum; each base retrieval measured once per case and reused for paired reranker comparison. Model load, indexing, persistence and HTTP/PDF parsing excluded.',
               'input_format': 'Public synthetic text resume fixtures; PDF parser quality is outside this benchmark.'}
    for name in ('dev', 'heldout'):
        split_rows = [row for row in rows if row['split'] == name]
        if split_rows:
            summary['metrics'][name] = {variant: aggregate([row for row in split_rows if row['variant'] == variant],
                                                          exhaustive_judgments=scope == 'panel') for variant in variants}
    save_json(output / 'summary.json', summary)
    save_json(output / 'case_metrics.json', {'cases': rows})
    save_json(output / 'checks.json', {'checks': checks})
    save_json(output / 'failures.json', {'failures': failures})
    (output / 'results.md').write_text(render_summary(summary))
    (output / 'failure_report.md').write_text(render_failures(failures))
    print(f'Evaluation saved: {output}', flush=True)
    return output, summary


def run(**kwargs):
    guard = nullcontext() if kwargs.get('with_qwen', False) else patch(
        'socket.socket.connect', side_effect=AssertionError('Unexpected evaluation network call'))
    with guard:
        return _run(**kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', type=Path, default=PROJECT_ROOT / 'evaluation')
    parser.add_argument('--split', choices=('all','dev','heldout'), default='all')
    parser.add_argument('--scope', choices=('panel','full'), default='panel')
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument('--qwen-probe', action='store_true', help='Optional bounded remote structured-plan probe for one accepted pair')
    args = parser.parse_args(argv)
    try:
        _, summary = run(directory=args.fixtures, split=args.split, scope=args.scope,
                         variants=tuple(args.variants), with_qwen=args.qwen_probe)
        return 0 if summary['checks']['passed'] == summary['checks']['total'] else 2
    except (OSError, ValueError, RuntimeError, KeyError):
        print('Evaluation failed: check frozen fixtures, prepared corpus, cached models and index consistency.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
