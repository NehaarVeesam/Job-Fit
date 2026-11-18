"""Pinned local-model Module 7 benchmark; development diagnostics only."""

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import PROJECT_ROOT, Settings, get_settings
from app.documents import job_document
from app.embeddings import load_embeddings
from app.hybrid_retrieval import retrieve_candidates
from app.matching import match_candidates
from app.profile import build_profile, load_prepared_jobs, skill_vocabulary
from app.reranking import load_reranker, rerank_candidates
from app.schemas import ParsedResume, Preferences
from app.vector_index import save_json


def run(config_path):
    config = json.loads(config_path.read_text())
    allowed = {'retrieval_mode', 'candidate_count', 'bm25_candidate_count', 'rerank_model',
               'rerank_revision', 'rerank_device', 'rerank_batch_size', 'rerank_max_tokens', 'rerank_cpu_threads'}
    if set(config['settings']) - allowed:
        raise ValueError('Unsupported benchmark setting')
    settings = Settings(_env_file=None, **(get_settings().model_dump() | config['settings']))
    settings = settings.model_copy(update={'rerank_enabled': True})
    if settings.qdrant_mode != 'local':
        raise ValueError('Offline benchmark requires local Qdrant')
    jobs, provenance = load_prepared_jobs(settings.artifacts_dir / 'dataset/all')
    vocabulary = skill_vocabulary(jobs)
    print('Loading cached embedding and cross-encoder models...', flush=True)
    start = time.perf_counter()
    embeddings, info = load_embeddings(settings)
    embedding_load = time.perf_counter() - start
    start = time.perf_counter()
    reranker = load_reranker(settings)
    reranker_load = time.perf_counter() - start
    output = settings.artifacts_dir / 'verification/module7'
    results = []
    for name in ('ai', 'backend', 'analyst', 'fresher', 'unrelated'):
        text = (PROJECT_ROOT / 'examples/resumes' / f'{name}.txt').read_text()
        preferences = (Preferences.model_validate_json((PROJECT_ROOT / 'examples/preferences.json').read_text())
                       if name == 'ai' else Preferences())
        prepared = build_profile(ParsedResume(pages=[{'page': 1, 'text': text}]), vocabulary, preferences)
        start = time.perf_counter()
        baseline = retrieve_candidates(settings, jobs, provenance, prepared,
                                        embeddings=embeddings, embedding_info=info)
        retrieval_seconds = time.perf_counter() - start
        baseline_response, baseline_audit = match_candidates(settings, jobs, provenance, prepared, baseline, use_llm=False)
        reranked = rerank_candidates(settings, prepared, baseline, reranker=reranker)
        if baseline['candidates'] and reranked['reranking']['status'] != 'applied':
            raise AssertionError('Real-model reranking fell back')
        response, audit = match_candidates(settings, jobs, provenance, prepared, reranked, use_llm=False)
        before = {item['job_id']: item for item in baseline_audit['assessments']}
        for item in audit['assessments']:
            for key in ('required_coverage', 'preferred_coverage', 'eligibility', 'experience', 'education', 'constraints', 'rejection_reasons'):
                assert item[key] == before[item['job_id']][key], (name, item['job_id'], key)
        assert {item['job_id'] for item in baseline['candidates']} == {item['job_id'] for item in reranked['candidates']}
        assert audit['generation_calls'] == 0
        for label, report in (('baseline_retrieval', baseline), ('reranked_retrieval', reranked),
                              ('baseline_response', baseline_response.model_dump(mode='json')),
                              ('reranked_response', response.model_dump(mode='json')), ('assessment', audit)):
            save_json(output / f'{name}_{label}.json', report)
        results.append({'profile': name, 'candidate_count': len(baseline['candidates']),
                        'baseline_top_5': [item['job_id'] for item in baseline['candidates'][:5]],
                        'reranked_top_5': [item['job_id'] for item in reranked['candidates'][:5]],
                        'baseline_recommendations': [item.job.job_id for item in baseline_response.recommendations],
                        'reranked_recommendations': [item.job.job_id for item in response.recommendations],
                        'retrieval_wall_seconds': round(retrieval_seconds, 4), 'reranking': reranked['reranking'],
                        'all_coverage_and_requirement_outcomes_unchanged': True})
        print(f"{name}: rerank={reranked['reranking']['status']}; recommendations=" +
              ','.join(item.job.job_id for item in response.recommendations), flush=True)
    judged = []
    by_id = {job.job_id: job for job in jobs}
    for case in config['pairwise_development_cases']:
        texts = [job_document(by_id[case[key]]).page_content for key in ('preferred_job_id', 'other_job_id')]
        scores, token_audit = reranker.score([(case['query'], text) for text in texts])
        # Matched BGE-M3 cosine baseline on the same two complete job texts.
        query_vector = embeddings.embed_query(case['query'])
        vectors = embeddings.embed_documents(texts)
        cosine = [sum(a * b for a, b in zip(query_vector, vector)) for vector in vectors]
        judged.append({**case, 'dense_cosines': cosine, 'reranker_logits': scores,
                       'dense_agrees': cosine[0] > cosine[1], 'reranker_agrees': scores[0] > scores[1],
                       'token_audit': token_audit})
    result = {'scope': config['scope'], 'rubric': config['rubric'],
              'benchmark_config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
              'configuration': config['settings'], 'corpus': provenance,
              'hardware': {'platform': platform.platform(), 'machine': platform.machine(), 'cpu_threads': settings.rerank_cpu_threads},
              'cold_model_load_seconds': {'embeddings': round(embedding_load, 4), 'reranker': round(reranker_load, 4)},
              'generation_calls': 0, 'socket_connections_blocked': True, 'profiles': results,
              'pairwise_development': judged,
              'pairwise_summary': {'case_count': len(judged), 'dense_agreements': sum(item['dense_agrees'] for item in judged),
                                   'reranker_agreements': sum(item['reranker_agrees'] for item in judged)},
              'limitations': ['Four easy topical pairs are not full suitability evaluation or evidence of general improvement.',
                             'Single-process timings are development observations, not robust latency benchmarks.',
                             'No new cross-encoder acceptance threshold was tuned; existing gates are unchanged.']}
    save_json(settings.artifacts_dir / 'verification/module7_live.json', result)
    print('Saved artifacts/verification/module7_live.json', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT / 'examples/reranker/benchmark.json')
    args = parser.parse_args()
    with patch('socket.socket.connect', side_effect=AssertionError('Unexpected network connection')):
        run(args.config)
