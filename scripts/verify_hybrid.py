"""Real-model Module 6 development comparison; no live Qwen or quality claims.

Requires prepared jobs and the locally cached BGE-M3 index/model. Uses only public
synthetic text resumes; blocks sockets and writes generated reports to artifacts/.
"""

import sys
import time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import PROJECT_ROOT, get_settings
from app.embeddings import load_embeddings
from app.hybrid_retrieval import MODES, retrieve_candidates
from app.matching import match_candidates
from app.profile import build_profile, load_prepared_jobs, skill_vocabulary
from app.schemas import ParsedResume, Preferences
from app.vector_index import save_json


def run():
    settings = get_settings()
    if settings.qdrant_mode != 'local':
        raise ValueError('Offline verification requires local Qdrant')
    jobs, provenance = load_prepared_jobs(settings.artifacts_dir / 'dataset/all')
    vocabulary = skill_vocabulary(jobs)
    print('Loading cached BGE-M3 for three-mode comparison...', flush=True)
    started = time.perf_counter()
    embeddings, info = load_embeddings(settings)
    model_load_seconds = round(time.perf_counter() - started, 3)
    outcomes = []
    directory = settings.artifacts_dir / 'verification/module6'
    for name in ('ai', 'backend', 'analyst', 'fresher', 'unrelated'):
        text = (PROJECT_ROOT / 'examples/resumes' / f'{name}.txt').read_text()
        preferences = (Preferences.model_validate_json((PROJECT_ROOT / 'examples/preferences.json').read_text())
                       if name == 'ai' else Preferences())
        prepared = build_profile(ParsedResume(pages=[{'page': 1, 'text': text}]), vocabulary, preferences)
        reports, summary = {}, {'profile': name, 'modes': {}}
        for mode in MODES:
            result = retrieve_candidates(settings, jobs, provenance, prepared, mode=mode,
                                         embeddings=embeddings, embedding_info=info)
            response, assessment = match_candidates(settings, jobs, provenance, prepared, result, use_llm=False)
            assert assessment['generation_calls'] == 0
            save_json(directory / f'{name}_{mode}_retrieval.json', result)
            save_json(directory / f'{name}_{mode}_response.json', response.model_dump(mode='json'))
            save_json(directory / f'{name}_{mode}_assessment.json', assessment)
            reports[mode] = result
            summary['modes'][mode] = {
                'eligible_count': result['eligible_count'], 'candidate_count': len(result['candidates']),
                'top_5': [item['job_id'] for item in result['candidates'][:5]],
                'recommendations': [item.job.job_id for item in response.recommendations],
                'status': response.status, 'timing_seconds': result['timing_seconds']}
        assert len({report['eligible_count'] for report in reports.values()}) == 1
        assert len({report['query_sha256'] for report in reports.values()}) == 1
        dense_ids = {item['job_id'] for item in reports['dense']['candidates']}
        bm25_ids = {item['job_id'] for item in reports['bm25']['candidates']}
        hybrid_ids = {item['job_id'] for item in reports['hybrid']['candidates']}
        summary.update(dense_bm25_overlap=len(dense_ids & bm25_ids),
                       hybrid_new_vs_dense=sorted(hybrid_ids - dense_ids))
        outcomes.append(summary)
        print(f"{name}: " + ' | '.join(f"{mode}: {','.join(summary['modes'][mode]['recommendations']) or 'no match'}"
                                     for mode in MODES), flush=True)
    save_json(settings.artifacts_dir / 'verification/module6_live.json', {
        'scope': 'Five synthetic development demos; no held-out suitability judgments.',
        'socket_connections_blocked': True, 'generation_calls': 0,
        'model_load_seconds': model_load_seconds, 'corpus': provenance,
        'settings': {'dense_window': settings.candidate_count, 'bm25_window': settings.bm25_candidate_count,
                     'rrf_constant': settings.rrf_constant, 'bm25_k1': settings.bm25_k1, 'bm25_b': settings.bm25_b},
        'queries': outcomes})
    print('Saved artifacts/verification/module6_live.json and per-profile reports.', flush=True)


if __name__ == '__main__':
    with patch('socket.socket.connect', side_effect=AssertionError('Unexpected network connection')):
        run()
