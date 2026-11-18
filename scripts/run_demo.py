"""Run public synthetic match/no-match examples with cached local models.

Defaults to hybrid retrieval, no reranking, and no Qwen calls. Generated PDFs,
profiles, retrieval records, and responses remain under ignored artifacts/.
"""

import argparse
import hashlib
import sys
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import PROJECT_ROOT, get_settings
from app.dataset_loader import CorpusOptions, prepare_corpus, write_corpus
from app.embeddings import load_embeddings
from app.hybrid_retrieval import retrieve_candidates
from app.matching import match_candidates
from app.profile import build_profile, load_prepared_jobs, skill_vocabulary
from app.reranking import load_reranker, rerank_candidates
from app.resume_parser import parse_resume
from app.schemas import Preferences
from app.vector_index import build_index, save_json
from scripts.create_demo_resumes import main as create_demo_resumes


def run(args):
    settings = get_settings().model_copy(update={'retrieval_mode': 'hybrid', 'rerank_enabled': args.rerank})
    output = settings.artifacts_dir / 'portfolio_demo'
    print('Preparing all 250 supplied synthetic jobs...', flush=True)
    corpus = output / 'dataset'
    write_corpus(prepare_corpus(settings.dataset_path, CorpusOptions()), corpus, settings.dataset_path)
    jobs, provenance = load_prepared_jobs(corpus)
    create_demo_resumes()
    vocabulary = skill_vocabulary(jobs)
    names = ('ai', 'unrelated') if args.profile == 'all' else (args.profile,)
    prepared = {}
    for name in names:
        extraction = parse_resume(PROJECT_ROOT / 'artifacts/demo_resumes' / f'{name}.pdf',
                                  max_upload_mb=settings.max_upload_mb, max_pages=settings.max_resume_pages)
        preferences = (Preferences.model_validate_json((PROJECT_ROOT / 'examples/preferences.json').read_text())
                       if name == 'ai' else Preferences())
        candidate = build_profile(extraction.parsed, vocabulary, preferences)
        candidate.warnings = extraction.warnings + candidate.warnings
        prepared[name] = candidate
        save_json(output / name / 'prepared_resume.json', candidate.model_dump(mode='json'))
    embeddings, info = None, None
    if any(candidate.retrieval_query for candidate in prepared.values()):
        print('Loading cached BGE-M3 and checking/reusing the job index...', flush=True)
        embeddings, info = load_embeddings(settings)
        index = build_index(settings, jobs, provenance, embeddings, info)
        print(f"Index ready; reused existing vectors={index['reused_existing_vectors']}", flush=True)
    reranker = load_reranker(settings) if args.rerank and embeddings is not None else None
    summaries = []
    for name, candidate in prepared.items():
        retrieval = retrieve_candidates(settings, jobs, provenance, candidate,
                                        embeddings=embeddings, embedding_info=info)
        retrieval = rerank_candidates(settings, candidate, retrieval, reranker=reranker)
        response, report = match_candidates(settings, jobs, provenance, candidate, retrieval, use_llm=args.with_llm)
        for filename, data in (('response.json', response.model_dump(mode='json')),
                               ('retrieval.json', retrieval), ('assessment.json', report)):
            save_json(output / name / filename, data)
        summaries.append({'profile': name, 'status': response.status,
                          'recommendation_ids': [match.job.job_id for match in response.recommendations],
                          'generation_calls': report['generation_calls'],
                          'reranking_status': retrieval['reranking']['status']})
        print(f"{name}: {response.status}; recommendations={len(response.recommendations)}", flush=True)
        for match in response.recommendations:
            print(f'  {match.job.job_id}: required coverage={match.required_coverage.percentage}%; eligibility={match.eligibility}')
        print(f'Response: {output / name / "response.json"}', flush=True)
    save_json(output / 'summary.json', {'dataset_id': settings.dataset_id, 'is_synthetic': True,
        'dataset_sha256': hashlib.sha256(settings.dataset_path.read_bytes()).hexdigest(),
        'corpus_size': len(jobs), 'retrieval_mode': 'hybrid', 'reranking_requested': args.rerank,
        'qwen_requested': args.with_llm, 'model': settings.embedding_model, 'model_revision': settings.embedding_revision,
        'cases': summaries, 'network_connections_blocked': not args.with_llm})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=('all','ai','unrelated'), default='all')
    parser.add_argument('--rerank', action='store_true')
    parser.add_argument('--with-llm', action='store_true', help='Enable the configured remote Qwen structured plans')
    args = parser.parse_args(argv)
    guard = nullcontext() if args.with_llm else patch('socket.socket.connect', side_effect=AssertionError('Unexpected demo network call'))
    try:
        with guard:
            run(args)
    except (OSError, ValueError, RuntimeError, KeyError, AssertionError):
        print('Demo failed: check local configuration, cached models and Qdrant availability. Download models before running.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
