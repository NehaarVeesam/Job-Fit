"""Run a searchable resume PDF through JobFit and retain a private stage trace."""

import argparse
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch
from uuid import uuid4

from langchain_core.embeddings import Embeddings
from app.config import get_settings
from app.dataset.dataset_loader import prepare_dataset, write_dataset
from app.retrieval.embeddings import load_embeddings
from app.retrieval.hybrid import retrieve_candidates
from app.matching.matcher import match_candidates
from app.resumes.profile import build_profile, load_prepared_jobs, skill_vocabulary
from app.retrieval.reranking import rerank_candidates
from app.resumes.resume_parser import ResumeError, parse_resume
from app.schemas import Preferences
from app.retrieval.vector_index import build_index, save_json


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class RunLog:
    def __init__(self, output):
        self.output = output

    def event(self, stage, status, **details):
        with (self.output / 'events.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps({'time_utc': utc_now(), 'stage': stage,
                                     'status': status, **details}, allow_nan=False) + '\n')
        print(f'{stage}: {status}', flush=True)

    @contextmanager
    def stage(self, name):
        started = time.perf_counter()
        self.event(name, 'started')
        try:
            yield
        except Exception:
            self.event(name, 'failed', duration_seconds=round(time.perf_counter() - started, 3))
            raise
        self.event(name, 'completed', duration_seconds=round(time.perf_counter() - started, 3))


class RecordedEmbeddings(Embeddings):
    """Record the exact resume query vector passed to Qdrant; no extra encode."""

    def __init__(self, delegate, output, info):
        self.delegate, self.output, self.info = delegate, output, info
        self.recorded = False

    def token_counts(self, texts):
        return self.delegate.token_counts(texts)

    def embed_documents(self, texts):
        return self.delegate.embed_documents(texts)

    def embed_query(self, text):
        vector = self.delegate.embed_query(text)
        save_json(self.output / 'query_embedding.json', {
            'status': 'embedded', 'query_sha256': hashlib.sha256(text.encode()).hexdigest(),
            'model': self.info['model'], 'revision': self.info['revision'],
            'dimension': len(vector), 'l2_norm': sum(x*x for x in vector) ** .5,
            'vector': vector,
        })
        self.recorded = True
        return vector


def write_manifest(output, **details):
    files = []
    for path in sorted(output.rglob('*')):
        if path.is_file() and path.name != 'run_manifest.json':
            files.append({'path': str(path.relative_to(output)), 'bytes': path.stat().st_size,
                          'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    save_json(output / 'run_manifest.json', {'updated_utc': utc_now(), **details, 'files': files})


def run(args, settings, output, log):
    with log.stage('dataset'):
        corpus = output / 'dataset'
        write_dataset(prepare_dataset(settings.dataset_path), corpus, settings.dataset_path)
        jobs, provenance = load_prepared_jobs(corpus)
    with log.stage('pdf_extraction'):
        extraction = parse_resume(args.resume.expanduser(), max_upload_mb=settings.max_upload_mb,
                                  max_pages=settings.max_resume_pages)
        save_json(output / 'parsed_resume.json', extraction.parsed.model_dump(mode='json'))
        (output / 'extracted_text.txt').write_text(
            '\n\n'.join(f'--- Page {page.page} ---\n{page.text}' for page in extraction.parsed.pages),
            encoding='utf-8')
    with log.stage('profile'):
        preferences = (Preferences.model_validate_json(args.preferences.expanduser().read_text(encoding='utf-8'))
                       if args.preferences else Preferences())
        prepared = build_profile(extraction.parsed, skill_vocabulary(jobs), preferences)
        prepared.warnings = extraction.warnings + prepared.warnings
        save_json(output / 'prepared_resume.json', prepared.model_dump(mode='json'))
        save_json(output / 'candidate_profile.json', prepared.candidate_profile.model_dump(mode='json'))
        save_json(output / 'preferences.json', preferences.model_dump(mode='json'))
        save_json(output / 'query.json', {'text': prepared.retrieval_query, 'warnings': prepared.warnings})
    embeddings, info = None, None
    if prepared.retrieval_query is not None and settings.retrieval_mode != 'bm25':
        with log.stage('embedding_model_and_job_index'):
            base, info = load_embeddings(settings)
            embeddings = RecordedEmbeddings(base, output, info)
            save_json(output / 'index_report.json', build_index(settings, jobs, provenance, embeddings, info))
    with log.stage('retrieval'):
        retrieval = retrieve_candidates(settings, jobs, provenance, prepared,
                                        embeddings=embeddings, embedding_info=info)
        save_json(output / 'base_retrieval.json', retrieval)
        if embeddings is None or not embeddings.recorded:
            save_json(output / 'query_embedding.json', {'status': 'skipped',
                'reason': 'BM25 mode, no supported query, or no eligible jobs; see retrieval status.',
                'retrieval_status': retrieval['status']})
    with log.stage('reranking'):
        retrieval = rerank_candidates(settings, prepared, retrieval)
        save_json(output / 'retrieval.json', retrieval)
    with log.stage('matching'):
        response, assessment = match_candidates(settings, jobs, provenance, prepared, retrieval,
                                                use_llm=args.with_llm)
        save_json(output / 'assessment.json', assessment)
        save_json(output / 'response.json', response.model_dump(mode='json'))
    log.event('run', 'completed', response_status=response.status,
              recommendations=len(response.recommendations))
    write_manifest(output, status='completed', resume_sha256=extraction.source_sha256,
                   dataset_sha256=hashlib.sha256(settings.dataset_path.read_bytes()).hexdigest(),
                   corpus_size=len(jobs), retrieval_mode=settings.retrieval_mode,
                   reranking_requested=args.rerank, qwen_requested=args.with_llm,
                   network_connections_blocked=not args.with_llm,
                   response_status=response.status, generation_calls=assessment['generation_calls'])
    print(f'Result: {response.status}; recommendations={len(response.recommendations)}')
    print(f'Response: {output / "response.json"}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('resume', type=Path, help='Path to your searchable, unlocked PDF')
    parser.add_argument('--preferences', type=Path, help='Optional preferences JSON; defaults to no filters')
    parser.add_argument('--retrieval-mode', choices=('dense', 'bm25', 'hybrid'), default='hybrid')
    parser.add_argument('--rerank', action='store_true')
    parser.add_argument('--with-llm', action='store_true', help='Allow remote Qwen evidence-selection calls')
    args = parser.parse_args(argv)
    output, log = None, None
    try:
        settings = get_settings().model_copy(update={
            'retrieval_mode': args.retrieval_mode, 'rerank_enabled': args.rerank})
        output = settings.artifacts_dir / 'resume_runs' / (
            datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid4().hex[:12])
        output.mkdir(parents=True, exist_ok=False)
        log = RunLog(output)
        print(f'Private run folder: {output}', flush=True)
        log.event('run', 'started')
        guard = nullcontext() if args.with_llm else patch(
            'socket.socket.connect', side_effect=RuntimeError('Offline run attempted a network connection'))
        with guard:
            run(args, settings, output, log)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        code = exc.code if isinstance(exc, ResumeError) else type(exc).__name__
        if log is not None:
            log.event('run', 'failed', error_code=code)
            write_manifest(output, status='failed', error_code=code)
        message = str(exc) if isinstance(exc, ResumeError) else (
            'Check preferences/configuration, cached models, and local Qdrant availability.')
        print(f'Run failed [{code}]: {message}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
