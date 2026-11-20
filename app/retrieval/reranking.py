"""Optional local cross-encoder ordering; original retrieval scores stay intact."""

import copy
import hashlib
import math
import time
from pathlib import Path

from app.retrieval.documents import job_document
from app.schemas import JobRecord


class PairTooLong(ValueError):
    def __init__(self, lengths, limit):
        super().__init__('Cross-encoder pair exceeds the explicit token budget')
        self.audit = {'pair_token_counts': lengths, 'max_tokens': limit,
                      'overflow_count': sum(length > limit for length in lengths),
                      'truncation_policy': 'reject_and_preserve_original_order'}


class CheckedCrossEncoder:
    def __init__(self, delegate, tokenizer, *, max_tokens=512, batch_size=8):
        self.delegate, self.tokenizer = delegate, tokenizer
        self.max_tokens, self.batch_size = max_tokens, batch_size

    def score(self, pairs, *, deadline=None):
        if not pairs:
            return [], {'pair_token_counts': [], 'overflow_count': 0, 'max_tokens': self.max_tokens}
        encoded = self.tokenizer([pair[0] for pair in pairs], [pair[1] for pair in pairs],
                                 add_special_tokens=True, truncation=False, padding=False,
                                 return_attention_mask=False)
        lengths = [len(ids) for ids in encoded['input_ids']]
        if len(lengths) != len(pairs):
            raise ValueError('Tokenizer count differs from pair count')
        if any(length > self.max_tokens for length in lengths):
            raise PairTooLong(lengths, self.max_tokens)
        scores = []
        for offset in range(0, len(pairs), self.batch_size):
            if deadline is not None and time.perf_counter() >= deadline:
                raise TimeoutError('Reranking deadline exceeded')
            # Identity activation is configured at load time: retain raw logits.
            values = self.delegate.predict(pairs[offset:offset + self.batch_size],
                                           batch_size=self.batch_size, show_progress_bar=False,
                                           convert_to_numpy=True)
            batch = [float(value) for value in values]
            if len(batch) != len(pairs[offset:offset + self.batch_size]) or not all(map(math.isfinite, batch)):
                raise ValueError('Expected one finite cross-encoder score per pair')
            scores.extend(batch)
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError('Reranking deadline exceeded')
        return scores, {'pair_token_counts': lengths, 'overflow_count': 0,
                        'max_tokens': self.max_tokens, 'truncation_policy': 'reject_and_preserve_original_order'}


def load_reranker(settings):
    import torch
    from huggingface_hub import snapshot_download
    from sentence_transformers import CrossEncoder

    if settings.rerank_device == 'cpu':
        torch.set_num_threads(settings.rerank_cpu_threads)
    snapshot = snapshot_download(repo_id=settings.rerank_model, revision=settings.rerank_revision,
                                 cache_dir=settings.rerank_cache_dir, local_files_only=True)
    if Path(snapshot).name != settings.rerank_revision:
        raise ValueError('Reranker snapshot revision mismatch')
    delegate = CrossEncoder(snapshot, max_length=settings.rerank_max_tokens,
                            device=settings.rerank_device, local_files_only=True,
                            trust_remote_code=False, activation_fn=torch.nn.Identity())
    limit = min(delegate.tokenizer.model_max_length, delegate.model.config.max_position_embeddings)
    if settings.rerank_max_tokens > limit or delegate.model.config.num_labels != 1:
        raise ValueError('Unexpected cross-encoder input/output contract')
    return CheckedCrossEncoder(delegate, delegate.tokenizer,
                               max_tokens=settings.rerank_max_tokens, batch_size=settings.rerank_batch_size)


def rerank_candidates(settings, prepared, retrieval, *, reranker=None, reranker_loader=None, deadline=None):
    """Return a new report, scoring every candidate or falling back as one unit.

    Scores reorder candidates only. No candidate is added/dropped and no original
    score, job text, skill evidence, or eligibility outcome is rewritten.
    """
    if 'reranking' in retrieval:
        raise ValueError('Expected a base retrieval report; do not rerank an already processed report')
    result = copy.deepcopy(retrieval)
    candidates = result.get('candidates', [])
    metadata = {'enabled': settings.rerank_enabled, 'status': 'disabled', 'model': settings.rerank_model,
                'revision': settings.rerank_revision, 'max_tokens': settings.rerank_max_tokens,
                'batch_size': settings.rerank_batch_size, 'device': settings.rerank_device,
                'cpu_threads': settings.rerank_cpu_threads, 'score_type': 'cross_encoder',
                'score_semantics': 'raw_logit_not_probability', 'candidate_count': len(candidates),
                'model_load_seconds': 0, 'inference_seconds': 0}
    result['reranking'] = metadata
    if not settings.rerank_enabled:
        return result
    expected_hash = hashlib.sha256((prepared.retrieval_query or '').encode()).hexdigest()
    if retrieval.get('query_sha256') != expected_hash or retrieval.get('applied_preferences') != prepared.preferences.model_dump(mode='json'):
        raise ValueError('Reranking query/preferences differ from prepared resume')
    if not candidates:
        metadata['status'] = 'skipped_empty'
        return result
    if prepared.retrieval_query is None:
        raise ValueError('Cannot rerank candidates without a query')
    pairs, seen = [], set()
    for candidate in candidates:
        job = JobRecord.model_validate(candidate['job'])
        if candidate['job_id'] != job.job_id or job.job_id in seen:
            raise ValueError('Duplicate or mismatched reranking candidate ID')
        if candidate.get('reranker_relevance') is not None:
            raise ValueError('Candidate already contains a reranker score')
        seen.add(job.job_id)
        pairs.append((prepared.retrieval_query, job_document(job).page_content))
    started = time.perf_counter()
    try:
        if deadline is not None and started >= deadline:
            raise TimeoutError('Reranking deadline exceeded')
        if reranker is None:
            reranker = reranker_loader() if reranker_loader is not None else load_reranker(settings)
        metadata['model_load_seconds'] = round(time.perf_counter() - started, 4)
        inference_start = time.perf_counter()
        scores, token_audit = reranker.score(pairs, deadline=deadline)
        metadata['inference_seconds'] = round(time.perf_counter() - inference_start, 4)
        if len(scores) != len(candidates) or not all(math.isfinite(score) for score in scores):
            raise ValueError('Invalid reranker output')
    except TimeoutError:
        raise
    except Exception as error:
        metadata.update(status='fallback', error_type=type(error).__name__,
                        elapsed_seconds=round(time.perf_counter() - started, 4))
        if isinstance(error, PairTooLong):
            metadata['token_audit'] = error.audit
        result.setdefault('warnings', []).append('Optional reranking unavailable or input invalid; original retrieval order retained.')
        return result
    for rank, (candidate, score) in enumerate(zip(candidates, scores), 1):
        candidate['original_rank'] = rank
        candidate['reranker_relevance'] = {'score_type': 'cross_encoder', 'value': float(score)}
    candidates.sort(key=lambda item: (-item['reranker_relevance']['value'], item['original_rank'], item['job_id']))
    for rank, candidate in enumerate(candidates, 1):
        candidate['rank'] = rank
    metadata.update(status='applied', token_audit=token_audit,
                    elapsed_seconds=round(time.perf_counter() - started, 4))
    return result
