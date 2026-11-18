"""Judgment-aware metrics: unknown labels are never silently negative."""

import math
import statistics


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def ranking_metrics(ranked_ids, accepted_ids, labels, *, k=5, corpus_ids=None):
    if k < 1 or len(set(ranked_ids)) != len(ranked_ids) or len(set(accepted_ids)) != len(accepted_ids):
        raise ValueError('Invalid k or duplicate output IDs')
    if not set(accepted_ids) <= set(ranked_ids):
        raise ValueError('Accepted jobs must belong to the ranking')
    if any(type(grade) is not int or grade not in (0, 1, 2) for grade in labels.values()):
        raise ValueError('Unexpected judgment grade')
    if corpus_ids is not None and not set(ranked_ids) <= set(corpus_ids):
        raise ValueError('Foreign ranked job')
    top = ranked_ids[:k]
    positive = sum(labels.get(job_id) == 2 for job_id in top)
    unknown = sum(job_id not in labels for job_id in top)
    accepted_positive = sum(labels.get(job_id) == 2 for job_id in accepted_ids)
    accepted_unknown = sum(job_id not in labels for job_id in accepted_ids)
    fully_judged = corpus_ids is not None and set(corpus_ids) <= set(labels)
    relevant_count = sum(labels.get(job_id) == 2 for job_id in (corpus_ids or labels))
    gains = [2 ** labels.get(job_id, 0) - 1 for job_id in top]
    dcg = sum(gain / math.log2(rank + 2) for rank, gain in enumerate(gains))
    ideal = sorted((2 ** labels[job_id] - 1 for job_id in (corpus_ids or labels)), reverse=True)[:k] if fully_judged else []
    idcg = sum(gain / math.log2(rank + 2) for rank, gain in enumerate(ideal))
    return {'precision_at_5': positive / k if not unknown else None,
            'precision_at_5_lower': positive / k, 'precision_at_5_upper': (positive + unknown) / k,
            'ranked_returned_at_5': len(top), 'ranked_judged_at_5': len(top) - unknown,
            'ranked_unjudged_at_5': unknown, 'accepted_count': len(accepted_ids),
            'accepted_relevant': accepted_positive, 'accepted_unjudged': accepted_unknown,
            'accepted_partial': sum(labels.get(job_id) == 1 for job_id in accepted_ids),
            'accepted_irrelevant': sum(labels.get(job_id) == 0 for job_id in accepted_ids),
            'accepted_precision': accepted_positive / len(accepted_ids) if accepted_ids and not accepted_unknown else None,
            'accepted_precision_lower': accepted_positive / len(accepted_ids) if accepted_ids else None,
            'accepted_precision_upper': (accepted_positive + accepted_unknown) / len(accepted_ids) if accepted_ids else None,
            'recall_at_5': positive / relevant_count if fully_judged and relevant_count else None,
            'ideal_precision_at_5': min(k, relevant_count) / k if fully_judged else None,
            'ndcg_at_5': dcg / idcg if fully_judged and idcg else None}


def aggregate(rows, *, exhaustive_judgments):
    if not rows:
        return {'case_count': 0}
    count = len(rows)
    returned = sum(row['metrics']['accepted_count'] for row in rows)
    positive = sum(row['metrics']['accepted_relevant'] for row in rows)
    unknown = sum(row['metrics']['accepted_unjudged'] for row in rows)
    def average(name):
        values = [row['metrics'][name] for row in rows]
        return sum(values) / count if all(value is not None for value in values) else None
    durations = [row['timing_seconds']['end_to_end'] for row in rows]
    result = {'case_count': count, 'macro_precision_at_5': average('precision_at_5'),
              'macro_ideal_precision_at_5': average('ideal_precision_at_5'),
              'precision_at_5_lower': average('precision_at_5_lower'), 'precision_at_5_upper': average('precision_at_5_upper'),
              'micro_accepted_precision': positive / returned if returned and not unknown else None,
              'accepted_precision_lower': positive / returned if returned else None,
              'accepted_precision_upper': (positive + unknown) / returned if returned else None,
              'mean_returned_count': returned / count, 'accepted_count': returned,
              'accepted_unjudged': unknown, 'judged_top_5_count': sum(row['metrics']['ranked_judged_at_5'] for row in rows),
              'unjudged_top_5_count': sum(row['metrics']['ranked_unjudged_at_5'] for row in rows),
              'latency_seconds': {'median': statistics.median(durations), 'p95': percentile(durations, .95),
                                  'n': len(durations), 'scope': 'One warm observation per distinct case; no concurrency/load testing.'}}
    if exhaustive_judgments:
        true_positive = sum(row['expected_no_match'] and row['predicted_no_match'] for row in rows)
        false_positive = sum(not row['expected_no_match'] and row['predicted_no_match'] for row in rows)
        false_negative = sum(row['expected_no_match'] and not row['predicted_no_match'] for row in rows)
        true_negative = count - true_positive - false_positive - false_negative
        result['no_match'] = {'true_positive': true_positive, 'false_positive': false_positive,
                              'false_negative': false_negative, 'true_negative': true_negative,
                              'precision': true_positive / (true_positive + false_positive) if true_positive + false_positive else None,
                              'recall': true_positive / (true_positive + false_negative) if true_positive + false_negative else None,
                              'accuracy': (true_positive + true_negative) / count}
    else:
        result['no_match'] = None
        result['no_match_note'] = 'Panel labels cannot establish no-match ground truth for the full 250-job corpus.'
    for name in ('recall_at_5', 'ndcg_at_5'):
        defined = [row['metrics'][name] for row in rows if row['metrics'][name] is not None]
        result['mean_' + name] = sum(defined) / len(defined) if defined else None
        result[name + '_defined_cases'] = len(defined)
    for stage in ('preparation', 'retrieval', 'reranking', 'assessment'):
        values = [row['timing_seconds'][stage] for row in rows]
        result['latency_seconds'][stage] = {'median': statistics.median(values), 'p95': percentile(values, .95)}
    return result
