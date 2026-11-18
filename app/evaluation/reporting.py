"""Compact, shareable summaries without private config or full resume text."""


def number(value):
    return 'N/A' if value is None else f'{value:.3f}'


def render_summary(summary):
    lines = ['# Module 8 evaluation results', '', summary['scope'], '',
             f"Suite: `{summary['suite_sha256']}`. Source: `{summary['dataset_sha256']}`.", '',
             'Labels are assistant-authored synthetic suitability judgments; they are not independent expert validation.', '',
             '| Split | Variant | Cases | P@5 / panel ceiling | nDCG@5 | Accepted precision | Mean returned | Median / p95 seconds |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for split, variants in summary['metrics'].items():
        for variant, row in variants.items():
            latency = row['latency_seconds']
            lines.append(f"| {split} | {variant} | {row['case_count']} | {number(row['macro_precision_at_5'])} / {number(row['macro_ideal_precision_at_5'])} | {number(row['mean_ndcg_at_5'])} | "
                         f"{number(row['micro_accepted_precision'])} | {row['mean_returned_count']:.3f} | "
                         f"{latency['median']:.3f} / {latency['p95']:.3f} |")
    lines += ['', '| Split | Variant | No-match precision | No-match recall | No-match accuracy | Judged / unjudged returned top-five slots |',
              '|---|---|---:|---:|---:|---:|']
    for split, variants in summary['metrics'].items():
        for variant, row in variants.items():
            no_match = row['no_match'] or {}
            lines.append(f"| {split} | {variant} | {number(no_match.get('precision'))} | {number(no_match.get('recall'))} | "
                         f"{number(no_match.get('accuracy'))} | {row['judged_top_5_count']} / {row['unjudged_top_5_count']} |")
    if any(row['unjudged_top_5_count'] for variants in summary['metrics'].values() for row in variants.values()):
        lines += ['', '| Split | Variant | P@5 lower / upper | Accepted precision lower / upper |',
                  '|---|---|---:|---:|']
        for split, variants in summary['metrics'].items():
            for variant, row in variants.items():
                lines.append(f"| {split} | {variant} | {number(row['precision_at_5_lower'])} / {number(row['precision_at_5_upper'])} | "
                             f"{number(row['accepted_precision_lower'])} / {number(row['accepted_precision_upper'])} |")
    lines += ['', 'Strict precision counts only grade 2. Grade 1 is partial/conditional. Missing top-five positions count as non-relevant.',
              'Accepted precision is N/A when nothing is returned. Unknown judgments produce bounds, not automatic negative labels.',
              'No-match metrics use separate expectations; conditional/partial opportunities prevent a no-match ground truth. The panel P@5 ceiling reflects how few suitable jobs exist under restrictive preferences. nDCG averages only cases with nonzero ideal gain; defined-case counts are in JSON.',
              '', f"Independent fixture checks: {summary['checks']['passed']}/{summary['checks']['total']} passed.",
              f"Reranker status counts: {summary['reranker_status_counts']}.", '',
              'Latency excludes model load/index build, includes preparation, retrieval, reranking and assessment, and uses one observation per case.',
              'This run does not evaluate OCR, arbitrary PDF layouts, remote Qwen reliability, real jobs, or population-level hiring suitability.',
              'No held-out settings or labels were changed after inspecting model outputs.', '']
    return '\n'.join(lines)


def render_failures(failures):
    lines = ['# Evaluation failure analysis', '',
             'These are disagreements with the authored rubric, plus failed deterministic checks. They are not automatically application bugs.', '']
    if not failures:
        lines.append('No disagreements observed in this scoped run; broader quality remains unproven.')
    for item in failures:
        lines.append('- ' + ' | '.join(f'{key}={value}' for key, value in item.items()))
    lines += ['', 'Do not tune using held-out failures. Collect separate development cases or version a future evaluation split.', '']
    return '\n'.join(lines)
