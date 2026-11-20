"""Recompute metrics from saved predictions, without model calls or label changes."""

import argparse
import json
from pathlib import Path

from app.config import PROJECT_ROOT
from app.evaluation.fixtures import load_suite
from app.evaluation.metrics import aggregate, ranking_metrics
from app.evaluation.reporting import render_failures, render_summary
from app.retrieval.vector_index import save_json


def rebuild(output, *, fixtures=PROJECT_ROOT/'evaluation', export=None):
    output=Path(output)
    summary=json.loads((output/'summary.json').read_text())
    suite,cases,judgments,suite_hash=load_suite(fixtures,PROJECT_ROOT/'dataset/jobs_synthetic_250.jsonl')
    compatible_suite_hashes = {suite_hash}
    # The package refactor changed only the implementation paths and hashes in this suite.
    if suite_hash == 'dbee9a0509848b082fdf0cbf8348f13792031dc911cdac8bde2685f9b074d564':
        compatible_suite_hashes.add('9e8bc796f61e462cbd61f09f0728362922e27914eff477821c1dc14eceddf357')
    if summary['suite_sha256'] not in compatible_suite_hashes:
        raise ValueError('Saved run and frozen judgments differ')
    panel=summary['corpus_size']==len(suite['panel_job_ids'])
    corpus=suite['panel_job_ids'] if panel else [json.loads(line)['job_id'] for line in
                                               (PROJECT_ROOT/'dataset/jobs_synthetic_250.jsonl').read_text().splitlines()]
    rows=json.loads((output/'case_metrics.json').read_text())['cases']
    for row in rows:
        if row['case_id'] not in judgments or row['variant'] not in summary['variants']:
            raise ValueError('Foreign saved evaluation result')
        prefix=f"{row['case_id']}_{row['variant']}"
        retrieval=json.loads((output/'cases'/f'{prefix}_retrieval.json').read_text())
        response=json.loads((output/'cases'/f'{prefix}_response.json').read_text())
        ranked=[item['job_id'] for item in retrieval['candidates']]
        accepted=[item['job']['job_id'] for item in response['recommendations']]
        if accepted != row['accepted_ids'] or ranked[:5] != row['top_5']:
            raise ValueError('Saved predictions differ from saved case log')
        row['metrics']=ranking_metrics(ranked,accepted,judgments[row['case_id']],corpus_ids=corpus)
    summary['metric_version']='jobfit-evaluation-metrics-v1'
    summary['metrics']={split:{variant:aggregate([row for row in rows if row['split']==split and row['variant']==variant],
                                               exhaustive_judgments=panel) for variant in summary['variants']}
                        for split in ('dev','heldout') if any(row['split']==split for row in rows)}
    failures=json.loads((output/'failures.json').read_text())['failures']
    save_json(output/'summary.json',summary)
    save_json(output/'case_metrics.json',{'cases':rows})
    (output/'results.md').write_text(render_summary(summary))
    (output/'failure_report.md').write_text(render_failures(failures))
    if export:
        export=Path(export).resolve()
        source=PROJECT_ROOT/'dataset'
        if export==source or source in export.parents:
            raise ValueError('Export must stay outside the source dataset')
        export.mkdir(parents=True,exist_ok=True)
        save_json(export/'results_summary.json',summary)
        (export/'results.md').write_text(render_summary(summary))
        (export/'failure_report.md').write_text(render_failures(failures))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory',type=Path)
    parser.add_argument('--fixtures',type=Path,default=PROJECT_ROOT/'evaluation')
    parser.add_argument('--export',type=Path)
    args=parser.parse_args()
    rebuild(args.run_directory,fixtures=args.fixtures,export=args.export)
    print('Metrics recomputed from unchanged saved predictions and frozen judgments.')
