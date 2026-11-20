# JobFit evaluation fixtures and results

This is a **controlled 17-job panel**, selected from the unchanged 250-job
synthetic corpus. It has 12 new synthetic text resumes (4 development, 8
held-out-for-this-suite) and 204 explicit graded pair judgments with rationales.

The AI assistant authored the resumes and labels before the first Module 8 model
run. These are not independent human expert judgments. No held-out labels or
matching/retrieval settings were adjusted after inspecting outputs.

- [Suite and frozen settings](suite.json)
- [Judgments](judgments.jsonl): 2 = suitable, 1 = partial/conditional, 0 = unsuitable.
- [Panel results](results.md) and [machine-readable summary](results_summary.json).
- [Panel failure analysis](failure_report.md).
- [Full 250-job diagnostics](full_corpus/results.md): incomplete judgment coverage,
  precision bounds, and undefined point metrics instead of fabricated negatives.

Run `python -m app.evaluation.runner` from the project root after dataset/model
preparation. It uses cached models, blocks network calls by default, and prints
its local output path. Add `--qwen-probe` only for the optional bounded live probe.

Raw records remain under ignored `artifacts/evaluation/`. The summary snapshots
here contain no credentials or real resume data. Do not interpret their metrics
as real-world hiring accuracy, full-corpus accuracy, or independent validation.
