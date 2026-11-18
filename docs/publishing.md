# GitHub publication policy

The final portfolio package contains source, public synthetic inputs, curated
outputs, and reproducible evaluation evidence. Generated raw runtime artifacts
remain local.

## Included in Git

| Path | Published content |
|---|---|
| `app/`, `python_only/` | Pipeline, API, evaluation tools, and educational comparison |
| `tests/`, `scripts/` | Automated checks, downloads, repeatable demo, verification |
| `docs/`, `README.md`, `plan.md` | Architecture, setup, module guides, scope and limits |
| `dataset/` | Original fictional 250-job JSONL/CSV and dataset card |
| `examples/resumes/` | Public synthetic text resumes |
| `examples/*.json` | Hand-authored contract examples |
| `examples/results/` | Two actual typed synthetic demo responses, metadata/checksums |
| `evaluation/suite.json`, `judgments.jsonl`, `resumes/` | Frozen settings, 204 assistant-authored labels, 12 synthetic text resumes |
| `evaluation/results*`, `failure_report.md`, `full_corpus/` | Sanitized scoped summaries, failures, and incomplete-label bounds |
| `.env.example`, `.gitignore`, `pyproject.toml`, `requirements.lock` | Safe template, exclusions, package and dependency pins |

The CSV and JSONL are two representations of the same 250 jobs. JSONL is the
application input. Source files remain unchanged. The 17-job evaluation panel is
a derived selection with explicit IDs, not a replacement for the input dataset.

The published response examples were validated as `MatchResponse`, use only the
public synthetic resumes/jobs, and contain no private configuration or complete
resume pages. Evaluation snapshots disclose their assistant-authored labels,
controlled-panel scope, source/suite hashes, and limits. They are not independent
human validation or real-world accuracy claims.

## Excluded from Git

- `.env`, other environment files except `.env.example`, credentials, key files.
- `.venv/`, caches, build output, logs, and local editor/agent/cloud settings.
- All `artifacts/`, including derived corpora, parsed pages/profiles, raw runs,
  temporary uploads, PDFs, vectors, databases, and downloaded model weights.
- Real candidate resumes or other private user data in any format.

Curated responses in `examples/results/` and summary snapshots in `evaluation/`
are deliberate reviewed exports. Do not force-add their raw source artifact
folders. `.gitignore` is an exclusion aid; review new text/JSON files too.

## Recreate reviewed outputs

```bash
python scripts/run_demo.py
python -m app.dataset_loader
python -m app.evaluation.runner
```

The demo writes to `artifacts/portfolio_demo/`. Only its public synthetic typed
responses were exported to `examples/results/`. Demo generation is deterministic
in behavior under the documented defaults; regenerated PDF checksums and measured
timings can vary. Input text and source dataset checksums identify the fixtures.

To recompute/export an evaluation summary from a saved run:

```bash
python -m app.evaluation.summarize PATH_TO_RUN_DIRECTORY --export evaluation
```

The exporter validates suite identity and saved prediction consistency. Full-corpus
summaries belong under `evaluation/full_corpus/`; they retain missing-judgment
coverage and undefined point metrics.

## Review an update before committing

This repository already has `main` and `origin` configured. Review an explicit
file list for the intended change, then inspect:

```bash
git status --short
git diff --cached --stat
git diff --cached --check
```

Check that no private files, model weights, PDFs or raw artifacts are staged.
Commit a description of the actual changes and push with `git push origin main`.
Do not embed authentication tokens in the remote URL, files, or commit message.
Ignore rules do not remove previously tracked files, so inspect the tracked list
as well as new additions.

No project license has been selected automatically. Keep claims about code reuse
and model/dependency distribution consistent with their applicable licenses.
