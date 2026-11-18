# Publishing JobFit to GitHub

Run all commands from the project root. These instructions prepare an initial
repository; they do not publish anything automatically.

## Files to commit

| Path | Reason |
|---|---|
| `app/` | Main implementation, including FastAPI |
| `python_only/` | Educational retrieval comparison and its tests |
| `tests/` | Automated module checks |
| `scripts/` | Reproducible model download, demo creation, verification |
| `docs/` | Module explanations, results, and publishing guide |
| `examples/` | Public synthetic text resumes and JSON contracts |
| `dataset/` | Supplied synthetic JSONL, CSV, and provenance card |
| `README.md`, `plan.md` | Project overview, setup, scope, and roadmap |
| `pyproject.toml`, `requirements.lock` | Package definition and pinned dependencies |
| `.gitignore`, `.env.example` | Exclusions and safe configuration template |

The CSV and JSONL dataset representations are intentionally both included.
All generated/prepared datasets and verification responses can be reproduced
locally and remain ignored under `artifacts/`.

## Files to keep local

- `.env` and other environment files, except `.env.example`.
- `.venv/`, Python caches, build output, logs, and test coverage files.
- `artifacts/`, `uploads/`, `models/`, model weights, and database files.
- Uploaded/real resumes and PDFs. Public demo PDFs are regenerated from synthetic
  text using `scripts/create_demo_resumes.py`.
- `.agents/`, `.codex/`, `.aws/`, `.ssh/`, editor settings, and credential files.

Ignore rules do not remove files already tracked in an existing repository.
Review the staged file list before every commit. Never force-add a secret or
publish a real resume merely because a file is accepted by an ignore pattern.

## Initial commit

```bash
cd /home/ubuntu/workdir/JF
git init
git branch -M main
git add .gitignore .env.example README.md plan.md pyproject.toml requirements.lock app python_only tests scripts docs examples dataset
git status --short
git diff --cached --stat
git diff --cached --check
```

Inspect the file list and staged changes. It should contain only the commit list
above, with no `.env`, generated artifacts, model weights, or real resume data.
Then commit:

```bash
git commit -m "Add JobFit baseline: resume matching with LangChain and FastAPI"
```

If Git requests an identity, set your own author name and email for this repo
with `git config user.name` and `git config user.email`, then repeat the commit.
If `git init` reports permission denied for `.git`, run these commands in your
normal writable local checkout; this workspace may protect its metadata folders.

## Push to GitHub

Create an **empty** repository on GitHub, for example `jobfit`. Leave README,
license, and `.gitignore` initialization unchecked because the local project
already has its own files. Choose public or private as you prefer.

Replace `YOUR_USERNAME` and, if needed, the repository name:

```bash
git remote add origin https://github.com/YOUR_USERNAME/jobfit.git
git push -u origin main
```

Authenticate using your configured GitHub credentials when Git prompts.
Do not embed a token in the remote URL or commit it in a file.
These commands assume a new repository with no existing `origin` or commits on
GitHub; inspect an existing setup before changing it, and do not force-push.

No license has been selected automatically. Add an appropriate license if you
want to grant reuse rights, and check the separate terms of model weights and
dependencies when distributing those components.
