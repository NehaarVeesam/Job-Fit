# JobFit

JobFit is a step-by-step Python project for matching a PDF resume to a catalog of synthetic job postings. The planned workflow extracts resume evidence, retrieves relevant jobs, and applies job requirements to decide which roles to recommend.

## Current status

Module 01 implements application settings, Pydantic models for jobs and candidate profiles, and a setup check. The next module will validate the full job catalog and write the prepared dataset and validation report.

## Setup on Windows

Run these commands in PowerShell from the repository root. Use Python 3.11 and reuse your existing `.venv` if you already created one.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install "pydantic-settings>=2,<3"
```

The default configuration works with the supplied dataset. To configure local values, copy [`.env.example`](.env.example) to `.env` if you haven't created it yet.

```powershell
if (-not (Test-Path -LiteralPath '.env')) {
    Copy-Item -LiteralPath '.env.example' -Destination '.env'
}
```

| Setting | Purpose |
| --- | --- |
| `JOBFIT_RAW_JOBS_PATH` | Path to the source JSONL file. |
| `JOBFIT_PREPARED_DATA_DIR` | Directory for outputs created by the dataset loader. |
| `JOBFIT_RETRIEVAL_TOP_K` | Default number of jobs to retrieve. |

## Verify the setup

```powershell
python -m app.cli.check
```

The check loads the configuration, validates the first job in the dataset, and creates a sample resume profile with a supporting quote. With the supplied dataset, the output includes:

```text
First job: syn_job_001 | Generative AI Engineer
Sample resume skill: FastAPI | page 1
Setup check passed.
```

## Project files

| Path | Purpose |
| --- | --- |
| [`app/config.py`](app/config.py) | Application settings and derived output paths. |
| [`app/schemas.py`](app/schemas.py) | Validates job records, candidate facts, and resume evidence. |
| [`app/cli/check.py`](app/cli/check.py) | Runs the Module 01 setup check. |
| [`dataset/jobs_synthetic_250.jsonl`](dataset/jobs_synthetic_250.jsonl) | The 250 fictional AI/ML and software postings used for development. |
| [`dataset/DATASET_CARD.md`](dataset/DATASET_CARD.md) | Describes the dataset, built-in test cases, and limitations. |

See the [reimplementation plan](JobFit_Reimplementation_Plan.md) for each module’s deliverables and checkpoints, and the [architecture roadmap](JobFit_Architecture_Learning_Roadmap.md) for the complete design.
