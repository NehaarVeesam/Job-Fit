# JobFit

JobFit is a step-by-step Python project for matching a PDF resume to a catalog of synthetic job postings. The planned workflow extracts resume evidence, retrieves relevant jobs, and applies job requirements to decide which roles to recommend.

## Current status

Module 01 provides application settings, data models, and a setup check. Module 02 validates the job catalog and saves the prepared jobs and skill evidence. Module 03 parses PDF resumes, builds evidence-backed candidate profiles, and creates a search query. Module 04 will build the dense job index.

## Setup on Windows

Run these commands in PowerShell from the repository root. Use Python 3.11 and reuse your existing `.venv` if you already created one.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install "pydantic-settings>=2,<3" pymupdf
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

## Prepare the job catalog

From the repository root with the virtual environment active, run:

```powershell
python -m app.dataset.dataset_loader
```

The loader uses Python’s standard library for file handling, duplicate detection, and skill text processing. Pydantic validates job records and serializes them to JSON. The loader keeps the first valid record for each job ID and records invalid and duplicate rows with their source line numbers.

The default output directory is `artifacts/prepared/`. The loader stages all three files before moving them into that directory. Git ignores the generated outputs, so regenerate them after cloning the repository.

| Output | Contents |
| --- | --- |
| `jobs.jsonl` | One validated job record per line. |
| `skill_evidence.jsonl` | One record per listed required or preferred skill, with the job ID and an exact quote from the description. |
| `validation_report.json` | Input, accepted, and rejected counts, the evidence count, and details of any rejected rows. |

With the supplied dataset, expect:

```text
Total rows: 250
Accepted jobs: 250
Rejected rows: 0
Skill evidence records: 1787
Prepared dataset: artifacts\prepared
```

The evidence count is across job postings. A skill required by two jobs contributes two records.

## Prepare a resume

Run the catalog preparation command first to generate the skill vocabulary. Then run the resume CLI with the path to a text-based PDF. Replace the example path and preferences with your own values.

```powershell
python -m app.cli.prepare_resume ".\path\to\resume.pdf" --target-role "Backend Engineer" --location-city "Bengaluru" --work-mode hybrid --output ".\artifacts\resume\prepared_resume.json"
```

The output contains the extracted pages, the candidate profile, the supplied preferences, and the retrieval query. JobFit checks every evidence quote against its cited page. The query combines the target role and extracted skills. Location and work mode stay available as search filters.

You can inspect the query and profile in PowerShell:

```powershell
$prepared = Get-Content -Raw -Encoding UTF8 ".\artifacts\resume\prepared_resume.json" | ConvertFrom-Json
$prepared.retrieval_query
$prepared.candidate_profile.skills | Select-Object -ExpandProperty text
```

Omit `--output` to print the JSON in the terminal. Saved output includes the extracted resume text, so keep it under the Git-ignored `artifacts/` directory.

## Project files

| Path | Purpose |
| --- | --- |
| [`app/config.py`](app/config.py) | Application settings and derived output paths. |
| [`app/schemas.py`](app/schemas.py) | Validates job records, candidate facts, and resume evidence. |
| [`app/cli/check.py`](app/cli/check.py) | Runs the Module 01 setup check. |
| [`app/dataset/dataset_loader.py`](app/dataset/dataset_loader.py) | Validates the catalog, extracts skill evidence, and writes the prepared outputs. |
| [`app/resumes/resume_parser.py`](app/resumes/resume_parser.py) | Extracts PDF text with its original page numbers. |
| [`app/resumes/profile.py`](app/resumes/profile.py) | Builds the candidate profile and retrieval query. |
| [`app/cli/prepare_resume.py`](app/cli/prepare_resume.py) | Runs resume preparation and returns the result as JSON. |
| [`dataset/jobs_synthetic_250.jsonl`](dataset/jobs_synthetic_250.jsonl) | The 250 fictional AI/ML and software postings used for development. |
| [`dataset/DATASET_CARD.md`](dataset/DATASET_CARD.md) | Describes the dataset, built-in test cases, and limitations. |

See the [reimplementation plan](JobFit_Reimplementation_Plan.md) for each module’s deliverables and checkpoints, and the [architecture roadmap](JobFit_Architecture_Learning_Roadmap.md) for the complete design.
