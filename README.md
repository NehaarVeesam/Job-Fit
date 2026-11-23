# JobFit

JobFit is a step-by-step Python project for matching a PDF resume to a catalog of synthetic job postings. The planned workflow extracts resume evidence, retrieves relevant jobs, and applies job requirements to decide which roles to recommend.

## Current status

Module 01 provides application settings, data models, and a setup check. Module 02 validates the job catalog and saves the prepared jobs and skill evidence. Module 03 parses PDF resumes, builds evidence-backed candidate profiles, and creates a search query. Module 04 builds and searches the local dense job index with BGE-M3 and Qdrant. Module 05 adds BM25 and hybrid retrieval with city and work-mode filters. Module 06 reranks the shortlist with a cross-encoder. Module 07 will evaluate job requirements and candidate evidence.

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

## Build and search the vector index

Prepare the job catalog first, then install the dependencies for dense search:

```powershell
python -m pip install langchain-core langchain-huggingface sentence-transformers langchain-qdrant
```

Build the index and run a sample search:

```powershell
python -m app.retrieval.vector_index
python -m app.cli.dense_cli "Backend Engineer Python FastAPI" --top-k 3
```

The first model load downloads BGE-M3. The build embeds each prepared job description as a 1,024-dimensional vector and saves it with the job metadata in the local `jobfit_jobs` collection under `artifacts/qdrant/`. The supplied catalog produces 250 points. Rebuilding updates points by their stable job IDs.

The search command embeds the query with the same model and prints ranked jobs with cosine similarity scores. You can pass the `retrieval_query` from your prepared resume as the search text.

## Compare BM25, dense, and hybrid retrieval

Install BM25 support and prepare the catalog:

```powershell
python -m pip install rank-bm25
python -m app.dataset.dataset_loader
```

Dense and hybrid searches also use the Qdrant index. Rebuild it after changing the prepared job catalog.

```powershell
@'
from app.config import Settings
from app.retrieval.hybrid import retrieve_candidates
from app.schemas import Preferences

count, hits = retrieve_candidates(
    "Backend Engineer Python FastAPI",
    3,
    Settings().prepared_jobs_path,
    mode="hybrid",
    preferences=Preferences(location_city="Bengaluru", work_mode="hybrid"),
)

print(f"Catalog jobs: {count}")
for document, score in hits:
    print(document.metadata["job_id"], document.metadata["title"], round(score, 5))
'@ | python -
```

Use `mode="bm25"` for keyword scores or `mode="dense"` for cosine similarity scores. Hybrid mode combines the two rankings using Reciprocal Rank Fusion. Both searches apply the city and work-mode filters before taking their final results.

## Rerank a shortlist

Run this after preparing the catalog and building the dense index:

```powershell
python -m app.cli.rerank_cli "Backend Engineer Python FastAPI" --mode hybrid --top-k 3 --location-city "Bengaluru" --work-mode hybrid
```

The first reranker load downloads `BAAI/bge-reranker-v2-m3`. The command prints the order from retrieval, then scores each query and job description together and prints the new order. A movement like `3 -> 1` means a job moved from third to first. Both lists contain the same job IDs. The original rank and score are available on each reranked hit.

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
| [`app/retrieval/documents.py`](app/retrieval/documents.py) | Converts validated job records to LangChain documents. |
| [`app/retrieval/embeddings.py`](app/retrieval/embeddings.py) | Configures BGE-M3 for job and query embeddings. |
| [`app/retrieval/vector_index.py`](app/retrieval/vector_index.py) | Builds and verifies the local Qdrant collection. |
| [`app/cli/dense_cli.py`](app/cli/dense_cli.py) | Runs a dense search and prints ranked jobs with similarity scores. |
| [`app/retrieval/dense.py`](app/retrieval/dense.py) | Searches the persistent Qdrant index. |
| [`app/retrieval/keyword_search.py`](app/retrieval/keyword_search.py) | Tokenizes job text and ranks keyword matches with BM25. |
| [`app/retrieval/filters.py`](app/retrieval/filters.py) | Applies city and work-mode preferences to job metadata. |
| [`app/retrieval/hybrid.py`](app/retrieval/hybrid.py) | Exposes all three search modes and combines rankings with RRF. |
| [`app/retrieval/reranking.py`](app/retrieval/reranking.py) | Scores query–job pairs and preserves their original retrieval ranks and scores. |
| [`app/cli/rerank_cli.py`](app/cli/rerank_cli.py) | Prints the shortlist before and after reranking. |
| [`dataset/jobs_synthetic_250.jsonl`](dataset/jobs_synthetic_250.jsonl) | The 250 fictional AI/ML and software postings used for development. |
| [`dataset/DATASET_CARD.md`](dataset/DATASET_CARD.md) | Describes the dataset, built-in test cases, and limitations. |

See the [reimplementation plan](JobFit_Reimplementation_Plan.md) for each module’s deliverables and checkpoints, and the [architecture roadmap](JobFit_Architecture_Learning_Roadmap.md) for the complete design.
