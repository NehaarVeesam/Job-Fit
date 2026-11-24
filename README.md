# JobFit

JobFit matches PDF resumes against 250 synthetic AI/ML and software job postings. It extracts skills and evidence from the resume, retrieves a shortlist, and applies eligibility and skill coverage rules. The CLI and FastAPI return recommendations, gaps, and explanations grounded in the source text.

JobFit supports BM25 keyword search, BGE-M3 dense search with local Qdrant, and hybrid search using Reciprocal Rank Fusion. A BGE cross-encoder can rerank the shortlist, and a local Qwen model can select evidence for explanations or review skill gaps.

## Setup

Install Python 3.11 and run these commands from the repository root.

### Windows (PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Linux (Bash)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### Install dependencies and prepare the catalog

Run the same commands in either shell after activating `.venv`.

```sh
python -m pip install "pydantic-settings>=2,<3" pymupdf rank-bm25 langchain-core langchain-huggingface langchain-qdrant langchain-openai sentence-transformers fastapi uvicorn python-multipart
python -m app.cli.check
python -m app.dataset.dataset_loader
```

The catalog loader should accept 250 jobs, reject zero rows, and write 1,787 skill-evidence records under `artifacts/prepared/`. The default settings work with the included catalog. To customize them, copy [`.env.example`](.env.example) to `.env` and edit the copy.

Build the local Qdrant index before using dense or hybrid matching. The first index build downloads BGE-M3. The first reranked run downloads the BGE reranker.

```sh
python -m app.retrieval.vector_index
```

## Run JobFit

### CLI

Replace `path/to/resume.pdf` with a text-based PDF. BM25 runs once the catalog is prepared. Hybrid also uses the local index.

```sh
python -m app.cli.run_resume "path/to/resume.pdf" --mode bm25 --top-k 3
python -m app.cli.run_resume "path/to/resume.pdf" --mode hybrid --top-k 3 --rerank --output "artifacts/match/result.json"
```

Use `--target-role` to guide retrieval and `--location-city` or `--work-mode` to filter results. The CLI prints JSON by default. Use a new filename when saving with `--output`.

### FastAPI

```sh
python -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Open [Swagger UI](http://127.0.0.1:8000/docs), choose `POST /match`, and upload a PDF in the `resume` field. Enter this JSON in `match_request` to run BM25.

```json
{"options": {"mode": "bm25", "top_k": 3}}
```

`GET /health` checks that the server is responding. `GET /ready?mode=bm25` checks for the prepared catalog files.

### Optional Qwen explanations

With Ollama installed and running locally, download the model and run the CLI with `--use-qwen`.

```sh
ollama pull qwen2.5:3b
python -m app.cli.run_resume "path/to/resume.pdf" --mode bm25 --top-k 3 --use-qwen
```

For the API, set `"use_qwen": true` inside `options`. Qwen selects evidence for the explanation, and JobFit renders the source quotes. The separate `--review-semantic-evidence` option suggests evidence for uncovered skills. These suggestions do not change recommendations.

## Test results

Six [synthetic resumes](evaluation/full_catalog_resumes.json) ran against all 250 job postings with BM25, dense, hybrid, and hybrid with reranking. The 24 runs returned 54 unique resume–job pairs. Each pair was graded once in the shared [ground truth](evaluation/full_catalog_judgments.json). [Per-resume outputs and scoring details](app/evaluation/results.md#full-catalog-benchmark)

| Variant | nDCG@5 | Known suitable retrieved | Strict recommendation precision | Citation errors |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 0.959 | 7/7 | 7/13 | 0 |
| Dense | 0.683 | 6/7 | 6/7 | 0 |
| Hybrid | 0.894 | 7/7 | 7/11 | 0 |
| Hybrid + reranker | 0.934 | 7/7 | 7/11 | 0 |

Each run shortlists up to five jobs. JobFit checks those jobs for eligibility and skill coverage to produce its final recommendations.

- **nDCG@5** measures how closely the shortlist order matches the ideal order from the ground truth.
- **Known suitable retrieved** counts fully suitable jobs in the shortlist out of the seven fully suitable pairs in the ground truth.
- **Strict recommendation precision** counts fully suitable jobs among the final recommendations.

Reranking improved hybrid's nDCG@5 from 0.894 to 0.934. Both retrieved all seven fully suitable ground-truth pairs.

After preparing the catalog and building the Qdrant index, run:

```sh
python -m app.evaluation.full_catalog collect
python -m app.evaluation.full_catalog score
```

## Project structure

- **Data and resumes** · [Synthetic job catalog](dataset/DATASET_CARD.md), [catalog preparation](app/dataset/dataset_loader.py), and [resume parsing and profiles](app/resumes/).
- **Retrieval and matching** · [Search and reranking](app/retrieval/) and the [matching and explanation pipeline](app/matching/pipeline.py).
- **Entry points** · [CLI](app/cli/run_resume.py) and [FastAPI](app/api/main.py).
- **Evaluation** · [Runners](app/evaluation/) and [synthetic fixtures](evaluation/).
