# JobFit - Evidence-Based Resume Matching

JobFit matches a searchable resume PDF to **250 fictional AI and software jobs based in India**. It finds candidate jobs using BGE-M3 vector search, BM25 keyword search, or a combination of both. It then checks skills, experience, education, and preferences. Each recommendation shows evidence from the resume and flags requirements it could not verify. Run JobFit from the command line or through a local FastAPI server.

## Results

### Sample outputs

| Resume | Result |
|---|---|
| [AI engineer](examples/results/ai_match_response.json) | Two recommendations. The resume showed evidence for 4 of 6 skills required by the Generative AI Engineer role and 3 of 6 required by the RAG Engineer role. |
| [Pastry chef resume](examples/results/unrelated_no_match_response.json) | `no_suitable_matches` with an empty recommendation list. |

Both runs used hybrid search without calling Qwen. The match response explains each recommendation using evidence from the resume and job description. [Output provenance](examples/results/metadata.json)

### Held-out evaluation

The benchmark uses **12 synthetic resumes and 204 suitability judgments** across a 17-job panel. An AI assistant labeled each resume-job pair. The table covers the eight held-out resumes. Strict accepted precision counts only recommendations labeled fully suitable.

| Retrieval method | Mean nDCG@5 | Strict accepted precision | Median time |
|---|---:|---:|---:|
| Dense | 0.965 | 0.500 | 0.219 s |
| BM25 | 0.974 | 0.500 | 0.056 s |
| Hybrid | 0.965 | 0.500 | 0.222 s |
| Hybrid + BGE reranker | 0.974 | 0.500 | 1.375 s |

All four methods correctly identified both no-match cases, one in the development split and one in the held-out split. The full-corpus report shows how many retrieved jobs had labels and gives a range for precision where labels were incomplete.

- [Panel results](evaluation/results.md)
- [Failure analysis](evaluation/failure_report.md)
- [Full-corpus diagnostics](evaluation/full_corpus/results.md)
- [Benchmark suite and judgments](evaluation/README.md)

### Reranker development benchmark

In a separate benchmark run on four CPU threads, four of five synthetic profiles each produced 30 candidate jobs. Hybrid search took **0.21 to 0.25 seconds** and reranking added **22.66 to 24.70 seconds**, excluding model loading. Dense search and the reranker both ranked the expected job first in **4 of 4** predefined pairs judged for topic relevance. The fifth profile produced no candidates to rerank. [Benchmark settings and labels](examples/reranker/benchmark.json)

## Run locally

### Setup

The pinned setup was verified on **Linux with Python 3.11** using CPU dependencies. The clone command preserves the LF line endings required to validate the benchmark files.

```bash
git clone --config core.autocrlf=false https://github.com/NehaarVeesam/Job-Fit.git
cd Job-Fit
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-deps --no-build-isolation -e .
if [ ! -f .env ]; then cp .env.example .env; fi
python -m app.check_setup
```

### Match a resume PDF

Use an unlocked PDF with selectable text. For a quick run without downloading an embedding model, use BM25 keyword search.

```bash
python -m app.cli.run_resume "/path/to/resume.pdf" --retrieval-mode bm25
```

To combine vector and keyword search, download BGE-M3 once and use the default hybrid mode.

```bash
python -m app.cli.download_models embeddings
python -m app.cli.run_resume "/path/to/resume.pdf"
```

The command prints the path to `response.json`. The run directory under `artifacts/resume_runs/` also contains extracted resume text and is ignored by Git. Use `--preferences examples/preferences.json` to try the sample filters, or `--help` to see the reranking and Qwen options.

### Run the local API

After downloading BGE-M3, prepare the job dataset, build the vector index, and start FastAPI.

```bash
python -m app.dataset_loader
python -m app.dense_cli index
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

In the interactive API, select `POST /match` and upload your PDF. Set `use_llm` to `false` to run the match locally without calling Qwen. [Interactive API](http://127.0.0.1:8000/docs)

### Reproduce the held-out evaluation

Keep the original benchmark files and their LF line endings. Once BGE-M3 is cached, download the reranker and run the held-out evaluation.

```bash
python -m app.cli.download_models reranker
python -m app.dataset_loader
python -m app.evaluation.runner --split heldout
```

The runner prints the directory under `artifacts/evaluation/` where it saves the metrics and failure report.

## Repository layout

| Folder | Contents |
|---|---|
| `app/` | Corpus preparation, resume parsing, retrieval, matching, CLI, API, and evaluation code |
| `dataset/` | Synthetic job records and [dataset card](dataset/DATASET_CARD.md) |
| `evaluation/` | Frozen fixtures, judgments, and published results |
| `examples/` | Synthetic inputs and exported outputs |
| `python_only/` | Direct retrieval implementation for comparison |
