# Run your own resume PDF

The same pipeline accepts a real resume; the jobs remain the supplied **250 synthetic postings**.
Use a searchable, unlocked PDF. Image-only scans need OCR, which is not implemented.
The configured defaults allow 5 MiB and 10 pages.

## Run

From the repository root, after the README setup:

```bash
source .venv/bin/activate
python scripts/check_artifacts.py
# Download once if the local embedding cache is absent:
python scripts/download_embeddings.py
python scripts/run_resume.py "/absolute/path/to/your_resume.pdf"
```

Replace the PDF path with your file. No personal resume is bundled. The default
command uses hybrid retrieval, local models and Qdrant, no reranking, no Qwen,
and blocks network connections. Use local Qdrant mode for this offline command.
It validates/prepares the source jobs automatically and builds or verifies/reuses
the matching job index. The input PDF is read without copying or modifying it.

Optional filters use the existing preferences schema. Omit filters initially to
avoid excluding useful jobs accidentally:

```bash
python scripts/run_resume.py "/absolute/path/to/your_resume.pdf" \
  --preferences examples/preferences.json
```

Optional BGE reranking and remote Qwen plans:

```bash
python scripts/download_reranker.py
python scripts/run_resume.py "/absolute/path/to/your_resume.pdf" --rerank
# Requires your existing private Qwen endpoint/key in .env:
python scripts/run_resume.py "/absolute/path/to/your_resume.pdf" --rerank --with-llm
```

`--with-llm` permits network access and sends compact assessments/evidence to
Qwen. Without it, all matching and deterministic explanations stay local.
`--retrieval-mode dense` selects semantic retrieval alone; `bm25` selects keyword
retrieval and deliberately skips query embeddings. Reranker failures can produce
a labeled fallback; check the retrieval report to confirm whether it was applied.

## What happens to the PDF?

1. **Extract:** PyMuPDF reads text, preserving page numbers.
2. **Structure:** existing rule-based extraction identifies supported skills,
   experience and education, with source quotes. Review warnings and the profile;
   it can miss information in unusual layouts or wording.
3. **Embed:** the profile produces a compact retrieval query. BGE-M3 embeds that
   query into a normalized **1024-dimensional** vector. This is the query actually
   used for search, rather than embedding the JSON syntax or every PDF page.
   Job documents are embedded separately and persisted in Qdrant.
4. **Retrieve:** semantic search and BM25 find candidates; RRF combines rankings.
5. **Assess:** optional reranking reorders candidates; existing skill-coverage and
   eligibility checks produce supported strengths and missing skills.
6. **Explain:** deterministic explanations by default; optional constrained Qwen
   plans. Scores remain distinct from skill coverage and hiring probabilities.

If no supported query exists, or no jobs survive filters, embeddings/search may
be skipped and the reason is saved. A completed run can validly return no matches.

## Files saved for each run

The command prints a fresh `artifacts/resume_runs/<timestamp>_<id>/` directory.
It never replaces a previous run.

| File | Contents |
|---|---|
| `events.jsonl` | UTC stage start/completion/failure events and durations |
| `dataset/` | Validated jobs, skill evidence and dataset validation report |
| `extracted_text.txt`, `parsed_resume.json` | Extracted text and page evidence |
| `candidate_profile.json`, `prepared_resume.json` | Structured fields, quotes and warnings |
| `preferences.json`, `query.json` | Applied filters and exact compact query |
| `index_report.json` | Model revision, token/vector checks, index identity and reuse; when needed |
| `query_embedding.json` | Actual query vector, dimension, norm, model revision and query hash; or skip reason |
| `base_retrieval.json`, `retrieval.json` | Rankings before/after optional reranking and scores |
| `assessment.json` | Coverage, eligibility, rejection reasons and generation-call counts |
| `response.json` | Final recommendations, evidence, gaps and explanations |
| `run_manifest.json` | Completion/failure status and checksums of files produced |

Failures retain the stage log and a failed manifest with available partial files;
subsequent-stage files will be absent. Logs omit resume text and exception dumps;
extracted text/profile/output files **contain personal information**. All remain
inside ignored `artifacts/`. Do not add real resume outputs to GitHub. The PDF and
`.env` are already ignored. There is no automatic retention/deletion for CLI runs.

## Existing artifacts and reproduction

`python scripts/check_artifacts.py` saves `artifacts/artifact_inventory.json` and
reports which expected artifact groups are present. This checks presence, not
model completeness or every past run. Historical console logs cannot be recovered
if they were never saved; the new command records its own structured events.

To regenerate other outputs independently:

```bash
python -m app.dataset_loader            # standard curated corpus
python -m app.dense_cli index           # job vectors and index manifest
python scripts/run_demo.py             # synthetic AI and no-match outputs
python scripts/run_resume.py "/absolute/path/to/your_resume.pdf"
# Optional, slower: download reranker first; regenerates four-variant evaluation
python -m app.evaluation.runner
```

Model downloads require network; other commands rely on cached weights. Local
Qdrant storage can be locked by another process: finish its request or stop your
API before running a CLI against the same storage. Older verification reports
remain useful development evidence; they are not reconstructed console logs.
