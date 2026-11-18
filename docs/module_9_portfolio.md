# Module 9 — Portfolio walkthrough and project completion

JobFit's planned offline portfolio scope is complete: Modules 0–9, including a
repeatable command-line demo, API, module guides, synthetic examples, and scoped
evaluation. The UI and hosted deployment were optional and are outside this
completed portfolio deliverable.

## Start here

For a quick review without downloading models:

1. Read the architecture and scope in the [README](../README.md).
2. Inspect the [actual AI demo response](../examples/results/ai_match_response.json)
   and [actual no-match response](../examples/results/unrelated_no_match_response.json).
3. Read the [evaluation table](../evaluation/results.md) and
   [failure analysis](../evaluation/failure_report.md).
4. Follow [Module 3](module_3_langchain.md) and the
   [Python-only comparison](../python_only/README.md) to understand LangChain's role.

The JSON files directly inside `examples/` are older, hand-authored contract
examples. The JSON files inside `examples/results/` are actual pipeline outputs
from public synthetic resumes, using all 250 source jobs. They are distinct from
the 17-job evaluation panel.

## Reproduce the demo

Use Linux and Python 3.11 for the verified environment. From a fresh checkout:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-deps --no-build-isolation -e .
cp .env.example .env
python -m app.check_setup
```

Copy the configuration template only on initial setup. Its placeholder Qwen
endpoint is sufficient for validation; the default demo makes no Qwen calls.
Download the pinned embedding model once:

```bash
python scripts/download_embeddings.py
python scripts/run_demo.py
```

The demo prepares all 250 jobs in a derived local folder, creates PDFs from the
public synthetic text resumes, extracts profiles, builds or reuses the Qdrant
index, runs hybrid retrieval, checks coverage/conditions, and saves match and
no-match responses. Default demo execution blocks network socket connections;
model files must already be cached. Run it with local Qdrant, one process at a
time, after finishing other CLI/index operations.

Outputs:

```text
artifacts/portfolio_demo/
  dataset/             Validated derived corpus; source dataset stays unchanged
  ai/
    prepared_resume.json
    retrieval.json
    assessment.json
    response.json
  unrelated/
    prepared_resume.json
    retrieval.json
    assessment.json
    response.json
  summary.json
```

Expected baseline AI result under the supplied preferences/default gates:

| Job | Required evidence | Missing required skills |
|---|---|---|
| Generative AI Engineer (`syn_job_001`) | 4/6 = 66.7% | Vector databases, Prompt engineering |
| RAG Engineer (`syn_job_105`) | 3/6 = 50% | Vector databases, Hybrid search, LLM evaluation |

Both have met experience/education requirements in this example. The unrelated
pastry-chef resume produces `no_suitable_matches`, an empty recommendation list,
and no generation calls. The default explanations are rendered from evidence
without Qwen. These examples demonstrate behavior, not held-out quality scores.

## Optional BGE reranking and Qwen plans

```bash
python scripts/download_reranker.py
python scripts/run_demo.py --profile ai --rerank
```

Reranking uses the pinned `BAAI/bge-reranker-v2-m3` model. Its scores reorder the
candidate pool while original retrieval scores still govern relevance acceptance.
The default complete-pair budget is 512 tokens, configurable up to 8,192.
In the recorded Module 7 CPU run, a 30-job pool took roughly 22.7–24.7 additional
seconds. Short, filtered evaluation cases take less time; these are different
workloads, not contradictory measurements.

For Qwen, configure `QWEN_BASE_URL`, `QWEN_API_KEY`, and the model ID in your local
`.env`, then explicitly enable it:

```bash
python scripts/run_demo.py --profile ai --with-llm
```

Canonical skill assessments go to the remote endpoint. Raw resume pages and job
company prose are excluded from that prompt. Provider/validation failures have a
labeled deterministic fallback. Do not commit your endpoint credentials.

The API remains available with `python -m uvicorn app.main:app --host 127.0.0.1
--port 8000 --workers 1 --no-access-log`; see [Module 5](module_5_api.md).

## Explain the architecture in an interview

> JobFit is a resume-to-job matching pipeline over 250 fictional India AI/software
> postings. It extracts evidence-backed candidate facts from searchable PDFs,
> retrieves using BGE-M3 and BM25 with rank fusion, optionally reranks with a BGE
> cross-encoder, and applies explicit skill-coverage and eligibility checks.
> LangChain connects documents, embeddings, Qdrant, and structured Qwen plans;
> Python validates selections and renders explanations with source quotes.
> FastAPI exposes the pipeline. I also implemented direct Python retrieval for
> comparison and a reproducible, scoped synthetic evaluation with failure analysis.

Defensible claims:

- Implemented dense, keyword, hybrid, and optional cross-encoder ranking.
- Preserved explicit unknown conditions and checked known mandatory failures.
- Used separate retrieval scores and evidence coverage; neither is a hiring probability.
- Compared four methods on a fully labeled, controlled 17-job panel with 12
  assistant-authored resumes and 204 suitability judgments.
- Reported partial/conditional recommendations and missing full-corpus judgments
  instead of presenting passed schema tests as recommendation accuracy.

The held-out panel's strict accepted precision is 50%: five fully suitable and
five partial/conditional accepted results, with no accepted grade-0 jobs.
Mean nDCG@5 is 0.965 for dense/hybrid and 0.974 for BM25/reranked hybrid; the small,
assistant-authored benchmark does not establish general model superiority.

## Published data and reproducibility boundaries

[Publishing policy](publishing.md) lists the exact categories. Public files
include original synthetic job data, public synthetic text resumes, typed demo
responses, frozen judgments/settings, aggregate results, and failure analysis.
Raw artifacts, PDFs, vector indexes, model weights, private configuration, and
real uploaded resumes remain local and ignored.

Model weights are downloaded from their upstream repositories. No project
license was selected automatically; do not describe the repository as granting
reuse rights beyond an applicable explicit license.

## Completion and future extensions

The final checks cover 136 main tests, five Python-only tests, configuration and
dependency consistency, a real cached-model match/no-match demo, and publication
review. The tests also passed in a clean source copy with template configuration
and no runtime artifacts, using the already installed pinned dependencies. Module 8 additionally records 829 scoped checks and a one-call live Qwen
probe with 14 rendered-support checks. Check totals describe different scopes;
they are not hundreds of independent suitability judgments.

Possible future work is broader human-reviewed full-corpus evaluation, better
critical-framework/seniority handling, OCR/layout support, and optional UI or
hosting. These are extensions to the completed offline educational project.
