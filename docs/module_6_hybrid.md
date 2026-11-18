# Module 6 — BM25 and hybrid retrieval

Module 6 adds a second way to search jobs: exact keyword retrieval with BM25.
It also combines that ranking with the existing BGE-M3 ranking using reciprocal
rank fusion (RRF). Dense search remains the default, so existing commands keep
their baseline behavior.

## The problem this solves

A vector search compares the meaning represented by embeddings. It can retrieve
related jobs even when their words differ. Keyword search rewards shared terms,
including distinctive technology names. These signals can disagree, and neither
is automatically the better ranking for every resume.

For example, an AI resume may mention `Python`, `LangChain`, `RAG`, and `FastAPI`.
Dense retrieval may put a broadly similar lead AI role near the top. BM25 can
raise a job whose description explicitly contains several of those terms.
Hybrid retrieval gives both rankings a vote, before the existing evidence and
eligibility checks decide which candidates can become recommendations.

## Inputs and outputs

Inputs are unchanged:

- Module 1's validated `jobs.jsonl`, `skill_evidence.jsonl`, and provenance report.
- Module 2's `prepared_resume.json`, including the compact query and preferences.
- For dense/hybrid search, Module 3's local BGE-M3 model and matching Qdrant index.

BM25 uses the same `job_document()` text as dense retrieval: title, role family,
required/preferred skills, and description. Job IDs and original records are
preserved. Both branches use the same location/work-mode/unknown-metadata filter.
The target role remains a query signal, rather than a new mandatory filter.

The main output is a candidate report. Each hybrid candidate includes its RRF
score, separate cosine score, BM25 score when it is in that branch's window,
source ranks, and each branch's contribution. Reports record the corpus,
query hash, preferences, tokenizer version, aliases, BM25 parameters, and windows.
The matching pipeline still produces the existing validated response contract.

## BM25, step by step

1. Filter jobs using the existing preference rules.
2. Tokenize the eligible job texts and query using the same tokenizer.
3. Count each term's frequency in each job and the number of jobs containing it.
4. Score query-term overlap, accounting for term rarity, repeated-term saturation,
   and document length.
5. Return positive-scoring jobs, highest score first; ties use job ID.

This implementation uses:

```text
IDF(term) = ln(1 + (N - df(term) + 0.5) / (df(term) + 0.5))

BM25(job, query) = sum over unique query terms of:
    IDF(term) * tf(term, job) * (k1 + 1)
    / (tf(term, job) + k1 * (1 - b + b * job_length / average_length))
```

`N` is the eligible job count; `df` counts jobs containing the term; `tf` counts
occurrences within a job. The positive-IDF variant avoids negative keyword
scores. Defaults: `k1=1.5`, `b=0.75`. Query terms are deduplicated so repeating a
skill in the query does not give it extra votes. Zero-overlap jobs are omitted.
Statistics are rebuilt over the eligible corpus on each search, which is small
here (at most 250 jobs). There is no additional persisted keyword database.

## Technical tokens and aliases

Ordinary word splitting could turn `C++`, `C#`, and `C` into the same token. Our
tokenizer keeps those distinct and preserves `.NET`, `Node.js`, `ASP.NET`, and
`CI/CD`. `ASP.NET Core` becomes two tokens: `asp.net` and `core`.
It lowercases terms and applies only this explicit alias table:

| Written alias | Canonical token |
|---|---|
| `nodejs` | `node.js` |
| `dotnet` | `.net` |
| `k8s` | `kubernetes` |
| `postgres` | `postgresql` |
| `sklearn` | `scikit-learn` |

Flask/FastAPI and Java/JavaScript remain different. This is retrieval-only
normalization: it does not broaden Module 2's extracted skill evidence or give
extra skill coverage. If profile extraction never includes a term in the query,
retrieval aliases cannot recover that term from the raw PDF.

## How hybrid search combines the results

The defaults retrieve the top 30 dense candidates and top 30 positive BM25
candidates. RRF merges their union by source job ID:

```text
RRF(job) = 1 / (60 + dense_rank) + 1 / (60 + bm25_rank)
```

A branch contributes nothing if the job is absent from its top window. Ranks
start at 1. The fused list is sorted by total score, with job ID breaking ties,
and truncated to the requested final candidate count (30 by default).

Example:

| Job | Dense rank | BM25 rank | RRF score |
|---|---|---|---|
| A | 1 | 3 | 1/61 + 1/63 = 0.032266 |
| B | 2 | 1 | 1/62 + 1/61 = 0.032522 |
| C | 3 | absent | 1/63 = 0.015873 |

B wins this example because both retrievers rank it highly. RRF combines ranks,
rather than adding raw BM25 and cosine scores, whose scales differ. The constant
60 controls how much rank differences matter; it is unrelated to the number of
retrieved jobs.

For the formula, see the [official RRF documentation](https://learn.microsoft.com/azure/search/hybrid-search-ranking).
For BM25 background, see [Robertson and Zaragoza's paper](https://www.nowpublishers.com/article/DownloadEBook/INR-019).

## Retrieval ranking versus recommendation acceptance

| Mode | Ordering score | Relevance acceptance check |
|---|---|---|
| Dense | Cosine | Existing cosine threshold (default 0.55) |
| BM25 | BM25 | Positive keyword score; cosine threshold explicitly not applied |
| Hybrid | RRF | Separate cosine score must still pass the existing threshold |

All modes still check coverage, minimum evidenced required skills, explicit
preferences, known requirement failures, and the configured unknown policy.
Qwen still only runs after acceptance, if enabled. It does not calculate BM25,
RRF, coverage, or eligibility.

For this small corpus, hybrid search obtains cosine scores for **all eligible
jobs**, then uses only the configured top dense window in RRF. Keeping the other
cosine scores allows a BM25-only candidate to face the same semantic gate.
This is intentional teaching-project overhead, not a scalable retrieval design.
At larger scale, score only the merged candidate pool separately.

BM25-only matching is an explicitly different relevance policy. More returned
recommendations do not demonstrate higher quality. The response labels this
policy and marks the cosine threshold `not_applied`. Score types remain named
`cosine`, `bm25`, and `rrf`; none is a hiring probability.

## Run it

Complete Modules 1–3 first for dense/hybrid. BM25 search needs only the prepared
corpus and prepared resume. Replace the input placeholder with a real local path:

```bash
# Keyword retrieval only; no embedding model, Qdrant index, or Qwen needed.
python -m app.search_cli --prepared-resume PATH_TO_PREPARED_RESUME.json --mode bm25

# Three-mode comparison, using the same query, corpus, preferences, and output k.
python -m app.search_cli --prepared-resume PATH_TO_PREPARED_RESUME.json --mode compare --k 30

# Hybrid retrieval followed by the existing checks and deterministic explanations.
python -m app.match_cli --prepared-resume PATH_TO_PREPARED_RESUME.json --retrieval-mode hybrid --no-llm

# For Qwen explanations, omit --no-llm.
```

Search/comparison reports are saved to
`artifacts/hybrid_retrieval/<prepared-input-hash>/{dense,bm25,hybrid,comparison}.json`.
Repeated runs with the same input overwrite those mode reports. Matching results
remain under `artifacts/matches/`. Cached reports can be passed to `match_cli`
with `--retrieval-report`; choose the same mode. Corpus, lexical configuration,
and dense manifest/model identities are checked before reuse.

`--k` controls final output count. The hybrid branch windows remain controlled by
`JOBFIT_CANDIDATE_COUNT` and `JOBFIT_BM25_CANDIDATE_COUNT`, rather than `--k`.
Thus requesting a final k larger than the union cannot invent more candidates.

For the API, set this locally and restart the API process:

```dotenv
JOBFIT_RETRIEVAL_MODE=hybrid
JOBFIT_CANDIDATE_COUNT=30
JOBFIT_BM25_CANDIDATE_COUNT=30
JOBFIT_BM25_K1=1.5
JOBFIT_BM25_B=0.75
JOBFIT_RRF_CONSTANT=60
```

No endpoint changes are required. `POST /match` uses the configured mode. Selected
job inspection preserves its retrieval score if cached; a job outside the hybrid
candidate pool is inspected with a separately computed cosine score, since it has
no fused rank. BM25 job inspection uses its lexical score, or zero for no overlap.
The API response's `score_type` identifies the score being shown.

## Files added or changed

| File | Purpose |
|---|---|
| `app/keyword_search.py` | Technical tokenizer, explicit aliases, BM25 arithmetic |
| `app/hybrid_retrieval.py` | Shared retrieval entry point, lexical identity, RRF, rank traces |
| `app/search_cli.py` | Search any mode and save three-mode comparisons |
| `app/config.py`, `.env.example` | Mode, windows, BM25, and RRF settings |
| `app/match_cli.py` | Choose a mode and validate cached report provenance |
| `app/matching.py`, `app/scoring.py` | Mode-aware relevance policy; coverage/conditions still shared |
| `app/api_service.py` | Configured mode in uploads and selected-job inspection |
| `tests/test_module6.py` | Arithmetic, technical terms, filters, provenance, gates, API checks |
| `scripts/verify_hybrid.py` | Actual local BGE-M3 comparison on five synthetic demos |

LangChain still provides the Documents, embedding adapter, and Qdrant integration
from Module 3. BM25 and RRF are deliberately small Python implementations so their
calculations are visible. No new dependency is needed, and the separate
`python_only/` Module 3 comparison remains unchanged.

## Verification and observed results

At this checkpoint, **101 main tests pass**, including **16 Module 6 tests**;
the five Python-only tests are separate. Checks cover a hand-calculated BM25
score, exact token distinctions/aliases, RRF arithmetic/rank reconstruction,
deduplication, matching query/corpus/filter provenance, no-query/no-match cases,
known failures, cosine rejection despite high fusion rank, and API upload/detail
in both added modes. Test vectors check plumbing, not semantic quality.

The real-model check uses cached BGE-M3, the existing Qdrant index, and five public
synthetic text resumes, with socket connections blocked and zero Qwen calls:

```bash
python scripts/verify_hybrid.py
```

Saved summary: `artifacts/verification/module6_live.json`. Detailed retrieval,
response, and assessment reports: `artifacts/verification/module6/`.

For the AI demo, all modes searched the same 41 eligible jobs and returned 30
candidates. The dense/BM25 top-30 lists overlapped in 24 jobs. Top retrieval IDs:

| Rank | Dense | BM25 | Hybrid |
|---|---|---|---|
| 1 | 001 | 001 | 001 |
| 2 | 028 | 005 | 105 |
| 3 | 105 | 105 | 028 |
| 4 | 127 | 042 | 005 |
| 5 | 180 | 028 | 127 |

IDs above abbreviate `syn_job_...`. RAG Engineer (105) moves from third to second
in hybrid retrieval. All three modes ultimately recommend 001 and 105 for this
demo, with unchanged 66.7% and 50% required coverage. Other demos have different
rankings and BM25-only recommendation counts. The unrelated resume produces no
recommendations in every mode.

These are development diagnostics, not a judgment that hybrid is more accurate.
Recency remains controlled by Module 1's chosen corpus; retrieval adds no live
job collection or automatic stale-job exclusion. Held-out suitability judgments,
quality metrics, and robust latency benchmarking belong to Module 8.

Module 6 is complete. Module 7 adds optional cross-encoder reranking; Modules 8–9
cover evaluation and a reproducible demo.
