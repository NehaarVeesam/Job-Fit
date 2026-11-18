# Module 7 — Optional cross-encoder reranking

Module 7 adds a local model that reorders already retrieved jobs. It is optional
and **disabled by default**. It does not replace BGE-M3, BM25, Qdrant, coverage
calculations, eligibility checks, or Qwen explanations.

## Why use a second model?

BGE-M3 encodes the query and job separately. Those vectors make initial retrieval
cheap because job vectors are stored in advance. BM25 offers a complementary
keyword ranking. Module 6 can merge their rankings using RRF.

A cross-encoder instead reads **both texts together**:

```text
Input pair: [compact resume query, one candidate job's text]
Output:     one relevance score
```

It can consider interactions between query and job words directly. It must do a
new forward pass for each pair, so we apply it to the candidate pool (normally
30 jobs), rather than treating it as the initial corpus-wide search engine.
The original candidate pool limits what it can find: a job absent from that pool
cannot be recovered by reranking.

## Updated flow

```text
Resume PDF + preferences
  -> Module 2: structured profile and compact query
  -> Modules 3/6: dense, BM25, or hybrid retrieval
  -> Module 7 (optional): score query/job pairs and reorder the candidate pool
  -> Module 4: unchanged skill coverage, eligibility, and acceptance checks
  -> optional grounded Qwen plans
  -> CLI JSON or FastAPI response
```

The reranker sees the compact skill/target-role query, not full resume pages.
Job text is the same `job_document()` representation used by dense/BM25 search.
Reranking does not use Qwen, call a remote generation service, or modify evidence.

## Model and settings

| Setting | Default |
|---|---|
| Enabled | `false` |
| Model | `BAAI/bge-reranker-v2-m3` |
| Revision | `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` |
| Runtime | Sentence Transformers `CrossEncoder`, PyTorch, local CPU |
| Application input budget | 512 tokens for the complete pair, including special tokens; configurable up to 8,192 |
| Batch size / CPU threads | 2 / 4 |
| Output | One raw logit per pair; identity activation |
| Cache | `artifacts/reranker_cache/` (ignored) |

The BGE reranker has roughly 2.27 GB of weights and supports multilingual
relevance scoring. It supports up to 8,192 pair tokens; our application defaults
to 512 because these job texts are short. The larger CPU model is suitable for
this offline portfolio workflow, with latency measured below. Its scores remain
uncalibrated for resume-job suitability.

Sources: [model card](https://huggingface.co/BAAI/bge-reranker-v2-m3),
[CrossEncoder API](https://www.sbert.net/docs/package_reference/cross_encoder/model.html).
The pinned tokenizer declares an 8,192-token model limit; the loader checks
the configured budget against both tokenizer and architecture limits. The
installed Sentence Transformers API was inspected before implementation.

## Keep the scores separate

Candidates keep their original `relevance` field (`cosine`, `bm25`, or `rrf`).
The additional `reranker_relevance` field is explicitly `cross_encoder`:

```json
{
  "job_id": "syn_job_105",
  "original_rank": 2,
  "rank": 3,
  "relevance": {"score_type": "rrf", "value": 0.031746031746031744},
  "reranker_relevance": {"score_type": "cross_encoder", "value": -0.11705614626407623}
}
```

Higher logits rank first. Negative values are valid: `-3` ranks above `-8`.
Neither positive nor negative logits are hiring probabilities. No new numerical
cross-encoder acceptance threshold is introduced or tuned on these demos.
Existing acceptance remains:

- Dense: original cosine threshold plus coverage and eligibility rules.
- BM25: original positive keyword-score gate plus coverage and eligibility rules.
- Hybrid: separate cosine threshold plus coverage and eligibility rules.

A high reranker score cannot override insufficient coverage or a failed mandatory
condition. All candidates are assessed after ordering, so a rejected top candidate
can be skipped while a lower-ranked acceptable candidate is selected. With the
five-result cap, changing order can change which passing jobs are recommended.

The original ranks and component scores remain in detailed retrieval reports.
The public match/ranked-candidate contracts expose the additional score, and
`applied_filters.reranking_status` identifies applied, disabled, skipped, or
fallback processing. Selected-job detail reuses the cached reranker score when
present; a job outside the candidate pool has no reranker score and does not
trigger new cross-encoder or Qwen calls just for inspection.

## Input budgets and failure behavior

Before inference, `CheckedCrossEncoder` tokenizes every complete pair with
`truncation=False`, counting both texts and special tokens. It refuses any pair
above the configured limit instead of allowing silent truncation.

If any pair is too long, model files are missing, or inference returns invalid
scores, the entire list retains its original order. The report labels this as
`fallback`, records a sanitized error class, and adds a response warning. Token
overflow records the pair lengths and limit. The candidate pool is never
partially reranked or partly dropped. Deadline expiry propagates as a timeout;
it is not disguised as a successful fallback.

Empty/no-query results skip model loading. Disabled reranking loads no additional
model. Normal inference loads only the pinned local snapshot, with remote code
disabled. The API caches the model once per process; batch boundaries check the
request deadline. An in-progress model forward pass cannot be forcibly stopped.

## Run it

Start from the existing prepared dataset, prepared resume, and retrieval index.
The optional model needs a one-time download:

```bash
python scripts/download_reranker.py

# Replace the placeholder with Module 2's prepared_resume.json path.
python -m app.match_cli \
  --prepared-resume PATH_TO_PREPARED_RESUME.json \
  --retrieval-mode hybrid --rerank --no-llm
```

`--no-llm` disables Qwen explanations, not the local reranker. Omit it to enable
normal Qwen explanations after matching. `--no-rerank` explicitly disables
reranking even if the local environment enables it.

To inspect retrieval alone:

```bash
python -m app.search_cli \
  --prepared-resume PATH_TO_PREPARED_RESUME.json \
  --mode hybrid --rerank
```

`--mode compare --rerank` applies the reranker separately to all three base
candidate lists. Base reports retain their usual filenames; additional files
are named `dense_reranked.json`, `bm25_reranked.json`, or `hybrid_reranked.json`
under `artifacts/hybrid_retrieval/<prepared-input-hash>/`.

The matching CLI saves `response.json`, `assessment_report.json`, and
`retrieval_report.json` under `artifacts/matches/<run-hash>/`. To reuse cached
retrieval with `--retrieval-report`, pass an unprocessed base search report;
already processed reports are rejected to prevent ambiguous double reranking.

For the API, update local settings and restart the process:

```dotenv
JOBFIT_RETRIEVAL_MODE=hybrid
JOBFIT_RERANK_ENABLED=true
JOBFIT_RERANK_MODEL=BAAI/bge-reranker-v2-m3
JOBFIT_RERANK_REVISION=953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e
JOBFIT_RERANK_CACHE_DIR=artifacts/reranker_cache
JOBFIT_RERANK_DEVICE=cpu
JOBFIT_RERANK_BATCH_SIZE=2
JOBFIT_RERANK_MAX_TOKENS=512
JOBFIT_RERANK_CPU_THREADS=4
```

The endpoints remain unchanged. No private `.env` values were overwritten by this
module. Defaults/templates specify the new settings.

## Files added or updated

| File | Responsibility |
|---|---|
| `app/reranking.py` | Offline model loader, complete-pair token checks, batched scoring, stable ordering, labeled fallback |
| `scripts/download_reranker.py` | Download only the pinned runtime files |
| `examples/reranker/benchmark.json` | Reproducible settings and four manually labeled topical development comparisons |
| `scripts/verify_reranker.py` | Real-model comparisons, latency observations, pair labels, preservation assertions |
| `tests/test_module7.py` | Budget, score, ordering, deadline, fallback, gate, and API checks |
| `app/config.py`, `.env.example` | Optional model/cache/device/runtime settings |
| `app/schemas.py`, `app/api_schemas.py` | Separate, typed cross-encoder scores |
| `app/match_cli.py`, `app/search_cli.py` | Optional reranking switches and reports |
| `app/api_service.py` | Per-process cache, request integration, selected-job score reuse |
| `app/matching.py`, `app/scoring.py` | Expose scores/status while preserving acceptance rules |

No new dependency was needed: Sentence Transformers/PyTorch were installed in
Module 3. LangChain still handles Documents, embedding integration, Qdrant, and
Qwen. This stage directly wraps `CrossEncoder` to keep pair token checks and all
original scores/evidence visible, instead of replacing the existing contracts
with a generic document compressor.

## Verification and observed tradeoffs

**118 main tests pass**, including 17 Module 7 tests; five Python-only tests are
separate. Controlled test scores check ordering and wiring, not model quality.

Run the separate cached-model benchmark with network socket connections blocked:

```bash
python scripts/verify_reranker.py
# Optional explicit configuration path:
python scripts/verify_reranker.py --config examples/reranker/benchmark.json
```

It checks five public synthetic resume profiles against the same hybrid
candidate pools, then compares dense cosine and cross-encoder ordering on four
predefined topical job pairs. Neither stage contacts Qwen.

Saved summary: `artifacts/verification/module7_live.json`.
Per-profile baseline/reranked retrieval, response, and assessment files:
`artifacts/verification/module7/`.

Observed CPU run (four threads, 30 candidates per nonempty profile):

| Profile | Hybrid retrieval wall time | Added BGE reranker inference | Complete-pair tokens |
|---|---|---|---|
| AI | 0.243 s | 24.699 s | 251–341 |
| Backend | 0.250 s | 22.976 s | 239–310 |
| Analyst | 0.226 s | 22.659 s | 233–289 |
| Fresher | 0.209 s | 23.068 s | 231–328 |
| Unrelated | 0.005 s | Skipped | No pairs |

These are single-run development observations, excluding cold model loading;
they are not robust latency guarantees. All observed pairs fit without
truncation. Artificial 513-token tests verify the overflow path separately.

For AI, the reranked top three are Generative AI Engineer, AI Engineer — Customer
Support Automation, and RAG Engineer. Only Generative AI Engineer and RAG Engineer
pass the recommendation checks, as before; required coverage stays 66.7% and 50%.
The analyst top-five recommendations change, illustrating the interaction of
ordering and the result cap. Changes are not automatically improvements.

Both dense cosine and cross-encoder scores agree with **4/4 easy topical pair
labels**. This shows no gain on that tiny development set. Those labels judge
role/technology relevance, not full resume-job suitability. They were declared in
the benchmark fixture before the run; no score threshold was tuned from them.

Coverage, experience, education, hard-constraint outcomes, and per-job rejection
reasons were asserted unchanged for every candidate in the five-profile run.
The unrelated resume still produces no recommendations.

The earlier MiniLM run is retained locally in
`artifacts/verification/module7_minilm_live.json` and `module7_minilm/` as historical
comparison evidence. Current `module7_live.json` and `module7/` files contain BGE
results. Both models agreed with the same four easy topical labels; those cases
show no quality gain for BGE. CPU timings differ substantially, and the two runs
used different batch sizes (MiniLM 8, BGE 2), so they are configuration-specific
observations rather than a controlled model-speed benchmark.

Module 7 is complete as an optional ordering stage. Leave it off by default until
Module 8's broader suitability evaluation justifies its quality/latency tradeoff.
The subsequent [Module 8 evaluation](module_8_evaluation.md) and
[Module 9 walkthrough](module_9_portfolio.md) document the completed portfolio scope.
