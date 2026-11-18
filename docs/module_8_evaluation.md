# Module 8 — Judgment-based evaluation

Module 8 measures retrieval ordering, accepted recommendations, no-match
behavior, evidence support, requirement checks, and warm text-pipeline latency.
It records disagreements with a stated suitability rubric and preserves the
inputs/settings that produced them.

## What evaluation adds

An automated test can show that 4/6 skills produces 66.7% coverage. Evaluation
asks whether the resulting job is suitable for the stated candidate. A React
candidate might meet four requirements of an Angular job and pass a numerical
coverage gate, while a suitability reviewer still considers that an important
gap. We now record such disagreements instead of treating a successful pipeline
run as evidence of recommendation accuracy.

## Inputs, scope, and labels

The application still uses all **250 synthetic jobs** by default. This first
fully judged benchmark uses an isolated **17-job panel** selected from those
unchanged source records. Its vector index has a distinct identity and local
storage under `artifacts/evaluation/qdrant/`.

There are **12 newly authored synthetic text resumes**:

- Four development cases: AI, junior Java, analyst, unrelated retail work.
- Eight held-out-for-this-suite cases: different city, MERN/MEAN distinction,
  graduation restriction, seniority target, unknown work mode, unknown advertised
  experience, Unity/C# versus Unreal/C++, and an Android posting containing an
  injection instruction.

Every resume/panel-job pair has an explicit judgment: **12 × 17 = 204 labels**.
Each row has a rationale, covering core technologies, experience, preferences,
and mandatory conditions. Labels were authored before the first Module 8 model
run; they were not calculated from application scores or coverage thresholds.

| Grade | Meaning | Strictly relevant? |
|---|---|---|
| 2 | Suitable on the stated core requirements and conditions | Yes |
| 1 | Partial or conditional: important gap, unresolved condition, or target-level mismatch | No |
| 0 | Unsuitable: wrong core stack, failed condition, insufficient experience, or preference mismatch | No |

**The labels and resumes are assistant-authored development fixtures.** They are
not independent human expert judgments or externally collected test data. The
held-out split is held out from tuning within this suite, not evidence of broad
out-of-distribution generalization.

This controlled panel makes exhaustive judgments manageable. It does not provide
an accuracy estimate for all 250 jobs, real vacancies, or arbitrary resumes.

## Freeze and traceability

`evaluation/suite.json` records:

- Development/held-out assignments and expected no-match outcomes.
- Source dataset checksum, resume checksums, and the explicit panel IDs.
- Fixed retrieval windows, aliases/tokenizer implementation, acceptance settings,
  model revisions, CPU settings, and token budget.
- Fingerprints of the pipeline implementation and dependency lock file.
- Independent expected coverage counts and requirement outcomes for selected pairs.

The loader rejects changed source/resume files, incomplete/duplicate labels,
exact resume duplication across cases, unsafe paths, contradictory no-match
expectations, and changes to the frozen pipeline implementation.

A run records a combined suite/judgment hash, package versions, hardware, model
load times, predictions, timings, checks, and failures. No thresholds, labels,
aliases, or retrieval settings were tuned after inspecting held-out results.
Future improvements require development work and a reviewed/versioned future
suite; do not silently adjust this test to improve a score.

## Compared variants

1. Dense BGE-M3 retrieval.
2. BM25 retrieval.
3. Hybrid retrieval using RRF.
4. Hybrid retrieval followed by BGE cross-encoder reranking.

Each uses the same query, job scope, and preference filters. The paired reranker
comparison reuses the same base hybrid candidate list. Existing relevance gates
remain as implemented in Modules 4/6: BM25-only has a keyword gate, while dense
and hybrid retain the cosine gate. Accepted-result comparisons therefore involve
both ordering and their declared acceptance policies.

## Metrics, with examples

**Precision@5:** number of grade-2 jobs in the first five candidates, divided by
five. Missing positions count as non-relevant. One suitable job and four empty
positions means 1/5 = 0.2, rather than 1.0.

Many panel cases have only one suitable job after explicit city/work-mode
preferences. We report the panel's ideal P@5 ceiling alongside the measured
value. The held-out average ceiling is 0.125; all methods reach it on this panel.
This is a property of the small test collection, not a general retrieval score.

**Accepted precision:** grade-2 accepted jobs divided by all accepted jobs,
micro-aggregated over a split. Zero accepted jobs produces `null`/`N/A`, never a
perfect precision score. Mean returned count is reported beside it.

**Recall@5:** recovered grade-2 jobs divided by all grade-2 jobs in an exhaustively
judged corpus. It is undefined when no grade-2 jobs exist or the corpus is only
partially labeled.

**nDCG@5:** discounted graded gain, using `2^grade - 1`, divided by ideal discounted
gain. It rewards placing suitable jobs above partial/irrelevant jobs. Cases with
zero ideal gain are undefined and excluded from its mean, with the number of
defined cases recorded. This fixture has seven defined held-out cases.

**No-match metrics:** no-match is the positive class. We record TP/FP/FN/TN,
precision, recall, and accuracy. Partial/conditional opportunities prevent a
no-match expectation even though they do not count as grade-2 strict matches.
Always returning nothing is penalized on cases that do have an opportunity.

**Latency:** median and linearly interpolated p95 of warm text-pipeline stage
sums across distinct cases. Preparation, retrieval, optional reranking, and
assessment are included. Model loading, index creation, persistence, PDF parsing,
HTTP handling, and Qwen probe latency are excluded. Each base retrieval is timed
once and reused for the paired reranker comparison. These are small, mixed-case
observations, not a repeated or concurrent production benchmark.

References: [ranked retrieval evaluation](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html),
[discounted cumulative gain](https://scikit-learn.org/stable/modules/model_evaluation.html#ranking-metrics).

## Missing judgments and the full corpus

The runner also supports all 250 jobs. Only the panel's pairs are currently
judged, so a new result outside the panel is **unjudged**, not automatically
irrelevant. The report shows judgment coverage and precision bounds:

```text
P@5 lower = known grade-2 jobs / 5
P@5 upper = (known grade-2 jobs + unjudged returned jobs) / 5
```

Accepted precision has analogous bounds. A point metric is `null` when its
returned positions include missing judgments. Recall/nDCG require exhaustive
corpus judgments. Full-corpus no-match ground truth is not inferred from the
panel's expectations: a candidate may have an opportunity outside the panel.

The saved full-corpus diagnostic run covers eight held-out resumes and three
retrieval variants. Its point precision is undefined for most cases, so we do
not publish a made-up full-corpus accuracy figure. See
[the coverage/bounds report](../evaluation/full_corpus/results.md).

## Evidence and edge-case audits

The panel run passed **829/829 checks**. These are repeated deterministic/evidence
checks across outputs, not 829 independent quality judgments. They include:

- Authored coverage-count and experience/condition expectations.
- Coverage arithmetic and job/skill citations.
- Resume quotes that exist on the cited page and actually mention the named skill.
- Support for the narrow rendered statement: “Your resume mentions X; this job
  lists it as a required/preferred skill.”
- Unknown metadata policies and India-wide remote location behavior.
- The fixed-date exclusion of four deliberately stale source postings.
- Mandatory graduation restrictions and the seniority ladder.
- C# versus C++ and related frameworks without automatic equivalence.
- Unchanged deterministic coverage/requirements when the injected instruction is
  removed, plus an inspected generation-prompt payload containing only skill data.

The injection audit does not assert that changing text leaves embeddings or ranks
unchanged. The explanation audit proves only closed-template literal support,
not proficiency, qualifications, free-form claim entailment, or hiring likelihood.

An optional live Qwen probe evaluates a bounded structured plan for one accepted
synthetic pair. The saved run made **one actual Qwen call**, returned a valid plan,
and passed **14/14 separate rendered-support checks**. This is not a broad LLM
reliability/faithfulness estimate. Default runs make no Qwen calls and block
network socket connections; models must already be cached locally.

## Observed results

The held-out panel contains eight cases:

| Variant | P@5 / ceiling | Mean nDCG@5 | Strict accepted precision | Mean returned |
|---|---:|---:|---:|---:|
| Dense | 0.125 / 0.125 | 0.965 | 0.500 | 1.250 |
| BM25 | 0.125 / 0.125 | 0.974 | 0.500 | 1.250 |
| Hybrid | 0.125 / 0.125 | 0.965 | 0.500 | 1.250 |
| Hybrid + BGE reranker | 0.125 / 0.125 | 0.974 | 0.500 | 1.250 |

Every method returns five grade-2 and five grade-1 recommendations across those
eight cases; none of the accepted panel jobs is grade 0. Strict accepted
precision is therefore **5/10 = 50%**. The partial/conditional results are:

- An Angular role for a React candidate with transferable backend skills.
- SDE I and SDE II for a candidate explicitly targeting senior work.
- A safety role whose work mode is unknown.
- An anomaly-detection role whose advertised experience requirement is missing.

Some of these are intentional consequences of the current conditional/unknown
policy. They are not all implementation bugs. They show why skill coverage and
passed schema checks do not establish full suitability.

Both explicit no-match cases were correctly handled: unrelated retail work and
a mandatory graduation-year mismatch. The reported 1.0 no-match metrics apply
only to these small, constructed cases. Development accepted precision is 1.0 on
four cases; the held-out drop is retained rather than hidden by changing rules.

The small nDCG change does not justify a claim of general BGE superiority. The
panel is tiny, the P@5 metric is saturated at its attainable ceiling, and labels
are assistant-authored. Keep reranking optional pending broader independently
judged evaluation. See the [complete table](../evaluation/results.md) and
[failure analysis](../evaluation/failure_report.md).

## Commands and output files

Complete Modules 1, 3, and 7's data/model preparation first:

```bash
python -m app.dataset_loader
python scripts/download_embeddings.py
python scripts/download_reranker.py

# Four methods, both splits, controlled panel; local inference, no Qwen calls.
python -m app.evaluation.runner

# Evaluate only the held-out split, keeping frozen settings unchanged.
python -m app.evaluation.runner --split heldout

# Optional: one bounded remote Qwen plan probe in addition to the benchmark.
python -m app.evaluation.runner --qwen-probe

# Full-corpus diagnostics with missing-judgment coverage and bounds.
python -m app.evaluation.runner --scope full --split heldout --variants dense bm25 hybrid
```

Each run prints its output directory under `artifacts/evaluation/`. It contains
summary JSON, per-case metrics, raw retrieval/response/assessment records,
evidence checks, failure JSON, and Markdown tables. All raw outputs stay ignored.

To recompute metrics and export a shareable snapshot from saved predictions:

```bash
python -m app.evaluation.summarize PATH_TO_RUN_DIRECTORY --export evaluation
```

This validates the frozen suite hash and consistency of saved predictions. It
makes no model calls, changes no labels, and preserves measured timings. The
tracked `evaluation/results_summary.json`, `results.md`, and `failure_report.md`
are sanitized snapshots; reproduce the raw runs locally. Full-corpus snapshots
live separately under `evaluation/full_corpus/`.

## Files and status

| Path | Responsibility |
|---|---|
| `evaluation/suite.json` | Frozen cases, panel, settings, hashes, expectations |
| `evaluation/resumes/` | Twelve new public synthetic text resumes |
| `evaluation/judgments.jsonl` | All 204 graded suitability labels and rationales |
| `app/evaluation/fixtures.py` | Fixture integrity and split/label validation |
| `app/evaluation/metrics.py` | Judgment-aware ranking, acceptance, no-match, latency metrics |
| `app/evaluation/audits.py` | Expected counts, evidence support, scoped edge-case checks |
| `app/evaluation/runner.py` | Real local-model comparisons and optional Qwen probe |
| `app/evaluation/reporting.py` | Readable tables and failure analysis |
| `app/evaluation/summarize.py` | Recompute/export without rerunning models |
| `tests/test_module8.py` | Metric arithmetic, missing labels, leakage, frozen inputs, audit failures, and actual keyword-run integration |

**136 main tests pass**, including 18 Module 8 tests; the five Python-only tests
are separate. No application matching thresholds or retrieval logic were changed.
Module 8 is complete as a reproducible, scoped evaluation foundation. Larger
human-reviewed full-corpus suitability judgments remain a documented extension.
The completed [Module 9 walkthrough](module_9_portfolio.md) packages these
results for the offline portfolio project; a UI is outside that chosen scope.
