# JobFit: Resume-to-Job Matcher — Fresh Build Plan

## 1. Problem statement

A job seeker wants to know: **Which jobs fit my resume, why do they fit, and
which requirements are not evidenced in my resume?**

JobFit takes two inputs:

1. **Job corpus:** the provided local dataset, `dataset/jobs_synthetic_250.jsonl`.
2. **Candidate input:** a resume PDF and optional preferences such as target
   role, city, work mode, and explicitly supplied eligibility information.

The application ranks relevant jobs, highlights up to five suitable matches,
shows strengths and requirements not evidenced in the resume, and provides
transparent relevance and requirement-coverage scores. Users can inspect the
same details for another job in the corpus. Return fewer matches or no suitable
matches when appropriate.

**This is a synthetic-data learning project.** These are fictional jobs and
employers, not current vacancies. Scores describe resume evidence and relevance;
they are not hiring probabilities or guarantees of qualification.

## 2. Fresh-start status and scope

The previous code, environment, indexes, and tests were removed, preserving
this plan and the supplied `dataset/` folder. **Fresh Modules 0–5 are now complete;
Modules 6–9 remain pending. Milestone A (baseline API) is complete.** Earlier implementation claims and test counts do
not apply to this fresh build. See [README.md](README.md) for the current setup.

- Use all **250 supplied jobs** by default. This replaces the previous live-feed
  scope and 100-job cap; no job collection or download is required.
- Keep `dataset/` unchanged. Generate processed records, reports, and indexes
  outside that folder.
- Use JSONL as the authoritative application input. The CSV is an alternative
  representation, not another 250 jobs to append.
- An optional quick-development subset can use `syn_job_001`–`syn_job_100`.
  Record the subset in reports and use the same subset in both retrieval indexes.
- Keep the app small: no scraping, live employer APIs, scheduled refresh,
  accounts, automated applications, agents, fine-tuning, or knowledge graph.
- PostgreSQL is unnecessary for this scope.

## 3. Dataset inventory and contract

Source documentation: [dataset/DATASET_CARD.md](dataset/DATASET_CARD.md).

| File | Purpose |
|---|---|
| `dataset/jobs_synthetic_250.jsonl` | Main input; one job object per line, with arrays preserved |
| `dataset/jobs_synthetic_250.csv` | Alternative export; list fields are joined with ` | ` |
| `dataset/DATASET_CARD.md` | Meaning, distribution, edge cases, and limitations |

Verified during planning:

| Property | Value |
|---|---|
| Dataset ID | `synthetic_india_ai_sw_v2` |
| Records / unique job IDs | 250 / 250 |
| Domains | 125 `AI/ML`, 125 `Software` |
| Fictional employers | 42, as documented in the dataset card |
| Seniority | Intern 11, Entry 35, Mid 131, Senior 59, Lead 14 |
| Jobs with hard constraints | 34 |
| Unknown experience | 4 records |
| Unknown work mode | 3 records |
| Null city | 4 India-wide remote records |
| Source URLs | All `null`; do not create application links |
| Synthetic marker | Every record has `is_synthetic = true` |

### Input schema

Retain the supplied fields and values:

| Field(s) | Handling |
|---|---|
| `job_id`, `dataset_id`, `is_synthetic` | Preserve identity and synthetic provenance |
| `title`, `company`, `description` | Preserve source text; require usable non-empty values |
| `domain`, `role_family` | Retain domain and role taxonomy; do not infer suitability from domain alone |
| `seniority`, `employment_type` | Retain explicit labels and distinguish internships/contracts |
| `location_city`, `location_country`, `work_mode` | Keep location and arrangement metadata; preserve unknown values |
| `experience_min_years`, `experience_max_years` | Non-negative numbers or null; validate ordering when both are known |
| `education` | Preserve wording and alternatives such as equivalent practical experience |
| `required_skills`, `preferred_skills` | Preserve separate string arrays and verify source evidence |
| `hard_constraints` | Preserve the list of conditions; assess explicitly, with unknown allowed |
| `source_url` | Nullable; currently null for every job |
| `posted_at` | Synthetic date, not proof of an actual posting |

The skill arrays are already structured input. **Do not ask an LLM to recreate
job requirements that the dataset supplies.** Each listed skill appears verbatim
in its description; validate this and retain supporting source quotes in derived
records. No skill should appear in both required and preferred lists for a job.

### Validation and provenance

- Validate record shape, types, unique IDs, dataset markers, arrays, experience
  ranges, and skill evidence. Report rejected or suspicious records with reasons.
- Check exact duplicates without merging different jobs merely because their
  titles match. Preserve same-role/different-city and seniority-ladder cases.
- Preserve `null` and `unknown`; do not turn missing experience into zero or a
  null city into a location mismatch for an India-wide remote job.
- Record the source filename, SHA-256 checksum, validation version, subset,
  counts, and optional filter configuration with every generated index/report.
- If supporting CSV later, explicitly split only list fields on ` | ` and verify
  parity with JSONL. Start with JSONL to avoid unnecessary conversion work.

### Dates and built-in edge cases

Use all 250 records by default so the supplied edge cases remain testable.
A separate optional recency mode can use a configurable reference date, fixed
to **2026-11-18** for the initial benchmark, and a 90-day window. It should flag
or exclude `syn_job_128`, `syn_job_157`, `syn_job_196`, and `syn_job_238`.
Do not silently shrink the default corpus or use the changing system date for
reproducible evaluations. Dates never make these fictional jobs live vacancies.

Retain and test:

- Unknown metadata: jobs 021, 076, 184 (work mode); 043, 095, 159, 232
  (experience); 075, 147, 197, 239 (India-wide remote city).
- Job 213: embedded prompt-injection text. Treat it as source data; its
  instruction must not alter ranking rules, scores, or generation instructions.
- Jobs 245–247: SDE I/II/III with overlapping skills but different experience.
- Jobs 001 and 109: similar role in different cities; preserve both.
- Related tools: FastAPI/Flask, MERN/MEAN, PyTorch/TensorFlow, and game-engine
  stacks are not automatically equivalent skills.
- Graduation, shifts, travel, office location, and contract conditions in
  `hard_constraints` need explicit handling, not just skill similarity.
- Technical tokens such as `C++`, `C#`, `.NET`, `Node.js`, `ASP.NET Core`, and
  `CI/CD` must survive cleaning and keyword tokenization.

## 4. Technology choices

| Component | Purpose |
|---|---|
| Python + Pydantic | Application logic, input validation, and output contracts |
| Standard-library JSON/JSONL loader | Read the supplied corpus; pandas is optional for analysis/CSV support |
| PyMuPDF | Extract resume text with page references |
| LangChain Core | Documents, prompts, runnable composition, model interfaces |
| `langchain-huggingface` + Sentence Transformers | Local BGE-M3 dense embeddings |
| Qdrant + `langchain-qdrant` | Persist embeddings and metadata; filtered dense retrieval |
| Qwen3.8-27B through a remote OpenAI-compatible endpoint | Planned evidence-backed explanations; endpoint and credentials stay in local configuration |
| FastAPI | Resume upload and result/detail endpoints |
| BM25 + reciprocal rank fusion | Later keyword/hybrid comparison |
| Cross-encoder reranker | Optional later refinement, enabled only if evaluation justifies it |
| Manual judgments + custom metrics | Matching, coverage, constraint, and latency evaluation |
| RAGAS / simple UI | Optional final additions |

Create and pin dependencies during implementation. Record exact model/runtime
versions and hardware. Do not install the entire optional stack before the
baseline needs it.

## 5. Embedding and storage specification

Use **`BAAI/bge-m3`**, **1024-dimensional L2-normalized dense vectors**, for both
job and resume-query embeddings. Qdrant uses vector size **1024** and **cosine**
distance. Use the same model and normalization configuration for both sides.

- Represent each job initially as one LangChain `Document`, with its ID,
  structured metadata, and source description available for evidence.
- Build searchable text consistently from title, role family, required/preferred
  skills, and description. Keep hard constraints separately available for checks.
- Inspect actual model-token lengths before deciding whether chunking is needed.
  BGE-M3 supports up to 8,192 tokens; do not silently truncate jobs or resumes.
- BGE-M3 does not require query instruction prefixes. Start with dense output;
  BM25 is the separate keyword baseline in a later module.
- Preserve `syn_job_*` IDs in payload metadata. Use deterministic UUID mappings
  for Qdrant point IDs rather than replacing the source identity.
- Use a clearly synthetic collection such as `jobfit_synthetic_india_ai_sw_v2`.
- Re-indexing the same source/configuration must not create duplicates. Rebuild
  affected embeddings when searchable text or embedding configuration changes.
- Store dataset checksum, model settings, and indexing-text version with the
  index. Keep vectors and model caches outside `dataset/`.

## 6. Architecture

```text
OFFLINE PREPARATION
Provided dataset/jobs_synthetic_250.jsonl
    -> validate and preserve source IDs + synthetic markers
    -> derived canonical jobs + validation report
    -> LangChain Documents
       |-> BGE-M3 -> Qdrant dense index
       |-> BM25 index                         [later comparison]

PER RESUME
Resume PDF + explicit preferences/eligibility facts
    -> extract text with page references
    -> build candidate profile with evidence + compact query
    -> evaluate explicit filters and constraints, preserving unknowns
    -> dense retrieval                        [baseline]
       + BM25 -> rank fusion                  [later]
    -> optional cross-encoder reranking       [later]
    -> deterministic skill coverage and eligibility assessment
    -> ranked results + up to five accepted recommendations
    -> LLM explanations using resume and job evidence
    -> validate claims, citations, and response schema
    -> FastAPI response / selected-job detail
```

LangChain connects document representations, embeddings, retrieval, prompts,
and generation. Domain rules and numerical scoring remain explicit Python
logic. This is RAG because explanations use retrieved job text together with
resume evidence. Retrieval and prompts alone do not guarantee grounded answers.

Do not generate detailed LLM explanations for all 250 jobs on every request.
Rank the corpus as needed, explain the top accepted results, and produce details
for another selected job on demand. Label those details clearly if the job
fails a constraint or has weak evidence.

## 7. Matching, constraints, and scoring

### Resume preparation

Extract original page-level text, then derive supported skills, projects,
experience, and education with quotes. Build a compact retrieval query without
copying names, email addresses, phone numbers, or resume addresses. Keep original
text local for evidence validation. Begin with deterministic extraction and a
vocabulary covering skills found in this dataset; report its limitations.

Professional experience remains unknown unless supported by an explicit,
unambiguous statement or a separately validated calculation. College projects
and internship duration must not automatically become full-time employment years.
Preferences are user choices, not facts inferred from contact details.

### Eligibility and constraints

- Apply identical hard filters to dense and BM25 retrieval.
- Separate preference matches, experience/education evidence, and mandatory
  constraints from semantic similarity.
- Evaluate constraints as `met`, `not_met`, or `unknown`, with evidence or a
  clearly identified user-supplied answer. Never invent missing answers.
- Travel, shift availability, graduation year, notice period, and contract
  willingness may require explicit user information. Surface unresolved conditions.
- A job's maximum experience range is not automatically a rejection rule for
  a more-experienced applicant unless the source states a strict cap.
- Education alternatives remain alternatives; a degree is not mandatory when
  equivalent practical experience is explicitly accepted.
- Define strict/lenient unknown-metadata policy and report which was used.
- Tune recommendation acceptance using development examples. A nearest neighbor
  is not necessarily suitable; allow fewer than five or no accepted matches.

### Transparent scores

Keep these measures separate initially; do not introduce an arbitrary combined
"fit percentage" before its behavior can be evaluated.

| Measure | Calculation / interpretation |
|---|---|
| Semantic relevance | Dense cosine similarity, or explicitly labeled retrieval/reranker score |
| Required-skill coverage | Required skills evidenced in resume / total required skills × 100 |
| Preferred-skill coverage | Preferred skills evidenced in resume / total preferred skills × 100 |
| Experience / education | Supported match, mismatch, or unknown; show supporting evidence |
| Hard constraints | Per-condition status with unresolved conditions visible |

Count each canonical skill once. Use true aliases only; do not equate related
frameworks. Track each skill match back to resume evidence and its original job
requirement. If a skill list is empty, report coverage as unavailable rather
than automatically assigning 100%. Missing resume evidence does not prove that
the candidate lacks the skill.

Calculate scores and constraint flags in code. The LLM explains supplied facts;
it must not assign or change the numerical scores. RRF, BM25, cosine, and
cross-encoder values have different scales and are never hiring probabilities.

## 8. Output contract

Return:

- `dataset_id`, `is_synthetic`, active corpus/subset, applied filters, and warnings.
- A ranked candidate list with clearly labeled relevance, coverage, and
  constraint results; up to five accepted jobs highlighted as recommendations.
- For each explained job: title, fictional employer, location, seniority,
  matched requirements, resume/page evidence, job evidence, and uncertainties.
- Required and preferred skills **not evidenced in the resume**, kept separate.
- `matches_found` or `no_suitable_matches` for the recommendation outcome.

Every view must identify the corpus as synthetic. Null `source_url` must not
produce a fake application link. Explanation citations use job IDs/description
quotes and resume pages. Never invent tools, achievements, experience, numerical
results, or job requirements. Optional later resume-bullet rewriting preserves
existing facts only.

Example display:

```text
Synthetic demo — Generative AI Engineer
Semantic relevance: [measured similarity]
Required-skill coverage: 4 of 6 (66.7%)
Not evidenced in resume: Vector databases, Prompt engineering
Experience: unknown
Hard constraints: none listed
Evidence: [actual resume quote/page] + [actual job requirement quote]
```

This is an output-format illustration, not a result already obtained.

## 9. Implementation modules and completion checkpoints

| Module | Simple responsibility | Status |
|---|---|---|
| 0 | Set up the environment, settings, and schemas | Complete |
| 1 | Load and validate the supplied job dataset | Complete |
| 2 | Parse the resume and prepare its search profile | Complete |
| 3 | Embed jobs/query with BGE-M3 and retrieve with Qdrant | Complete |
| 4 | Calculate coverage/constraints and generate grounded explanations | Complete |
| 5 | Expose upload, matching, and selected-job detail through FastAPI | Complete |
| 6 | Add BM25 and compare hybrid retrieval | Pending |
| 7 | Add and evaluate optional cross-encoder reranking | Pending |
| 8 | Measure retrieval, explanations, constraints, and latency | Pending |
| 9 | Add an optional UI and reproducible documentation | Pending |

### Module 0 — Setup and contracts

**Tasks:** create `.venv`, dependency files, `.env.example`, `.gitignore`,
configuration, job/resume/preference/evidence/score/response schemas, and a short
README. Configure Qwen/runtime settings; record hardware.

**Deliverables:** importable project skeleton and validated sample contracts.

- [x] Configuration loads without requiring models or services at import time.
- [x] All supplied dataset fields are represented without losing arrays/nulls.
- [x] Response contracts cover synthetic provenance, unknowns, and no-match cases.

**Implemented:** `.venv`, dependency pins, configuration, schemas covering all
21 source fields, four synthetic contract examples, and an offline setup check.
All 19 contract/configuration tests pass, including schema compatibility with
all 250 records. Dataset file hashes are unchanged. See the
[Module 0 guide](docs/module_0_setup.md). Dataset loading/reporting is implemented in Module 1.
Resume parsing and LangChain dense retrieval are implemented in Modules 2–3.
Grounded generation and the baseline API are implemented in Modules 4–5.

### Module 1 — Local dataset loading and validation

**Tasks:** read the JSONL, validate all 250 records, preserve IDs and source
text, check skill evidence/duplicates, report unknowns and lengths, and save
validated derived data plus checksum/configuration metadata. Add optional
100-job subset and fixed-date recency mode; retain the full default corpus.

**Deliverables:** loader/validation CLI, derived records, validation report.

- [x] Default input is `dataset/jobs_synthetic_250.jsonl`; all 250 records are accounted for.
- [x] Source files are unchanged; every output record retains synthetic provenance.
- [x] Duplicate/invalid records have explicit reasons; edge-case records are not silently dropped.
- [x] Optional 90-day mode at 2026-11-18 identifies exactly the four known stale IDs.
- [x] Repeated runs preserve IDs and deterministic output ordering.

**Implemented:** `app/dataset_loader.py`, ten dataset tests, and the
[Module 1 guide](docs/module_1_dataset.md). Default preparation produces 250
validated jobs, skill-evidence records, and a checksum-linked validation report.
The optional subset and recency modes produce 100 and 246 jobs. All 29 tests
across Modules 0 and 1 pass; original dataset file hashes remain unchanged.

### Module 2 — Resume parsing and profile preparation

**Tasks:** extract PDF page text, build an evidence-backed profile and compact
query, accept explicit preferences, and validate cited quotes. Use synthetic or
anonymized resumes for development. Handle scanned, empty, corrupt, and locked
PDFs with useful errors; OCR is deferred.

**Deliverables:** parser, profile builder, CLI, example prepared profiles.

- [x] AI, backend, analyst, fresher, and unrelated resumes produce useful profiles or clear errors.
- [x] Every extracted fact has valid page evidence; contact details do not enter the automatic query.
- [x] Unknown experience stays unknown; project duration is not treated as employment.

**Implemented:** local PDF extraction, rule-based corpus-vocabulary skill mentions,
quoted section facts, conservative explicit professional duration, contact-free
query construction, and a preparation CLI. Five synthetic PDF demos and 13 new
tests cover normal profiles, evidence, PDF failures, and limitations. All 42 tests
across Modules 0–2 pass. See [Module 2 guide](docs/module_2_resume.md).

### Module 3 — Embeddings and dense retrieval

**Tasks:** measure actual token lengths; build stable job Documents; embed using
normalized 1024-dimensional BGE-M3 vectors; persist to Qdrant; implement dense
retrieval and shared filter rules. Record indexing/query latency and configuration.

**Deliverables:** index builder and dense retrieval command/function.

- [x] Vectors have correct dimensions/norms and re-indexing does not duplicate jobs.
- [x] The collection identifies its dataset/subset/checksum and survives restarts.
- [x] Filters preserve intentional unknown and India-wide-remote behavior.
- [x] Results trace back to source IDs and baseline quality/latency is recorded.

**Implemented:** LangChain Documents, checked Hugging Face BGE-M3 embeddings,
persistent local Qdrant, stable UUIDs/provenance, token limits, shared city/work-mode
filters, and index/search CLIs. The real 250-job index has unit-normalized
1024-dimensional vectors; text lengths are 209–317 model tokens. Re-indexing
reuses verified vectors. Five real-model demo queries/reuse checks run with socket
connections blocked; 56 fast tests pass across Modules 0–3. Demo inspection and
latency are recorded, without held-out suitability claims. See the
[LangChain tutorial](docs/module_3_langchain.md) and
[development results](docs/module_3_results.md). Server mode remains unverified.

### Module 4 — Coverage, constraints, and grounded generation

**Tasks:** calculate required/preferred coverage, evaluate constraints with
unknown support, create job-side evidence from supplied arrays/text, then build
a LangChain prompt/LLM chain for explanations. Validate citations and structured
output; use bounded retries or clear errors. Treat job/resume text as untrusted data.

**Deliverables:** scoring/constraint helpers, generation chain, response builder.

- [x] Scores are deterministic and include correct denominators and evidence.
- [x] Related skills, education alternatives, and mandatory constraints behave correctly.
- [x] The LLM cannot change scores or introduce unsupported facts.
- [x] Job 213 cannot override instructions or receive a score boost from its embedded instruction.
- [x] Empty/unsuitable results produce a valid no-match response.

**Implemented:** deterministic evidence coverage, conservative experience/education
and explicit mandatory-condition checks, baseline acceptance gates, and matching
CLI. LangChain/Qwen produces an allowlisted explanation plan; Python renders
factual sentences with validated citations. Arbitrary model prose/scores never
enter output, and invalid/unavailable generation has a labeled bounded fallback.
The real AI demo accepted two of 30 candidates, with two successful remote Qwen
calls. All 72 main tests pass, including 16 Module 4 tests; five Python-only
comparison tests are separate. Threshold calibration and held-out suitability
remain pending evaluation. See [Module 4 guide](docs/module_4_matching.md).

### Module 5 — FastAPI baseline

**Tasks:** add health, resume-upload/match, and selected-job-detail endpoints.
Validate preferences, bound uploads and execution, handle unavailable services,
and clean temporary uploads. Avoid logging raw resumes.

**Deliverables:** runnable API with example requests/responses.

- [x] Upload -> parse -> retrieve -> score -> explain works end to end.
- [x] Another job can be inspected without regenerating all 250 explanations.
- [x] Errors and synthetic labels are visible and responses validate.

**Implemented:** FastAPI routes for health/readiness, bounded PDF upload/matching,
selected-job inspection, and run deletion. One processing slot guards local
Qdrant access; response deadlines retain the slot until worker cleanup. Runs are
bounded/expiring in memory, and temporary PDFs are removed. All 85 main tests
plus five Python-only tests pass. Real HTTP upload through BGE-M3/Qdrant/scoring/
remote Qwen returned two grounded recommendations; off-pool job inspection and
run deletion were verified. See [Module 5 guide](docs/module_5_api.md).

**Milestone A is complete: Modules 0–5 pass the baseline checkpoints.**

### Module 6 — BM25 and hybrid retrieval

**Tasks:** build BM25 on the same eligible corpus, preserve technical tokens,
retrieve a starting top 30 from each method, combine using reciprocal rank
fusion, and deduplicate by source job ID. Candidate counts are tunable settings.

**Deliverables:** BM25/hybrid modes and reproducible comparison inputs.

- [ ] Dense, BM25, and hybrid use the same source, subset, and filters.
- [ ] Keyword tests cover symbol-bearing skills and true aliases.
- [ ] Merged candidates have unique job IDs and traceable ranks.

### Module 7 — Optional cross-encoder reranking

**Tasks:** choose a reranker that fits the hardware; record its input limits;
score candidate-profile/job pairs; inspect truncation; tune acceptance on
separate development examples and measure the latency tradeoff.

**Deliverables:** optional reranker wrapper and benchmark configuration.

- [ ] Reranking preserves job IDs, coverage values, and mandatory constraint outcomes.
- [ ] Quality and latency are compared with the baseline; improvement is not assumed.
- [ ] Weak/irrelevant matches can still be rejected.

### Module 8 — Evaluation

**Tasks:** prepare development/held-out resumes, judge resume–job suitability,
compare retrieval variants, verify coverage arithmetic and evidence, assess
constraints/no-match behavior, and save failures/configurations.

**Deliverables:** judgments, evaluation runner, comparison table, failure report.

- [ ] Evaluation uses true suitability judgments, not domain labels alone.
- [ ] Held-out resumes do not tune thresholds, aliases, or retrieval settings.
- [ ] Synthetic corpus limitations and edge-case outcomes are reported explicitly.
- [ ] No-match precision is not reported as perfect when nothing is returned.

**Milestone B is reached after the retrieval comparisons and evaluation.**

### Module 9 — Demo and documentation

**Tasks:** optionally add a small upload/results/detail UI; document setup,
model downloads, commands, synthetic provenance, local data handling, and
measured limits. Optional resume rewrites must preserve facts.

**Deliverables:** reproducible README, demo, and final report.

- [ ] Another person can reproduce the demo from this dataset.
- [ ] All displayed jobs are clearly fictional and no application links are invented.
- [ ] Portfolio claims match actual saved evaluation results.

**Milestone C is reached when the demo and documentation are reproducible.**

## 10. Evaluation design

Start with 10–20 synthetic/anonymized resumes spanning AI/ML, backend,
frontend, analyst, fresher, and unrelated profiles. Include deliberate gaps,
strong skill overlap with wrong seniority, restrictive location preferences,
unknown metadata, and unresolved constraints.

1. Separate development and held-out resume sets before tuning.
2. Pool candidates from each retrieval variant and manually label each
   resume–job pair relevant, partially relevant, or irrelevant using a written
   rubric covering requirements, experience, and explicit preferences/constraints.
3. Compare ranked-list Precision@5 before acceptance gating. Missing positions
   count as non-relevant. Define which labels count as relevant beforehand.
4. Separately report accepted-result precision, mean returned count, correct
   no-match decisions, coverage correctness, and constraint handling.
5. Record retrieval/reranking and end-to-end median/p95 latency, hardware,
   model versions, source checksum, subset, filters, and warm/cold-start policy.

| Mode | Precision@5 | Accepted precision | Mean returned | Median / p95 latency |
|---|---|---|---|---|
| Dense only | TBD | TBD | TBD | TBD |
| BM25 only | TBD | TBD | TBD | TBD |
| Hybrid | TBD | TBD | TBD | TBD |
| Hybrid + reranker | TBD | TBD | TBD | TBD |

Check the supplied unknowns, stale dates, seniority ladder, city variants,
related-tool cases, technical tokens, hard constraints, and prompt injection.
For job 213, an injected instruction must not change deterministic scores or
system policy; it may still rank naturally on legitimate job requirements.

Manually review explanation support using both resume and job text. Optional
RAGAS faithfulness is an additional contextual-support check, not a guarantee
of suitability. Record its judge/runtime and any external data transmission.

The descriptions are short and uniform and requirements are cleanly listed.
Good results here do not establish performance on real postings or arbitrary
resume layouts. Any future real-data evaluation must be reported separately.

## 11. Planned repository structure

Modules 0–5 have created preparation, retrieval, matching/generation, FastAPI,
settings/contracts, tests, README, and module guides. Hybrid retrieval, evaluation,
and UI files in this tree remain proposed and will be created module by module:

```text
JobFit/
  plan.md
  dataset/                         # Provided input; preserve unchanged
    DATASET_CARD.md
    jobs_synthetic_250.jsonl
    jobs_synthetic_250.csv
  README.md
  pyproject.toml
  requirements.lock
  .env.example
  .gitignore
  app/
    config.py
    schemas.py
    dataset_loader.py
    resume_parser.py
    profile.py
    prepare_resume.py
    documents.py
    embeddings.py
    filters.py
    vector_index.py
    dense_cli.py
    retrieval.py
    scoring.py
    constraints.py
    generation.py
    matching.py
    match_cli.py
    reranking.py
    main.py
    api_service.py
    api_schemas.py
  scripts/
    download_embeddings.py
    verify_dense.py
    create_demo_resumes.py
  artifacts/                       # Derived records, reports, indexes; ignored
  examples/                        # Synthetic candidate examples
  evaluation/
    judgments.jsonl
    run_evaluation.py
    results.md
  tests/
  ui/                              # Optional
```

Keep real resumes, secrets, temporary uploads, vectors, model weights, caches,
and private evaluation artifacts outside public commits. Preserve source input
and its checksum; do not overwrite `dataset/` with generated output.

## 12. Portfolio wording and references

Use only after implementation:

> Built JobFit, a RAG-based resume-to-job matcher using LangChain, BGE-M3,
> Qdrant, and FastAPI, evaluated on 250 synthetic India AI/software postings,
> with evidence-backed explanations, transparent skill coverage, and explicit
> handling of eligibility constraints and unknown information.

Add measured improvements only after evaluation. State the metric, baseline,
sample size, and whether the change is percentage points or relative percent.

References:

- [Provided dataset card](dataset/DATASET_CARD.md)
- [BGE-M3 model card](https://huggingface.co/BAAI/bge-m3)
- [LangChain Hugging Face embeddings](https://docs.langchain.com/oss/python/integrations/embeddings/huggingfacehub)
- [Qdrant collection configuration](https://qdrant.tech/documentation/manage-data/collections/)
- [RAGAS faithfulness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/)
