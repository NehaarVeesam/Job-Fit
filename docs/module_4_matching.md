# Module 4 — Evidence Coverage, Requirements, and Qwen Explanations

## Purpose

Module 3 returns nearby jobs. Module 4 asks which of those candidates pass our
explicit baseline policy and explains their evidenced strengths and gaps.

```text
Prepared resume + Module 3 dense candidates
    -> verify query, preferences, job IDs, and source records
    -> assess each required/preferred skill using actual resume quotes
    -> calculate coverage in Python
    -> assess experience, education, and mandatory conditions
    -> apply acceptance policy, preserving dense rank
    -> select up to five recommendations
    -> LangChain prompt -> remote Qwen -> structured explanation plan
    -> validate allowed skill selections -> render factual sentences with quotes
    -> MatchResponse + assessment report
```

Rejected jobs remain in the assessment report. No model call is made for them.
The generation client is initialized lazily, so a no-match outcome makes zero
Qwen calls. All job data remains explicitly synthetic.

## Coverage is a deterministic calculation

For each required/preferred source skill, find the same canonical skill in the
candidate profile and recheck its supporting quote. Matching is case-insensitive
with symbol-aware boundaries. A quote must actually mention that skill and must
not be an obvious contact/negative/learning line. The prepared-resume contract
also verifies the quote against its cited page.

Examples: JavaScript does not imply Java; Flask does not imply FastAPI; a profile
claiming Python with a quote mentioning only JavaScript does not count.

```text
Required skills evidenced: 4
Required skills in job:    6
Required coverage:         4 / 6 × 100 = 66.7%
```

Preferred coverage has a separate denominator. A zero-length requirement list
has a null percentage, not 100%. Each assessment contains both job-side and
resume-side evidence. Missing skills use `not_evidenced_in_resume`, which does
not assert that the candidate lacks the skill in real life.

This is a literal-mention baseline, not proficiency measurement. Aliases,
unusual phrasing, mixed positive/negative lines, and the Module 2 vocabulary
limit remain limitations.

## Requirements and unknowns

Every decision is `met`, `not_met`, or `unknown`, with resume evidence or explicit
user-fact keys for a known outcome.

**Experience:** compare a supported explicit professional duration to the job's
minimum. Project/internship duration and employment dates are not summed. Missing
years remain unknown. Exceeding the advertised maximum is not an automatic
rejection because the source does not state a strict cap.

**Education:** narrow positive rules recognize a degree plus an applicable field
in an education quote. Missing/different degrees remain unknown; the system does
not reject a candidate just because an "equivalent practical experience"
alternative requires review. Complex research requirements and current-study
conditions remain unknown without an explicit answer.

**Mandatory conditions:** all 21 distinct source phrases can be answered using
their exact wording in `preferences.eligibility_facts`:

```json
{
  "eligibility_facts": {
    "constraint:6-month contract with possible extension": true,
    "constraint:Rotational on-call for network incidents": false,
    "education_requirement_met:syn_job_001": true
  }
}
```

These are explicit candidate attestations, not externally verified qualifications.
Use a condition key only when it actually appears in the selected job. The
education answer must account for all stated alternatives in that requirement.

Some simple conditions also accept typed fields:

| Source condition | Explicit input |
|---|---|
| Final-year students only | `is_final_year`: boolean |
| Final-year students graduating in 2027 | `is_final_year` plus integer `graduation_year` |
| Open to 2025 and 2026 graduates only | Integer `graduation_year` |
| Must be able to join within 60 days | Integer `notice_period_days` |
| Up to 25% / 30% travel | Integer `max_travel_percent` |
| Must be based in India | `based_in_india`: boolean |
| Must work from the Jaipur office | `willing_jaipur_office`: boolean |
| Rotational shifts including nights | `willing_rotational_night_shifts`: boolean |

An exact-condition boolean takes precedence over the convenience fields. Invalid
numeric ranges/types do not establish a known decision. Specific rotations,
regional travel, contracts, background checks, and other compound conditions
require the exact-condition answer. A resume address or preferred city never
becomes an answer to a mandatory condition.

Aggregate eligibility is `not_met` if any check fails, `met` if all checks pass,
and `unknown` otherwise. `met` describes these implemented checks; it is not an
employer's hiring decision.

## Recommendation policy

Defaults, configurable through `.env`:

| Setting | Default |
|---|---|
| `JOBFIT_CANDIDATE_COUNT` | 30 retrieved jobs |
| `JOBFIT_MIN_COSINE_SIMILARITY` | 0.55 |
| `JOBFIT_MIN_REQUIRED_COVERAGE` | 50 percent |
| `JOBFIT_MIN_EVIDENCED_REQUIRED` | 2, capped at the job's actual requirement count |
| `JOBFIT_MAX_RECOMMENDATIONS` | 5 |
| `JOBFIT_GENERATION_ATTEMPTS` | 2 per selected job |

A recommendation must pass explicit preferences, the relevance/coverage gates,
and the evidenced-required-skill minimum. A job with zero evidenced required
skills or no required-skill denominator is rejected. Known requirement failures
are rejected. Unknown requirements are included with warnings under
`include_with_warning` and rejected under `exclude`.

We preserve the original dense ordering among accepted candidates. There is no
hidden weighted score combining coverage and cosine similarity. All thresholds
are transparent development heuristics; they have not been calibrated on
held-out suitability judgments. Those evaluations belong to Module 8.

`ranked_candidates` represents the retrieved pool, including rejected jobs.
`recommendations` contains only accepted jobs. The assessment report records
every rejection reason and identifies otherwise passing jobs omitted because the
recommendation limit was reached. Fewer than five and no-match outputs are valid.

## What Qwen does, and how LangChain is used

The chain in `app/generation.py` is:

```python
prompt = ChatPromptTemplate.from_messages([...])
structured_model = model.with_structured_output(
    ExplanationPlan.model_json_schema(), method="json_schema", strict=True
)
chain = prompt | structured_model
plan = chain.invoke({"data": assessment_json})
```

- `ChatPromptTemplate` builds system and human messages.
- `ChatOpenAI` connects to your existing OpenAI-compatible Qwen endpoint.
- `with_structured_output` requests the `ExplanationPlan` JSON shape.
- `|` composes the prompt and model into a runnable chain.
- `invoke` executes that chain once.

See the [official LangChain structured-output reference](https://reference.langchain.com/python/langchain-openai/chat_models/base/ChatOpenAI/with_structured_output).

Qwen receives the source job ID, allowed evidenced skill names, missing skill
names, and their required/preferred classifications. It chooses one to three
strengths to emphasize and returns the complete gap list. It receives no raw PDF,
contact details, employer prose, or job-description instructions.

Example plan:

```json
{
  "strengths": ["RAG", "LangChain", "Python"],
  "gaps": ["Vector databases", "Prompt engineering", "Qdrant", "LLM evaluation"]
}
```

This is deliberately a **constrained explanation plan**, not unrestricted
model-written prose. Python checks every selected name against the assessment
and renders a sentence such as:

> Your resume mentions RAG; this job lists it as a required skill.

The sentence includes actual resume/job citations. Coverage, eligibility, all
missing skills, and requirement warnings come from Python. A model-supplied
score, extra field, invented skill, duplicate selection, or incomplete gap list
is rejected. Literal source presence is the supported claim; we do not upgrade
it into proficiency or qualification.

This design prevents arbitrary model claims from entering the response. It also
limits narrative variety: Qwen prioritizes the explanation while Python controls
its final factual wording. Free-form semantic explanation verification is not
implemented or claimed.

The source injection in job 213 cannot alter deterministic scoring. Its full
description never enters the generation prompt. This is not a general proof of
prompt-injection resistance for future unrestricted prompts or tool use.

## Failure handling and configuration

Your `.env` endpoint, API key, and Qwen options are preserved. Generation uses
`Qwen/Qwen3.8-27B` through the remote OpenAI-compatible endpoint. API credentials
are never placed in prompts/reports. LangSmith tracing is disabled around calls.

The configured JSON modes `json_schema`, `json_object`/true, and plain JSON parsing
are supported. The verified endpoint used `json_schema`. Requests include the
configured thinking flag, output-token limit, timeout, and optional sampling
values. `QWEN_MAX_MODEL_LEN` remains server configuration metadata; it is not
sent as an unsupported per-request API parameter.

SDK retries are disabled, and application attempts are bounded per job. Parse,
transport, and allowlist failures produce a labeled deterministic fallback after
the configured attempts. `generation_status` is one of:

- `deterministic`: explicitly offline mode.
- `qwen_grounded_plan`: a valid Qwen plan was accepted.
- `fallback`: generation failed; deterministic evidence sentences were used.

Reports record attempted calls, sanitized error class names, generation time,
policy, corpus/index identity, and per-job assessments. They do not save API
keys, private endpoint values, or raw provider error bodies. No-match cases make
zero calls. Ollama inference is not implemented by Module 4's generation adapter;
its configured remote OpenAI-compatible path is the supported runtime.

## Run it

From the project root, use the existing prepared AI example:

```bash
# Fully offline scoring/explanations; cached BGE-M3 and local Qdrant are used.
.venv/bin/python -m app.match_cli \
  --prepared-resume artifacts/resumes/86ef5a01645b45129f5cc0a61a1cf313e2a6bcb68dc89291f1b240e563d27bcb/prepared_resume.json \
  --no-llm

# Same pipeline with remote Qwen plans for accepted jobs.
.venv/bin/python -m app.match_cli \
  --prepared-resume artifacts/resumes/86ef5a01645b45129f5cc0a61a1cf313e2a6bcb68dc89291f1b240e563d27bcb/prepared_resume.json

.venv/bin/python -m unittest discover -s tests -v
```

Replace the prepared path for another resume. `--corpus-dir` explicitly selects
an alternative prepared/indexed corpus. Optionally use `--retrieval-report PATH`
to reuse a Module 3 result instead of loading embeddings. Its query checksum,
preferences, source records, ready manifest, corpus, and model must agree.
Older Module 3 reports without the newly added query checksum must be regenerated.
These checks establish consistency, not tamper-proof authenticity of local files.

Outputs:

```text
artifacts/matches/<RUN_HASH>/
  response.json                 Validated MatchResponse
  assessment_report.json        All retrieved assessments and rejection reasons
```

Scores and classifications appear in structured fields. The response's `skills`
array separates required/preferred requirements and missing evidence. The output
folder changes with input/configuration/report contents; the CLI prints its path.
Outputs remain ignored local artifacts and may contain resume evidence quotes.

## Files and verification

| File | Responsibility |
|---|---|
| `app/constraints.py` | Conservative experience, education, and explicit condition decisions |
| `app/scoring.py` | Coverage, job evidence, acceptance gates, and sentence rendering |
| `app/generation.py` | LangChain/Qwen structured plan, allowlist validation, bounded attempts |
| `app/matching.py` | Candidate assessment, selection, response/audit construction |
| `app/match_cli.py` | Dense retrieval -> assessment -> generation CLI |
| `app/schemas.py` | Added education/experience assessments and generation status |
| `app/retrieval.py` | Added query checksum for safe result reuse |
| `tests/test_module4.py` | 16 tests for policy, requirements, grounding, retries, and injection handling |

All **72 tests** across Modules 0–4 pass; the Python-only Module 3 comparison
has its own five tests. Tests exercise real LangChain prompt/model/parser
composition with a fake chat model, distinct from the live remote verification.

The real AI demo retrieved 30 candidates and accepted two: `syn_job_001` at
66.7% required coverage and `syn_job_105` at 50%. The remote Qwen check completed
two valid calls, and both responses were labeled `qwen_grounded_plan`.
See `artifacts/verification/module4_ai_qwen_response.json` and
`artifacts/verification/module4_live.json` for saved development evidence.
Original dataset checksums remain unchanged.

Module 5 is complete: see [the API guide](module_5_api.md).
Next is Module 6, BM25/hybrid comparison. Held-out quality evaluation, broader
extraction, and threshold calibration remain later work.
