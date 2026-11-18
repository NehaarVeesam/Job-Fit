# Module 0 — Setup and Contracts

## Implemented scope

- `.venv`, `pyproject.toml`, and a tested dependency lock.
- Local `.env`, public `.env.example`, and `.gitignore`.
- Explicit configuration loading and Pydantic data contracts.
- Four hand-authored, synthetic JSON examples plus an offline setup checker.
- Contract tests, including compatibility with the 250 supplied source objects.

## Defaults

| Setting | Value |
|---|---|
| Input | `dataset/jobs_synthetic_250.jsonl` |
| Dataset ID | `synthetic_india_ai_sw_v2` |
| Subset | `all` (250 jobs); optional Module 1 loader mode `first_100` |
| Recency filter | Disabled; benchmark reference date is 2026-11-18 |
| Generated artifacts | `artifacts/`, outside the input folder |
| Embeddings | `BAAI/bge-m3`, 1024 dimensions, normalized, cosine |
| Embedding device / batch | CPU / 4 |
| Qdrant | `http://127.0.0.1:6333`, synthetic collection name |
| Generation | Remote OpenAI-compatible endpoint, `Qwen/Qwen3.8-27B` |
| Recommendations | Up to 5 |
| Future PDF upload bounds | 5 MiB / 10 pages |

Qwen is configured through `QWEN_MODEL`, `QWEN_BASE_URL`, and `QWEN_API_KEY`.
The private endpoint and key are preserved only in local `.env` settings.
`JOBFIT_LLM_MODEL` can override the model name. JSON mode accepts `json_schema`,
`json_object`, `none`, or booleans. Blank optional context-length, temperature,
and top-p values retain server defaults. API keys are masked in settings output.
The loader supports this configuration; inference and connectivity are untested.

Settings precedence: constructor arguments, environment variables, root `.env`,
then defaults. Relative data/artifact paths resolve against the project root,
not the shell's current directory. Unknown `.env` keys fail validation.
Settings reject a generated-artifact directory inside the configured dataset
folder, wrong vector dimensions, or disabled normalization.

Module 1 implements subset/date filtering. These settings alone do not implement embeddings, uploads,
network calls, or model inference. They are contracts for later modules.

## Contract rules

`JobRecord` contains all 21 source fields. It validates explicit synthetic
markers, India metadata, experience ranges, separate skill arrays, and verbatim
skill presence in descriptions. Source URLs remain null for this synthetic set.
Description text is preserved rather than silently rewritten.

Candidate facts cite resume pages. Constraint statuses support `met`, `not_met`,
and `unknown`, with evidence or explicit user-fact references for decisions.
Coverage uses counts and percentages rounded to one decimal; an empty requirement
list has no percentage. Scores have explicit types and no hiring-probability field.

Response contracts reject inconsistent statuses, duplicate IDs, more than five
recommendations, fabricated job quotes, coverage inconsistent with assessed
skills, and known-ineligible accepted recommendations. Their ranked entries
must agree with the recommendations. Actual resume quote checking, requirement
interpretation, and recommendation acceptance policy are not implemented yet.

## Development-host observations

Observed on 2026-11-18; this is the development host, not necessarily the user's
personal machine:

- Python 3.11.13, Linux, 32 logical CPUs.
- Approximately 117 GiB RAM visible to the host.
- NVIDIA tooling cannot communicate with a working driver; GPU availability is unverified.
- Ollama is not installed on this host.

No embedding or generation weights were downloaded. No Qdrant/Ollama services
were started and no inference was benchmarked. Parsing and embeddings are planned locally. Generation is now configured for
your remote Qwen endpoint; when implemented, generation sends its prompt context
to that server. Module 0 checks do not send data to it.

## Completion checklist

- [x] Environment and configuration work without models/services at import time.
- [x] Schemas represent all source fields without losing arrays/nulls.
- [x] Examples cover synthetic provenance, unknown information, scores, and no matches.
- [x] Setup and contract checks pass; input dataset hashes remain unchanged.

**Module 1 is also complete:** see [its guide](module_1_dataset.md) for dataset
loading, validation accounting, provenance, and derived-output reporting.
Module 2 is complete as well: see [its guide](module_2_resume.md).
Module 3 is complete: see [the LangChain guide](module_3_langchain.md).
Module 4 is complete: see [its guide](module_4_matching.md).
Module 5 is complete: see [the API guide](module_5_api.md).
Next is Module 6, BM25/hybrid comparison. No implementation from
before the reset is reused.
