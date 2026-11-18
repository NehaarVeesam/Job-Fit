# Module 5 — FastAPI Upload, Matching, and Job Inspection

## What FastAPI does

Modules 1–4 contain the matching logic. FastAPI makes that workflow available
through HTTP so a browser, future UI, or another program can submit a resume
and receive structured results.

```text
HTTP PDF + preferences + use_llm
  -> validate upload and preferences
  -> extract resume and prepare profile
  -> cached BGE-M3 + persistent Qdrant retrieval
  -> coverage, conditions, selection, grounded explanation
  -> JSON response + temporary run ID

Run ID + another job ID
  -> reuse prepared resume
  -> inspect that job, including missing evidence and failures
```

FastAPI handles requests, response validation, errors, upload cleanup, execution
limits, and temporary runs. `MatchingService` connects the existing Python
functions; FastAPI itself does not calculate embeddings or scores.

Files and preferences use multipart form data, with preferences encoded as a
JSON string. This requires `python-multipart`:
[official FastAPI upload documentation](https://fastapi.tiangolo.com/tutorial/request-files/).

## Start it

The workspace already has the pinned dependencies installed. A new checkout
needs the updated dependency lock and the prepared corpus/model/index from
Modules 1–3.

```bash
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1 --no-access-log
```

Open `http://127.0.0.1:8000/docs` for interactive documentation and requests.
`/openapi.json` exposes the API specification. Model loading happens lazily on
the first query and is cached per process. Startup/health do not initialize
BGE-M3 or contact Qwen. This API uses the default all-jobs prepared corpus.

Use one worker: runs are process-local and local Qdrant access is serialized.
This is a local learning API without accounts/authentication; keep the documented
localhost binding. Multiworker session storage and deployment are separate work.

## Endpoints

| Method | Path | Responsibility |
|---|---|---|
| GET | `/health` | Liveness, synthetic marker, and busy status |
| GET | `/ready` | Confirm the validated corpus was loaded |
| POST | `/match` | Upload, prepare, retrieve, score, and explain |
| GET | `/runs/{run_id}/jobs/{job_id}` | Inspect any corpus job against the cached resume |
| DELETE | `/runs/{run_id}` | Forget the cached run; returns 204 |

Readiness is explicitly a lightweight corpus check. Models/indexes are verified
when used; `/ready` does not claim a healthy remote Qwen service or audited index.
If the corpus was missing at startup, prepare it and restart the API. A running
process uses its loaded corpus snapshot.

## Example requests

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/match \
  -F 'resume=@artifacts/demo_resumes/ai.pdf;type=application/pdf' \
  -F 'preferences=<examples/preferences.json' \
  -F 'use_llm=false'
```

`use_llm=false` gives deterministic explanations. Its default is true; use true
to request remote Qwen plans. Generation sends the canonical skills and
assessment labels described in Module 4, not raw PDF pages.

The response contains `run_id`, `expires_in_seconds`, `result` (the existing
`MatchResponse`), and `preparation_warnings`. Use its run ID for another job:

```bash
curl http://127.0.0.1:8000/runs/PASTE_RUN_ID/jobs/syn_job_234
curl -X DELETE http://127.0.0.1:8000/runs/PASTE_RUN_ID
```

Replace `PASTE_RUN_ID` with the actual value. The run in saved verification
responses was deleted and cannot be reused.

## Selected-job inspection

Details work for jobs outside the original candidate pool, preference failures,
known requirement failures, and zero skill overlap. Output includes the original
job, relevance when available, both coverage measures, all skill/condition
assessments, rejection reasons, warnings, and:

- `passes_policy`: whether this job passes the current acceptance gates.
- `was_recommended`: whether it was actually selected in the original result.

A passing job can still have been omitted because of the top-five limit. A
zero-overlap job has an empty positive-reasons list rather than a fabricated
strength. A resume without a query gets null relevance.

An existing cosine score is reused. Otherwise only that job is searched in the
current index, without preference exclusions that would prevent inspection;
preference mismatches are explicitly reported by scoring. Detail makes no new
Qwen calls and never regenerates explanations for all jobs. Inspection is not
an accepted recommendation.

## Limits and retained data

| Setting/behavior | Default |
|---|---|
| PDF size / pages | 5 MiB / 10 |
| Extracted text | 300,000 characters |
| Preferences JSON | 16 KiB |
| Total multipart body | PDF size limit + 64 KiB envelope allowance |
| HTTP processing deadline | 120 s (`JOBFIT_API_TIMEOUT_SECONDS`) |
| Heavy processing slots | 1; concurrent work receives 429 |
| Run lifetime | 1800 s (`JOBFIT_API_SESSION_TTL_SECONDS`) |
| Maximum cached runs | 20 (`JOBFIT_API_MAX_SESSIONS`) |

The body limiter counts incoming chunks, including requests without
Content-Length, before oversized uploads accumulate in the multipart parser.
PDF bytes are checked separately. A generated temporary filename is used under
`artifacts/uploads/`; user filenames never become storage paths. Actual PDF
content is checked independently of extension/MIME type.

Temporary PDFs are removed after success/failure. Multipart files are closed.
API requests do not write raw pages or results to the CLI artifact folders.
Prepared resume data remains in the bounded in-memory run cache for evidence
checking. Deletion, expiry, capacity eviction, or restart removes runs. Expiry
is pruned on requests rather than a background timer. Returned evidence quotes
are available to the caller holding the run ID; this is a local demo capability,
not an authenticated multiuser security model.

Blocking model/database work runs in a worker thread. Health checks remain
responsive. A response deadline cannot forcibly terminate a Python thread:
after 504 or disconnection, a current CPU/provider operation may finish later.
The processing slot stays occupied, the result is discarded, and worker cleanup
removes its PDF. Further Qwen attempts check the deadline. This prevents
overlapping local database work after timeout; it is not hard process cancellation.

The application does not log raw resumes, preferences, keys, or provider error
bodies. The documented server command disables access logs.

## Errors

| Status | Meaning |
|---|---|
| 413 | PDF or total request too large |
| 422 | Invalid form/preferences or invalid/empty/locked/unreadable/oversized-page PDF |
| 429 | Another heavy request is processing |
| 503 | Corpus, cached model, vector index, or temporary storage unavailable |
| 504 | HTTP processing deadline exceeded |
| 404 | Unknown/expired/deleted run or unknown job |
| 500 | Unexpected processing error with a generic response |

Errors have stable codes/messages without raw input/provider bodies. No-match is
a successful 200 response. Qwen failure also yields 200 when Python matching
succeeds, but its recommendations explicitly show `generation_status=fallback`
and a warning. Clients must inspect that status.

## Files and verification

| File | Purpose |
|---|---|
| `app/main.py` | Routes, upload/body limits, gate/deadline, and temporary runs |
| `app/api_service.py` | Connect modules, cache embeddings, inspect selected jobs |
| `app/api_schemas.py` | Typed HTTP contracts, including failed/zero-overlap detail |
| `tests/test_module5.py` | 13 HTTP tests with real PDFs and local Qdrant |
| `scripts/verify_api.py` | Real HTTP synthetic verification against a running server |

All **85 main tests** pass, plus five Python-only comparison tests. API tests use
controlled embeddings and generation failures, actual PDFs/application logic,
and persistent local Qdrant. They cover valid matching/detail, zero overlap,
limits with/without Content-Length, invalid PDFs/preferences, missing services,
fallback, busy responses, timeout cleanup, expiry/deletion, and capacity eviction.

A real HTTP server was separately verified using real BGE-M3, Qdrant, scoring,
and remote Qwen. The AI PDF returned two `qwen_grounded_plan` recommendations;
`syn_job_234` was inspected as a rejected job. The run was deleted and no
temporary PDFs remained. Saved evidence:

```text
artifacts/verification/module5_ai_http_response.json
artifacts/verification/module5_selected_job_http_response.json
artifacts/verification/module5_http_verification.json
```

Reproduce with a running server:

```bash
.venv/bin/python scripts/verify_api.py
.venv/bin/python scripts/verify_api.py --use-llm
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m unittest discover -s python_only -v
```

TestClient stalled at event-loop startup inside the development sandbox; HTTP
tests were run outside that sandbox. Actual verification used localhost port
8765; the normal start command above uses port 8000.

Module 5 completes the baseline API milestone. Next is Module 6: BM25 and
dense/hybrid comparison. Held-out evaluation and deployment remain separate work.
