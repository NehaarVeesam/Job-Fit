# Module 1 — Dataset Loading and Validation

## Purpose and architecture

Module 1 turns the supplied JSONL into a validated, reproducible job corpus for
later resume matching. It uses Python's standard library and the existing
Pydantic `JobRecord` contract. No new dependencies are required.

```text
dataset/jobs_synthetic_250.jsonl       Original input; never modified
          |
          v
Read every line -> validate -> check duplicate IDs/content
          |
          v
Sort by source job ID -> optional subset/date filtering
          |
          v
artifacts/dataset/<mode>/
  jobs.jsonl                         Validated selected records
  skill_evidence.jsonl               Verbatim job-side evidence
  validation_report.json             Provenance, accounting, flags, statistics
```

These are fictional India AI/software jobs. Dates do not establish real openings.
The CSV is an alternate export and is not loaded or appended to JSONL.

## Run it

From the project root:

```bash
# Default: all 250 jobs, no date exclusions
.venv/bin/python -m app.dataset_loader

# Optional development subset: IDs 001–100, regardless of file ordering
.venv/bin/python -m app.dataset_loader --subset first_100

# Optional fixed-date benchmark: 246 jobs
.venv/bin/python -m app.dataset_loader --recency --reference-date 2026-11-18 --window-days 90

# Explicitly override any subset/date-filter values in .env
.venv/bin/python -m app.dataset_loader --subset all --no-recency

# Verify both modules
.venv/bin/python -m unittest discover -s tests -v
```

The CLI reads `JOBFIT_DATASET_PATH`, `JOBFIT_ARTIFACTS_DIR`,
`JOBFIT_CORPUS_SUBSET`, `JOBFIT_RECENCY_FILTER_ENABLED`,
`JOBFIT_REFERENCE_DATE`, and `JOBFIT_RECENCY_WINDOW_DAYS` from existing settings.
CLI arguments override subset/date settings for that invocation without editing
`.env`. No changing system date is used. The CLI validates application settings,
including the configured remote Qwen endpoint, but never contacts that endpoint.

## Files and functions

| File or function | Responsibility |
|---|---|
| `app/dataset_loader.py` | Loader, evidence builder, report writer, and module CLI |
| `CorpusOptions` | Subset and reproducible date-filter configuration |
| `prepare_corpus(source, options)` | Read/validate records; return `PreparedCorpus` containing jobs, evidence, and report |
| `write_corpus(corpus, output_dir, source)` | Write artifacts outside the source folder, with output checksums |
| `tests/test_module1.py` | Ten tests for preservation, accounting, filters, evidence, and failure cases |
| `docs/module_1_dataset.md` | This implementation and usage guide |

`app/schemas.py` supplies the existing validation rules. `app/config.py` supplies
settings. Neither an LLM nor LangChain is needed to read structured job records.
LangChain Documents, tokenizer measurement, BGE-M3, and Qdrant enter in Module 3.

## Validation and duplicate policy

Every physical JSONL line is accounted for. Blank lines, invalid UTF-8, malformed
JSON, repeated JSON fields, non-finite JSON numbers, and schema violations are
rejected with line numbers and reasons. Schema checks include synthetic markers,
skill evidence, separate required/preferred skills, and ordered experience bounds.

For a repeated job ID, the first valid record wins. Later identical occurrences
are reported as `duplicate_record`; changed occurrences are reported as
`conflicting_job_id`. This is an explicit review policy, not automatic merging.
Identical normalized content under different IDs is flagged as suspicious and
retained. Sharing a title alone is never grounds for removing a job.

Nulls, skill spelling, source descriptions, hard constraints, and source IDs are
preserved. Pydantic serializes dates as ISO strings and experience numbers as
numbers; formatting and JSON key order may differ from input. No description
cleaning, skill equivalence inference, or requirement regeneration occurs.

The report contains separate summaries before and after filtering. Accounting is:

```text
input_lines = rejected + valid_unique
valid_unique = excluded + output
```

`requires_review` and exit code 1 indicate rejected/suspicious records or an empty
input. Valid records and a report are still written so problems can be reviewed.
Exit code 0 means validation completed without those review conditions. A valid
corpus filtered down to zero records is reported as zero output with explicit
exclusions; it is distinct from an empty source file.

## Optional filters

All 250 jobs are retained by default. The report still flags stale/future dates.

`first_100` selects `syn_job_001` through `syn_job_100` by ID, rather than taking
the first 100 physical lines. Subset selection precedes date exclusions; each
excluded job receives the first applicable reason.

Recency mode accepts dates in the inclusive interval:

```text
reference_date - window_days <= posted_at <= reference_date
```

At 2026-11-18 with a 90-day window, the excluded stale jobs are:

- `syn_job_128`
- `syn_job_157`
- `syn_job_196`
- `syn_job_238`

Future-dated jobs would receive `after_reference_date` in recency mode.
The supplied corpus has none. These are corpus preparation filters, not candidate
location, eligibility, or suitability decisions.

## Output files

The default directory is `artifacts/dataset/all/`. Other modes have separate
directories: `first_100/` and `all_recent_2026-11-18_90d/`, for example.

**`jobs.jsonl`** contains one validated selected `JobRecord` per line, preserving
all 21 fields. It is sorted by `job_id` and is the prepared corpus for later work.

**`skill_evidence.jsonl`** contains each job ID, its original one-based source line,
and evidence for every required/preferred skill. Each entry records the exact
skill, a full source line containing it, and zero-based Python character offsets
`start`/`end` into `description`; `end` is exclusive. These offsets locate the skill,
not the entire quote. The first exact occurrence is used. This validates literal
job-side presence, not resume evidence or semantic equivalence between tools.

**`validation_report.json`** contains:

- Source filename, byte size, SHA-256, dataset ID, and synthetic marker.
- Validation version, subset, reference date, and recency settings.
- Input/valid/rejected/excluded/output counts and explicit reasons.
- Suspicious duplicates, stale/future IDs, and unknown metadata IDs.
- Domain, seniority, work-mode, company, and hard-constraint statistics.
- Description lengths in characters and whitespace-separated words.
- SHA-256 checksums linking the report to the two generated JSONL files.

Word counts are **not model token counts**. Actual tokenizer checks are pending
Module 3. Reports contain no API keys or remote endpoint values.

The writer rejects output inside the input folder and symlink output files.
Each file is replaced atomically; the three files are not a transactional bundle,
so check report checksums before consuming outputs after an interrupted run.
Rerunning the same input/options produces byte-identical artifacts and replaces
the mode's existing files. Generated artifacts are ignored by Git.

## Verified results

| Property | Result |
|---|---|
| Default input / valid / output | 250 / 250 / 250 |
| Rejected records / suspicious duplicates | 0 / 0 |
| AI/ML / Software | 125 / 125 |
| Fictional companies | 42 |
| Jobs with hard constraints | 34 |
| Unknown work mode / experience / city | 3 / 4 / 4 |
| Description characters, minimum / maximum | 727 / 1045 |
| Optional ID subset output | 100 |
| Optional 90-day output | 246 |
| Tests across Modules 0 and 1 | 29 passing (19 + 10) |
| Original dataset files | SHA-256 checks unchanged |

Tests preserve the job 213 injection text as inert data, the same-role/different-
city cases, the seniority ladder, and tokens including `C++`, `C#`, `.NET`,
`Node.js`, and `CI/CD`. This stage never executes source instructions; resistance
of future generation prompts to injection must be tested in Module 4.

## Status and next step

Module 1 is complete. It prepares jobs; it does not parse resumes, embed text,
rank jobs, assess candidates, or generate explanations. Qwen is configured through
the existing remote OpenAI-compatible endpoint, and no inference was performed.

Module 2 is now implemented: see [its guide](module_2_resume.md) for PDF
extraction, evidence-backed candidate profiles, and retrieval queries.
Module 3 is complete: see [the LangChain guide](module_3_langchain.md).
Module 4 is complete: see [its guide](module_4_matching.md).
Module 5 is complete: see [the API guide](module_5_api.md).
Next is Module 6: BM25/hybrid comparison.
