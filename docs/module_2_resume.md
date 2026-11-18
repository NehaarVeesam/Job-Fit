# Module 2 — Resume Parsing and Candidate Preparation

## What it does

Module 1 prepared the job side. Module 2 prepares the resume side:

```text
Resume PDF + optional explicit preferences
    -> local PDF text extraction, preserving page numbers
    -> corpus-vocabulary skill mentions and section facts with quotes
    -> conservative professional-experience duration
    -> compact retrieval query
    -> validated JSON artifacts saved locally
```

Module 2 uses Python, Pydantic, and pinned `PyMuPDF==1.26.7`.
Text extraction uses `page.get_text("text", sort=True)` from the
[official PyMuPDF API](https://pymupdf.readthedocs.io/en/latest/page.html).
No Qwen request, embedding, retrieval, job score, or suitability decision occurs.
LangChain and BGE-M3 enter in Module 3; matching and generation enter in Module 4.

## Inputs

1. A searchable PDF with selectable text, within configured limits (default
   5 MiB and 10 pages).
2. A successful Module 1 output directory, default `artifacts/dataset/all/`.
   Its job records supply the skill vocabulary. Both `jobs.jsonl` and
   `skill_evidence.jsonl` are checked against `validation_report.json` hashes.
3. Optional preferences JSON using the existing `Preferences` contract.

The original raw jobs, prepared jobs, and resume PDF are never rewritten.
The prepared corpus must have status `ok`; a changed artifact or unresolved
Module 1 review condition stops preparation. Skill evidence is integrity-checked
here; its actual job quotes will be consumed for matching later.

## Run the synthetic demo

From the project root:

```bash
# Prepare jobs if not already done
.venv/bin/python -m app.dataset_loader

# Turn five public synthetic text examples into actual PDFs
.venv/bin/python scripts/create_demo_resumes.py

# Parse one PDF and build the candidate profile
.venv/bin/python -m app.prepare_resume artifacts/demo_resumes/ai.pdf --preferences examples/preferences.json

# Additional example: no implied professional experience from college projects
.venv/bin/python -m app.prepare_resume artifacts/demo_resumes/fresher.pdf

# No recognized corpus skills produces a valid profile with a null query
.venv/bin/python -m app.prepare_resume artifacts/demo_resumes/unrelated.pdf

# Use a different prepared corpus explicitly
.venv/bin/python -m app.prepare_resume artifacts/demo_resumes/backend.pdf --corpus-dir artifacts/dataset/first_100

# All tests
.venv/bin/python -m unittest discover -s tests -v
```

For a new checkout, install the updated dependency lock first:

```bash
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
```

Use your own resume by replacing the PDF path. Preferences are independent user
choices: a resume address never becomes a preferred city. Without a preferences
file, preferences remain empty/default. `--corpus-dir` chooses an alternative
prepared subset; it does not regenerate jobs or apply additional filters.

## Output location and files

The CLI prints the output folder:

```text
artifacts/resumes/<SHA-256 of the input PDF>/
  parsed_resume.json
  candidate_profile.json
  preferences.json
  prepared_resume.json
  preparation_report.json
```

| File | Purpose |
|---|---|
| `parsed_resume.json` | Complete extracted page text with one-based page numbers |
| `candidate_profile.json` | Skills, projects, work/education lines, and duration with supporting quotes |
| `preferences.json` | Validated explicit preferences, separate from resume facts |
| `prepared_resume.json` | Combined contract including extracted pages, profile, preferences, query, and warnings |
| `preparation_report.json` | PDF hash, parser/profile version, corpus provenance, vocabulary hash, counts, and warnings |

The combined `prepared_resume.json` is the main handoff to future modules.
For example, a skill fact looks like:

```json
{
  "text": "Python",
  "evidence": [{"quote": "Python, LangChain, RAG, FastAPI, Docker", "page": 1}]
}
```

The AI example with supplied preferences produces this query:

```text
Target role: Generative AI Engineer
Skills: Docker, FastAPI, LangChain, Python, RAG
```

Skill order is deterministic for the extracted input. The query uses canonical
skill names and an explicit target role; raw project text, name, email, phone,
resume address, and employer names are not copied into the automatic query.
This is a compact baseline query; richer contextual query preparation can be
evaluated later.

Raw pages and facts do retain personal information for local evidence checking.
All outputs are under ignored `artifacts/`; the CLI prints counts and paths,
not raw resume content. Nothing is sent to the configured remote Qwen server.
Each output file is replaced atomically, but the files are not a transactional
bundle. Preparing the same PDF again overwrites its folder, including preferences.

## Extraction rules and limitations

### Pages and errors

- Preserve every page number and its extracted text, including blank pages.
- Warn about pages with no extractable text; partially image-based resumes may
  have incomplete profiles.
- Reject entirely blank/image-only PDFs with a message requesting searchable
  text. OCR is deferred.
- Reject locked PDFs, non-PDF input, unreadable/empty files, corrupt PDFs, and
  PDFs requiring structural repair.
- Enforce configured byte/page limits and a 300,000 extracted-character guard.
  Exceeding a limit produces an error, never silent truncation.

Text ordering is a basic PDF extraction baseline. Complex columns, tables,
unusual fonts, or layouts may require later improvement. Character limits are
resource guards, not model-token validation; actual tokenization is Module 3.

### Skill mentions and section facts

- Recognize common headings such as Skills, Technical Skills, Summary, Projects,
  Work Experience, Internships, and Education, including `Skills: Python, SQL`.
- Skip header text before a recognized section. Unrecognized layouts may miss
  facts or produce no query; warnings request review rather than inventing skills.
- Match corpus skills case-insensitively with boundaries protecting `C++`, `C#`,
  `.NET`, `Node.js`, and `CI/CD`. Java does not match JavaScript; FastAPI does not
  imply Flask. No inferred synonyms or related-tool equivalences are added.
- Exclude obvious contact lines and conservatively skip positive skill evidence
  on lines containing negative, learning, or aspirational language. This is a
  line-level heuristic and may skip valid skills on the same line.
- Record project, experience, and education lines as quoted section facts.
  These are source statements, not normalized employers, qualifications, or
  verified achievements.
- Treat all resume text as inert source data; instructions embedded in it never
  trigger tools or model calls.

**A skill mention is not verified expertise.** Mention detection does not establish
proficiency, recency, job suitability, or complete understanding of negation.
Short/common skill names and unusual headings remain limitations. The active
corpus bounds the vocabulary, so skills absent from that corpus are not extracted.

### Experience years

Accept only a narrow explicit statement in Summary or Experience, for example:

```text
Professional experience: 3 years
2.5 years of full-time experience
```

Save its exact quote. Conflicting explicit values leave years unknown. Do not sum
employment dates, infer years from seniority, or convert college-project/internship
durations to professional years. Statements such as "over 3 years" remain unknown
because this schema stores a single exact number, not a lower bound.

### Evidence and empty queries

`PreparedResume` validates every profile quote against its cited extracted page.
A nonexistent page or fabricated quote fails validation. This proves literal
source presence, not semantic correctness or truth of a candidate's claim.

Without recognized skills or an explicit target role, `retrieval_query` is null
and warnings explain why. This remains a valid preparation result; later retrieval
must handle an unavailable query explicitly. No fake skills or generic fallback
job query are inserted. An explicit target role alone can produce a query, but
it never becomes evidence of qualification.

## Files added and verification

| File | Responsibility |
|---|---|
| `app/resume_parser.py` | Bounded PDF extraction and actionable input errors |
| `app/profile.py` | Corpus integrity/vocabulary, rule-based facts, and query construction |
| `app/prepare_resume.py` | CLI orchestration and local JSON outputs |
| `app/schemas.py` | Added `PreparedResume` with cross-page quote validation |
| `scripts/create_demo_resumes.py` | Reproducible synthetic text-to-PDF demo workflow |
| `examples/resumes/*.txt` | AI, backend, analyst, fresher, unrelated public synthetic inputs |
| `tests/test_module2.py` | 13 tests using real generated PDFs and profile failure cases |

Verified: all **42 tests pass** across Modules 0–2 (19 + 10 + 13). All five demo
profiles round-trip through the combined contract. The AI example has five
evidenced skill mentions and three explicitly stated years; the fresher retains
unknown years; the unrelated example has no automatic query. Tests also cover
PDF errors/limits, contact-free queries, symbol-bearing skills, uncertain claims,
corpus tampering, and fabricated quotes/pages.

## Next

Module 2 is complete as a deterministic baseline. Module 3 is now implemented: see the [LangChain tutorial](module_3_langchain.md)
for Documents, token counts, normalized BGE-M3 embeddings, and Qdrant retrieval.
Module 4 is complete: see [its guide](module_4_matching.md).
Module 5 is complete: see [the API guide](module_5_api.md).
Next is Module 6: BM25/hybrid comparison.
