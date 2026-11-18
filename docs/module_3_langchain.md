# Module 3 — Learn LangChain Through Dense Job Retrieval

## The goal

Modules 1 and 2 prepared the two inputs. Module 3 answers:

> Given this resume's search query, which jobs are most similar in meaning?

It returns retrieval candidates and cosine scores. Requirement coverage,
eligibility, recommendation acceptance, and Qwen explanations are Module 4.
All jobs are synthetic, and retrieval scores are not hiring probabilities.

## 1. How the pieces fit together

```text
OFFLINE INDEXING — run once per corpus/model/text configuration

Module 1 jobs.jsonl + validation report + evidence checksum
    -> one LangChain Document per job
    -> measure actual tokenizer lengths
    -> HuggingFaceEmbeddings running BAAI/bge-m3 locally
    -> check 1024 dimensions, finite values, and unit L2 norm
    -> QdrantVectorStore writes vectors + text + metadata
    -> persistent Qdrant data and a provenance manifest

PER RESUME

Module 2 prepared_resume.json
    -> compact retrieval_query + explicit preferences
    -> shared Python rules produce eligible job IDs
    -> same BGE-M3 model embeds the query
    -> Qdrant searches eligible vectors by cosine similarity
    -> ranked candidates with original job records and warnings
```

LangChain supplies common interfaces between these components. It is a Python
library; it is not an embedding model, database, or LLM. You still explicitly
choose the model, the searchable text, metadata, filters, and output rules.

## 2. The LangChain concepts used in our code

### Document: text plus metadata

A LangChain `Document` is a container:

```python
from langchain_core.documents import Document

document = Document(
    page_content="Generative AI Engineer. Required skills: Python, LangChain, RAG.",
    metadata={"job_id": "syn_job_001", "is_synthetic": True},
)
```

`page_content` is the text we embed. `metadata` preserves information for source
traceability, filtering, and later assessment. Converting JSON into a Document
does not generate an embedding or call an LLM.

Our implementation is `app/documents.py::job_document`. Its searchable text is:

```text
Title: <source title>
Role family: <source role family>
Required skills: <source array joined for search>
Preferred skills: <source array joined for search>
Description:
<complete original description>
```

Metadata retains all 21 source fields, plus text version/checksum, index ID,
source checksum, and corpus subset. Structured skill arrays and constraints stay
available; searchable text does not replace the source record. Requirements also
occur in descriptions, so this initial representation repeats them intentionally.
Alternative text representations can be evaluated later.

### Embeddings: turning meaning into a numeric representation

An embedding model maps text to a list of numbers. Our chosen model is
`BAAI/bge-m3`, producing 1024-dimensional dense vectors. The model card documents
up to 8192 tokens and dense usage through Sentence Transformers:
[official BGE-M3 model card](https://huggingface.co/BAAI/bge-m3).

```text
"Python backend API developer"
    -> embedding model
    -> [0.012, -0.037, ..., 0.008]   # illustrative, not an actual saved vector
```

This is a fixed-length representation, not a JSON summary and not generated
prose. Dimensions are learned features; we do not assign one coordinate to
"Python" and another to "Java". Similar meaning often produces nearby vectors,
but this is not a guarantee that a candidate qualifies for a job.

LangChain's `Embeddings` interface has two important methods:

```python
job_vectors = embeddings.embed_documents([job_text_1, job_text_2])
query_vector = embeddings.embed_query(resume_query)
```

The first takes a list of texts and returns a list of vectors. The second takes
one query and returns one vector. Both sides use the same model, revision,
dimension, and normalization settings in this project.

### HuggingFaceEmbeddings: the integration adapter

`langchain-huggingface` provides `HuggingFaceEmbeddings`, which satisfies the
LangChain interface by calling Sentence Transformers. Its local inference path
runs the model in-process:
[official LangChain integration](https://docs.langchain.com/oss/python/integrations/embeddings/huggingfacehub).

Conceptually:

```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True, "batch_size": 4},
)
```

That illustrative constructor can download model files. **Our actual loader**
first resolves the pinned local snapshot and passes its directory as `model_name`,
with `local_files_only=True` and `trust_remote_code=False`. Downloading is an
explicit separate script. This avoids an unwanted tokenizer metadata request
observed when loading the remote repository name with the pinned libraries.

`app/embeddings.py::CheckedEmbeddings` wraps the adapter and also implements the
same LangChain interface. Before encoding it checks actual tokenizer lengths;
afterward it checks count, dimension, finiteness, and vector norm. Qdrant can use
our checked wrapper wherever it expects LangChain `Embeddings`.

The runtime uses the pinned snapshot:

```text
5617a9f61b028005a4858fdac845db406aefb181
```

Only dense Sentence Transformers files are downloaded; BGE-M3 sparse/ColBERT
output and learned reranking are not implemented here.

### Tokens and why we check lengths first

Models read tokens, not Python characters or whitespace words. The 850-character
description statistics from Module 1 cannot establish model input length.

Module 3 tokenizes the complete searchable job text with special tokens and no
truncation. It checks the query too. The limit is the smaller of the loaded
Sentence Transformer limit and 8192.

Each current job fits as one Document, so we do not split it into chunks. An
oversized future job or query raises an error requesting explicit chunking.
We never silently truncate a job's requirements or a resume query.

### Normalization and cosine similarity

Every vector must have L2 norm approximately 1:

```text
sqrt(v1² + v2² + ... + v1024²) ≈ 1
```

Cosine similarity compares direction. For unit vectors it equals their dot
product. A two-dimensional illustration:

```text
Query: [1.0, 0.0]
Job:   [0.8, 0.6]
Both lengths are 1; cosine similarity is 0.8.
```

Our vectors have 1024 dimensions, not two. Scores range from -1 to 1, with higher
values indicating closer embedding directions. A score of 0.8 is not "80% of
requirements met". Module 4 will calculate separate required/preferred coverage
from evidence and counts.

### Vector store: LangChain's connection to Qdrant

Qdrant stores each job's vector, searchable text, and metadata. LangChain's
`QdrantVectorStore` supplies the common add/search interface. The integration
supports persistent local Qdrant:
[official Qdrant/LangChain guide](https://qdrant.tech/documentation/frameworks/langchain/).

The relevant shape of our code is:

```python
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

client = QdrantClient(path="artifacts/qdrant")
# Our index builder creates/verifies a 1024-dimensional cosine collection first.
store = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=embeddings,
    distance=models.Distance.COSINE,
    validate_collection_config=False,
)

store.add_documents(documents, ids=stable_point_ids, batch_size=4)
results = store.similarity_search_with_score(query, k=5, filter=eligible_filter)
client.close()
```

Our own checks verify collection size, distance, full provenance, IDs, and
payloads. That is why this constructor skips the integration's extra validation
embedding of dummy text. `add_documents` calls `embed_documents` internally;
the search method calls `embed_query` internally. We do not need to manually
generate vectors and also pass them to those same methods.

Search returns `(Document, score)` pairs. We recover original job records by
their preserved `syn_job_*` IDs and return traceable candidates.

### Retriever: a reusable query-to-documents interface

LangChain can expose a vector store as a retriever:

```python
retriever = store.as_retriever(
    search_kwargs={"k": 5, "filter": eligible_filter}
)
documents = retriever.invoke(query)
```

This returns Documents and is convenient when connecting retrieval to a later
chain. Our current application uses `similarity_search_with_score` because we
want explicit scores in the candidate output. These are two interfaces to the
same underlying retrieval, not two different embedding models.

No prompt template, generation chain, agent, or Qwen call is needed for Module 3.

## 3. Persistence, IDs, and reproducibility

`app/vector_index.py` manages:

- Deterministic UUIDv5 Qdrant point IDs derived from dataset ID and source job ID.
- A full index identity recording corpus checksum/subset/filter settings,
  searchable-text version/checksum, and semantic embedding settings.
- A collection name built from the configured synthetic base plus 16 characters
  of the index hash. Full metadata checks guard against mismatched identities.
- A ready manifest written only after verifying complete point coverage,
  correct payloads, dimensions, and unit vector norms.
- Reuse of a verified complete index without embedding the jobs again.
- Recovery of a partial index by upserting stable IDs; no duplicate creation.

Changing the corpus, subset, dates, searchable text, model revision, or other
semantic vector configuration produces a separate collection. Existing
collections are not deleted. Device/batch/thread timing settings are recorded
but do not change semantic identity. A corrupted/mismatched payload or foreign
collection is rejected rather than silently overwritten.

Persistent local Qdrant uses `artifacts/qdrant/`. Only one process should open that
directory at once; run indexing and search sequentially. Local mode is suitable
for this small learning corpus and uses its local search implementation. Its
timings do not represent a deployed Qdrant server.

Server configuration is also available through `JOBFIT_QDRANT_MODE=server` and
`JOBFIT_QDRANT_URL`. It uses the same collection/vector-store code but is not
covered by the local live-run evidence. Multiworker API deployment is later work.

## 4. Metadata filters and unknowns

`app/filters.py` defines shared Python rules, then passes eligible job IDs as a
Qdrant filter so exclusions happen before top-k selection. Later BM25 can reuse
the same ID set.

- City matching is case-insensitive, with surrounding whitespace ignored.
- A known different city fails an explicit city preference.
- A null-city remote job is explicitly India-wide and passes a city preference.
- Unknown city/work-mode metadata follows `include_with_warning` or `exclude`.
- Known work-mode mismatches are excluded.
- Target role influences the prepared semantic query; it is not an exact-title
  hard filter.
- Experience, education, and mandatory constraints are retained for Module 4;
  semantic similarity does not establish their outcomes.

The result reports eligible counts, applied preferences, and unknown/India-wide
warnings. A resume with no query returns `no_query`; filters leaving zero jobs
return `no_eligible_jobs`. Neither fabricates candidates.

## 5. Run it yourself

Install the updated lock in a new checkout before running these commands. The
lock uses the official CPU PyTorch wheel index and describes the verified
Linux/Python 3.11 environment; GPU/platform-specific packaging is separate work.

```bash
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .

# One-time network download of model files (roughly 2.3 GB of dense weights).
.venv/bin/python scripts/download_embeddings.py

# Prepare jobs if needed, then build the full index.
.venv/bin/python -m app.dataset_loader
.venv/bin/python -m app.dense_cli index

# Generate and prepare a demo resume if needed.
.venv/bin/python scripts/create_demo_resumes.py
.venv/bin/python -m app.prepare_resume artifacts/demo_resumes/ai.pdf --preferences examples/preferences.json

# Use the prepared_resume.json path printed by the previous command.
.venv/bin/python -m app.dense_cli search --prepared-resume artifacts/resumes/<PDF_HASH>/prepared_resume.json --k 5

# Use exactly the same alternative corpus for indexing and retrieval.
.venv/bin/python -m app.dense_cli index --corpus-dir artifacts/dataset/first_100
.venv/bin/python -m app.dense_cli search --corpus-dir artifacts/dataset/first_100 --prepared-resume artifacts/resumes/<PDF_HASH>/prepared_resume.json --k 5

.venv/bin/python -m unittest discover -s tests -v
```

Replace `<PDF_HASH>` with the actual folder name; it is a placeholder, not a
literal shell command argument. Default input is the all-jobs corpus; explicitly
select an alternative corpus directory in both commands. The Qwen endpoint/key
are preserved and unused by this module. Normal cached local-mode operation
does not require an embedding API or model-server connection.

## 6. Files and outputs

| File | Responsibility |
|---|---|
| `app/documents.py` | LangChain Document construction, text checksum/version, stable UUIDs |
| `app/embeddings.py` | Cached Hugging Face integration plus tokenizer/vector guards |
| `app/filters.py` | Shared preference/unknown rules and Qdrant ID filter |
| `app/vector_index.py` | Qdrant persistence, manifest, idempotency, and audits |
| `app/retrieval.py` | Filtered dense candidates with cosine scores and provenance |
| `app/dense_cli.py` | Index/search commands and concise progress |
| `scripts/download_embeddings.py` | Explicit model download at the pinned revision |
| `tests/test_module3.py` | Fast offline tests with controlled vectors and real local Qdrant |

Generated output, all ignored by Git:

```text
artifacts/model_cache/                 Pinned embedding model files
artifacts/qdrant/                      Persistent Qdrant collections
artifacts/indexes/<INDEX_ID>.json       Ready index manifest and timing/audit report
artifacts/retrieval/<INPUT_HASH>_5.json Ranked candidates for a prepared resume
```

The model cache, Qdrant path, batch size, CPU threads, and backend are settings
in `app/config.py` and `.env.example`. Paths are resolved against the project root
and generated data is rejected inside the configured raw dataset folder.
Remote Qwen endpoint/credentials were not replaced.

The retrieval output contains original job IDs/records, cosine values, rank,
corpus/index identity, eligible count, applied preferences, warnings, and timing.
It is a candidate report rather than the Module 4 recommendation contract.
Resume text is not printed in progress logs; reports remain local artifacts.

## 7. Verification and limits

Unit/integration tests use deterministic test vectors to verify mechanics:
Document preservation, UUIDs, input token guards, vector checks, index identity,
disk reopening, unchanged-index reuse, partial-index recovery, filters, empty
outcomes, and provenance mismatch rejection. Test vectors are not evidence of
BGE-M3 quality. Live BGE-M3 evidence is recorded separately in index/retrieval
artifacts and the [companion results note](module_3_results.md).
All 56 fast tests pass. The real index contains 250 points, with 209–317 model
tokens per searchable job; verified vector norms range from 0.999999939 to
1.000000104. Re-indexing reused all vectors, and a five-profile real-model check
succeeded with socket connections blocked.

Timing fields distinguish model load (CLI search output), embedding/search, and
index embedding/upsert. Index times start after model load. Initial downloads
are not included. The CLI records model-load time separately in its report. Startup/warmup and CPU cache affect measurements; a few demo
queries do not establish production latency or held-out matching quality.

Short, uniform synthetic descriptions and clean skill lists make this corpus
easier than real jobs. Job 213's embedded instruction stays source text; no
instruction is executed by retrieval. It can still affect semantic similarity
as text. Module 4 must separately test generation resistance to prompt injection.

## 8. Learning exercises

1. Open `job_document` and inspect the difference between `page_content` and
   `metadata`. Which fields affect the embedded text directly?
2. Run indexing twice. Check the manifest's reuse flag and point count. Why does
   a stable source ID need a UUID representation for Qdrant?
3. Search the AI and backend demos. Compare titles and cosine scores. Do not
   interpret the scores as requirement percentages.
4. Change explicit city/work-mode preferences by preparing the PDF with another
   preferences file. Compare eligible counts and warnings.
5. Follow one result's `job_id` back to Module 1's source and skill evidence.
6. Read `CheckedEmbeddings`: why should token guards wrap both document and
   query methods rather than only the indexing command?

Module 4 is now implemented: see [its guide](module_4_matching.md) for coverage,
constraints, acceptance, and grounded Qwen plans. Module 5 is complete: see [the API guide](module_5_api.md).
Next is Module 6: BM25/hybrid comparison.
