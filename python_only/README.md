# Module 3 with ordinary Python and direct library calls

This is a second implementation of Module 3 for learning. Read `engine.py` from
top to bottom, then `cli.py`. It uses Sentence Transformers and QdrantClient
directly. It does not import LangChain, wrap its classes, or call the main
LangChain indexing/retrieval modules.

"Python only" here means **without LangChain**. We still use libraries for
embedding inference, database operations, and input validation. Reimplementing
PyTorch or a vector database would obscure the retrieval workflow.

## What is shared

The examples deliberately reuse the completed framework-independent code:

- `app/config.py`: settings, including model cache and revision.
- `app/schemas.py`: Pydantic job/resume contracts.
- `app/resumes/profile.py::load_prepared_jobs`: validated Module 1 corpus and checksums.
- `app/retrieval/filters.py`: identical city/work-mode/unknown policies.

These imports do not bring in LangChain. The implementation has its own text
container, embedding calls, Qdrant point construction, upsert, and query logic.
Modules 0–2 were already implemented without LangChain; this folder demonstrates
the Module 3 part where LangChain first entered the main project.

## Read the flow

```text
JobRecord
    -> JobText(text, metadata)             Ordinary Python dataclass
    -> SentenceTransformer.encode(...)    Explicit embedding call
    -> PointStruct(id, vector, payload)   Explicit database record construction
    -> QdrantClient.upsert(...)           Explicit storage call

Prepared resume query
    -> SentenceTransformer.encode(...)
    -> QdrantClient.query_points(...)
    -> Read payload + score from each returned point
    -> Candidate JSON
```

1. `prepare_job()` builds the same searchable job text as the main implementation.
2. `Encoder` loads the same pinned BGE-M3 snapshot from the existing local cache.
   It counts tokens, calls the model directly, requests normalization, and checks
   the resulting 1024-dimensional vectors.
3. `JobIndex.build()` creates the cosine collection, encodes batches, constructs
   `PointStruct` objects, and calls `upsert`. Stable UUIDs prevent duplicates.
4. `JobIndex.search()` applies the same preference rules, encodes the query,
   calls `query_points`, and manually maps points into candidate dictionaries.

The code also audits payloads/vectors, reuses complete indexes, recovers incomplete
indexes, preserves source IDs, and rejects mismatched collection provenance.
These are application responsibilities in both implementations.

## Exactly what LangChain saves

| Task | This folder | Main LangChain implementation |
|---|---|---|
| Text and metadata container | Our `JobText` dataclass | Standard `Document` |
| Model access | `SentenceTransformer` directly | `HuggingFaceEmbeddings` adapter |
| Batch job embedding | Explicit `encoder.encode(texts)` | Called within `add_documents()` |
| Database point construction | Explicit `PointStruct` and payload dictionaries | Qdrant integration constructs these |
| Database writes | Explicit `client.upsert()` | `store.add_documents()` |
| Query embedding | Explicit `encoder.encode([query])[0]` | Called within similarity search |
| Search and result mapping | `client.query_points()`, then map points | `similarity_search_with_score()` returns Document/score pairs |
| Connecting later components | Our functions and data structures | Common retriever/runnable interfaces |
| Token guards, provenance, filters, scoring policy | Our Python rules | Still our Python rules |

For example, the direct implementation does:

```python
vectors = encoder.encode(texts)
points = [
    models.PointStruct(id=identifier, vector=vector, payload=payload)
    for identifier, vector, payload in zip(ids, vectors, payloads)
]
client.upsert(collection_name, points=points)
```

The equivalent LangChain operation is:

```python
store.add_documents(documents, ids=ids)
```

Similarly, direct query encoding, `query_points`, and result conversion become:

```python
results = store.similarity_search_with_score(query, k=5, filter=eligible_filter)
```

LangChain reduces integration glue and provides common interfaces when models,
vector stores, retrievers, and generation chains are connected. It does not make
BGE-M3 intrinsically more accurate or remove the need for domain rules and tests.
For a small fixed pipeline, direct Python can be equally understandable and gives
you explicit control. Neither approach is automatically better for every project.

## Run from the project root

The existing `.venv` already contains the dependencies and cached model.
No download or new environment is needed in this workspace.

```bash
.venv/bin/python -m python_only.cli index

# Use the prepared_resume.json path printed by Module 2:
.venv/bin/python -m python_only.cli search --prepared-resume PATH_TO_PREPARED_RESUME.json --k 5

.venv/bin/python -m unittest discover -s python_only -v
```

Replace `PATH_TO_PREPARED_RESUME.json` with a real path. Both commands use the prepared 250-job dataset.

To run in an environment without LangChain installed, create a separate virtual
environment and install `python_only/requirements.txt` from the project root.
Run modules from the project root so shared `app` modules are importable. This
file lists direct dependency pins; the main project's lock records the fully
tested installed environment. Model files must first be downloaded with the
existing framework-independent `python -m app.cli.download_models embeddings`.

## Isolation and limits

Outputs are separate:

```text
artifacts/python_only/qdrant/   Persistent learning index
artifacts/python_only/reports/  Index and search reports
```

The main LangChain index and reports are not modified. The existing model cache
is reused, so there is no second model download. This learning implementation
uses persistent local Qdrant only; its backend does not follow the main server-mode
setting. Only one process can open its local directory at once.

The same source text, pinned model, normalization, newline handling, and filters
make score comparisons meaningful. Floating-point tolerances apply; equal-score
ties can appear in different orders. Candidate retrieval still does not assess
coverage or suitability, and no Qwen calls are made.

The report writer is intentionally simple and replaces a report after a run;
it does not provide transactional writes. Review the main implementation and
module guide for the fuller CLI/reporting layer.

## Verified in this workspace

- Built the separate 250-job index with real BGE-M3 embeddings; job texts use
  209–317 tokens.
- Repeated indexing reused all 250 points.
- The AI demo returned the same five job IDs in the same order as the main
  LangChain implementation, with a maximum absolute cosine-score difference of
  **0.0** for this run. This is one development comparison, not a quality benchmark.
- Cached index reuse and search succeeded with socket connections blocked.
- No LangChain or LangSmith modules were imported during that real run.
- This folder includes five tests for the direct retrieval implementation.
- Original dataset file checksums remain unchanged.

Evidence: `artifacts/python_only/reports/comparison.json`. The AI search report
and index report are beside it. Run the following against the existing AI demo:

```bash
.venv/bin/python -m python_only.cli search \
  --prepared-resume artifacts/resumes/86ef5a01645b45129f5cc0a61a1cf313e2a6bcb68dc89291f1b240e563d27bcb/prepared_resume.json \
  --k 5
```
