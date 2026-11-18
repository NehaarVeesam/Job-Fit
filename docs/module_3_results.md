# Module 3 — Verified Development Results

These results use the real pinned BAAI/bge-m3 model and persistent local Qdrant on the development host. They are development demonstrations, not held-out suitability or production-performance evaluation.

## Index verification

| Measurement | Result |
|---|---|
| Corpus | 250 synthetic India jobs |
| Vector dimensions / distance | 1024 / cosine |
| Vector norm minimum / maximum | 0.999999939 / 1.000000104 |
| Searchable job tokens, min / max / mean | 209 / 317 / 253.9 |
| Initial embedding + upsert | 169.692 s |
| Initial total after model load | 169.797 s |
| Verification process model load | 7.972 s |
| Verified repeated index | 250 points, reused existing vectors |
| Repeat total after model load | 0.198 s |
| Socket connections during verification | Blocked; cached operation succeeded |

CPU float32 inference, four PyTorch CPU threads, batch size four; local Qdrant. Model loading, initial download, embedding/upsert, and search are different measurements. These single development runs are not latency percentiles.

## Demo retrieval

| Resume | Status | Eligible jobs | First retrieved job | Embedding + search (s) |
|---|---|---|---|---|
| ai | candidates_found | 41 | syn_job_001: Generative AI Engineer (0.7705) | 0.163 |
| backend | candidates_found | 250 | syn_job_195: Full Stack Developer (Java/Angular) (0.6320) | 0.129 |
| analyst | candidates_found | 250 | syn_job_010: Data Scientist (0.6074) | 0.111 |
| fresher | candidates_found | 250 | syn_job_218: SDET (Python) (0.5795) | 0.098 |
| unrelated | no_query | 250 | No query; no retrieval | 0 |

The AI demo uses the explicit Bengaluru/hybrid/Generative AI Engineer preferences; other demos use empty preferences. Candidate counts therefore differ. The unrelated resume has no recognized corpus skills or target role and is not embedded/searched.

Inspect candidate lists critically: the AI demo also retrieves a Golang Developer in position five, and a fresher can retrieve jobs requiring professional experience. This demonstrates why nearest neighbors must pass Module 4 coverage/constraint/acceptance checks. No job has been accepted as a suitable recommendation here.

The default corpus retains stale synthetic postings, so stale IDs can appear in retrieval. Recency is an explicit alternative Module 1 corpus and needs its own index.

## Saved evidence and reproduction

- `artifacts/verification/module3_initial_index.json`: preserved initial build audit and timing.
- `artifacts/verification/module3_live.json`: real-model reuse and five-profile checks.
- `artifacts/indexes/`: current ready index manifest.
- `artifacts/retrieval/`: individual result JSON files.
- `scripts/verify_dense.py`: reproducible local check with socket connections blocked.
- `tests/test_module3.py`: 14 fast tests with controlled embeddings and real local Qdrant; no quality claims from test vectors.

Reproduction: prepare Module 1 jobs, download the pinned model, generate demo PDFs, then run `.venv/bin/python scripts/verify_dense.py`. A fresh run first builds the index, then verifies reuse. See [the tutorial](module_3_langchain.md).

All 56 tests across Modules 0–3 pass. Original dataset file checksums are unchanged. Qwen generation, coverage scoring, suitability evaluation, hybrid retrieval, and an API remain future modules.
