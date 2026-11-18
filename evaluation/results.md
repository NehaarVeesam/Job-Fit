# Module 8 evaluation results

Controlled 17-job panel selected from the unchanged 250-job synthetic corpus; panel metrics do not estimate full-corpus or real-world accuracy.

Suite: `9e8bc796f61e462cbd61f09f0728362922e27914eff477821c1dc14eceddf357`. Source: `162b6e5e27be9bb2260325de0e16901ffb0b5e5db57b4160cfcf198e65a41143`.

Labels are assistant-authored synthetic suitability judgments; they are not independent expert validation.

| Split | Variant | Cases | P@5 / panel ceiling | nDCG@5 | Accepted precision | Mean returned | Median / p95 seconds |
|---|---|---:|---:|---:|---:|---:|---:|
| dev | dense | 4 | 0.300 / 0.300 | 1.000 | 1.000 | 1.500 | 0.212 / 0.291 |
| dev | bm25 | 4 | 0.300 / 0.300 | 1.000 | 1.000 | 1.500 | 0.055 / 0.060 |
| dev | hybrid | 4 | 0.300 / 0.300 | 1.000 | 1.000 | 1.500 | 0.215 / 0.286 |
| dev | hybrid_reranked | 4 | 0.300 / 0.300 | 1.000 | 1.000 | 1.500 | 1.926 / 12.222 |
| heldout | dense | 8 | 0.125 / 0.125 | 0.965 | 0.500 | 1.250 | 0.219 / 0.277 |
| heldout | bm25 | 8 | 0.125 / 0.125 | 0.974 | 0.500 | 1.250 | 0.056 / 0.058 |
| heldout | hybrid | 8 | 0.125 / 0.125 | 0.965 | 0.500 | 1.250 | 0.222 / 0.279 |
| heldout | hybrid_reranked | 8 | 0.125 / 0.125 | 0.974 | 0.500 | 1.250 | 1.375 / 7.314 |

| Split | Variant | No-match precision | No-match recall | No-match accuracy | Judged / unjudged returned top-five slots |
|---|---|---:|---:|---:|---:|
| dev | dense | 1.000 | 1.000 | 1.000 | 9 / 0 |
| dev | bm25 | 1.000 | 1.000 | 1.000 | 9 / 0 |
| dev | hybrid | 1.000 | 1.000 | 1.000 | 9 / 0 |
| dev | hybrid_reranked | 1.000 | 1.000 | 1.000 | 9 / 0 |
| heldout | dense | 1.000 | 1.000 | 1.000 | 16 / 0 |
| heldout | bm25 | 1.000 | 1.000 | 1.000 | 16 / 0 |
| heldout | hybrid | 1.000 | 1.000 | 1.000 | 16 / 0 |
| heldout | hybrid_reranked | 1.000 | 1.000 | 1.000 | 16 / 0 |

Strict precision counts only grade 2. Grade 1 is partial/conditional. Missing top-five positions count as non-relevant.
Accepted precision is N/A when nothing is returned. Unknown judgments produce bounds, not automatic negative labels.
No-match metrics use separate expectations; conditional/partial opportunities prevent a no-match ground truth. The panel P@5 ceiling reflects how few suitable jobs exist under restrictive preferences. nDCG averages only cases with nonzero ideal gain; defined-case counts are in JSON.

Independent fixture checks: 829/829 passed.
Reranker status counts: {'applied': 11, 'skipped_empty': 1}.

Latency excludes model load/index build, includes preparation, retrieval, reranking and assessment, and uses one observation per case.
This run does not evaluate OCR, arbitrary PDF layouts, remote Qwen reliability, real jobs, or population-level hiring suitability.
No held-out settings or labels were changed after inspecting model outputs.
