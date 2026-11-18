# Module 8 evaluation results

Full 250-job retrieval; only panel pairs are labeled. Unjudged outputs have bounds, not assumed negative labels.

Suite: `9e8bc796f61e462cbd61f09f0728362922e27914eff477821c1dc14eceddf357`. Source: `162b6e5e27be9bb2260325de0e16901ffb0b5e5db57b4160cfcf198e65a41143`.

Labels are assistant-authored synthetic suitability judgments; they are not independent expert validation.

| Split | Variant | Cases | P@5 / panel ceiling | nDCG@5 | Accepted precision | Mean returned | Median / p95 seconds |
|---|---|---:|---:|---:|---:|---:|---:|
| heldout | dense | 8 | N/A / N/A | N/A | N/A | 2.125 | 0.281 / 0.337 |
| heldout | bm25 | 8 | N/A / N/A | N/A | N/A | 2.250 | 0.056 / 0.076 |
| heldout | hybrid | 8 | N/A / N/A | N/A | N/A | 2.125 | 0.276 / 0.338 |

| Split | Variant | No-match precision | No-match recall | No-match accuracy | Judged / unjudged returned top-five slots |
|---|---|---:|---:|---:|---:|
| heldout | dense | N/A | N/A | N/A | 12 / 24 |
| heldout | bm25 | N/A | N/A | N/A | 12 / 24 |
| heldout | hybrid | N/A | N/A | N/A | 12 / 24 |

| Split | Variant | P@5 lower / upper | Accepted precision lower / upper |
|---|---|---:|---:|
| heldout | dense | 0.125 / 0.725 | 0.294 / 0.706 |
| heldout | bm25 | 0.125 / 0.725 | 0.278 / 0.722 |
| heldout | hybrid | 0.125 / 0.725 | 0.294 / 0.706 |

Strict precision counts only grade 2. Grade 1 is partial/conditional. Missing top-five positions count as non-relevant.
Accepted precision is N/A when nothing is returned. Unknown judgments produce bounds, not automatic negative labels.
No-match metrics use separate expectations; conditional/partial opportunities prevent a no-match ground truth. The panel P@5 ceiling reflects how few suitable jobs exist under restrictive preferences. nDCG averages only cases with nonzero ideal gain; defined-case counts are in JSON.

Independent fixture checks: 660/660 passed.
Reranker status counts: {}.

Latency excludes model load/index build, includes preparation, retrieval, reranking and assessment, and uses one observation per case.
This run does not evaluate OCR, arbitrary PDF layouts, remote Qwen reliability, real jobs, or population-level hiring suitability.
No held-out settings or labels were changed after inspecting model outputs.
