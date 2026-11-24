# JobFit evaluation results

The starter suite uses two synthetic resumes, five manual job judgments, and a shortlist size of three. The same cases were run through BM25, dense, and hybrid retrieval, followed by JobFit’s matching rules and citation audit.

## Smoke test outputs

| Test case | Mode | Retrieved job IDs in order | Matching output | Recommendations | Citation errors |
| --- | --- | --- | --- | ---: | ---: |
| `rag_engineer` | BM25 | `syn_job_109`, `syn_job_001`, `syn_job_105` | `matches_found` | 3 | 0 |
| `rag_engineer` | Dense | `syn_job_105`, `syn_job_002`, `syn_job_001` | `matches_found` | 2 | 0 |
| `rag_engineer` | Hybrid | `syn_job_105`, `syn_job_109`, `syn_job_001` | `matches_found` | 3 | 0 |
| `pastry_chef` | BM25 | None | `no_suitable_matches` | 0 | 0 |
| `pastry_chef` | Dense | `syn_job_196`, `syn_job_116`, `syn_job_174` | `no_suitable_matches` | 0 | 0 |
| `pastry_chef` | Hybrid | `syn_job_196`, `syn_job_116`, `syn_job_174` | `no_suitable_matches` | 0 | 0 |

## Retrieval for the RAG engineer

| Mode | Results judged | Judged precision | Known relevant recall | Judged nDCG | Labeled suitable jobs recommended |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 3/3 | 1.000 | 0.750 | 1.000 | 3 |
| Dense | 2/3 | 1.000 | 0.500 | 1.000 | 2 |
| Hybrid | 3/3 | 1.000 | 0.750 | 1.000 | 3 |

BM25 and hybrid retrieved all three jobs labeled suitable. Known relevant recall was 0.750 because the ground truth also included one adjacent MLOps role, giving four known relevant jobs in total. Dense returned two known relevant jobs and one unjudged job. Judged precision and nDCG describe only the labeled portion of the shortlist.

## No-match behavior

All three modes returned `no_suitable_matches` for the pastry chef. BM25 retrieved zero jobs. Dense and hybrid each retrieved three candidates, and the matcher excluded all of them.

## Evidence quality

The citation audit reported zero missing job or resume quotes across all six case and retrieval mode combinations.

In a separate semantic evidence check, Qwen correctly linked a resume line about automated testing and deployment to CI/CD. In the tests-only case, Qwen incorrectly marked CI/CD as supported and cited an unrelated line about Docker, MLflow, and AWS. The cited line existed, so checking its source alone would have missed the error. Semantic findings are advisory. The matcher uses exact skill evidence for coverage and recommendations.

## Conclusion

All six runs produced the expected match or no-match outcome. No job labeled unsuitable was recommended. On the RAG engineer case, BM25 and hybrid found more labeled suitable jobs than dense search. With two resumes and partial labels, this suite gives us a starting point for regression checks.

To reproduce the retrieval and matching results, run `python -m app.evaluation.runner --mode all` from the project root after preparing the catalog and building the local Qdrant index.

## Full-catalog benchmark

Six [synthetic resumes](../../evaluation/full_catalog_resumes.json) cover RAG, MLOps, FastAPI, Spark data engineering, Playwright QA, and Android. Each resume ran through JobFit as a PDF with all four retrieval variants. All 24 runs searched 250 jobs and assessed the top five with the same matching rules and template explanations.

The top-five outputs contained 54 unique resume–job pairs after counting repeated jobs for the same resume once. The [ground truth](../../evaluation/full_catalog_judgments.json) records one assistant-authored grade and evidence rationale per pair. Grade 2 means suitable, 1 means partial, and 0 means unsuitable. Each variant is scored against the same ground truth. Unreturned jobs are outside the scope of the judgments.

| Variant | nDCG@5 | Known suitable retrieved | Strict recommendation precision | Citation errors |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 0.959 | 7/7 | 7/13 | 0 |
| Dense | 0.683 | 6/7 | 6/7 | 0 |
| Hybrid | 0.894 | 7/7 | 7/11 | 0 |
| Hybrid + reranker | 0.934 | 7/7 | 7/11 | 0 |

- **nDCG@5** rewards higher positions for suitable and partial jobs. The ideal order comes from the judged results for that resume across all four variants.
- **Known suitable retrieved** counts how many of the seven grade-2 pairs each variant retrieved.
- **Strict recommendation precision** counts grade-2 jobs among the final recommendations.
- **Citation errors** counts matching evidence quotes that could not be found in their cited sources.

Reranking moved the suitable Spark job (`syn_job_073`) from second to first. Dense search left it outside the top five, so it was never assessed. All 24 runs had zero citation errors.

### Outputs by resume

| Resume | Variant | Top five | Final recommendations |
| --- | --- | --- | --- |
| rag_evaluation | BM25 | syn_job_105, syn_job_109, syn_job_001, syn_job_136, syn_job_152 | syn_job_105, syn_job_109, syn_job_001, syn_job_136 |
| rag_evaluation | Dense | syn_job_002, syn_job_132, syn_job_107, syn_job_105, syn_job_023 | syn_job_105 |
| rag_evaluation | Hybrid | syn_job_105, syn_job_002, syn_job_109, syn_job_136, syn_job_107 | syn_job_105, syn_job_109, syn_job_136 |
| rag_evaluation | Hybrid + reranker | syn_job_105, syn_job_109, syn_job_002, syn_job_107, syn_job_136 | syn_job_105, syn_job_109, syn_job_136 |
| ml_platform | BM25 | syn_job_008, syn_job_140, syn_job_141, syn_job_067, syn_job_071 | syn_job_008, syn_job_140, syn_job_141, syn_job_067 |
| ml_platform | Dense | syn_job_008, syn_job_140, syn_job_046, syn_job_044, syn_job_165 | syn_job_008, syn_job_140 |
| ml_platform | Hybrid | syn_job_008, syn_job_140, syn_job_036, syn_job_034, syn_job_046 | syn_job_008, syn_job_140 |
| ml_platform | Hybrid + reranker | syn_job_008, syn_job_140, syn_job_034, syn_job_036, syn_job_046 | syn_job_008, syn_job_140 |
| fastapi_service | BM25 | syn_job_177, syn_job_062, syn_job_099, syn_job_197, syn_job_051 | syn_job_177 |
| fastapi_service | Dense | syn_job_177, syn_job_183, syn_job_001, syn_job_180, syn_job_109 | syn_job_177 |
| fastapi_service | Hybrid | syn_job_177, syn_job_062, syn_job_197, syn_job_180, syn_job_178 | syn_job_177, syn_job_178 |
| fastapi_service | Hybrid + reranker | syn_job_177, syn_job_197, syn_job_178, syn_job_062, syn_job_180 | syn_job_177, syn_job_178 |
| spark_data | BM25 | syn_job_073, syn_job_100, syn_job_006, syn_job_074, syn_job_211 | syn_job_073 |
| spark_data | Dense | syn_job_074, syn_job_164, syn_job_075, syn_job_031, syn_job_077 | None |
| spark_data | Hybrid | syn_job_074, syn_job_073, syn_job_077, syn_job_100, syn_job_164 | syn_job_073 |
| spark_data | Hybrid + reranker | syn_job_073, syn_job_100, syn_job_077, syn_job_074, syn_job_164 | syn_job_073 |
| playwright_qa | BM25 | syn_job_083, syn_job_218, syn_job_219, syn_job_067, syn_job_243 | syn_job_083 |
| playwright_qa | Dense | syn_job_250, syn_job_083, syn_job_062, syn_job_167, syn_job_093 | syn_job_083 |
| playwright_qa | Hybrid | syn_job_083, syn_job_250, syn_job_082, syn_job_093, syn_job_218 | syn_job_083 |
| playwright_qa | Hybrid + reranker | syn_job_083, syn_job_218, syn_job_082, syn_job_093, syn_job_250 | syn_job_083 |
| android_compose | BM25 | syn_job_078, syn_job_213, syn_job_217, syn_job_181, syn_job_079 | syn_job_078, syn_job_213 |
| android_compose | Dense | syn_job_078, syn_job_181, syn_job_217, syn_job_186, syn_job_213 | syn_job_078, syn_job_213 |
| android_compose | Hybrid | syn_job_078, syn_job_181, syn_job_217, syn_job_213, syn_job_186 | syn_job_078, syn_job_213 |
| android_compose | Hybrid + reranker | syn_job_078, syn_job_213, syn_job_217, syn_job_181, syn_job_186 | syn_job_078, syn_job_213 |

### Reproduce the benchmark

After preparing the catalog and building the Qdrant index, run:

```sh
python -m app.evaluation.full_catalog collect
python -m app.evaluation.full_catalog score
```

The collection step saves the outputs under `artifacts/evaluation/`. If a future run returns a new pair, add its judgment before scoring.
