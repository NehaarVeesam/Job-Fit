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

BM25 and hybrid retrieved all three jobs labeled suitable. Known relevant recall was 0.750 because the answer key also included one adjacent MLOps role, giving four known relevant jobs in total. Dense returned two known relevant jobs and one unjudged job. Judged precision and nDCG describe only the labeled portion of the shortlist.

## No-match behavior

All three modes returned `no_suitable_matches` for the pastry chef. BM25 retrieved zero jobs. Dense and hybrid each retrieved three candidates, and the matcher excluded all of them.

## Evidence quality

The citation audit reported zero missing job or resume quotes across all six case and retrieval mode combinations.

In a separate semantic evidence check, Qwen correctly linked a resume line about automated testing and deployment to CI/CD. In the tests-only case, Qwen incorrectly marked CI/CD as supported and cited an unrelated line about Docker, MLflow, and AWS. The cited line existed, so checking its source alone would have missed the error. Semantic findings are advisory. The matcher uses exact skill evidence for coverage and recommendations.

## Conclusion

All six runs produced the expected match or no-match outcome. No job labeled unsuitable was recommended. On the RAG engineer case, BM25 and hybrid found more labeled suitable jobs than dense search. With two resumes and partial labels, this suite gives us a starting point for regression checks.

To reproduce the retrieval and matching results, run `python -m app.evaluation.runner --mode all` from the project root after preparing the catalog and building the local Qdrant index.

## Held-out benchmark

The benchmark uses 12 synthetic resumes and a panel of 17 jobs drawn from the 250-job catalog. Each of the 204 resume–job pairs has an assistant-authored suitability grade. Four resumes form the development split and eight form the held-out split. All four variants use the same matching rules after retrieval.

| Retrieval variant | Mean nDCG@5 | Strict accepted precision | Expected match outcomes |
| --- | ---: | ---: | ---: |
| BM25 | 1.000 | 72.7% | 8/8 |
| Dense | 0.872 | 70.0% | 7/8 |
| Hybrid | 0.962 | 72.7% | 8/8 |
| Hybrid + reranker | 1.000 | 72.7% | 8/8 |

Reranking improved hybrid's top-five ordering and tied BM25 on this panel. Mean nDCG@5 measures ranking quality over cases with at least one job graded partial or fully suitable. Strict accepted precision counts only recommendations labeled fully suitable. All four variants recommended some partial matches. Dense retrieval also left the fully suitable `syn_job_141` outside the `junior_mlops` shortlist.

## Development benchmark

| Retrieval variant | Mean nDCG@5 | Strict accepted precision | Expected match outcomes |
| --- | ---: | ---: | ---: |
| BM25 | 1.000 | 50.0% | 4/4 |
| Dense | 0.766 | 50.0% | 4/4 |
| Hybrid | 0.964 | 50.0% | 4/4 |
| Hybrid + reranker | 1.000 | 50.0% | 4/4 |

## Reproduce the benchmark

Prepare the job catalog, then run both splits from the project root.

```sh
python -m app.evaluation.benchmark --split dev
python -m app.evaluation.benchmark --split heldout
```
