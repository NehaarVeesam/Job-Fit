# JobFit evaluation results

The starter suite uses two synthetic resumes, five manual job judgments, and a shortlist size of three. The same cases were run through BM25, dense, and hybrid retrieval, followed by JobFit’s matching rules and citation audit.

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