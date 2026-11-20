# Actual synthetic demo outputs

These files are published outputs from a synthetic demo run:

- [AI match response](ai_match_response.json): two accepted jobs from all 250
  source jobs, with 66.7% and 50% required-skill coverage.
- [Unrelated no-match response](unrelated_no_match_response.json): no suitable
  jobs and an empty recommendation list.
- [Metadata](metadata.json): source/model/input checksums and execution settings.

All source jobs and candidate text are fictional. Responses validate against
`app.schemas.MatchResponse`. No Qwen calls or reranking were used; default demo
execution blocked network socket connections and reused cached local embeddings.
Explanations were rendered deterministically from evidence.

`relevance.score_type = rrf` identifies the hybrid ordering score. The cosine
acceptance threshold was checked separately; an RRF value is not a cosine value,
a skill-coverage percentage, or a hiring probability. Missing evidence is listed
in `skills` with `status = not_evidenced_in_resume`, separated by requirement type.

The metadata records the original inputs and settings. The local resume CLI is documented in the [project README](../../README.md#run-locally).

These demonstration outputs are distinct from the frozen 17-job evaluation panel
and from the older hand-authored contract examples directly inside `examples/`.
