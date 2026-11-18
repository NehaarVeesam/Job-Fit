# Actual synthetic demo outputs

These files were exported from a successful `python scripts/run_demo.py` run:

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

Run the demo locally to regenerate outputs under ignored
`artifacts/portfolio_demo/`. Generated PDFs, profiles, full retrieval reports,
vector indexes, and model weights are excluded from publication. PDF hashes can
change when regenerated; the text-source checksums here identify the fixed inputs.

These demonstration outputs are distinct from the frozen 17-job evaluation panel
and from the older hand-authored contract examples directly inside `examples/`.
