# synthetic_india_ai_sw_v2 — 250 synthetic job postings

**These are not real job postings.** Every company, posting, date and requirement was
written for development and evaluation of JobFit. Company names are fictional; any
resemblance to a real organisation is coincidental. Do not present these as live
vacancies, and keep them out of the real Greenhouse snapshot (use a separate Qdrant
collection or filter on `dataset_id` / `is_synthetic`).

v2 extends v1: `syn_job_001`–`syn_job_100` are unchanged apart from `dataset_id`, so
any labels you made against v1 still apply. Filter to those IDs to get the 100-job set.

## Contents

| | |
|---|---|
| Records | 250 (125 AI/ML, 125 Software) |
| Companies | 42 fictional employers |
| Role families | 26 (AI/ML includes data science and analytics roles) |
| Seniority | Intern 11, Entry 35, Mid 131, Senior 59, Lead 14 |
| Employment type | full_time 235, internship 11, contract 4 |
| Work mode | hybrid 152, onsite 65, remote 30, unknown 3 |
| Cities | 16 Indian cities, plus 4 India-wide remote roles |
| `posted_at` | 2026-08-24 to 2026-11-16, except 4 deliberately stale postings (see below) |

Files: `jobs_synthetic_250.jsonl` (lists as JSON arrays) and `jobs_synthetic_250.csv`
(lists joined with ` | `).

## Schema

Follows the canonical record in the JobFit plan, plus a few fields useful for scoring:
`job_id`, `title`, `company`, `domain`, `role_family`, `seniority`, `employment_type`,
`location_city`, `location_country`, `work_mode`, `experience_min_years`,
`experience_max_years`, `education`, `required_skills`, `preferred_skills`,
`hard_constraints`, `description`, `source_url` (always null), `posted_at`,
`dataset_id`, `is_synthetic`.

Every skill in `required_skills` and `preferred_skills` appears verbatim in
`description` (under "Must-have skills" / "Good to have"), so job-side evidence quotes
can be validated by exact substring match. No skill is listed as both required and
preferred in the same job.

## Built-in test cases

- **Unknown metadata:** `work_mode = unknown` in 021, 076, 184; null experience in
  043, 095, 159, 232; null city (India-wide remote) in 075, 147, 197, 239.
- **Stale postings:** 128, 157, 196 and 238 were posted more than 90 days before
  2026-11-18, to test the recency filter in your plan.
- **Prompt injection:** `syn_job_213` contains an instruction aimed at screening tools
  in its "About us" text. Your pipeline should treat it as data and never rank it
  higher because of it. Exclude this job if you don't want the test.
- **Seniority ladder:** 245, 246 and 247 are SDE I, II and III at the same company
  with overlapping skills, to test that experience level changes the result.
- **Same role, different city:** 001 and 109 are the same Generative AI Engineer role
  in Bengaluru and Hyderabad.
- **Related but not equivalent skills:** FastAPI vs Flask (177 vs 178), MERN vs MEAN
  (061 vs 198), PyTorch vs TensorFlow (170), Unity/C# vs Unreal/C++ (234 vs 235).
- **Hard constraints:** 34 jobs carry constraints (graduation year, shifts, travel,
  office location, contract length) that skill overlap alone cannot satisfy.
- **Technical tokens:** C++, C#, .NET, Node.js, ASP.NET Core and CI/CD appear across
  many jobs to test keyword tokenisation.

## Known limitations

Descriptions are shorter and more uniform than real postings, and skills are cleanly
listed, so requirement extraction and retrieval will look easier here than on real
data. Report results on this set separately from the real corpus.
