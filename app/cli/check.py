"""Run a smoke check for settings, the job dataset, and the resume models."""

from app.config import Settings
from app.schemas import CandidateProfile, JobRecord


def main() -> None:
    """Validate the first dataset row and a sample profile, then print a summary."""
    settings = Settings()

    with settings.raw_jobs_path.open(encoding="utf-8") as dataset:
        first_line = dataset.readline()

    if not first_line:
        raise SystemExit(f"Job dataset is empty: {settings.raw_jobs_path}")

    job = JobRecord.model_validate_json(first_line)
    # Use sample evidence so this check can run before PDF parsing is available.
    profile = CandidateProfile.model_validate(
        {
            "skills": [
                {
                    "text": "FastAPI",
                    "evidence": [
                        {"quote": "Built Python APIs with FastAPI.", "page": 1}
                    ],
                }
            ]
        }
    )
    skill = profile.skills[0]
    evidence = skill.evidence[0]

    print(f"Dataset: {settings.raw_jobs_path}")
    print(f"Prepared jobs path: {settings.prepared_jobs_path}")
    print(f"Retrieval limit: {settings.retrieval_top_k}")
    print(f"First job: {job.job_id} | {job.title}")
    print(f"Sample resume skill: {skill.text} | page {evidence.page}")
    print("Setup check passed.")

if __name__ == "__main__":
    main()
