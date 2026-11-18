"""Verify a running local API using the public synthetic AI resume."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from app.api_schemas import JobDetailResponse, UploadMatchResponse
from app.config import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--use-llm", action="store_true", help="Also require successful remote Qwen plans")
    args = parser.parse_args()
    folder = PROJECT_ROOT / "artifacts/verification"
    folder.mkdir(parents=True, exist_ok=True)
    with httpx.Client(base_url=args.url, timeout=140) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 200
        with (PROJECT_ROOT / "artifacts/demo_resumes/ai.pdf").open("rb") as resume:
            response = client.post("/match", files={"resume": ("ai.pdf", resume, "application/pdf")}, data={
                "preferences": (PROJECT_ROOT / "examples/preferences.json").read_text(),
                "use_llm": str(args.use_llm).lower(),
            })
        response.raise_for_status()
        result = UploadMatchResponse.model_validate(response.json())
        try:
            assert result.result.recommendations
            if args.use_llm:
                assert all(match.generation_status == "qwen_grounded_plan" for match in result.result.recommendations)
            detail = client.get(f"/runs/{result.run_id}/jobs/syn_job_234")
            detail.raise_for_status()
            selected = JobDetailResponse.model_validate(detail.json())
            assert selected.job.job_id == "syn_job_234"
            assert selected.generation_status == "deterministic"
            assert not selected.was_recommended
            assert not list((PROJECT_ROOT / "artifacts/uploads").glob("*.pdf"))
            (folder / "module5_ai_http_response.json").write_text(result.model_dump_json(indent=2) + "\n")
            (folder / "module5_selected_job_http_response.json").write_text(selected.model_dump_json(indent=2) + "\n")
        finally:
            assert client.delete(f"/runs/{result.run_id}").status_code == 204
        assert client.get(f"/runs/{result.run_id}/jobs/syn_job_234").status_code == 404
    summary = {"scope": "Real HTTP synthetic development demo; not held-out quality evaluation",
               "use_llm": args.use_llm, "recommendations": len(result.result.recommendations),
               "generation_statuses": [match.generation_status for match in result.result.recommendations],
               "selected_job": selected.job.job_id, "selected_job_passes_policy": selected.passes_policy,
               "temporary_uploads_cleaned": True, "run_deleted": True}
    (folder / "module5_http_verification.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
