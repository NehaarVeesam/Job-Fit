"""Assess dense candidates, gate recommendations, then explain accepted jobs."""

import hashlib

from app.generation import GroundedExplainer
from app.schemas import MatchResponse, RankedCandidate, RelevanceScore
from app.scoring import assess_job, make_match, rejection_reasons


def match_candidates(settings, jobs, provenance, prepared, retrieval, *, use_llm=True, explainer=None):
    if retrieval.get("applied_preferences") != prepared.preferences.model_dump(mode="json"):
        raise ValueError("Retrieval preferences differ from the prepared resume")
    if retrieval.get("query_sha256") != hashlib.sha256((prepared.retrieval_query or "").encode()).hexdigest():
        raise ValueError("Retrieval query differs from the prepared resume; rerun Module 3")
    if retrieval.get("corpus_size") != len(jobs) or retrieval.get("is_synthetic") is not True:
        raise ValueError("Retrieval corpus metadata mismatch")
    by_id = {job.job_id: job for job in jobs}
    ranked, recommendations, audit = [], [], []
    seen = set()
    explainer = explainer or (GroundedExplainer(settings) if use_llm else None)
    if prepared.retrieval_query is None and retrieval.get("candidates"):
        raise ValueError("Cannot accept candidates without a prepared query")
    for candidate in retrieval.get("candidates", []):
        job_id = candidate["job_id"]
        if job_id in seen or job_id not in by_id:
            raise ValueError("Duplicate or foreign candidate ID")
        seen.add(job_id)
        job = by_id[job_id]
        if candidate.get("job") != job.model_dump(mode="json"):
            raise ValueError("Candidate source record differs from the prepared corpus")
        relevance = RelevanceScore.model_validate(candidate["relevance"])
        assessment = assess_job(job, prepared, relevance)
        ranked.append(RankedCandidate(job_id=job_id, relevance=relevance, eligibility=assessment.eligibility))
        rejected = rejection_reasons(assessment, prepared, settings)
        selected = not rejected and len(recommendations) < settings.max_recommendations
        audit.append({"job_id": job_id, "selected": selected, "rejection_reasons": rejected,
                      "selection_limit_reached": not rejected and not selected,
                      "required_coverage": assessment.required.model_dump(mode="json"),
                      "preferred_coverage": assessment.preferred.model_dump(mode="json"),
                      "eligibility": assessment.eligibility,
                      "experience": assessment.experience.model_dump(mode="json"),
                      "education": assessment.education.model_dump(mode="json"),
                      "constraints": [item.model_dump(mode="json") for item in assessment.constraints],
                      "skills": [item.model_dump(mode="json") for item in assessment.skills]})
        if not selected:
            continue
        order, status, warnings = None, "deterministic", []
        if use_llm:
            plan = explainer.explain(assessment)
            if plan is not None:
                order, status = plan.strengths, "qwen_grounded_plan"
            else:
                status = "fallback"
                warnings.append("Qwen generation unavailable or invalid after bounded attempts; deterministic evidence explanations used.")
        recommendations.append(make_match(assessment, order, status, warnings))
    response = MatchResponse(
        corpus_size=len(jobs), corpus_subset=provenance["configuration"]["subset"],
        status="matches_found" if recommendations else "no_suitable_matches",
        applied_preferences=prepared.preferences,
        applied_filters={"min_cosine_similarity": str(settings.min_cosine_similarity),
                         "min_required_coverage": str(settings.min_required_coverage),
                         "min_evidenced_required": settings.min_evidenced_required,
                         "unknown_policy": prepared.preferences.unknown_metadata_policy},
        ranked_candidates=ranked, recommendations=recommendations,
        warnings=["Synthetic demo. Acceptance thresholds are development heuristics, not calibrated suitability judgments.",
                  "Coverage measures literal resume evidence, not proficiency or hiring probability."]
        + (["Some recommendations have unresolved requirements; inspect their eligibility and warnings."]
           if any(item.eligibility == "unknown" for item in recommendations) else []),
    )
    report = {"policy_version": "jobfit-acceptance-v1", "corpus": provenance,
              "retrieval_index_id": retrieval.get("index_id"), "assessments": audit,
              "generation_requested": use_llm, "generation_calls": explainer.calls if explainer else 0,
              "generation_error_types": explainer.failures if explainer else [],
              "generation_seconds": round(explainer.seconds, 3) if explainer else 0,
              "generation_model": settings.llm_model if use_llm else None}
    return response, report
