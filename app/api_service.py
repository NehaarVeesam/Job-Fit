"""Blocking pipeline called in a worker; models are cached per API process."""

import time
from dataclasses import dataclass

from app.api_schemas import JobDetailResponse
from app.generation import GroundedExplainer
from app.matching import match_candidates
from app.profile import build_profile, load_prepared_jobs, skill_vocabulary
from app.resume_parser import parse_resume
from app.schemas import MatchResponse, PreparedResume, RelevanceScore
from app.scoring import assess_job, make_match, rejection_reasons


class ServiceUnavailable(RuntimeError):
    pass


@dataclass
class MatchRun:
    prepared: PreparedResume
    retrieval: dict
    response: MatchResponse


class MatchingService:
    def __init__(self, settings):
        self.settings = settings
        self.embeddings = None
        self.embedding_info = None
        self.jobs, self.provenance = [], None
        try:
            self.jobs, self.provenance = load_prepared_jobs(settings.artifacts_dir / "dataset/all")
        except (OSError, ValueError, KeyError):
            pass

    def ready(self):
        # Lightweight readiness; inference and a full index audit happen on use.
        return bool(self.jobs)

    def _embeddings(self):
        if self.embeddings is None:
            from app.embeddings import load_embeddings
            try:
                self.embeddings, self.embedding_info = load_embeddings(self.settings)
            except Exception as error:
                raise ServiceUnavailable("Cached embedding model is unavailable") from error
        return self.embeddings, self.embedding_info

    def match(self, path, preferences, use_llm, deadline):
        if not self.ready():
            raise ServiceUnavailable("Prepare the job corpus with Module 1 first")
        extraction = parse_resume(path, max_upload_mb=self.settings.max_upload_mb,
                                  max_pages=self.settings.max_resume_pages)
        prepared = build_profile(extraction.parsed, skill_vocabulary(self.jobs), preferences)
        prepared.warnings = extraction.warnings + prepared.warnings
        from app.hybrid_retrieval import retrieve_candidates
        # Keyword-only and no-query requests do not need a vector model/index.
        embeddings, info = ((None, None) if prepared.retrieval_query is None or
                            self.settings.retrieval_mode == "bm25" else self._embeddings())
        if time.perf_counter() >= deadline:
            raise TimeoutError("Processing deadline exceeded")
        try:
            retrieval = retrieve_candidates(self.settings, self.jobs, self.provenance, prepared,
                                            embeddings=embeddings, embedding_info=info)
        except Exception as error:
            raise ServiceUnavailable("Retrieval is unavailable; check prepared corpus and selected index") from error
        if time.perf_counter() >= deadline:
            raise TimeoutError("Processing deadline exceeded")
        bounded_settings = self.settings.model_copy(update={
            "llm_timeout": min(self.settings.llm_timeout, max(0.1, deadline - time.perf_counter()))})
        explainer = GroundedExplainer(bounded_settings, deadline=deadline) if use_llm else None
        response, _ = match_candidates(self.settings, self.jobs, self.provenance, prepared, retrieval,
                                       use_llm=use_llm, explainer=explainer)
        return MatchRun(prepared, retrieval, response)

    def detail(self, run, job_id):
        job = next((job for job in self.jobs if job.job_id == job_id), None)
        if job is None:
            raise KeyError(job_id)
        candidate = next((item for item in run.retrieval["candidates"] if item["job_id"] == job_id), None)
        score = RelevanceScore.model_validate(candidate["relevance"]) if candidate else None
        dense_score = (RelevanceScore.model_validate(candidate["dense_relevance"])
                       if candidate and candidate.get("dense_relevance") else None)
        if score is None and run.prepared.retrieval_query and run.retrieval.get("retrieval_mode") == "bm25":
            from app.hybrid_retrieval import retrieve_candidates
            report = retrieve_candidates(self.settings, self.jobs, self.provenance, run.prepared, mode="bm25", k=250)
            item = next((item for item in report["candidates"] if item["job_id"] == job_id), None)
            score = RelevanceScore.model_validate(item["relevance"]) if item else RelevanceScore(value=0, score_type="bm25")
        if score is None and run.prepared.retrieval_query:
            from app.filters import qdrant_filter
            from app.vector_index import (collection_name, open_client, prepare_documents,
                                          vector_store, verify_collection)
            embeddings, info = self._embeddings()
            documents, identity = prepare_documents(self.jobs, self.provenance, info)
            client = open_client(self.settings)
            try:
                name = collection_name(self.settings, identity)
                if not verify_collection(client, name, identity, documents)["complete"]:
                    raise ServiceUnavailable("Dense index is incomplete")
                found = vector_store(client, name, embeddings).similarity_search_with_score(
                    run.prepared.retrieval_query, k=1, filter=qdrant_filter([job_id]))
                if not found or found[0][0].metadata["job_id"] != job_id:
                    raise ServiceUnavailable("Selected job is unavailable in the index")
                score = RelevanceScore(value=max(-1, min(1, found[0][1])), score_type="cosine")
            except ServiceUnavailable:
                raise
            except Exception as error:
                raise ServiceUnavailable("Dense index is unavailable") from error
            finally:
                client.close()
        assessed = assess_job(job, run.prepared, score)
        rejected = rejection_reasons(assessed, run.prepared, self.settings, dense_relevance=dense_score)
        reasons = make_match(assessed).reasons if any(s.status == "evidenced" for s in assessed.skills) else []
        return JobDetailResponse(
            job=job, relevance=score, required_coverage=assessed.required,
            preferred_coverage=assessed.preferred, eligibility=assessed.eligibility,
            experience_assessment=assessed.experience, education_assessment=assessed.education,
            constraints=assessed.constraints, skills=assessed.skills, reasons=reasons,
            passes_policy=not rejected, rejection_reasons=rejected,
            was_recommended=any(match.job.job_id == job_id for match in run.response.recommendations),
            warnings=assessed.warnings + ["Selected-job inspection is not a recommendation; no new Qwen calls were made."],
        )
