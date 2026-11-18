"""Coverage, requirement decisions, acceptance, and bounded model grounding."""

import copy
import hashlib
import json
import unittest

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.config import PROJECT_ROOT, Settings
from app.constraints import education_requirement, experience_requirement, hard_constraint
from app.generation import ExplanationPlan, GroundedExplainer
from app.matching import match_candidates
from app.profile import build_profile, skill_vocabulary
from app.schemas import JobRecord, MatchResponse, ParsedResume, Preferences, RelevanceScore
from app.scoring import assess_job, make_match, rejection_reasons


def profile(text=None, preferences=None):
    text = text or (PROJECT_ROOT / "examples/resumes/ai.txt").read_text()
    return build_profile(ParsedResume(pages=[{"page": 1, "text": text}]), VOCABULARY, preferences)


JOBS = [JobRecord.model_validate_json(line) for line in
        (PROJECT_ROOT / "dataset/jobs_synthetic_250.jsonl").read_text().splitlines()]
VOCABULARY = skill_vocabulary(JOBS)
PROVENANCE = {"configuration": {"subset": "all"}}


def settings(**kwargs):
    return Settings(_env_file=None, llm_provider="ollama", **kwargs)


def retrieval(prepared, jobs=None, score=0.8):
    return {"applied_preferences": prepared.preferences.model_dump(mode="json"),
            "query_sha256": hashlib.sha256((prepared.retrieval_query or "").encode()).hexdigest(),
            "is_synthetic": True, "corpus_size": len(JOBS), "index_id": "test-index",
            "candidates": [{"job_id": job.job_id, "job": job.model_dump(mode="json"),
                            "relevance": {"value": score, "score_type": "cosine"}}
                           for job in (jobs if jobs is not None else [JOBS[0]])]}


class CoverageTests(unittest.TestCase):
    def test_ai_counts_match_actual_evidence_and_source_quotes(self):
        prepared = profile()
        assessed = assess_job(JOBS[0], prepared, RelevanceScore(value=0.8, score_type="cosine"))
        self.assertEqual((assessed.required.evidenced_count, assessed.required.total_count, assessed.required.percentage), (4, 6, 66.7))
        self.assertEqual(assessed.preferred.evidenced_count, 1)
        match = make_match(assessed)
        self.assertEqual(match.eligibility, "met")
        for reason in match.reasons:
            self.assertIn(reason.resume_evidence.quote, prepared.parsed_resume.pages[0].text)
            self.assertIn(reason.job_evidence.quote, JOBS[0].description)

    def test_related_skills_and_false_profile_claim_do_not_count(self):
        prepared = profile("Skills\nFlask, JavaScript\nProjects\nBuilt a Flask app")
        # A cited quote's existence does not establish that it supports Python.
        prepared.candidate_profile.skills[0].text = "Python"
        assessed = assess_job(JOBS[0], prepared)
        self.assertEqual(assessed.required.evidenced_count, 0)
        self.assertTrue(all(item.status == "not_evidenced_in_resume" for item in assessed.skills))

    def test_unknown_and_above_maximum_experience_are_handled_conservatively(self):
        fresher = profile((PROJECT_ROOT / "examples/resumes/fresher.txt").read_text())
        self.assertEqual(experience_requirement(JOBS[0], fresher).status, "unknown")
        senior = profile("Summary\n20 years of professional experience\nSkills\nPython")
        self.assertEqual(experience_requirement(JOBS[0], senior).status, "met")
        junior = profile("Summary\n1 year of professional experience\nSkills\nPython")
        self.assertEqual(experience_requirement(JOBS[0], junior).status, "not_met")

    def test_education_alternatives_are_not_rejected_without_a_degree(self):
        prepared = profile("Skills\nPython, LangChain, RAG, FastAPI\nExperience\nBuilt APIs")
        self.assertEqual(education_requirement(JOBS[0], prepared).status, "unknown")
        prepared.preferences.eligibility_facts["education_requirement_met:syn_job_001"] = True
        self.assertEqual(education_requirement(JOBS[0], prepared).status, "met")


class RequirementTests(unittest.TestCase):
    def test_all_source_conditions_support_unknown_and_explicit_true_false(self):
        for condition in {text for job in JOBS for text in job.hard_constraints}:
            with self.subTest(condition=condition):
                self.assertEqual(hard_constraint(condition, {}).status, "unknown")
                for answer, expected in ((True, "met"), (False, "not_met")):
                    result = hard_constraint(condition, {"constraint:" + condition: answer})
                    self.assertEqual(result.status, expected)
                    self.assertEqual(result.user_fact_keys, ["constraint:" + condition])

    def test_graduation_notice_and_travel_rules_use_explicit_typed_answers(self):
        cases = [
            ("Final-year students graduating in 2027", {"is_final_year": True}, "unknown"),
            ("Final-year students graduating in 2027", {"is_final_year": True, "graduation_year": 2027}, "met"),
            ("Final-year students graduating in 2027", {"is_final_year": False}, "not_met"),
            ("Open to 2025 and 2026 graduates only", {"graduation_year": True}, "unknown"),
            ("Must be able to join within 60 days", {"notice_period_days": 90}, "not_met"),
            ("Up to 30% travel to client sites", {"max_travel_percent": 25}, "not_met"),
        ]
        for condition, facts, expected in cases:
            self.assertEqual(hard_constraint(condition, facts).status, expected)

    def test_resume_address_and_preferences_do_not_answer_conditions(self):
        prepared = profile("Address: India\nSkills\nPython", Preferences(location_city="Jaipur"))
        self.assertEqual(hard_constraint("Must be based in India", prepared.preferences.eligibility_facts).status, "unknown")
        self.assertEqual(hard_constraint("Must work from the Jaipur office", prepared.preferences.eligibility_facts).status, "unknown")


class MatchingTests(unittest.TestCase):
    def test_acceptance_low_scores_gaps_and_no_match_status(self):
        prepared = profile()
        response, report = match_candidates(settings(), JOBS, PROVENANCE, prepared, retrieval(prepared), use_llm=False)
        self.assertEqual(response.status, "matches_found")
        self.assertEqual(report["generation_calls"], 0)
        weak, _ = match_candidates(settings(), JOBS, PROVENANCE, prepared, retrieval(prepared, score=0.1), use_llm=False)
        self.assertEqual(weak.status, "no_suitable_matches")
        self.assertEqual(weak.recommendations, [])
        self.assertEqual(MatchResponse.model_validate_json(response.model_dump_json()), response)

    def test_known_failure_is_rejected_and_unknown_policy_is_explicit(self):
        prepared = profile("Summary\n1 year of professional experience\nSkills\nPython, LangChain, RAG, FastAPI")
        response, _ = match_candidates(settings(), JOBS, PROVENANCE, prepared, retrieval(prepared), use_llm=False)
        self.assertEqual(response.status, "no_suitable_matches")
        unknown = profile("Skills\nPython, LangChain, RAG, FastAPI")
        response, _ = match_candidates(settings(), JOBS, PROVENANCE, unknown, retrieval(unknown), use_llm=False)
        self.assertEqual(response.recommendations[0].eligibility, "unknown")
        unknown.preferences.unknown_metadata_policy = "exclude"
        strict, _ = match_candidates(settings(), JOBS, PROVENANCE, unknown, retrieval(unknown), use_llm=False)
        self.assertEqual(strict.status, "no_suitable_matches")

    def test_foreign_duplicate_changed_source_and_query_are_rejected(self):
        prepared = profile()
        for mutate in (
            lambda r: r["candidates"].append(copy.deepcopy(r["candidates"][0])),
            lambda r: r["candidates"][0]["job"].update(title="Invented"),
            lambda r: r.update(query_sha256="wrong"),
        ):
            report = retrieval(prepared)
            mutate(report)
            with self.assertRaises(ValueError):
                match_candidates(settings(), JOBS, PROVENANCE, prepared, report, use_llm=False)

    def test_no_query_no_match_and_zero_generation_calls(self):
        prepared = profile((PROJECT_ROOT / "examples/resumes/unrelated.txt").read_text())
        response, report = match_candidates(settings(), JOBS, PROVENANCE, prepared, retrieval(prepared, []))
        self.assertEqual(response.status, "no_suitable_matches")
        self.assertEqual(report["generation_calls"], 0)

    def test_recommendation_limit_and_empty_required_list(self):
        prepared = profile()
        jobs = [JOBS[0].model_copy(update={"job_id": f"syn_job_{900+i}"}) for i in range(6)]
        result = retrieval(prepared, jobs)
        result["corpus_size"] = len(jobs)
        response, _ = match_candidates(settings(), jobs, PROVENANCE, prepared, result, use_llm=False)
        self.assertEqual(len(response.recommendations), 5)
        empty = JOBS[0].model_copy(update={"required_skills": []})
        assessed = assess_job(empty, prepared, RelevanceScore(value=0.8, score_type="cosine"))
        self.assertIsNone(assessed.required.percentage)
        self.assertTrue(rejection_reasons(assessed, prepared, settings()))


def fake_chain(responses):
    # Exercise an actual LangChain prompt/model/parser composition without HTTP.
    return (ChatPromptTemplate.from_messages([("human", "{data}")])
            | FakeListChatModel(responses=responses)
            | PydanticOutputParser(pydantic_object=ExplanationPlan))


class GroundedGenerationTests(unittest.TestCase):
    def setUp(self):
        self.prepared = profile()
        self.assessed = assess_job(JOBS[0], self.prepared)
        self.gaps = [item.skill for item in self.assessed.skills if item.status == "not_evidenced_in_resume"]

    def test_valid_plan_changes_only_reason_selection(self):
        explainer = GroundedExplainer(settings(), fake_chain([json.dumps({"strengths": ["RAG", "Python"], "gaps": self.gaps})]))
        response, report = match_candidates(settings(), JOBS, PROVENANCE, self.prepared,
                                            retrieval(self.prepared), explainer=explainer)
        match = response.recommendations[0]
        self.assertEqual(match.required_coverage.percentage, 66.7)
        self.assertEqual(match.generation_status, "qwen_grounded_plan")
        self.assertIn("mentions RAG", match.reasons[0].explanation)
        self.assertEqual(report["generation_calls"], 1)

    def test_invented_skill_or_score_field_is_rejected_with_bounded_fallback(self):
        for output in ({"strengths": ["Kubernetes"], "gaps": self.gaps},
                       {"strengths": ["Python"], "gaps": self.gaps, "score": 100},
                       {"strengths": ["Python"], "gaps": []}):
            explainer = GroundedExplainer(settings(), fake_chain([json.dumps(output)]))
            response, report = match_candidates(settings(), JOBS, PROVENANCE, self.prepared,
                                                retrieval(self.prepared), explainer=explainer)
            match = response.recommendations[0]
            self.assertEqual(match.generation_status, "fallback")
            self.assertEqual(match.required_coverage.percentage, 66.7)
            self.assertEqual(report["generation_calls"], 2)
            self.assertNotIn("Kubernetes", " ".join(item.explanation for item in match.reasons))

    def test_invalid_then_valid_output_is_retried(self):
        explainer = GroundedExplainer(settings(), fake_chain([
            "not JSON", json.dumps({"strengths": ["Python"], "gaps": self.gaps})]))
        self.assertIsNotNone(explainer.explain(self.assessed))
        self.assertEqual(explainer.calls, 2)

    def test_injection_job_text_cannot_change_scores_or_become_instructions(self):
        job = JOBS[212]
        plain = job.model_copy(update={"description": "\n".join(
            "Must-have skills: " + skill for skill in job.required_skills + job.preferred_skills)})
        text = "Skills\n" + ", ".join(job.required_skills)
        candidate = profile(text)
        original = assess_job(job, candidate)
        cleaned = assess_job(plain, candidate)
        self.assertEqual(original.required, cleaned.required)
        self.assertEqual(original.preferred, cleaned.preferred)
        self.assertEqual(original.eligibility, cleaned.eligibility)
        class Capture:
            def invoke(self, inputs):
                data = json.loads(inputs["data"])
                self.data = data
                return {"strengths": data["allowed_strengths"][:1], "gaps": data["allowed_gaps"]}
        chain = Capture()
        self.assertIsNotNone(GroundedExplainer(settings(), chain).explain(original))
        self.assertNotIn(job.description, json.dumps(chain.data))
        self.assertEqual(set(chain.data), {"job_id", "allowed_strengths", "allowed_gaps", "requirements"})


if __name__ == "__main__":
    unittest.main()
