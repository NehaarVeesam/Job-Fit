"""Independent fixture expectations and narrowly scoped evidence/policy audits."""

import json
import re

from app.matching.constraints import hard_constraint
from app.config import DATASET_JOB_IDS
from app.matching.generation import GroundedExplainer
from app.schemas import RelevanceScore
from app.matching.scoring import assess_job


def expectation_audit(suite, prepared_by_case, jobs):
    by_id = {job.job_id: job for job in jobs}
    checks = []
    for expected in suite['expectations']:
        case_id, job_id = expected['case_id'], expected['job_id']
        assessed = assess_job(by_id[job_id], prepared_by_case[case_id])
        observations = {'required_evidenced': assessed.required.evidenced_count,
                        'required_total': assessed.required.total_count, 'experience': assessed.experience.status}
        if 'constraint_status' in expected:
            observations['constraint_status'] = ('not_met' if any(c.status == 'not_met' for c in assessed.constraints)
                                                  else 'unknown' if any(c.status == 'unknown' for c in assessed.constraints) else 'met')
        for key, value in expected.items():
            if key in ('case_id', 'job_id'):
                continue
            actual = any(value in warning for warning in assessed.warnings) if key == 'warning_contains' else observations[key]
            checks.append({'case_id': case_id, 'job_id': job_id, 'check': key,
                           'expected': True if key == 'warning_contains' else value, 'actual': actual,
                           'passed': actual == (True if key == 'warning_contains' else value)})
    return checks


def evidence_audit(response, prepared):
    checks = []
    pages = {page.page: page.text for page in prepared.parsed_resume.pages}
    for match in response.recommendations:
        for kind, names, coverage in (('required', match.job.required_skills, match.required_coverage),
                                      ('preferred', match.job.preferred_skills, match.preferred_coverage)):
            entries = [item for item in match.skills if item.requirement_type == kind]
            supported = sum(item.status == 'evidenced' for item in entries)
            expected = round(100 * supported / len(names), 1) if names else None
            checks.append({'job_id': match.job.job_id, 'check': f'{kind}_coverage_arithmetic',
                           'passed': coverage.evidenced_count == supported and coverage.total_count == len(names)
                           and coverage.percentage == expected})
        for skill in match.skills:
            pattern = re.compile(r'(?<![\w.+#])' + re.escape(skill.skill) + r'(?![\w+#])', re.I)
            valid = skill.job_evidence.quote in match.job.description and bool(pattern.search(skill.job_evidence.quote))
            if skill.status == 'evidenced':
                valid = valid and bool(skill.resume_evidence) and all(
                    item.quote in pages.get(item.page, '') and bool(pattern.search(item.quote)) for item in skill.resume_evidence)
            else:
                valid = valid and not skill.resume_evidence
            checks.append({'job_id': match.job.job_id, 'check': 'skill_literal_support', 'passed': bool(valid)})
        for reason in match.reasons:
            parsed = re.fullmatch(r'Your resume mentions (.+); this job lists it as a (required|preferred) skill\.', reason.explanation)
            valid = bool(parsed)
            if parsed:
                name, kind = parsed.groups()
                pattern = re.compile(r'(?<![\w.+#])' + re.escape(name) + r'(?![\w+#])', re.I)
                names = match.job.required_skills if kind == 'required' else match.job.preferred_skills
                valid = name in names and bool(pattern.search(reason.resume_evidence.quote)) and bool(pattern.search(reason.job_evidence.quote))
                valid = valid and reason.resume_evidence.quote in pages.get(reason.resume_evidence.page, '') and reason.job_evidence.quote in match.job.description
            checks.append({'job_id': match.job.job_id, 'check': 'closed_template_explanation_support', 'passed': bool(valid)})
    return checks


def edge_audit(settings, full_jobs, prepared_by_case):
    by_id = {job.job_id: job for job in full_jobs}
    checks = []
    # This tests deterministic policy and prompt isolation, not invariant dense
    # rankings after changing text: embeddings may legitimately change with text.
    job = by_id['syn_job_213']
    prepared = prepared_by_case.get('heldout_android_injection')
    if prepared:
        original = assess_job(job, prepared, RelevanceScore(value=.8, score_type='cosine'))
        clean = job.model_copy(update={'description': '\n'.join(line for line in job.description.splitlines()
                                                                if 'ignore' not in line.casefold() and 'screening' not in line.casefold())})
        changed = assess_job(clean, prepared, RelevanceScore(value=.8, score_type='cosine'))
        same = (clean.description != job.description and original.required == changed.required and original.preferred == changed.preferred
                and original.eligibility == changed.eligibility and original.constraints == changed.constraints)
        checks.append({'check': 'injection_does_not_change_deterministic_requirements', 'passed': same})
        class Capture:
            data = None
            def invoke(self, data):
                self.data = json.loads(data['data'])
                return {'strengths': self.data['allowed_strengths'][:1], 'gaps': self.data['allowed_gaps']}
        capture = Capture()
        plan = GroundedExplainer(settings, chain=capture).explain(original)
        checks.append({'check': 'generation_prompt_contains_only_allowlisted_skill_data',
                       'passed': plan is not None and set(capture.data) == {'job_id','allowed_strengths','allowed_gaps','requirements'}
                       and 'description' not in capture.data and 'resume' not in capture.data})
    checks.append({'check': 'full_dataset_keeps_all_250_synthetic_jobs',
                   'passed': len(full_jobs) == len(DATASET_JOB_IDS) and set(by_id) == DATASET_JOB_IDS})
    condition = 'Open to 2025 and 2026 graduates only'
    checks.append({'check': 'unanswered_mandatory_condition_remains_unknown',
                   'passed': hard_constraint(condition, {}).status == 'unknown'})
    from app.retrieval.filters import job_passes_preferences
    from app.schemas import Preferences
    checks.append({'check': 'strict_unknown_work_mode_excluded',
                   'passed': not job_passes_preferences(by_id['syn_job_021'], Preferences(work_mode='hybrid', unknown_metadata_policy='exclude'))})
    checks.append({'check': 'india_wide_remote_is_not_an_unknown_city_rejection',
                   'passed': job_passes_preferences(by_id['syn_job_075'], Preferences(location_city='Bengaluru', work_mode='remote', unknown_metadata_policy='exclude'))})
    return checks
