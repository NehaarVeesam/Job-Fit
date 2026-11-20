"""Load frozen, public synthetic fixtures; reject leakage and incomplete labels."""

import hashlib
import json
import re
from pathlib import Path

from app.schemas import Preferences


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_suite(directory, source_path):
    directory = Path(directory).resolve()
    suite_path, judgments_path = directory / 'suite.json', directory / 'judgments.jsonl'
    suite = json.loads(suite_path.read_text())
    from app.config import PROJECT_ROOT
    expected_files = {
        'app/matching/constraints.py',
        'app/matching/generation.py',
        'app/matching/matcher.py',
        'app/matching/scoring.py',
        'app/resumes/profile.py',
        'app/retrieval/dense.py',
        'app/retrieval/documents.py',
        'app/retrieval/filters.py',
        'app/retrieval/hybrid.py',
        'app/retrieval/keyword_search.py',
        'app/retrieval/reranking.py',
        'app/schemas.py',
    }
    if set(suite.get('implementation_sha256', {})) != expected_files:
        raise ValueError('Frozen pipeline fingerprints are incomplete')
    if any(digest(PROJECT_ROOT / name) != value for name, value in suite['implementation_sha256'].items()):
        raise ValueError('Pipeline changed after fixture freeze; version/review the evaluation suite before rerunning')
    if digest(PROJECT_ROOT / 'requirements.lock') != suite['requirements_lock_sha256']:
        raise ValueError('Dependency lock changed after fixture freeze')
    if digest(source_path) != suite['dataset_sha256']:
        raise ValueError('Evaluation source differs from frozen dataset')
    panel = suite['panel_job_ids']
    if not panel or len(panel) != len(set(panel)):
        raise ValueError('Invalid evaluation panel')
    cases, identities, resume_hashes, splits = [], set(), set(), set()
    for item in suite['cases']:
        case = dict(item)
        if not re.fullmatch(r'[a-z][a-z0-9_]{0,63}', case['case_id']):
            raise ValueError('Invalid case identifier')
        path = (directory / case['resume_file']).resolve()
        if directory not in path.parents or path.suffix != '.txt':
            raise ValueError('Fixture resume path must be a text file inside evaluation directory')
        if digest(path) != case['resume_sha256']:
            raise ValueError('Resume differs from frozen fixture')
        if case['case_id'] in identities or case['resume_sha256'] in resume_hashes:
            raise ValueError('Duplicate case or exact resume leakage across splits')
        if case['split'] not in ('dev', 'heldout') or type(case['expected_no_match']) is not bool:
            raise ValueError('Invalid split or expected outcome')
        Preferences.model_validate(case['preferences'])
        identities.add(case['case_id']); resume_hashes.add(case['resume_sha256']); splits.add(case['split'])
        case['text'] = path.read_text()
        cases.append(case)
    if splits != {'dev', 'heldout'}:
        raise ValueError('Both development and held-out splits are required')
    labels = {case_id: {} for case_id in identities}
    for line in judgments_path.read_text().splitlines():
        item = json.loads(line)
        case_id, job_id, grade = item['case_id'], item['job_id'], item['grade']
        if case_id not in labels or job_id not in panel or job_id in labels[case_id]:
            raise ValueError('Foreign or duplicate judgment')
        if type(grade) is not int or grade not in (0, 1, 2) or not item['rationale'].strip():
            raise ValueError('Invalid judgment')
        labels[case_id][job_id] = grade
    if any(set(row) != set(panel) for row in labels.values()):
        raise ValueError('Every panel pair must be explicitly judged')
    for case in cases:
        if case['expected_no_match'] != (not any(grade >= 1 for grade in labels[case['case_id']].values())):
            raise ValueError('No-match expectation differs from authored suitability labels')
    for item in suite['expectations']:
        if item['case_id'] not in identities or item['job_id'] not in panel:
            raise ValueError('Foreign diagnostic expectation')
    suite_hash = hashlib.sha256((digest(suite_path) + digest(judgments_path)).encode()).hexdigest()
    return suite, cases, labels, suite_hash
