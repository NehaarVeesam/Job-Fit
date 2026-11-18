"""Inventory local runtime artifacts without loading models or reading resume content."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import get_settings
from app.vector_index import save_json


def inventory(settings):
    root = settings.artifacts_dir
    groups = {
        'prepared_dataset': (root / 'dataset/all', ['jobs.jsonl', 'skill_evidence.jsonl', 'validation_report.json']),
        'embedding_cache': (settings.embedding_cache_dir / 'models--BAAI--bge-m3' / 'snapshots' / settings.embedding_revision,
                            ['config.json', 'tokenizer.json', 'pytorch_model.bin']),
        'reranker_cache': (settings.rerank_cache_dir / 'models--BAAI--bge-reranker-v2-m3' / 'snapshots' / settings.rerank_revision,
                           ['config.json', 'tokenizer.json', 'model.safetensors']),
        'index_manifests': (root / 'indexes', ['*.json']),
        'qdrant_storage': (settings.qdrant_path, ['collection/*']),
        'prepared_resumes': (root / 'resumes', ['*/prepared_resume.json']),
        'matching_results': (root / 'matches', ['*/response.json']),
        'verification_reports': (root / 'verification', ['*.json']),
        'evaluation_runs': (root / 'evaluation', ['*/summary.json']),
        'portfolio_demo': (root / 'portfolio_demo', ['summary.json', 'ai/response.json', 'unrelated/response.json']),
        'pdf_run_traces': (root / 'resume_runs', ['*/events.jsonl', '*/run_manifest.json']),
    }
    rows = []
    for name, (directory, patterns) in groups.items():
        matches = {pattern: sum(path.exists() for path in directory.glob(pattern)) for pattern in patterns}
        rows.append({'group': name, 'directory': str(directory), 'expected_patterns': matches,
                     'status': 'present' if all(matches.values()) else 'missing_or_partial'})
    return {'scope': 'Presence inventory only; not a checksum, model-integrity, or historical-log completeness audit.',
            'historical_logs': 'Older commands did not all persist logs. New PDF runs save events.jsonl.',
            'groups': rows}


def main():
    settings = get_settings()
    report = inventory(settings)
    target = settings.artifacts_dir / 'artifact_inventory.json'
    save_json(target, report)
    for row in report['groups']:
        print(f"{row['group']}: {row['status']}")
    print(f'Inventory: {target}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
