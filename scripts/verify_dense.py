"""Real-model development check; separate from fast tests and held-out evaluation.

Requires downloaded model files, prepared jobs, and generated demo PDFs.
Blocks socket connections to verify cached local operation.
"""

import hashlib
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import PROJECT_ROOT, get_settings
from app.embeddings import load_embeddings
from app.profile import build_profile, load_prepared_jobs, skill_vocabulary
from app.resume_parser import parse_resume
from app.retrieval import retrieve
from app.schemas import Preferences
from app.vector_index import build_index, manifest_path, prepare_documents, save_json


def run():
    settings = get_settings()
    if settings.qdrant_mode != "local":
        raise ValueError("This offline verification requires Qdrant local mode")
    jobs, provenance = load_prepared_jobs(settings.artifacts_dir / "dataset/all")
    vocabulary = skill_vocabulary(jobs)
    started = time.perf_counter()
    embeddings, info = load_embeddings(settings)
    load_seconds = round(time.perf_counter() - started, 3)
    _, identity = prepare_documents(jobs, provenance, info)
    path = manifest_path(settings, identity)
    first_snapshot = settings.artifacts_dir / "verification/module3_initial_index.json"
    if path.exists() and not first_snapshot.exists():
        save_json(first_snapshot, json.loads(path.read_text()))
    print("Checking index reuse...", flush=True)
    index = build_index(settings, jobs, provenance, embeddings, info)
    if not index["reused_existing_vectors"]:
        # On a fresh build, immediately repeat to verify idempotency too.
        save_json(first_snapshot, index)
        index = build_index(settings, jobs, provenance, embeddings, info)
    assert index["reused_existing_vectors"]
    outcomes = []
    for name in ("ai", "backend", "analyst", "fresher", "unrelated"):
        pdf = settings.artifacts_dir / "demo_resumes" / (name + ".pdf")
        extraction = parse_resume(pdf)
        preferences = (Preferences.model_validate_json((PROJECT_ROOT / "examples/preferences.json").read_text())
                       if name == "ai" else Preferences())
        prepared = build_profile(extraction.parsed, vocabulary, preferences)
        result = retrieve(settings, jobs, provenance, embeddings, info, prepared, k=5)
        input_hash = hashlib.sha256(prepared.model_dump_json().encode()).hexdigest()
        result_path = settings.artifacts_dir / "retrieval" / f"{input_hash}_5.json"
        save_json(result_path, result)
        summary = {"profile": name, "status": result["status"],
                   "eligible_count": result["eligible_count"],
                   "embedding_and_search_seconds": result["timing_seconds"]["embedding_and_search"],
                   "top_candidates": [{"job_id": item["job_id"], "title": item["job"]["title"],
                                       "cosine": item["relevance"]["value"]} for item in result["candidates"]]}
        outcomes.append(summary)
        print(f"{name}: {result['status']}; " + ", ".join(item["job_id"] for item in result["candidates"]), flush=True)
    report = {"scope": "Development demos; not held-out suitability evaluation",
              "socket_connections_blocked": True, "model_load_seconds": load_seconds,
              "index_repeat": index, "queries": outcomes}
    output = settings.artifacts_dir / "verification/module3_live.json"
    save_json(output, report)
    print(f"Saved real-model verification: {output}")


if __name__ == "__main__":
    with patch("socket.socket.connect", side_effect=AssertionError("Unexpected network connection")):
        run()
