"""Download only BGE-M3 dense Sentence Transformers files into the project cache."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings


def main():
    from huggingface_hub import HfApi, snapshot_download

    settings = get_settings()
    info = HfApi().model_info(settings.embedding_model, revision=settings.embedding_revision or "main")
    print(f"Downloading {settings.embedding_model} revision {info.sha}", flush=True)
    path = snapshot_download(
        repo_id=settings.embedding_model, revision=info.sha,
        cache_dir=settings.embedding_cache_dir,
        allow_patterns=["*.json", "*.safetensors", "pytorch_model.bin", "sentencepiece.bpe.model"],
        ignore_patterns=["onnx/*", "openvino/*"],
    )
    print(f"Model snapshot: {path}")
    print("Set JOBFIT_EMBEDDING_REVISION=" + info.sha + " to pin this snapshot.")


if __name__ == "__main__":
    main()
