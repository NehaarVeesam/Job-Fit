"""Cache the pinned local models used by JobFit."""

import argparse

from app.config import get_settings


def download_embeddings(settings):
    from huggingface_hub import HfApi, snapshot_download

    info = HfApi().model_info(settings.embedding_model, revision=settings.embedding_revision or "main")
    print(f"Downloading {settings.embedding_model} revision {info.sha}", flush=True)
    path = snapshot_download(
        repo_id=settings.embedding_model,
        revision=info.sha,
        cache_dir=settings.embedding_cache_dir,
        allow_patterns=["*.json", "*.safetensors", "pytorch_model.bin", "sentencepiece.bpe.model"],
        ignore_patterns=["onnx/*", "openvino/*"],
    )
    print(f"Model snapshot: {path}")
    print("Set JOBFIT_EMBEDDING_REVISION=" + info.sha + " to pin this snapshot.")


def download_reranker(settings):
    from huggingface_hub import snapshot_download

    print(f"Downloading {settings.rerank_model} at {settings.rerank_revision}", flush=True)
    snapshot = snapshot_download(
        repo_id=settings.rerank_model,
        revision=settings.rerank_revision,
        cache_dir=settings.rerank_cache_dir,
        allow_patterns=[
            "config.json", "model.safetensors", "tokenizer.json",
            "tokenizer_config.json", "special_tokens_map.json", "sentencepiece.bpe.model",
        ],
    )
    print(f"Reranker snapshot: {snapshot}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("embeddings", "reranker"))
    args = parser.parse_args(argv)
    settings = get_settings()
    if args.model == "embeddings":
        download_embeddings(settings)
    else:
        download_reranker(settings)


if __name__ == "__main__":
    main()
