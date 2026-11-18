"""Download the pinned optional cross-encoder; normal inference stays local."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import get_settings


def main():
    from huggingface_hub import snapshot_download
    settings = get_settings()
    print(f'Downloading {settings.rerank_model} at {settings.rerank_revision}', flush=True)
    snapshot = snapshot_download(repo_id=settings.rerank_model, revision=settings.rerank_revision,
                                 cache_dir=settings.rerank_cache_dir,
                                 allow_patterns=['config.json', 'model.safetensors', 'tokenizer.json',
                                                 'tokenizer_config.json', 'special_tokens_map.json', 'sentencepiece.bpe.model'])
    print(f'Reranker snapshot: {snapshot}')


if __name__ == '__main__':
    main()
