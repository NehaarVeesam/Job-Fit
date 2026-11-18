"""Load local configuration explicitly; no model/database calls on import."""

from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, HttpUrl, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="JOBFIT_", env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8", extra="forbid", str_strip_whitespace=True,
        populate_by_name=True,
    )

    dataset_path: Path = PROJECT_ROOT / "dataset" / "jobs_synthetic_250.jsonl"
    dataset_id: Literal["synthetic_india_ai_sw_v2"] = "synthetic_india_ai_sw_v2"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    corpus_subset: Literal["all", "first_100"] = "all"
    recency_filter_enabled: bool = False
    reference_date: date = date(2026, 11, 18)
    recency_window_days: int = Field(default=90, ge=1)

    embedding_model: Literal["BAAI/bge-m3"] = "BAAI/bge-m3"
    embedding_dimension: int = Field(default=1024, ge=1024, le=1024)
    normalize_embeddings: bool = True
    vector_distance: Literal["cosine"] = "cosine"
    embedding_device: Literal["cpu", "cuda", "mps"] = "cpu"
    embedding_batch_size: int = Field(default=4, ge=1)
    embedding_revision: str = Field(default="5617a9f61b028005a4858fdac845db406aefb181",
                                    pattern=r"^[0-9a-f]{40}$")
    embedding_cache_dir: Path = PROJECT_ROOT / "artifacts" / "model_cache"
    embedding_cpu_threads: int = Field(default=4, ge=1, le=32)
    qdrant_mode: Literal["local", "server"] = "local"
    qdrant_path: Path = PROJECT_ROOT / "artifacts" / "qdrant"
    qdrant_url: HttpUrl = HttpUrl("http://127.0.0.1:6333")
    qdrant_collection: str = Field(default="jobfit_synthetic_india_ai_sw_v2", min_length=1)

    llm_provider: Literal["ollama", "openai_compatible"] = "openai_compatible"
    llm_model: str = Field(
        default="Qwen/Qwen3.8-27B", min_length=1,
        validation_alias=AliasChoices("JOBFIT_LLM_MODEL", "QWEN_MODEL"),
    )
    llm_base_url: HttpUrl | None = Field(default=None, validation_alias="QWEN_BASE_URL")
    llm_api_key: SecretStr | None = Field(default=None, validation_alias="QWEN_API_KEY", repr=False)
    llm_json_mode: Literal["json_schema", "json_object", "none"] | bool = Field(
        default="json_schema", validation_alias="QWEN_JSON_MODE",
    )
    llm_max_tokens: int = Field(default=2048, ge=1, validation_alias="QWEN_MAX_TOKENS")
    llm_max_model_len: int | None = Field(default=None, ge=1, validation_alias="QWEN_MAX_MODEL_LEN")
    llm_enable_thinking: bool = Field(default=False, validation_alias="QWEN_ENABLE_THINKING")
    llm_temperature: float | None = Field(default=None, ge=0, le=2, validation_alias="QWEN_TEMPERATURE")
    llm_top_p: float | None = Field(default=None, gt=0, le=1, validation_alias="QWEN_TOP_P")
    llm_timeout: float = Field(default=120, gt=0, validation_alias="QWEN_TIMEOUT")
    ollama_base_url: HttpUrl = HttpUrl("http://127.0.0.1:11434")
    max_recommendations: int = Field(default=5, ge=1, le=5)
    candidate_count: int = Field(default=30, ge=1, le=250)
    retrieval_mode: Literal["dense", "bm25", "hybrid"] = "dense"
    bm25_candidate_count: int = Field(default=30, ge=1, le=250)
    bm25_k1: float = Field(default=1.5, gt=0, allow_inf_nan=False)
    bm25_b: float = Field(default=0.75, ge=0, le=1, allow_inf_nan=False)
    rrf_constant: int = Field(default=60, ge=1, le=10000)
    min_cosine_similarity: float = Field(default=0.55, ge=-1, le=1)
    min_required_coverage: float = Field(default=50, ge=0, le=100)
    min_evidenced_required: int = Field(default=2, ge=1)
    generation_attempts: int = Field(default=2, ge=1, le=3)
    api_timeout_seconds: float = Field(default=120, gt=0, le=600)
    api_session_ttl_seconds: int = Field(default=1800, ge=1, le=86400)
    api_max_sessions: int = Field(default=20, ge=1, le=100)
    max_upload_mb: int = Field(default=5, ge=1, le=50)
    max_resume_pages: int = Field(default=10, ge=1, le=100)

    @field_validator("llm_max_model_len", "llm_temperature", "llm_top_p", mode="before")
    @classmethod
    def blank_means_server_default(cls, value):
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator("dataset_path", "artifacts_dir", "embedding_cache_dir", "qdrant_path")
    @classmethod
    def resolve_project_path(cls, value: Path) -> Path:
        value = value.expanduser()
        return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()

    @field_validator("normalize_embeddings")
    @classmethod
    def require_normalized_vectors(cls, value: bool) -> bool:
        if not value:
            raise ValueError("This project requires normalized embeddings")
        return value

    @model_validator(mode="after")
    def protect_input_folder(self):
        if self.llm_provider == "openai_compatible" and self.llm_base_url is None:
            raise ValueError("Set QWEN_BASE_URL for the OpenAI-compatible model server")
        source_folder = self.dataset_path.parent
        if self.artifacts_dir == source_folder or source_folder in self.artifacts_dir.parents:
            raise ValueError("Write generated artifacts outside the input dataset folder")
        for path in (self.embedding_cache_dir, self.qdrant_path):
            if path == source_folder or source_folder in path.parents:
                raise ValueError("Model cache and Qdrant data must be outside the input dataset folder")
        if "synthetic" not in self.qdrant_collection.casefold():
            raise ValueError("Use a collection name that clearly identifies synthetic data")
        return self


def get_settings() -> Settings:
    return Settings()
