from pathlib import Path
from pydantic import PositiveInt
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="JOBFIT_",
        env_file=".env",
        env_file_encoding="utf-8"
    )

    raw_jobs_path: Path = Path("dataset/jobs_synthetic_250.jsonl")
    prepared_data_dir: Path = Path("artifacts/prepared")
    retrieval_top_k: PositiveInt = 10
    
    @property
    def prepared_jobs_path(self) -> Path:
        return self.prepared_data_dir / "jobs.jsonl"
    