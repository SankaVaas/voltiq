"""
core/config.py — centralised settings loaded from environment / .env file.
All other modules import `settings` from here; never read os.environ directly.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ──────────────────────────────────────────────────────────────────
    app_env: Literal["development", "staging", "production"] = "development"
    app_name: str = "voltiq"
    app_version: str = "0.1.0"
    log_level: str = "INFO"

    # ── API ──────────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    api_secret_key: str = Field(default="dev-secret", min_length=8)

    # ── External API keys ────────────────────────────────────────────────────
    anthropic_api_key: str = Field(default="", repr=False)
    entso_e_api_key: str = Field(default="", repr=False)

    # ── Qdrant ───────────────────────────────────────────────────────────────
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_incidents: str = "grid_incidents"
    qdrant_collection_reports: str = "entso_reports"

    # ── MLflow ───────────────────────────────────────────────────────────────
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_forecast: str = "voltiq_forecast"
    mlflow_experiment_anomaly: str = "voltiq_anomaly"

    # ── Azure ────────────────────────────────────────────────────────────────
    azure_storage_account: str = ""
    azure_storage_key: str = Field(default="", repr=False)
    azure_container_name: str = "voltiq-data"

    # ── AWS ──────────────────────────────────────────────────────────────────
    aws_access_key_id: str = Field(default="", repr=False)
    aws_secret_access_key: str = Field(default="", repr=False)
    aws_default_region: str = "eu-west-1"
    s3_bucket: str = "voltiq-data"

    # ── Model hyper-params ───────────────────────────────────────────────────
    forecast_horizon: int = 48
    forecast_lookback: int = 168
    anomaly_threshold: float = 0.95
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    llm_model: str = "claude-3-5-haiku-20241022"
    llm_max_tokens: int = 2048

    # ── Paths ────────────────────────────────────────────────────────────────
    data_raw_dir: Path = ROOT_DIR / "data" / "raw"
    data_processed_dir: Path = ROOT_DIR / "data" / "processed"
    model_artifact_dir: Path = ROOT_DIR / "data" / "artifacts"

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return upper

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def qdrant_url(self) -> str:
        return f"http://{self.qdrant_host}:{self.qdrant_port}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings singleton. Use this everywhere."""
    return Settings()


settings = get_settings()
