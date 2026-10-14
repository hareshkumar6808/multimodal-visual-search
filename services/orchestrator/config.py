from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"
    perception_mode: str = "real"
    perception_module: str = "services.perception"
    perception_confidence_threshold: float = Field(default=0.45, ge=0, le=1)
    ml_router_enabled: bool = True
    ml_router_artifact_path: str = "services/orchestrator/artifacts/task_router_v1.joblib"

    nvidia_api_key: str | None = None
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_model: str | None = None
    nvidia_supports_vision: bool = False
    nvidia_reasoning_model: str | None = None
    nvidia_vision_model: str | None = None
    nvidia_daily_budget: int = Field(default=0, ge=0)
    nvidia_max_tokens: int = Field(default=384, ge=32, le=4096)

    gemini_api_key: str | None = None
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    gemini_model: str | None = "gemini-2.5-flash"
    gemini_daily_budget: int = Field(default=0, ge=0)

    local_base_url: str | None = None
    local_model: str | None = None
    local_api_key: str | None = None
    local_supports_vision: bool = False
    local_daily_budget: int = Field(default=0, ge=0)
    local_provider_timeout_seconds: float = Field(default=30.0, gt=0)
    local_max_tokens: int = Field(default=512, ge=32, le=4096)

    provider_timeout_seconds: float = Field(default=10.0, gt=0)
    conversation_db_path: str = ".tools/stage1/conversations.db"
    provider_performance_db_path: str = ".tools/stage1/conversations.db"
    provider_ewma_alpha: float = Field(default=0.25, gt=0, le=1)
    provider_ttft_weight: float = Field(default=0.6, ge=0)
    provider_total_latency_weight: float = Field(default=0.4, ge=0)
    provider_failure_penalty_ms: float = Field(default=8_000, ge=0)
    provider_min_samples: int = Field(default=3, ge=1)


@lru_cache
def get_settings() -> Settings:
    return Settings()
