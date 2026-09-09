from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"
    perception_mode: str = "real"
    perception_module: str = "context_perception"
    perception_confidence_threshold: float = Field(default=0.45, ge=0, le=1)

    nvidia_api_key: str | None = None
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_model: str | None = None
    nvidia_supports_vision: bool = False
    nvidia_daily_budget: int = Field(default=0, ge=0)

    gemini_api_key: str | None = None
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    gemini_model: str | None = None
    gemini_daily_budget: int = Field(default=0, ge=0)

    local_base_url: str | None = None
    local_model: str | None = None
    local_api_key: str | None = None
    local_supports_vision: bool = False
    local_daily_budget: int = Field(default=0, ge=0)

    provider_timeout_seconds: float = Field(default=45.0, gt=0)


@lru_cache
def get_settings() -> Settings:
    return Settings()
