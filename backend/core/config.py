"""Настройки приложения из переменных окружения и файла `.env`."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Типизированная конфигурация API и будущих RAG-сервисов."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "KPFU Web Assistant"
    app_env: Literal["development", "testing", "production"] = "development"
    debug: bool = True

    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, ge=1, le=65535)
    api_prefix: str = "/api"
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:8501"]

    database_url: str = (
        "postgresql+asyncpg://kpfu:kpfu_local_password@localhost:5432/"
        "kpfu_assistant"
    )

    llm_provider: str = "ollama"
    llm_model: str = "qwen2.5:7b"
    ollama_base_url: str = "http://localhost:11434"
    external_llm_api_base_url: str | None = None
    external_llm_api_key: str | None = None

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "kpfu_documents"
    redis_url: str = "redis://localhost:6379/0"

    jwt_secret_key: str = "change-me-in-local-env"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = Field(default=60, ge=1)

    log_level: str = "INFO"
    log_file: str = "logs/app.log"

    @property
    def log_path(self) -> Path:
        """Вернуть путь к лог-файлу относительно корня проекта."""

        return Path(self.log_file)


@lru_cache
def get_settings() -> Settings:
    """Создать настройки один раз и переиспользовать их в приложении."""

    return Settings()


settings = get_settings()
