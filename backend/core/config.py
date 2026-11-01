"""Типизированные настройки приложения из переменных окружения."""

from functools import lru_cache
from typing import List, Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings

_DEFAULT_JWT_SECRET = "change-me-in-production"
_ALLOWED_ENGINES = {"auto", "ollama", "huggingface"}


class Settings(BaseSettings):
    """Конфигурация приложения в формате методички этапа 2."""

    PROJECT_NAME: str = "RUNG"
    DEBUG: bool = True
    CORS_ORIGINS: str = Field(default="", validation_alias="CORS_ORIGINS")

    @property
    def cors_origins_list(self) -> List[str]:
        """Преобразовать строку origins из `.env` в список URL."""

        if not self.CORS_ORIGINS:
            return []
        return [item.strip() for item in self.CORS_ORIGINS.split(",") if item.strip()]

    DATABASE_URL: str = Field(default="sqlite:///./rung.db", validation_alias="DATABASE_URL")

    @property
    def ASYNC_DATABASE_URL(self) -> str:
        """Вернуть URL базы данных с асинхронным драйвером SQLAlchemy."""

        url = self.DATABASE_URL
        if url.startswith("postgresql://"):
            return url.replace("postgresql://", "postgresql+asyncpg://", 1)
        if url.startswith("sqlite:///"):
            return url.replace("sqlite:///", "sqlite+aiosqlite:///", 1)
        return url

    TRANSLATION_ENGINE: str = Field(default="auto", validation_alias="TRANSLATION_ENGINE")
    FALLBACK_TO_HF_ON_ERROR: bool = Field(default=True, validation_alias="FALLBACK_TO_HF_ON_ERROR")
    AGENT_ENGINE: str = Field(default="ollama", validation_alias="AGENT_ENGINE")
    LLM_MODEL: str = Field(default="qwen2.5:7b", validation_alias="LLM_MODEL")
    CRITIC_MODEL: Optional[str] = Field(default=None, validation_alias="CRITIC_MODEL")
    EDITOR_MODEL: Optional[str] = Field(default=None, validation_alias="EDITOR_MODEL")
    ENSEMBLE_MODELS: str = Field(
        default="qwen2.5:7b,mistral:7b-instruct,llama3.1:8b",
        validation_alias="ENSEMBLE_MODELS",
    )

    @property
    def ensemble_models_list(self) -> List[str]:
        """Преобразовать строку ансамбля в список имён моделей."""

        if not self.ENSEMBLE_MODELS:
            return []
        return [item.strip() for item in self.ENSEMBLE_MODELS.split(",") if item.strip()]

    HF_TRANSLATION_MODEL: str = Field(
        default="facebook/nllb-200-distilled-600M", validation_alias="HF_TRANSLATION_MODEL"
    )
    HF_CRITIC_MODEL: Optional[str] = Field(default=None, validation_alias="HF_CRITIC_MODEL")
    HF_EDITOR_MODEL: Optional[str] = Field(default=None, validation_alias="HF_EDITOR_MODEL")
    HF_TOKEN: Optional[str] = Field(default=None, validation_alias="HF_TOKEN")

    OLLAMA_HOST: str = Field(default="localhost", validation_alias="OLLAMA_HOST")
    OLLAMA_PORT: int = Field(default=11434, validation_alias="OLLAMA_PORT")

    @property
    def ollama_url(self) -> str:
        """Вернуть URL endpoint Ollama для генерации."""

        return f"http://{self.OLLAMA_HOST}:{self.OLLAMA_PORT}/api/generate"

    QDRANT_HOST: str = Field(default="localhost", validation_alias="QDRANT_HOST")
    QDRANT_PORT: int = Field(default=6333, validation_alias="QDRANT_PORT")
    QDRANT_COLLECTION_NAME: str = Field(default="tm_vectors", validation_alias="QDRANT_COLLECTION_NAME")
    QDRANT_API_KEY: Optional[str] = Field(default=None, validation_alias="QDRANT_API_KEY")

    @property
    def qdrant_url(self) -> str:
        """Вернуть базовый URL Qdrant."""

        return f"http://{self.QDRANT_HOST}:{self.QDRANT_PORT}"

    REDIS_URL: str = Field(default="redis://localhost:6379/0", validation_alias="REDIS_URL")
    JWT_SECRET_KEY: str = Field(default=_DEFAULT_JWT_SECRET, validation_alias="JWT_SECRET_KEY")
    JWT_ALGORITHM: str = Field(default="HS256", validation_alias="JWT_ALGORITHM")
    JWT_EXPIRATION_MINUTES: int = Field(default=60 * 24 * 7, validation_alias="JWT_EXPIRATION_MINUTES")
    EVAL_MODE: str = Field(default="full", validation_alias="EVAL_MODE")
    ACTIVE_LEARNING_ENABLED: bool = Field(default=True, validation_alias="ACTIVE_LEARNING_ENABLED")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
        "extra": "ignore",
    }

    @field_validator("TRANSLATION_ENGINE", "AGENT_ENGINE")
    @classmethod
    def validate_engine(cls, value: str) -> str:
        """Проверить допустимый движок."""

        if value not in _ALLOWED_ENGINES:
            raise ValueError(f"engine must be one of {sorted(_ALLOWED_ENGINES)}, got '{value}'")
        return value

    @model_validator(mode="after")
    def validate_production_secrets(self) -> "Settings":
        """Запретить небезопасный JWT-секрет в production-режиме."""

        if not self.DEBUG:
            if self.JWT_SECRET_KEY == _DEFAULT_JWT_SECRET:
                raise ValueError("JWT_SECRET_KEY must be set to a non-default value when DEBUG=false")
            if len(self.JWT_SECRET_KEY) < 32:
                raise ValueError("JWT_SECRET_KEY must be at least 32 characters when DEBUG=false")
        return self


@lru_cache()
def get_settings() -> Settings:
    """Создать настройки один раз и переиспользовать их в приложении."""

    return Settings()


settings = get_settings()
