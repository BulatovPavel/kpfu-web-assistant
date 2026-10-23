"""ORM-модели для хранения данных RAG-ассистента."""

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Float, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Базовый класс всех SQLAlchemy-моделей приложения."""


class User(Base):
    """Пользователь приложения и его роль доступа."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    username: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(30), default="user")
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class KnowledgeBaseEntry(Base):
    """Эталонная запись базы знаний или фрагмент источника."""

    __tablename__ = "knowledge_base"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_text: Mapped[str] = mapped_column(Text)
    target_text: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(100), index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class GlossaryEntry(Base):
    """Термин и его определение для нормализации запросов."""

    __tablename__ = "glossary"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    term: Mapped[str] = mapped_column(String(255), index=True)
    definition_translation: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(100), index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class UncertainExample(Base):
    """Сомнительный пример для последующей проверки человеком."""

    __tablename__ = "uncertain_examples"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    input_text: Mapped[str] = mapped_column(Text)
    model_output: Mapped[str] = mapped_column(Text)
    critic_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(30), default="pending", index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class CacheEntry(Base):
    """Кэшированный результат обработки пользовательского запроса."""

    __tablename__ = "cache"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    input_key: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    output: Mapped[str] = mapped_column(Text)
    hits: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class EvaluationResult(Base):
    """Результат оценки качества модели или режима работы."""

    __tablename__ = "evaluation_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_name: Mapped[str] = mapped_column(String(100), index=True)
    mode: Mapped[str] = mapped_column(String(50), index=True)
    sample_size: Mapped[int] = mapped_column(Integer)
    metrics_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    evaluation_time: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class ModelComparison(Base):
    """Сравнение нескольких моделей или режимов генерации."""

    __tablename__ = "model_comparisons"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    models_compared: Mapped[list[str]] = mapped_column(JSON)
    sample_size: Mapped[int] = mapped_column(Integer)
    results_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    winner: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class ReasoningLog(Base):
    """Краткая трассировка этапа работы агента или супервизора."""

    __tablename__ = "reasoning_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    request_id: Mapped[str] = mapped_column(String(100), index=True)
    agent_name: Mapped[str] = mapped_column(String(100), index=True)
    input_summary: Mapped[str] = mapped_column(Text)
    output_summary: Mapped[str] = mapped_column(Text)
    model_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    parameters: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    status: Mapped[str] = mapped_column(String(30), index=True)


class LLMCall(Base):
    """Метаданные отдельного вызова LLM без хранения секретных промптов."""

    __tablename__ = "llm_calls"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    request_id: Mapped[str] = mapped_column(String(100), index=True)
    agent_name: Mapped[str] = mapped_column(String(100), index=True)
    model_name: Mapped[str] = mapped_column(String(100))
    provider: Mapped[str] = mapped_column(String(50))
    prompt_hash: Mapped[str] = mapped_column(String(128))
    response_hash: Mapped[str] = mapped_column(String(128))
    tokens_used: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(30), index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
