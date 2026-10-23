"""Проверки создания схемы базы данных."""

from pathlib import Path

import pytest
from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import create_async_engine

from backend.db.models import Base

REQUIRED_TABLES = {
    "users",
    "knowledge_base",
    "glossary",
    "uncertain_examples",
    "cache",
    "evaluation_results",
    "model_comparisons",
    "reasoning_logs",
    "llm_calls",
}


@pytest.mark.asyncio
async def test_all_required_tables_are_created(tmp_path: Path) -> None:
    """Проверить создание всех таблиц схемы на SQLite."""

    database_path = tmp_path / "test.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{database_path}")

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
        table_names = await connection.run_sync(
            lambda sync_connection: set(inspect(sync_connection).get_table_names())
        )

    await engine.dispose()

    assert REQUIRED_TABLES <= table_names
