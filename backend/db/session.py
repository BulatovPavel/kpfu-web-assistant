"""Асинхронное подключение к базе данных."""

from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Final

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from backend.core.config import settings
from backend.db.models import Base

DATABASE_ENGINE: Final[AsyncEngine] = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    pool_pre_ping=True,
)
SESSION_FACTORY: Final[async_sessionmaker[AsyncSession]] = async_sessionmaker(
    bind=DATABASE_ENGINE,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Предоставить асинхронную сессию базы данных для одного запроса."""

    async with SESSION_FACTORY() as session:
        yield session


async def create_tables() -> None:
    """Создать все таблицы, описанные в ORM-моделях."""

    if settings.database_url.startswith("sqlite"):
        database_path = settings.database_url.split("///", maxsplit=1)[-1]
        if database_path != ":memory:":
            Path(database_path).parent.mkdir(parents=True, exist_ok=True)

    async with DATABASE_ENGINE.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)


async def dispose_engine() -> None:
    """Освободить соединения с базой данных при остановке приложения."""

    await DATABASE_ENGINE.dispose()
