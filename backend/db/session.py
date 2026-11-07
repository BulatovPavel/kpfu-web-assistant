from pathlib import Path
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from backend.core.config import get_settings
from backend.db.models import Base

settings = get_settings()

async_engine = create_async_engine(
    settings.ASYNC_DATABASE_URL,
    echo=settings.DEBUG,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
)

AsyncSessionLocal = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that yields an async session.

    This is the single source of truth for async session management
    across the application. Routers and helpers import it from here
    (directly or via ``backend.api.deps``).
    """
    async with AsyncSessionLocal() as session:
        yield session


async def create_tables() -> None:
    """Создать все таблицы, описанные в ORM-моделях."""

    if settings.ASYNC_DATABASE_URL.startswith("sqlite"):
        database_path = settings.ASYNC_DATABASE_URL.split("///", maxsplit=1)[-1]
        if database_path != ":memory:":
            Path(database_path).parent.mkdir(parents=True, exist_ok=True)

    async with async_engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)


async def dispose_engine() -> None:
    """Освободить соединения с базой данных при остановке приложения."""

    await async_engine.dispose()

get_db = get_async_session