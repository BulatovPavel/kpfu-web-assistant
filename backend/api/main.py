"""Точка входа FastAPI-приложения."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.core.config import settings
from backend.core.constants import APP_VERSION, HEALTHCHECK_PATH, LANGUAGES
from backend.db.session import create_tables, dispose_engine
from backend.utils.logging import configure_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Настроить сервисы приложения при запуске и завершении работы."""

    configure_logging(settings)
    await create_tables()
    logger.info("Application started: %s", settings.app_name)
    try:
        yield
    finally:
        await dispose_engine()
        logger.info("Application stopped")


app = FastAPI(
    title=settings.app_name,
    version=APP_VERSION,
    description="RAG-ассистент по официальным материалам КФУ",
    debug=settings.debug,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": f"{settings.app_name} is running!"}

@app.get(HEALTHCHECK_PATH, tags=["system"])
async def healthcheck() -> dict[str, str]:
    """Вернуть состояние сервиса для проверки доступности приложения."""

    return {"status": "ok", "service": settings.app_name, "version": APP_VERSION}

@app.get("/languages")
async def languages():
    return {"languages": LANGUAGES}
