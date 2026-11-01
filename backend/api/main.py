"""Точка входа FastAPI-приложения."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.core.config import get_settings
from backend.core.constants import LANGUAGES
from backend.core.exceptions import RUNGException
from backend.db.session import create_tables, dispose_engine
from backend.utils.logging import configure_logging

logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Настроить сервисы приложения при запуске и завершении работы."""

    configure_logging(settings)
    await create_tables()
    logger.info("Application started: %s", settings.PROJECT_NAME)
    try:
        yield
    finally:
        await dispose_engine()
        logger.info("Application stopped")


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="RAG-ассистент по официальным материалам КФУ",
    debug=settings.DEBUG,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.DEBUG else settings.cors_origins_list,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RUNGException)
async def rung_exception_handler(
    _: Request,
    exception: RUNGException,
) -> JSONResponse:
    """Преобразовать доменное исключение в структурированный HTTP-ответ."""

    return JSONResponse(
        status_code=exception.status_code,
        content=exception.to_dict(),
    )

@app.get("/")
async def root():
    return {"message": f"{settings.PROJECT_NAME} is running!"}

@app.get("/healthz", tags=["system"])
async def healthcheck() -> dict[str, str]:
    """Вернуть состояние сервиса для проверки доступности приложения."""

    return {"status": "ok"}

@app.get("/languages")
async def languages():
    return {"languages": LANGUAGES}
