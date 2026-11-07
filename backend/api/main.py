import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.routes import health
from backend.api.routes import auth
from backend.core.config import get_settings
from backend.core.constants import LANGUAGES
from backend.core.exceptions import RUNGException
from backend.db.session import create_tables, dispose_engine
from backend.utils.logging import configure_logging

logger = logging.getLogger(__name__)
settings = get_settings()

_POST_CREATE_STATEMENTS = (
    "CREATE INDEX IF NOT EXISTS ix_cache_eval_mode "
    "ON translation_cache (eval_mode)",
)


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
    request: Request,
    exc: RUNGException,
) -> JSONResponse:
    """
    Render any RUNGException as a structured JSON response.

    Shape:
        {"detail": "...", "code": "...", "context": {...}}

    The status code is taken from the exception class, so subclasses
    (UncertainExampleNotFoundError, EngineFallbackTriggeredError, ...)
    keep their declared HTTP status.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_dict(),
    )


app.include_router(health.router)
app.include_router(auth.router)

@app.get("/")
async def root():
    return {"message": f"{settings.PROJECT_NAME} is running!"}


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/languages")
async def languages():
    return {"languages": LANGUAGES}