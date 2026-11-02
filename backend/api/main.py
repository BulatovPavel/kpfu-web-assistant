from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text

from backend.api.routes import health
from backend.core.config import get_settings
from backend.core.constants import LANGUAGES
from backend.core.exceptions import RUNGException
from backend.db.models import Base
from backend.db.session import async_engine

settings = get_settings()

# Idempotent index migrations for deployments on an existing volume.
#
# ``Base.metadata.create_all`` only creates tables and indexes that do
# not yet exist. When a new Index is added to a model, existing
# databases never receive it — the volume survives the container
# restart, but its schema is frozen at the moment of the first start.
#
# Each statement here is a no-op on a fresh database (create_all
# already created the index) and a safe migration on an existing one.
# ``IF NOT EXISTS`` is understood by both Postgres and SQLite, so the
# same code path works in production and in tests.
_POST_CREATE_STATEMENTS = (
    # translation_cache.eval_mode is filtered on by get_cached() and
    # clear_cache(), but the UniqueConstraint places it in the middle
    # of a 6-column key, so a prefix scan cannot use it.
    "CREATE INDEX IF NOT EXISTS ix_cache_eval_mode "
    "ON translation_cache (eval_mode)",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        for statement in _POST_CREATE_STATEMENTS:
            await conn.execute(text(statement))
    try:
        yield
    finally:
        # Этап 8: сюда добавится await close_ollama_clients()
        pass


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


@app.get("/")
async def root():
    return {"message": f"{settings.PROJECT_NAME} is running!"}


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/languages")
async def languages():
    return {"languages": LANGUAGES}