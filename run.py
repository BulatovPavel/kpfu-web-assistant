"""Запуск приложения в режиме разработки."""

import uvicorn

from backend.core.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "backend.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )
