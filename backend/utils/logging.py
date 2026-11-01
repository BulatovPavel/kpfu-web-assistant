"""Настройка цветного консольного и ротируемого файлового логирования."""

from __future__ import annotations

import logging
import logging.config
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from backend.core.config import Settings, get_settings

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / f"rung_{datetime.now().strftime('%Y%m%d')}.log"
ERROR_FILE = LOG_DIR / f"errors_{datetime.now().strftime('%Y%m%d')}.log"

settings = get_settings()
DEFAULT_LEVEL = logging.DEBUG if settings.DEBUG else logging.INFO
_JSON_MODE = os.getenv("LOG_JSON", "false").lower() == "true"


class CustomFormatter(logging.Formatter):
    """Форматтер с опциональной цветной раскраской консоли."""

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    def __init__(
        self,
        fmt: str | None = None,
        datefmt: str | None = None,
        use_colors: bool = True,
    ) -> None:
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.use_colors = use_colors and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        """Добавить цвет к сообщению в зависимости от уровня лога."""

        if not self.use_colors:
            return super().format(record)

        levelno = record.levelno
        if levelno >= logging.CRITICAL:
            prefix = self.bold_red
        elif levelno >= logging.ERROR:
            prefix = self.red
        elif levelno >= logging.WARNING:
            prefix = self.yellow
        else:
            prefix = self.grey

        return f"{prefix}{super().format(record)}{self.reset}"


LOGGING_CONFIG: dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "()": CustomFormatter,
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "json": {
            "()": "pythonjsonlogger.json.JsonFormatter",
            "format": (
                "asctime: %(asctime)s "
                "name: %(name)s "
                "levelname: %(levelname)s "
                "message: %(message)s "
                "module: %(module)s "
                "funcName: %(funcName)s "
                "lineno: %(lineno)d "
                "process: %(process)d "
                "thread: %(thread)d"
            ),
            "datefmt": "%Y-%m-%dT%H:%M:%S%z",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "default",
            "stream": sys.stdout,
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "json" if _JSON_MODE else "default",
            "filename": str(LOG_FILE),
            "maxBytes": 10_485_760,
            "backupCount": 10,
            "encoding": "utf-8",
        },
        "error_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "ERROR",
            "formatter": "json" if _JSON_MODE else "default",
            "filename": str(ERROR_FILE),
            "maxBytes": 10_485_760,
            "backupCount": 5,
            "encoding": "utf-8",
        },
    },
    "loggers": {
        "sqlalchemy.engine": {"level": "WARNING", "propagate": False},
        "httpcore": {"level": "WARNING", "propagate": False},
        "httpx": {"level": "WARNING", "propagate": False},
        "urllib3": {"level": "WARNING", "propagate": False},
        "transformers": {"level": "WARNING", "propagate": False},
        "sentence_transformers": {"level": "WARNING", "propagate": False},
        "huggingface_hub": {"level": "WARNING", "propagate": False},
    },
    "root": {
        "level": "DEBUG",
        "handlers": ["console", "file", "error_file"],
    },
}


def setup_logging(level: int | str | None = None) -> logging.Logger:
    """Применить централизованную конфигурацию логирования."""

    logging_level = DEFAULT_LEVEL if level is None else level
    logging.config.dictConfig(LOGGING_CONFIG)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging_level)
    return root_logger


def configure_logging(app_settings: Settings) -> None:
    """Настроить логирование с учётом режима DEBUG приложения."""

    setup_logging(logging.DEBUG if app_settings.DEBUG else logging.INFO)


logger = setup_logging()

__all__ = ["configure_logging", "logger", "setup_logging"]
