"""Настройка логирования в консоль и в ротируемый файл."""

import logging
from logging.handlers import RotatingFileHandler

from backend.core.config import Settings
from backend.core.constants import DEFAULT_LOG_FORMAT


def configure_logging(app_settings: Settings) -> None:
    """Один раз настроить логирование в консоль и файл."""

    log_path = app_settings.log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(DEFAULT_LOG_FORMAT)
    root_logger = logging.getLogger()
    root_logger.setLevel(app_settings.log_level.upper())

    if not root_logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)

        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)
        return

    for handler in root_logger.handlers:
        handler.setFormatter(formatter)
