"""Специализированные исключения приложения."""


class KPFUAssistantError(Exception):
    """Базовое исключение для ожидаемых ошибок приложения."""


class ConfigurationError(KPFUAssistantError):
    """Возникает при некорректной конфигурации приложения."""


class LLMProviderError(KPFUAssistantError):
    """Возникает, если LLM-провайдер не может обработать запрос."""


class RetrievalError(KPFUAssistantError):
    """Возникает при ошибке поиска в базе знаний."""
