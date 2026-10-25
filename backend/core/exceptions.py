from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import status


class RUNGException(Exception):
    code: str = "rung_error"
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR

    def __init__(
        self,
        message: str,
        *,
        code: Optional[str] = None,
        status_code: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        if status_code is not None:
            self.status_code = status_code
        self.context: Dict[str, Any] = context or {}

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "detail": self.message,
            "code": self.code,
        }
        if self.context:
            payload["context"] = self.context
        return payload

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"code={self.code!r}, message={self.message!r}, "
            f"context={self.context!r})"
        )


class ConfigurationError(RUNGException):
    code = "configuration_error"
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR


class UnknownEvalModeError(ConfigurationError):
    code = "unknown_eval_mode"
    status_code = status.HTTP_400_BAD_REQUEST


class EngineError(RUNGException):
    code = "engine_error"
    status_code = status.HTTP_502_BAD_GATEWAY


class EngineUnavailableError(EngineError):
    code = "engine_unavailable"


class EngineFallbackTriggeredError(EngineError):
    code = "engine_fallback_triggered"
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE


class ModelNotLoadedError(EngineError):
    code = "model_not_loaded"


class LLMTimeoutError(EngineError):
    code = "llm_timeout"
    status_code = status.HTTP_504_GATEWAY_TIMEOUT


class LLMResponseError(EngineError):
    code = "llm_response_error"


class RetrievalError(RUNGException):
    code = "retrieval_error"
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE


class VectorStoreUnavailableError(RetrievalError):
    code = "vector_store_unavailable"


class EmbeddingError(RetrievalError):
    code = "embedding_error"


class BM25IndexNotInitializedError(RetrievalError):
    code = "bm25_not_initialized"


class KnowledgeBaseError(RUNGException):
    code = "knowledge_base_error"
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR


class EntryNotFoundError(KnowledgeBaseError):
    code = "entry_not_found"
    status_code = status.HTTP_404_NOT_FOUND


class DuplicateEntryError(KnowledgeBaseError):
    code = "duplicate_entry"
    status_code = status.HTTP_409_CONFLICT


class ImportFormatError(KnowledgeBaseError):
    code = "import_format_error"
    status_code = status.HTTP_400_BAD_REQUEST


class AgentError(RUNGException):
    code = "agent_error"
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR


class ValidationFailedError(AgentError):
    code = "validation_failed"
    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT


class CriticFailureError(AgentError):
    code = "critic_failure"


class EditorFailureError(AgentError):
    code = "editor_failure"


class HITLError(RUNGException):
    code = "hitl_error"
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR


class UncertainExampleNotFoundError(HITLError):
    code = "uncertain_example_not_found"
    status_code = status.HTTP_404_NOT_FOUND


class CorrectionConflictError(HITLError):
    code = "correction_conflict"
    status_code = status.HTTP_409_CONFLICT


class WorkerError(RUNGException):
    code = "worker_error"
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR


class IndexRebuildError(WorkerError):
    code = "index_rebuild_error"
