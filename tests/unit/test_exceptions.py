"""Unit tests for backend.core.exceptions."""

from fastapi import status

from backend.core.exceptions import (
    ConfigurationError,
    EngineFallbackTriggeredError,
    EngineUnavailableError,
    RUNGException,
    UncertainExampleNotFoundError,
    UnknownEvalModeError,
)


class TestRUNGException:
    def test_default_code_and_status(self):
        e = RUNGException("boom")
        assert e.code == "rung_error"
        assert e.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert e.context == {}

    def test_to_dict_without_context(self):
        e = RUNGException("boom")
        assert e.to_dict() == {"detail": "boom", "code": "rung_error"}

    def test_to_dict_with_context(self):
        e = RUNGException("boom", context={"x": 1})
        assert e.to_dict() == {
            "detail": "boom",
            "code": "rung_error",
            "context": {"x": 1},
        }

    def test_override_code(self):
        e = RUNGException("boom", code="custom")
        assert e.code == "custom"

    def test_override_status_code(self):
        e = RUNGException("boom", status_code=418)
        assert e.status_code == 418


class TestSubclasses:
    def test_engine_unavailable_is_502(self):
        e = EngineUnavailableError("down")
        assert e.status_code == status.HTTP_502_BAD_GATEWAY
        assert e.code == "engine_unavailable"

    def test_engine_fallback_is_503(self):
        e = EngineFallbackTriggeredError("switched")
        assert e.status_code == status.HTTP_503_SERVICE_UNAVAILABLE

    def test_unknown_eval_mode_is_400(self):
        e = UnknownEvalModeError("bad mode")
        assert e.status_code == status.HTTP_400_BAD_REQUEST
        assert isinstance(e, ConfigurationError)

    def test_uncertain_not_found_is_404(self):
        e = UncertainExampleNotFoundError("missing")
        assert e.status_code == status.HTTP_404_NOT_FOUND

    def test_all_inherit_from_rung(self):
        for cls in [
            EngineUnavailableError,
            UnknownEvalModeError,
            UncertainExampleNotFoundError,
        ]:
            assert issubclass(cls, RUNGException)