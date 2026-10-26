"""
Unit tests for backend.core.config.

Every test creates Settings with _env_file=None to isolate from the
developer's .env file. Without this, a leftover PROJECT_NAME=RUNG-Test
from a manual experiment would break the tests.
"""
# TODO: Переписать закомменченные тесты под поля Settings

import pytest

from backend.core.config import Settings, get_settings


def make_settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


class TestSettingsDefaults:
    def test_project_name_default(self):
        s = make_settings()
        assert s.app_name == "KPFU Web Assistant"

    def test_debug_default(self):
        s = make_settings()
        assert isinstance(s.debug, bool)

    # def test_translation_engine_default(self):
    #     s = make_settings()
    #     assert s.TRANSLATION_ENGINE in {"auto", "ollama", "huggingface"}

    def test_agent_engine_default(self):
        s = make_settings()
        assert s.llm_provider == "ollama"

    # def test_eval_mode_is_plain_string(self):
    #     """EVAL_MODE is not validated at Settings level.

    #     The 8 ablation modes are validated at pipeline runtime with
    #     UnknownEvalModeError, so any string passes through here.
    #     """
    #     s = make_settings(EVAL_MODE="some_future_mode")
    #     assert s.EVAL_MODE == "some_future_mode"


# class TestAsyncDatabaseUrl:
#     def test_sqlite_to_aiosqlite(self):
#         s = make_settings(DATABASE_URL="sqlite:///./test.db")
#         assert s.ASYNC_DATABASE_URL == "sqlite+aiosqlite:///./test.db"

#     def test_postgresql_to_asyncpg(self):
#         s = make_settings(DATABASE_URL="postgresql://user:pass@host:5432/db")
#         assert s.ASYNC_DATABASE_URL == (
#             "postgresql+asyncpg://user:pass@host:5432/db"
#         )

#     def test_already_async_unchanged(self):
#         s = make_settings(DATABASE_URL="postgresql+asyncpg://u:p@h:5432/db")
#         assert s.ASYNC_DATABASE_URL == "postgresql+asyncpg://u:p@h:5432/db"


# class TestUrls:
#     def test_ollama_url(self):
#         s = make_settings(OLLAMA_HOST="myhost", OLLAMA_PORT=9999)
#         assert s.ollama_url == "http://myhost:9999/api/generate"

#     def test_qdrant_url(self):
#         s = make_settings(QDRANT_HOST="qdrant.local", QDRANT_PORT=6334)
#         assert s.qdrant_url == "http://qdrant.local:6334"


# class TestEnsembleModels:
#     def test_three_models(self):
#         s = make_settings(ENSEMBLE_MODELS="a,b,c")
#         assert s.ensemble_models_list == ["a", "b", "c"]

#     def test_with_spaces(self):
#         s = make_settings(ENSEMBLE_MODELS="a , b , c")
#         assert s.ensemble_models_list == ["a", "b", "c"]

#     def test_empty(self):
#         s = make_settings(ENSEMBLE_MODELS="")
#         assert s.ensemble_models_list == []

#     def test_default_includes_mistral(self):
#         s = make_settings()
#         assert "mistral:7b-instruct" in s.ensemble_models_list


# class TestCorsOrigins:
#     def test_empty_returns_empty_list(self):
#         s = make_settings(CORS_ORIGINS="")
#         assert s.cors_origins_list == []

#     def test_single_origin(self):
#         s = make_settings(CORS_ORIGINS="https://a.com")
#         assert s.cors_origins_list == ["https://a.com"]

#     def test_multiple_with_spaces(self):
#         s = make_settings(CORS_ORIGINS="https://a.com , https://b.com")
#         assert s.cors_origins_list == ["https://a.com", "https://b.com"]

#     def test_ignores_empty_entries(self):
#         s = make_settings(CORS_ORIGINS="https://a.com,,https://b.com,")
#         assert s.cors_origins_list == ["https://a.com", "https://b.com"]


class Test_llm_provider:
    # def test_invalid_translation_engine_raises(self):
    #     with pytest.raises(ValueError, match="engine must be one of"):
    #         make_settings(llm_provider="google")

    # def test_invalid_agent_engine_raises(self):
    #     with pytest.raises(ValueError, match="engine must be one of"):
    #         make_settings(AGENT_ENGINE="google")

    def test_valid_translation_engine(self):
        s = make_settings(llm_provider="huggingface")
        assert s.llm_provider == "huggingface"

    # def test_valid_agent_engine(self):
    #     s = make_settings(AGENT_ENGINE="auto")
    #     assert s.AGENT_ENGINE == "auto"


class TestProductionSecrets:
    def test_debug_true_allows_default_secret(self):
        s = make_settings(debug=True)
        assert s.jwt_secret_key == "change-me-in-local-env"

    def test_debug_false_rejects_default_secret(self):
        with pytest.raises(ValueError, match="non-default value"):
            make_settings(debug=False)

    def test_debug_false_rejects_short_secret(self):
        with pytest.raises(ValueError, match="at least 32"):
            make_settings(debug=False, jwt_secret_key="short")

    def test_debug_false_accepts_long_secret(self):
        s = make_settings(debug=False, jwt_secret_key="x" * 64)
        assert s.debug is False


class TestGetSettingsSingleton:
    def test_returns_same_object(self):
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_returns_settings_instance(self):
        assert isinstance(get_settings(), Settings)
