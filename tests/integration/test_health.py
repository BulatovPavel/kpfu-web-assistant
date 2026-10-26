"""
Integration tests for healthcheck and info endpoints.

Uses TestClient (sync wrapper over httpx) to hit the FastAPI app
directly, without network and without uvicorn.
"""

from fastapi.testclient import TestClient

from backend.api.main import app
from backend.core.constants import APP_VERSION
from backend.core.config import settings


client = TestClient(app)


class TestRoot:
    def test_root_returns_200(self):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_contains_message(self):
        response = client.get("/")
        data = response.json()
        assert "message" in data
        assert "is running!" in data["message"]


class TestHealthz:
    def test_healthz_returns_200(self):
        response = client.get("/healthz")
        assert response.status_code == 200

    def test_healthz_returns_ok_status(self):
        response = client.get("/healthz")
        assert response.json() == {"status": "ok", "service": settings.app_name, "version": APP_VERSION}


class TestLanguages:
    def test_languages_returns_200(self):
        response = client.get("/languages")
        assert response.status_code == 200

    def test_languages_contains_english(self):
        response = client.get("/languages")
        data = response.json()
        assert "languages" in data
        assert data["languages"]["en"] == "English"

    def test_languages_has_all_18_codes(self):
        response = client.get("/languages")
        data = response.json()
        # 18 supported languages; a missing one breaks the UI dropdown
        assert len(data["languages"]) == 18

    def test_languages_does_not_expose_known(self):
        """KNOWN_LANGUAGES is exposed via /health, not /languages.

        The public dropdown shows all 18 languages; the "Ollama-ready"
        subset is an internal detail rendered only on the admin page.
        """
        response = client.get("/languages")
        data = response.json()
        assert "known_languages" not in data