"""Общие константы приложения."""

from typing import Dict, List


APP_VERSION = "0.1.0"
HEALTHCHECK_PATH = "/healthz"
DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DEFAULT_CORS_ORIGINS = ["http://localhost:3000", "http://localhost:8501"]

# All languages accepted by the API. Validated in backend.core.models.
LANGUAGES: Dict[str, str] = {
    "en": "English",
    "ru": "Russian",
    "tg": "Tajik",
    "tt": "Tatar",
    "fa": "Persian",
    "os": "Ossetian",
    "ar": "Arabic",
    "av": "Avar",
    "ba": "Bashkir",
    "bg": "Bulgarian",
    "id": "Indonesian",
    "kk": "Kazakh",
    "ky": "Kyrgyz",
    "ms": "Malay",
    "tk": "Turkmen",
    "tr": "Turkish",
    "uz": "Uzbek",
    "udm": "Udmurt",
}

# Languages for which the default Ollama model is reliable. Requests
# targeting a language outside this set are routed to HuggingFace.
KNOWN_LANGUAGES: List[str] = [
    "en", "ru", "fa",
    "ar", "bg", "id", "kk", "ky", "ms", "tr", "uz",
    "tt", "ba", "tg",
]

# ---------------------------------------------------------------------------
# Quality thresholds (Critic / Supervisor)
# ---------------------------------------------------------------------------

# A translation is accepted when overall_score >= threshold.
CRITIC_PASS_THRESHOLD: float = 7.0

# Adaptive iteration budget for the Editor stage in Supervisor:
#   score is None or score < ADAPTIVE_BUDGET_LOW   -> up to 3 iterations
#   score < ADAPTIVE_BUDGET_MID                    -> up to 2 iterations
#   score >= ADAPTIVE_BUDGET_MID, but validation
#     failed or the critic produced actionable
#     suggestions                                  -> up to 1 iteration
#   score >= ADAPTIVE_BUDGET_MID and no outstanding
#     issues                                       -> 0 iterations
ADAPTIVE_BUDGET_LOW: float = 4.0
ADAPTIVE_BUDGET_MID: float = 7.0