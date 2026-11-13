import json
import logging
import re
from typing import Any

from backend.agents.base import Agent
from backend.utils.hf_llm import get_hf_llm

logger = logging.getLogger(__name__)

_FENCED_RESPONSE_RE = re.compile(r"\A```(?:json)?\s*(.*?)\s*```\Z", re.DOTALL | re.IGNORECASE)
_ANSWER_PREFIX_RE = re.compile(r"\A(?:answer|ответ)\s*:\s*", re.IGNORECASE)


class Editor(Agent):
    hf_model_setting = "HF_EDITOR_MODEL"
    ollama_model_setting = "EDITOR_MODEL"

    def __init__(self, model_name: str | None = None, engine: str | None = None):
        super().__init__(name="Editor", engine=engine)
        if self.engine == "huggingface":
            self.model_name = model_name or self._hf_model_name()
            self.hf_llm = get_hf_llm(self.model_name)
        else:
            self.model_name = (
                model_name or self._ollama_model_name() or self.settings.LLM_MODEL
            ).replace("ollama/", "")
            self.hf_llm = None
        self.temperature = 0.3

    @staticmethod
    def _format_issues(
        errors: Any, warnings: Any, suggestions: Any
    ) -> str:
        sections = []
        for label, items in (
            ("Validation errors", errors),
            ("Validation warnings", warnings),
            ("Critic suggestions", suggestions),
        ):
            if not isinstance(items, list):
                items = [items] if items else []
            if not items:
                continue
            rendered = [
                item
                if isinstance(item, str)
                else json.dumps(item, ensure_ascii=False, default=str)
                for item in items
            ]
            sections.append(f"{label}:\n" + "\n".join(f"- {item}" for item in rendered))
        return "\n\n".join(sections)

    @staticmethod
    def _build_prompt(
        question: str,
        answer: str,
        retrieved_context: Any,
        citations: Any,
        errors: Any,
        warnings: Any,
        suggestions: Any,
    ) -> str:
        context = (
            json.dumps(retrieved_context, ensure_ascii=False, indent=2, default=str)
            if retrieved_context
            else "[]"
        )
        citation_data = (
            json.dumps(citations, ensure_ascii=False, indent=2, default=str)
            if citations
            else "[]"
        )
        issues = Editor._format_issues(errors, warnings, suggestions)
        if not issues:
            issues = "No specific issues were provided."

        return (
            "You edit answers for an assistant that answers questions about "
            "Kazan Federal University using retrieved website content.\n"
            "Revise the current answer to address the listed issues. Treat the "
            "retrieved context as the only factual source: do not add unsupported "
            "facts, URLs, or citations. If the context does not support a claim, "
            "remove it or clearly state that the available sources do not provide "
            "that information. Preserve valid citation markers and do not invent "
            "new ones. Return only the revised answer, without JSON or commentary.\n\n"
            f"User question:\n{question}\n\n"
            f"Retrieved context:\n{context}\n\n"
            f"Existing citations:\n{citation_data}\n\n"
            f"Current answer:\n{answer}\n\n"
            f"Issues to address:\n{issues}"
        )

    @staticmethod
    def _extract_answer(response: str) -> str:
        raw = response.strip()
        fenced_match = _FENCED_RESPONSE_RE.fullmatch(raw)
        if fenced_match:
            raw = fenced_match.group(1).strip()

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            for key in ("answer", "text"):
                value = parsed.get(key)
                if isinstance(value, str):
                    return value.strip()

        return _ANSWER_PREFIX_RE.sub("", raw).strip()

    async def process(self, state: dict[str, Any]) -> dict[str, Any]:
        logger.info("[%s] Starting...", self.name)

        if state.get("critic_failed"):
            logger.info("[%s] Critic failed, skipping edit.", self.name)
            state["editor_changed"] = False
            state["editor_failed"] = False
            state.setdefault("reasoning_log", []).append({
                "agent": self.name,
                "step": "editing",
                "changed": False,
                "note": "Skipped: critic_failed",
            })
            return state

        answer_value = state.get("answer", "")
        answer = answer_value if isinstance(answer_value, str) else ""
        errors = state.get("validator_errors", []) or []
        warnings = state.get("validator_warnings", []) or []
        suggestions = state.get("critic_suggestions", []) or []

        if not errors and not warnings and not suggestions:
            state["editor_changed"] = False
            state["editor_failed"] = False
            state.setdefault("reasoning_log", []).append({
                "agent": self.name,
                "step": "editing",
                "changed": False,
                "note": "No issues to fix",
            })
            return state

        question_value = state.get("question", "")
        question = question_value if isinstance(question_value, str) else ""
        prompt = self._build_prompt(
            question=question,
            answer=answer,
            retrieved_context=state.get("retrieved_context", []),
            citations=state.get("citations", []),
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
        )

        editor_failed = False
        try:
            raw_corrected = await self._call_llm(prompt)
            corrected = self._extract_answer(raw_corrected)
        except Exception:
            logger.exception("[%s] LLM call failed", self.name)
            corrected = answer
            editor_failed = True

        if not corrected.strip():
            logger.warning("[%s] Empty correction, keeping original.", self.name)
            corrected = answer
            editor_failed = True

        changed = corrected != answer
        state["editor_failed"] = editor_failed
        state["editor_changed"] = changed
        state["editor_original"] = answer
        state["editor_corrected"] = corrected
        state["editor_iterations"] = state.get("editor_iterations", 0) + 1

        if changed:
            state["answer"] = corrected

        self._last_result = {
            "changed": changed,
            "failed": editor_failed,
            "original": answer,
            "corrected": corrected,
        }
        state.setdefault("reasoning_log", []).append({
            "agent": self.name,
            "step": "editing",
            "original": answer,
            "corrected": corrected,
            "changed": changed,
            "failed": editor_failed,
        })
        return state

    # ВРЕМЕННО (убрать в Этапе 8): заглушка
    async def _call_llm(self, prompt: str) -> str:
        return "Заглушка: реальный вызов LLM в Этапе 8."
