import json
import logging
import re
from typing import Any, Dict, List, Optional

import httpx

from backend.agents.base import Agent
from backend.core.constants import CRITIC_PASS_THRESHOLD
from backend.utils.hf_llm import get_hf_llm

logger = logging.getLogger(__name__)

_EXPECTED_KEYS = frozenset({
    "adequacy",
    "fluency",
    "terminology",
    "overall_score",
})

_UNIVERSAL_FEWSHOT = """Example:
{
  "adequacy": 8,
  "fluency": 8,
  "terminology": 8,
  "overall_score": 8,
  "reasoning": "The translation preserves the source meaning and reads naturally.",
  "suggestions": [],
  "missing_terms": []
}"""

_LANGUAGE_FEWSHOT = {
    "ru": (
        "For Russian, use natural Russian syntax and standard spelling; "
        "preserve names and established terminology."
    ),
}

_COMMON_RULES = """Score each dimension from 0 to 10:
- adequacy: preservation of the source meaning and important details;
- fluency: grammatical, idiomatic, and readable target-language text;
- terminology: accurate and consistent use of domain and glossary terms.
The overall_score is the overall translation quality, also from 0 to 10.
Do not reward fluent wording that changes or invents source facts."""

_DOMAIN_RULES = {
    "medical": (
        "For medical content, carefully check negation, symptoms, diagnoses, "
        "dosages, units, and instructions; do not infer clinical facts."
    ),
}


class Critic(Agent):
    hf_model_setting = "HF_CRITIC_MODEL"
    ollama_model_setting = "CRITIC_MODEL"

    def __init__(self, model_name: Optional[str] = None, engine: Optional[str] = None):
        super().__init__(name="Critic", engine=engine)

        if self.engine == "huggingface":
            self.model_name = model_name or self._hf_model_name()
            self.hf_llm = get_hf_llm(self.model_name)
        else:
            self.model_name = (
                model_name or self._ollama_model_name() or self.settings.LLM_MODEL
            ).replace("ollama/", "")
            self.hf_llm = None

        self.temperature = 0.0

    @staticmethod
    def _build_few_shot(target_lang: str, domain: str) -> str:
        parts = [_UNIVERSAL_FEWSHOT]
        language_example = _LANGUAGE_FEWSHOT.get(target_lang.casefold())
        if language_example:
            parts.append(language_example)
        return "\n\n".join(parts)

    @staticmethod
    def _build_scoring_rules(domain: str, audience: str) -> str:
        parts = [_COMMON_RULES]
        domain_rule = _DOMAIN_RULES.get(domain.casefold())
        if domain_rule:
            parts.append(domain_rule)
        if audience and audience.casefold() != "general":
            parts.append(
                f"Match the register and terminology to the intended audience: "
                f"{audience}."
            )
        return "\n\n".join(parts)

    @staticmethod
    def _format_glossary(glossary_entries: Any) -> str:
        if not isinstance(glossary_entries, list):
            return "None"

        formatted = []
        for entry in glossary_entries:
            if isinstance(entry, dict):
                source_term = entry.get("en", "")
                target_term = entry.get("tgt", "")
            else:
                source_term = getattr(entry, "en", "")
                target_term = getattr(entry, "tgt", "")
            if not isinstance(source_term, str) or not isinstance(target_term, str):
                continue
            source_term = source_term.strip()
            target_term = target_term.strip()
            if source_term and target_term:
                formatted.append(f"- {source_term} → {target_term}")
        return "\n".join(formatted) if formatted else "None"

    def _build_prompt(
        self,
        source: str,
        translation: str,
        source_lang: str,
        target_lang: str,
        glossary_entries: Any,
        reference: Any,
        domain: str,
        audience: str,
    ) -> str:
        reference_text = (
            reference
            if isinstance(reference, str)
            else json.dumps(reference, ensure_ascii=False, default=str)
            if reference is not None
            else "None"
        )
        scoring_rules = self._build_scoring_rules(domain, audience)
        few_shot = self._build_few_shot(target_lang, domain)
        glossary = self._format_glossary(glossary_entries)

        return (
            "You are a professional translation critic. Evaluate the translation "
            "against the source; do not rewrite it.\n"
            f"Source language: {source_lang}\n"
            f"Target language: {target_lang}\n"
            f"Domain: {domain}\n"
            f"Audience: {audience}\n\n"
            f"Scoring rules:\n{scoring_rules}\n\n"
            f"Source:\n{source}\n\n"
            f"Translation:\n{translation}\n\n"
            f"Required glossary:\n{glossary}\n\n"
            f"Reference translation:\n{reference_text}\n\n"
            f"Example evaluation:\n{few_shot}\n\n"
            "Return one JSON object with numeric adequacy, fluency, terminology, "
            "and overall_score fields from 0 to 10, plus reasoning, suggestions, "
            "and missing_terms. Do not include markdown fences or text outside "
            "the JSON object."
        )

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("[%s] Starting...", self.name)

        source_text = state.get("source_text", "")
        translation = state.get("translation", "")
        source_lang = state.get("source_lang", "en")
        target_lang = state.get("target_lang", "ru")
        glossary_entries = state.get("glossary_used", []) or []
        reference = state.get("reference")
        domain = state.get("domain", "general")
        audience = state.get("audience", "general")

        prompt = self._build_prompt(
            source=source_text, translation=translation,
            source_lang=source_lang, target_lang=target_lang,
            glossary_entries=glossary_entries, reference=reference,
            domain=domain, audience=audience,
        )

        failed = False
        try:
            response = await self._call_llm(prompt)
            evaluation = self._parse_json_response(response)
        except httpx.TimeoutException:
            logger.error("[%s] LLM timeout", self.name)
            evaluation = self._get_fallback_evaluation("Timeout error")
            failed = True
        except httpx.HTTPStatusError as e:
            logger.error("[%s] LLM HTTP error: %s", self.name, e.response.status_code)
            evaluation = self._get_fallback_evaluation(
                f"HTTP error: {e.response.status_code}"
            )
            failed = True
        except Exception as e:
            logger.error("[%s] Unexpected error: %s", self.name, e, exc_info=True)
            evaluation = self._get_fallback_evaluation("Unexpected error")
            failed = True

        if evaluation.get("failed"):
            failed = True

        suggestions = evaluation.get("suggestions") or []
        missing_terms = evaluation.get("missing_terms") or []
        overall_score = evaluation.get("overall_score")

        state["critic_adequacy"] = evaluation.get("adequacy")
        state["critic_fluency"] = evaluation.get("fluency")
        state["critic_terminology"] = evaluation.get("terminology")
        state["critic_score"] = overall_score
        state["critic_reasoning"] = evaluation.get("reasoning")
        state["critic_suggestions"] = suggestions
        state["critic_missing_terms"] = missing_terms
        state["critic_failed"] = failed
        state["critic_passed"] = (
            overall_score is not None
            and overall_score >= CRITIC_PASS_THRESHOLD
            and not failed
        )

        self._last_result = {
            "adequacy": evaluation.get("adequacy"),
            "fluency": evaluation.get("fluency"),
            "terminology": evaluation.get("terminology"),
            "overall_score": overall_score,
            "reasoning": evaluation.get("reasoning"),
            "suggestions": suggestions,
            "missing_terms": missing_terms,
            "failed": failed,
        }

        state.setdefault("reasoning_log", []).append(self._last_result | {
            "agent": self.name, "step": "critique",
        })

        return state

    def _get_fallback_evaluation(self, reason: str) -> Dict[str, Any]:
        return {
            "adequacy": None, "fluency": None, "terminology": None,
            "overall_score": None,
            "reasoning": f"Evaluation failed: {reason}",
            "suggestions": [], "missing_terms": [],
            "failed": True,
        }

    @staticmethod
    def _fallback_extract(response: str) -> dict[str, Any]:
        score_pattern = r"[-+]?\d+(?:[.,]\d+)?"

        def find_score(label_pattern: str) -> float | None:
            match = re.search(
                rf"^\s*(?:{label_pattern})\s*[:=]\s*({score_pattern})",
                response,
                re.IGNORECASE | re.MULTILINE,
            )
            if not match:
                return None
            return float(match.group(1).replace(",", "."))

        evaluation = {
            "adequacy": find_score("adequacy"),
            "fluency": find_score("fluency"),
            "terminology": find_score("terminology"),
            "overall_score": find_score(r"overall(?:[\s_-]+)?score|score"),
            "reasoning": response.strip() or None,
            "suggestions": [],
            "missing_terms": [],
            "failed": False,
        }
        if evaluation["overall_score"] is None:
            evaluation["failed"] = True
        return evaluation

    @staticmethod
    def _extract_json_spans(text: str) -> List[str]:
        spans: List[str] = []
        i = 0
        n = len(text)
        while i < n:
            if text[i] != "{":
                i += 1
                continue
            depth = 0
            in_string = False
            escaped = False
            j = i
            while j < n:
                ch = text[j]
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = not in_string
                elif not in_string:
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            spans.append(text[i:j + 1])
                            break
                j += 1
            i = j + 1
        return spans

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        candidates = self._extract_json_spans(response)
        if not candidates:
            logger.error("No JSON object found in response")
            return self._fallback_extract(response)

        parsed_candidates: List[Dict[str, Any]] = []
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed, dict):
                continue
            parsed.pop("failed", None)
            parsed_candidates.append(parsed)

        if not parsed_candidates:
            logger.error("All JSON candidates failed to parse")
            return self._fallback_extract(response)

        for parsed in parsed_candidates:
            if _EXPECTED_KEYS.issubset(parsed.keys()):
                return parsed

        logger.warning("No candidate had all expected keys; using first parsed object")
        return parsed_candidates[0]

    async def _call_llm(self, prompt: str) -> str:
        return '{"adequacy": 5.0, "fluency": 5.0, "terminology": 5.0, "overall_score": 5.0, "reasoning": "Stub LLM: real call in Этап 8.", "suggestions": [], "missing_terms": []}'