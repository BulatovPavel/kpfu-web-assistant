import logging
import re
from collections.abc import Mapping
from itertools import pairwise
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from backend.agents.base import BaseAgent
from backend.data.embeddings import embedding_model

logger = logging.getLogger(__name__)

MIN_TERM_LEN = 2
MIN_TERM_LEN_FOR_SEMANTIC = 5
GLOSSARY_SEMANTIC_THRESHOLD = 0.7

_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")


def _normalise_text(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip().casefold()


def _normalise_url(url: str) -> str:
    parsed = urlsplit(url.strip())
    if not parsed.scheme or not parsed.netloc:
        return url.strip().rstrip("/").casefold()
    path = parsed.path.rstrip("/") or "/"
    return urlunsplit(
        (parsed.scheme.casefold(), parsed.netloc.casefold(), path, parsed.query, "")
    )


def _source_id(reference: Any) -> str | None:
    if isinstance(reference, str):
        value = reference.strip()
    elif isinstance(reference, Mapping):
        raw_value = reference.get("source_id", reference.get("id"))
        value = str(raw_value).strip() if raw_value is not None else ""
    else:
        value = ""
    return value or None


def _contains_word(text: str, term: str) -> bool:
    if not text or not term:
        return False
    pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"
    return re.search(pattern, text, re.IGNORECASE) is not None


def _glossary_pair(entry: Any) -> tuple[str, str]:
    if isinstance(entry, Mapping):
        source_term = entry.get("en") or ""
        target_term = entry.get("tgt") or ""
    else:
        source_term = getattr(entry, "en", "") or ""
        target_term = getattr(entry, "tgt", "") or ""
    if not isinstance(source_term, str) or not isinstance(target_term, str):
        return "", ""
    return source_term.strip(), target_term.strip()


def _semantic_match(
    term: str,
    translation: str,
    threshold: float = GLOSSARY_SEMANTIC_THRESHOLD,
) -> bool:
    if not term or not translation:
        return False
    try:
        vectors = embedding_model.encode([term, translation])
        term_vec, translation_vec = vectors[0], vectors[1]
        similarity = sum(
            float(term_value) * float(translation_value)
            for term_value, translation_value in zip(term_vec, translation_vec)
        )
        return similarity >= threshold
    except (TypeError, ValueError, IndexError, RuntimeError) as error:
        logger.warning("Semantic match failed for term '%s': %s", term, error)
        return False


class Validator(BaseAgent):
    """Rule-based validation of answer structure and references to retrieved context."""

    def __init__(self, engine: str | None = None):
        super().__init__(name="Validator")

    @staticmethod
    def _check_glossary(
        source_text: str,
        translation: str,
        glossary_entries: list[Any],
    ) -> list[str]:
        errors = []
        for entry in glossary_entries:
            source_term, target_term = _glossary_pair(entry)
            if (
                len(source_term) < MIN_TERM_LEN
                or len(target_term) < MIN_TERM_LEN
                or not _contains_word(source_text, source_term)
            ):
                continue
            if _contains_word(translation, target_term):
                continue
            if (
                len(target_term) >= MIN_TERM_LEN_FOR_SEMANTIC
                and _semantic_match(target_term, translation)
            ):
                continue
            errors.append(
                f"Glossary term '{source_term}' must be translated as "
                f"'{target_term}'."
            )
        return errors

    @staticmethod
    def _check_repetitions(answer: str) -> list[str]:
        words = _WORD_RE.findall(answer.casefold())
        repeated = sorted(
            {
                word
                for previous, word in pairwise(words)
                if previous == word
            }
        )
        if not repeated:
            return []
        return [f"Repeated adjacent word(s): {', '.join(repeated)}"]

    @staticmethod
    def _read_context(
        context: Any, errors: list[str]
    ) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
        context_by_id: dict[str, dict[str, str]] = {}
        context_by_url: dict[str, dict[str, str]] = {}

        if context is None:
            return context_by_id, context_by_url
        if not isinstance(context, list):
            errors.append("'retrieved_context' must be a list.")
            return context_by_id, context_by_url

        for index, item in enumerate(context):
            if not isinstance(item, Mapping):
                errors.append(f"retrieved_context[{index}] must be an object.")
                continue

            source_id = _source_id(item)
            text = item.get("text")
            url = item.get("url", "")
            if not source_id:
                errors.append(
                    f"retrieved_context[{index}] is missing 'source_id'."
                )
                continue
            if not isinstance(text, str) or not text.strip():
                errors.append(
                    f"retrieved_context[{index}] is missing non-empty 'text'."
                )
                continue
            if not isinstance(url, str):
                errors.append(f"retrieved_context[{index}] has an invalid 'url'.")
                continue
            if source_id in context_by_id:
                errors.append(f"Duplicate retrieved source_id: {source_id}.")
                continue

            source = {
                "source_id": source_id,
                "text": text,
                "url": url.strip(),
            }
            context_by_id[source_id] = source
            if source["url"]:
                context_by_url[_normalise_url(source["url"])] = source

        return context_by_id, context_by_url

    @staticmethod
    def _check_references(
        references: Any,
        field: str,
        context_by_id: dict[str, dict[str, str]],
        context_by_url: dict[str, dict[str, str]],
        errors: list[str],
    ) -> int:
        if references is None:
            return 0
        if not isinstance(references, list):
            errors.append(f"'{field}' must be a list.")
            return 0

        checked = 0
        for index, reference in enumerate(references):
            if isinstance(reference, str):
                source_id = reference.strip()
                url = ""
                quote = ""
            elif isinstance(reference, Mapping):
                source_id = _source_id(reference) or ""
                raw_url = reference.get("url", "")
                raw_quote = reference.get("quote", "")
                if not isinstance(raw_url, str):
                    errors.append(f"{field}[{index}] has an invalid 'url'.")
                    continue
                if not isinstance(raw_quote, str):
                    errors.append(f"{field}[{index}] has an invalid 'quote'.")
                    continue
                url = raw_url.strip()
                quote = raw_quote
            else:
                errors.append(f"{field}[{index}] must be a source ID or object.")
                continue

            source = context_by_id.get(source_id) if source_id else None
            if source is None and url:
                source = context_by_url.get(_normalise_url(url))
            if source is None:
                label = source_id or url or "<empty>"
                errors.append(
                    f"{field}[{index}] references a source absent from "
                    f"'retrieved_context': {label}."
                )
                continue

            if url and (
                not source["url"]
                or _normalise_url(url) != _normalise_url(source["url"])
            ):
                errors.append(
                    f"{field}[{index}] URL does not match retrieved source "
                    f"'{source['source_id']}'."
                )
                continue

            if quote and _normalise_text(quote) not in _normalise_text(source["text"]):
                errors.append(
                    f"{field}[{index}] quote is not present in retrieved source "
                    f"'{source['source_id']}'."
                )
                continue

            checked += 1

        return checked

    async def process(self, state: dict[str, Any]) -> dict[str, Any]:
        answer_value = state.get("answer", "")
        answer = answer_value if isinstance(answer_value, str) else ""
        errors: list[str] = []
        warnings: list[str] = []

        if not answer.strip():
            errors.append("Answer is empty.")

        context_by_id, context_by_url = self._read_context(
            state.get("retrieved_context"), errors
        )
        references_checked = self._check_references(
            state.get("citations"),
            "citations",
            context_by_id,
            context_by_url,
            errors,
        )
        references_checked += self._check_references(
            state.get("sources"),
            "sources",
            context_by_id,
            context_by_url,
            errors,
        )

        if answer.strip():
            warnings.extend(self._check_repetitions(answer))

        if not context_by_id:
            warnings.append(
                "No retrieved context; the answer could not be checked against sources."
            )
        elif references_checked == 0:
            warnings.append(
                "No valid citations or sources supplied; answer-to-context support "
                "was not checked."
            )

        state["validator_errors"] = errors
        state["validator_warnings"] = warnings
        state["validation_passed"] = not errors

        self._last_result = {
            "passed": state["validation_passed"],
            "errors": errors,
            "warnings": warnings,
        }

        reasoning_log = state.setdefault("reasoning_log", [])
        if not isinstance(reasoning_log, list):
            logger.warning("Replacing non-list reasoning_log in agent state.")
            reasoning_log = []
            state["reasoning_log"] = reasoning_log
        reasoning_log.append({
            "agent": self.name,
            "step": "validation",
            "errors": errors,
            "warnings": warnings,
            "passed": state["validation_passed"],
        })
        return state



# import json, logging, re
# from typing import Any, Dict, List, Optional, Set

# from backend.agents.base import BaseAgent
# from backend.data.embeddings import embedding_model

# logger = logging.getLogger(__name__)

# MIN_TERM_LEN = 2
# MIN_TERM_LEN_FOR_SEMANTIC = 5
# MIN_CONTENT_WORD_LEN = 4
# GLOSSARY_SEMANTIC_THRESHOLD = 0.7
# _MAX_LEFTOVER_REPORTED = 5
# _MIN_TARGET_SCRIPT_RATIO = 0.5

# _COMMON_STOPWORDS = set(
#     """
#     a about above after again against all am an and any are as at be because been
#     before being below between both but by can could did do does doing down during
#     each few for from further had has have having he her here hers herself him
#     himself his how i if in into is it its itself just me more most my myself no
#     nor not of off on once only or other our ours ourselves out over own same she
#     should so some such than that the their theirs them themselves then there
#     these they this those through to too under until up very was we were what when
#     where which while who whom why with would you your yours yourself yourselves
#     а без более бы был была были было быть в вам вас весь во вот все всего всех
#     вы где да даже для до его ее её если есть еще ещё же за здесь и из или им
#     ими их к как какая какие какой кем когда кого кому который которая которые
#     которое кто ли либо мне много может можно мы на над надо наш наша наши наше
#     не него нее неё нет ни них но о об один она они оно от по под при про с сам
#     сама сами себе себя свой свою свои свое своё со так такая такие такой там
#     те тем то того той только том ты у уже хотя чего чем что чтобы эта эти это
#     этот я
#     """.split()
# )

# _SCRIPT_PATTERNS = {
#     "cyrillic": re.compile(r"[\u0400-\u052F]"),
#     "arabic": re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]"),
#     "latin": re.compile(r"[A-Za-z\u00C0-\u024F\u1E00-\u1EFF]"),
# }
# _LANG_SCRIPT = {
#     "en": "latin",
#     "ru": "cyrillic",
#     "tg": "cyrillic",
#     "tt": "cyrillic",
#     "fa": "arabic",
#     "os": "cyrillic",
#     "ar": "arabic",
#     "av": "cyrillic",
#     "ba": "cyrillic",
#     "bg": "cyrillic",
#     "id": "latin",
#     "kk": "cyrillic",
#     "ky": "cyrillic",
#     "ms": "latin",
#     "tk": "latin",
#     "tr": "latin",
#     "uz": "latin",
#     "udm": "cyrillic",
# }
# _CJK_RE = re.compile(r"[\u3040-\u30FF\u3400-\u4DBF\u4E00-\u9FFF]")
# _PLACEHOLDER_RE = re.compile(r"%\d+\$s|%s")
# _LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)


# def _contains_word(text: str, term: str) -> bool:
#     """Whole-word, case-insensitive match of ``term`` inside ``text``."""
#     if not term or not text:
#         return False
#     pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"
#     return re.search(pattern, text, re.IGNORECASE) is not None


# def _extract_content_words(text: str) -> Set[str]:
#     if not text:
#         return set()
#     words = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
#     return {
#         w for w in words
#         if len(w) >= MIN_CONTENT_WORD_LEN and w not in _COMMON_STOPWORDS
#     }


# def _glossary_pair(entry: Any) -> tuple[str, str]:
#     if isinstance(entry, dict):
#         source_term = entry.get("en") or ""
#         target_term = entry.get("tgt") or ""
#     else:
#         source_term = getattr(entry, "en", "") or ""
#         target_term = getattr(entry, "tgt", "") or ""
#     return source_term.strip(), target_term.strip()


# def _semantic_match(term: str, translation: str, threshold: float = GLOSSARY_SEMANTIC_THRESHOLD) -> bool:
#     if not term or not translation:
#         return False
#     try:
#         vectors = embedding_model.encode([term, translation])
#         term_vec, trans_vec = vectors[0], vectors[1]
#         cos = sum(a * b for a, b in zip(term_vec, trans_vec))
#         return cos >= threshold
#     except Exception as e:
#         logger.warning("Semantic match failed for term '%s': %s", term, e)
#         return False


# class Validator(BaseAgent):
#     """
#     Rule-based validator. Does NOT call an LLM.
#     """

#     def __init__(self, engine: Optional[str] = None):
#         super().__init__(name="Validator")

#     async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
#         logger.info("[%s] Starting...", self.name)

#         source_text = state.get("source_text", "") or ""
#         translation = state.get("translation", "") or ""
#         target_lang = state.get("target_lang", "") or ""
#         glossary_entries = state.get("glossary_used", []) or []

#         errors: List[str] = []
#         warnings: List[str] = []

#         errors.extend(self._check_glossary(source_text, translation, glossary_entries))
#         if not translation.strip():
#             errors.append("Translation is empty")
#         errors.extend(self._check_echo(source_text, translation))
#         errors.extend(self._check_placeholders(source_text, translation))
#         errors.extend(self._check_script(target_lang, translation))

#         warnings.extend(self._check_repetitions(translation))
#         warnings.extend(self._check_leftover_source(source_text, translation))

#         state["validator_errors"] = errors
#         state["validator_warnings"] = warnings
#         state["validation_passed"] = len(errors) == 0

#         self._last_result = {
#             "passed": state["validation_passed"],
#             "errors": errors,
#             "warnings": warnings,
#         }

#         reasoning_entry = {
#             "agent": self.name,
#             "step": "validation",
#             "errors": errors,
#             "warnings": warnings,
#             "passed": state["validation_passed"],
#         }
#         state.setdefault("reasoning_log", []).append(reasoning_entry)
#         return state

#     def _check_glossary(
#         self,
#         source_text: str,
#         translation: str,
#         glossary_entries: List[Any],
#     ) -> List[str]:
#         errors = []
#         for entry in glossary_entries:
#             if isinstance(entry, str):
#                 try:
#                     entry = json.loads(entry)
#                 except json.JSONDecodeError:
#                     logger.warning("Ignoring malformed glossary JSON entry.")
#                     continue
#             source_term, target_term = _glossary_pair(entry)
#             if (
#                 len(source_term) < MIN_TERM_LEN
#                 or len(target_term) < MIN_TERM_LEN
#                 or not _contains_word(source_text, source_term)
#             ):
#                 continue
#             if _contains_word(translation, target_term):
#                 continue
#             if (
#                 len(target_term) >= MIN_TERM_LEN_FOR_SEMANTIC
#                 and _semantic_match(target_term, translation)
#             ):
#                 continue
#             errors.append(
#                 f"Glossary term '{source_term}' must be translated as "
#                 f"'{target_term}'."
#             )
#         return errors

#     @staticmethod
#     def _check_echo(source_text: str, translation: str) -> List[str]:
#         normalized_source = re.sub(r"[\W_]+", "", source_text.casefold())
#         normalized_translation = re.sub(r"[\W_]+", "", translation.casefold())
#         if normalized_source and normalized_source == normalized_translation:
#             return ["Translation is identical to the source text."]
#         return []

#     @staticmethod
#     def _check_placeholders(source_text: str, translation: str) -> List[str]:
#         source_placeholders = sorted(_PLACEHOLDER_RE.findall(source_text))
#         translated_placeholders = sorted(_PLACEHOLDER_RE.findall(translation))
#         if source_placeholders != translated_placeholders:
#             return [
#                 "Placeholders differ between source and translation "
#                 f"(source={source_placeholders}, "
#                 f"translation={translated_placeholders})."
#             ]
#         return []

#     @staticmethod
#     def _check_script(target_lang: str, translation: str) -> List[str]:
#         script = _LANG_SCRIPT.get(target_lang.lower())
#         if not script:
#             return [
#                 f"Unsupported target language for script validation: '{target_lang}'."
#             ]

#         if script != "cjk" and _CJK_RE.search(translation):
#             return [
#                 "Translation contains CJK characters for a non-CJK target language."
#             ]

#         letters = _LETTER_RE.findall(translation)
#         if not letters:
#             return ["Translation contains no letters from the target script."]

#         target_letters = sum(
#             bool(_SCRIPT_PATTERNS[script].fullmatch(char)) for char in letters
#         )
#         ratio = target_letters / len(letters)
#         if ratio < _MIN_TARGET_SCRIPT_RATIO:
#             return [
#                 f"Target-script letter ratio is {ratio:.0%}; "
#                 f"at least {_MIN_TARGET_SCRIPT_RATIO:.0%} is required "
#                 f"for {target_lang}."
#             ]

#         return []

#     @staticmethod
#     def _check_repetitions(translation: str) -> List[str]:
#         words = re.findall(r"\w+", translation.casefold(), flags=re.UNICODE)
#         repeated = sorted({
#             word
#             for previous, word in zip(words, words[1:])
#             if previous == word
#         })
#         if not repeated:
#             return []
#         return [f"Repeated consecutive word(s): {', '.join(repeated)}."]

#     @staticmethod
#     def _check_leftover_source(source_text: str, translation: str) -> List[str]:
#         source_words = _extract_content_words(source_text)
#         translation_words = _extract_content_words(translation)
#         leftovers = sorted(source_words & translation_words)
#         if not leftovers:
#             return []
#         reported = leftovers[:_MAX_LEFTOVER_REPORTED]
#         suffix = "..." if len(leftovers) > _MAX_LEFTOVER_REPORTED else ""
#         return [
#             "Possible untranslated source words in translation: "
#             f"{', '.join(reported)}{suffix}."
#         ]
