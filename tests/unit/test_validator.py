import pytest

import backend.agents.validator as validator_module
from backend.agents.base import BaseAgent
from backend.agents.validator import Validator


@pytest.mark.asyncio
async def test_validator_is_rule_based_and_accepts_engine_argument():
    validator = Validator(engine="huggingface")

    assert isinstance(validator, BaseAgent)
    result = await validator.process({"answer": "Добро пожаловать"})

    assert result["validation_passed"] is True
    assert result["validator_errors"] == []
    assert result["validator_warnings"] == [
        "No retrieved context; the answer could not be checked against sources."
    ]


@pytest.mark.asyncio
async def test_empty_answer_fails_validation():
    state = {"answer": "  ", "retrieved_context": []}

    result = await Validator().process(state)

    assert result["validation_passed"] is False
    assert "Answer is empty." in result["validator_errors"]


@pytest.mark.asyncio
async def test_citation_and_quote_must_match_retrieved_context():
    state = {
        "answer": "Университет основан в 1804 году.",
        "retrieved_context": [{
            "source_id": "history",
            "url": "https://kpfu.ru/about/history",
            "text": "Казанский университет основан в 1804 году.",
        }],
        "citations": [{
            "source_id": "history",
            "url": "https://kpfu.ru/about/history/",
            "quote": "Казанский университет основан в 1804 году.",
        }],
        "sources": [{"source_id": "history"}],
    }

    result = await Validator().process(state)

    assert result["validation_passed"] is True
    assert result["validator_errors"] == []


@pytest.mark.asyncio
async def test_unknown_citation_source_fails_validation():
    state = {
        "answer": "Ответ.",
        "retrieved_context": [{
            "source_id": "history",
            "text": "История университета.",
        }],
        "citations": [{"source_id": "unrelated"}],
    }

    result = await Validator().process(state)

    assert result["validation_passed"] is False
    assert any(
        "absent from 'retrieved_context'" in error
        for error in result["validator_errors"]
    )


@pytest.mark.asyncio
async def test_quote_not_present_in_source_fails_validation():
    state = {
        "answer": "Ответ.",
        "retrieved_context": [{
            "source_id": "history",
            "text": "История университета.",
        }],
        "citations": [{
            "source_id": "history",
            "quote": "Факультет открылся в 2020 году.",
        }],
    }

    result = await Validator().process(state)

    assert result["validation_passed"] is False
    assert any(
        "quote is not present" in error for error in result["validator_errors"]
    )


@pytest.mark.asyncio
async def test_adjacent_repeated_words_are_a_warning_not_an_error():
    state = {
        "answer": "Это очень очень важная информация.",
        "retrieved_context": [{
            "source_id": "doc-1",
            "text": "Важная информация.",
        }],
        "citations": ["doc-1"],
    }

    result = await Validator().process(state)

    assert result["validation_passed"] is True
    assert result["validator_errors"] == []
    assert any("очень" in warning for warning in result["validator_warnings"])


def test_check_repetitions_warns_for_repeated_russian_conjunction():
    assert Validator._check_repetitions("и и") == [
        "Repeated adjacent word(s): и"
    ]


def test_check_repetitions_does_not_warn_for_different_words():
    assert Validator._check_repetitions("и в") == []


@pytest.mark.asyncio
async def test_sources_must_belong_to_retrieved_context():
    state = {
        "answer": "Ответ.",
        "retrieved_context": [{
            "source_id": "doc-1",
            "url": "https://kpfu.ru/page",
            "text": "Текст страницы.",
        }],
        "sources": [{
            "source_id": "doc-1",
            "url": "https://example.com/fake",
        }],
    }

    result = await Validator().process(state)

    assert result["validation_passed"] is False
    assert any(
        "URL does not match" in error for error in result["validator_errors"]
    )


@pytest.mark.asyncio
async def test_invalid_context_shape_is_reported():
    result = await Validator().process({
        "answer": "Ответ.",
        "retrieved_context": {"source_id": "doc-1"},
    })

    assert result["validation_passed"] is False
    assert "'retrieved_context' must be a list" in result["validator_errors"][0]


def test_check_glossary_reports_missing_translation(monkeypatch):
    monkeypatch.setattr(validator_module, "_semantic_match", lambda *args: False)

    errors = Validator._check_glossary(
        "The university is old.",
        "Это старое учреждение.",
        [{"en": "university", "tgt": "университет"}],
    )

    assert errors == [
        "Glossary term 'university' must be translated as 'университет'."
    ]


def test_check_glossary_accepts_literal_translation(monkeypatch):
    def semantic_match(*args):
        pytest.fail("semantic matching must not run for a literal match")

    monkeypatch.setattr(validator_module, "_semantic_match", semantic_match)

    errors = Validator._check_glossary(
        "The university is old.",
        "Это старый университет.",
        [{"en": "university", "tgt": "университет"}],
    )

    assert errors == []


def test_check_glossary_accepts_semantic_translation(monkeypatch):
    monkeypatch.setattr(validator_module, "_semantic_match", lambda *args: True)

    errors = Validator._check_glossary(
        "The university is old.",
        "Это старый вуз.",
        [{"en": "university", "tgt": "университет"}],
    )

    assert errors == []


def test_check_glossary_skips_term_shorter_than_minimum(monkeypatch):
    def semantic_match(*args):
        pytest.fail("semantic matching must not run for a short term")

    monkeypatch.setattr(validator_module, "_semantic_match", semantic_match)

    errors = Validator._check_glossary(
        "A student arrived.",
        "Прибыл студент.",
        [{"en": "a", "tgt": "я"}],
    )

    assert errors == []
