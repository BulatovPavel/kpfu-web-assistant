import logging

import httpx
import pytest

import backend.agents.critic as critic_module
from backend.agents.critic import Critic


@pytest.fixture
def critic():
    return Critic(engine="ollama")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"a": 1}', ['{"a": 1}']),
        ("{a: {b: 1}}", ["{a: {b: 1}}"]),
        ("{a} {b}", ["{a}", "{b}"]),
        ('{"reasoning": "with {braces}"}', ['{"reasoning": "with {braces}"}']),
        (
            r'{"reasoning": "with \"escaped\" quotes"}',
            [r'{"reasoning": "with \"escaped\" quotes"}'],
        ),
    ],
)
def test_extract_json_spans(text, expected):
    assert Critic._extract_json_spans(text) == expected


def test_parse_json_response_returns_object_with_expected_keys(critic):
    response = (
        '{"adequacy": 7, "fluency": 8, "terminology": 9, '
        '"overall_score": 8}'
    )

    assert critic._parse_json_response(response) == {
        "adequacy": 7,
        "fluency": 8,
        "terminology": 9,
        "overall_score": 8,
    }


def test_parse_json_response_accepts_json_code_fence(critic):
    response = (
        "```json\n"
        '{"adequacy": 7, "fluency": 8, "terminology": 9, '
        '"overall_score": 8}\n'
        "```"
    )

    result = critic._parse_json_response(response)

    assert result["overall_score"] == 8
    assert result["terminology"] == 9


def test_parse_json_response_finds_json_inside_surrounding_text(critic):
    response = (
        'Some text {"adequacy": 7, "fluency": 8, "terminology": 9, '
        '"overall_score": 8} more text'
    )

    assert critic._parse_json_response(response)["overall_score"] == 8


def test_parse_json_response_uses_fallback_for_invalid_json(critic, monkeypatch):
    fallback = {"overall_score": 7}
    calls = []

    def fallback_extract(response):
        calls.append(response)
        return fallback

    monkeypatch.setattr(critic, "_fallback_extract", fallback_extract, raising=False)
    response = "Not valid JSON: {invalid}"

    assert critic._parse_json_response(response) == fallback
    assert calls == [response]


def test_parse_json_response_warns_and_returns_first_object_without_all_keys(
    critic, caplog
):
    first_object = {"adequacy": 7}
    response = '{"adequacy": 7} {"fluency": 6}'

    with caplog.at_level(logging.WARNING):
        result = critic._parse_json_response(response)

    assert result == first_object
    assert any("using first parsed object" in record.message for record in caplog.records)


def test_parse_json_response_removes_failed_key(critic):
    result = critic._parse_json_response(
        '{"adequacy": 7, "fluency": 8, "terminology": 9, '
        '"overall_score": 8, "failed": true}'
    )

    assert result["overall_score"] == 8
    assert "failed" not in result


def test_fallback_extract_reads_scores_from_text(critic):
    result = critic._fallback_extract("SCORE: 7\nADEQUACY: 8")

    assert result["overall_score"] == 7
    assert result["adequacy"] == 8


def test_get_fallback_evaluation_has_no_scores_and_marks_failure(critic):
    result = critic._get_fallback_evaluation("timeout")

    assert result["adequacy"] is None
    assert result["fluency"] is None
    assert result["terminology"] is None
    assert result["overall_score"] is None
    assert result["failed"] is True


@pytest.mark.asyncio
async def test_process_marks_timeout_as_failure(critic, monkeypatch):
    monkeypatch.setattr(critic, "_build_prompt", lambda **kwargs: "prompt", raising=False)

    async def timeout(_prompt):
        raise httpx.TimeoutException("timed out")

    monkeypatch.setattr(critic, "_call_llm", timeout)

    result = await critic.process({})

    assert result["critic_failed"] is True
    assert result["critic_score"] is None


def test_build_few_shot_includes_universal_and_language_examples(critic):
    result = critic._build_few_shot("ru", "general")

    assert critic_module._UNIVERSAL_FEWSHOT in result
    assert critic_module._LANGUAGE_FEWSHOT["ru"] in result


def test_build_scoring_rules_includes_common_and_domain_rules(critic):
    result = critic._build_scoring_rules("medical", "general")

    assert critic_module._COMMON_RULES in result
    assert critic_module._DOMAIN_RULES["medical"] in result


def test_build_prompt_places_target_language_and_source(critic):
    source = "SOURCE_SENTINEL"
    target_lang = "TARGET_LANGUAGE_SENTINEL"

    prompt = critic._build_prompt(
        source=source,
        translation="TRANSLATION_SENTINEL",
        source_lang="en",
        target_lang=target_lang,
        glossary_entries=[],
        reference=None,
        domain="general",
        audience="general",
    )

    assert f"Target language: {target_lang}" in prompt
    assert f"Source:\n{source}" in prompt
    assert prompt.index(source) < prompt.index("TRANSLATION_SENTINEL")
