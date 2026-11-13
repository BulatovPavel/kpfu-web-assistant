from unittest.mock import AsyncMock

import pytest

from backend.agents.editor import Editor


@pytest.mark.asyncio
async def test_editor_revises_answer_without_changing_retrieved_sources(monkeypatch):
    editor = Editor(engine="ollama")
    call_llm = AsyncMock(return_value='{"answer": "Исправленный ответ [1]."}')
    monkeypatch.setattr(editor, "_call_llm", call_llm)
    citations = [{"source_id": "page-1", "quote": "Фрагмент"}]
    context = [{"source_id": "page-1", "text": "Фрагмент", "url": "https://kpfu.ru"}]
    state = {
        "question": "Что известно?",
        "answer": "Черновой ответ.",
        "retrieved_context": context,
        "citations": citations,
        "validator_errors": ["Claim is not supported."],
    }

    result = await editor.process(state)

    assert result["answer"] == "Исправленный ответ [1]."
    assert result["retrieved_context"] is context
    assert result["citations"] is citations
    assert result["editor_changed"] is True
    assert result["editor_failed"] is False
    call_llm.assert_awaited_once()
    prompt = call_llm.await_args.args[0]
    assert "Что известно?" in prompt
    assert "Claim is not supported." in prompt
    assert "do not add unsupported facts" in prompt


@pytest.mark.asyncio
async def test_editor_skips_when_critic_failed(monkeypatch):
    editor = Editor(engine="ollama")
    call_llm = AsyncMock()
    monkeypatch.setattr(editor, "_call_llm", call_llm)
    state = {"answer": "Оригинал.", "critic_failed": True}

    result = await editor.process(state)

    assert result["answer"] == "Оригинал."
    assert result["editor_changed"] is False
    assert result["editor_failed"] is False
    call_llm.assert_not_awaited()


@pytest.mark.asyncio
async def test_editor_skips_when_there_are_no_issues(monkeypatch):
    editor = Editor(engine="ollama")
    call_llm = AsyncMock()
    monkeypatch.setattr(editor, "_call_llm", call_llm)
    state = {"answer": "Ответ."}

    result = await editor.process(state)

    assert result["answer"] == "Ответ."
    assert result["editor_changed"] is False
    call_llm.assert_not_awaited()


@pytest.mark.asyncio
async def test_editor_keeps_original_and_marks_failure_on_llm_error(monkeypatch):
    editor = Editor(engine="ollama")
    call_llm = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
    monkeypatch.setattr(editor, "_call_llm", call_llm)
    state = {
        "answer": "Оригинал.",
        "validator_warnings": ["Repeated word."],
    }

    result = await editor.process(state)

    assert result["answer"] == "Оригинал."
    assert result["editor_failed"] is True
    assert result["editor_changed"] is False


def test_extract_answer_handles_fenced_json():
    assert Editor._extract_answer('```json\n{"answer": "Ответ"}\n```') == "Ответ"


def test_extract_answer_preserves_non_json_unicode_response():
    response = "Ответ содержит 中文 текст и цитату «漢字»."

    assert Editor._extract_answer(response) == response
