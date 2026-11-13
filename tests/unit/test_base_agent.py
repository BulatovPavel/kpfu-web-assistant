from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import backend.agents.base as base_module
from backend.agents.base import Agent, BaseAgent
from backend.core.exceptions import ModelNotLoadedError


class ConcreteAgent(Agent):
    hf_model_setting = "HF_CRITIC_MODEL"
    ollama_model_setting = "CRITIC_MODEL"

    async def process(self, state):
        return state


def configure_settings(monkeypatch, **overrides):
    values = {
        "AGENT_ENGINE": "ollama",
        "HF_CRITIC_MODEL": "test/hf-model",
        "CRITIC_MODEL": "test-ollama-model",
        "LLM_MODEL": "test-default-model",
        "ollama_url": "http://ollama.test/api/generate",
    }
    values.update(overrides)
    settings = SimpleNamespace(**values)
    monkeypatch.setattr(base_module, "get_settings", lambda: settings)
    return settings


def make_agent(monkeypatch, **settings):
    configure_settings(monkeypatch, **settings)
    return ConcreteAgent(engine="ollama")


def test_base_agent_process_is_abstract():
    assert "process" in BaseAgent.__abstractmethods__


@pytest.mark.parametrize(
    ("hf_model", "expected_engine"),
    [
        ("test/hf-model", "huggingface"),
        ("", "ollama"),
        (None, "ollama"),
    ],
)
def test_auto_engine_uses_huggingface_only_when_model_is_configured(
    monkeypatch, hf_model, expected_engine
):
    configure_settings(monkeypatch, HF_CRITIC_MODEL=hf_model)
    agent = ConcreteAgent(engine="auto")

    assert agent.engine == expected_engine


def test_huggingface_engine_falls_back_to_ollama_without_model(monkeypatch, caplog):
    configure_settings(monkeypatch, HF_CRITIC_MODEL="")

    agent = ConcreteAgent(engine="huggingface")

    assert agent.engine == "ollama"
    assert "falling back to ollama" in caplog.text


def test_ollama_engine_is_preserved(monkeypatch):
    configure_settings(monkeypatch)

    agent = ConcreteAgent(engine="ollama")

    assert agent.engine == "ollama"


def test_unknown_engine_falls_back_to_ollama_with_warning(monkeypatch, caplog):
    configure_settings(monkeypatch)

    agent = ConcreteAgent(engine="google")

    assert agent.engine == "ollama"
    assert "unknown engine='google'" in caplog.text


@pytest.mark.asyncio
async def test_call_llm_pins_ollama_seed_to_42(monkeypatch):
    agent = make_agent(monkeypatch)
    response = Mock()
    response.json.return_value = {"response": " generated text "}
    response.raise_for_status.return_value = None
    client = Mock()
    client.post = AsyncMock(return_value=response)
    get_client = Mock(return_value=client)
    monkeypatch.setattr(base_module, "get_ollama_client", get_client)
    agent.model_name = "test-ollama-model"

    result = await agent._call_llm("test prompt")

    assert result == "generated text"
    get_client.assert_called_once_with()
    client.post.assert_awaited_once_with(
        "http://ollama.test/api/generate",
        json={
            "model": "test-ollama-model",
            "prompt": "test prompt",
            "stream": False,
            "options": {"temperature": 0.0, "seed": 42},
        },
    )


@pytest.mark.asyncio
async def test_call_llm_raises_model_not_loaded_without_hf_instance(
    monkeypatch,
):
    configure_settings(monkeypatch, HF_CRITIC_MODEL="test/hf-model")
    agent = ConcreteAgent(engine="huggingface")
    assert agent.engine == "huggingface"
    agent.hf_llm = None
    sleep = AsyncMock()
    monkeypatch.setattr("backend.utils.async_utils.asyncio.sleep", sleep)

    with pytest.raises(ModelNotLoadedError):
        await agent._call_llm("test prompt")

    assert sleep.await_count == 2
