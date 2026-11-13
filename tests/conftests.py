import pytest

@pytest.fixture(autouse=True)
def clean_hf_models():
    yield
    from backend.utils.hf_llm import unload_hf_llm
    unload_hf_llm()

