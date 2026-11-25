import httpx
import pytest

from app import llm
from app.config import settings
from app.retrieval import Hit

HITS = [
    Hit(1, "a.md", "Travel > Hotels", "The London hotel limit is 250 USD.", 0.9),
    Hit(2, "b.md", "", "Unrelated passage.", 0.1),
]


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setattr(llm, "_sleep", lambda seconds: None)

    def use(name: str) -> None:
        monkeypatch.setattr(settings, "llm_provider", name)
        monkeypatch.setattr(settings, "llm_model", "test-model")
        monkeypatch.setattr(settings, "anthropic_api_key", "anthropic-key")
        monkeypatch.setattr(settings, "openai_api_key", "openai-key")

    return use


def fake_post(monkeypatch, outcomes):
    """Replace httpx.post; each call takes the next outcome (exception, or (status, json))."""
    calls = []
    queue = iter(outcomes)

    def post(url, **kwargs):
        calls.append((url, kwargs))
        outcome = next(queue)
        if isinstance(outcome, Exception):
            raise outcome
        status, body = outcome
        return httpx.Response(status, json=body, request=httpx.Request("POST", url))

    monkeypatch.setattr(llm.httpx, "post", post)
    return calls


OPENAI_OK = (200, {"choices": [{"message": {"content": "250 USD [1]"}}]})


def test_no_hits_returns_no_results_message():
    assert llm.answer("anything", []) == llm.NO_RESULTS


def test_without_a_provider_the_best_passage_is_the_answer(provider):
    provider("none")
    assert llm.answer("hotel limit?", HITS) == "The London hotel limit is 250 USD."


def test_prompt_numbers_sources_with_heading_and_includes_question():
    prompt = llm.build_prompt("What is the hotel limit?", HITS)
    assert "[1] a.md (Travel > Hotels)\nThe London hotel limit is 250 USD." in prompt
    assert "[2] b.md (no heading)" in prompt
    assert prompt.endswith("Question: What is the hotel limit?")


def test_anthropic_request_and_response(provider, monkeypatch):
    provider("anthropic")
    calls = fake_post(monkeypatch, [(200, {"content": [{"type": "text", "text": "250 USD [1]"}]})])

    assert llm.answer("hotel limit?", HITS) == "250 USD [1]"
    url, kwargs = calls[0]
    assert url == "https://api.anthropic.com/v1/messages"
    assert kwargs["headers"]["x-api-key"] == "anthropic-key"
    assert kwargs["json"]["model"] == "test-model"
    assert "[1] a.md" in kwargs["json"]["messages"][0]["content"]


def test_openai_request_and_response(provider, monkeypatch):
    provider("openai")
    calls = fake_post(monkeypatch, [OPENAI_OK])

    assert llm.answer("hotel limit?", HITS) == "250 USD [1]"
    url, kwargs = calls[0]
    assert url == "https://api.openai.com/v1/chat/completions"
    assert kwargs["headers"]["Authorization"] == "Bearer openai-key"
    assert kwargs["json"]["model"] == "test-model"


def test_unknown_provider_is_rejected(provider):
    provider("mystery")
    with pytest.raises(ValueError, match="mystery"):
        llm.answer("q", HITS)


def test_retries_network_errors_rate_limits_and_server_errors(provider, monkeypatch):
    provider("openai")
    calls = fake_post(
        monkeypatch, [httpx.ConnectError("connection refused"), (429, {}), (503, {}), OPENAI_OK]
    )

    assert llm.answer("q", HITS) == "250 USD [1]"
    assert len(calls) == 4


def test_gives_up_after_max_attempts(provider, monkeypatch):
    provider("openai")
    calls = fake_post(monkeypatch, [(503, {})] * llm.MAX_ATTEMPTS)

    with pytest.raises(httpx.HTTPStatusError):
        llm.answer("q", HITS)
    assert len(calls) == llm.MAX_ATTEMPTS


def test_network_error_on_every_attempt_is_raised(provider, monkeypatch):
    provider("openai")
    calls = fake_post(monkeypatch, [httpx.ConnectError("down")] * llm.MAX_ATTEMPTS)

    with pytest.raises(httpx.ConnectError):
        llm.answer("q", HITS)
    assert len(calls) == llm.MAX_ATTEMPTS


def test_does_not_retry_client_errors(provider, monkeypatch):
    provider("anthropic")
    calls = fake_post(monkeypatch, [(401, {"error": "invalid key"})])

    with pytest.raises(httpx.HTTPStatusError):
        llm.answer("q", HITS)
    assert len(calls) == 1


def test_backoff_doubles_between_attempts(provider, monkeypatch):
    provider("openai")
    waits = []
    monkeypatch.setattr(llm, "_sleep", waits.append)
    fake_post(monkeypatch, [(500, {}), (500, {}), OPENAI_OK])

    llm.answer("q", HITS)
    assert waits == [1, 2]
