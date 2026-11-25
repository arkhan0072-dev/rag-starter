"""Answer generation. Without an API key (LLM_PROVIDER=none) the best passage is returned."""

import time

import httpx

from .config import settings
from .retrieval import Hit

PROMPT = """Answer the question using only the numbered sources below.
Cite the sources you use like [1] or [2]. If the sources do not contain the answer,
say that you don't know instead of guessing.

{sources}

Question: {question}"""

NO_RESULTS = "I couldn't find anything relevant in the indexed documents."

MAX_ATTEMPTS = 4
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_sleep = time.sleep  # replaced in tests


def build_prompt(question: str, hits: list[Hit]) -> str:
    sources = "\n\n".join(
        f"[{i}] {h.source} ({h.heading or 'no heading'})\n{h.content}"
        for i, h in enumerate(hits, start=1)
    )
    return PROMPT.format(sources=sources, question=question)


def _post(url: str, headers: dict, payload: dict) -> dict:
    """POST with retries on network errors, rate limits and server errors.

    Waits 1s, 2s, 4s between attempts. Client errors (bad key, bad request) fail at once.
    """
    for attempt in range(MAX_ATTEMPTS - 1):
        try:
            resp = httpx.post(url, headers=headers, json=payload, timeout=60)
            if resp.status_code not in RETRYABLE_STATUS:
                break
        except httpx.TransportError:
            pass
        _sleep(2**attempt)
    else:
        # Final attempt: network errors and error statuses propagate to the caller.
        resp = httpx.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


def answer(question: str, hits: list[Hit]) -> str:
    if not hits:
        return NO_RESULTS
    if settings.llm_provider == "none":
        return hits[0].content

    message = {"role": "user", "content": build_prompt(question, hits)}

    if settings.llm_provider == "anthropic":
        data = _post(
            "https://api.anthropic.com/v1/messages",
            {"x-api-key": settings.anthropic_api_key, "anthropic-version": "2023-06-01"},
            {"model": settings.llm_model, "max_tokens": 800, "messages": [message]},
        )
        return data["content"][0]["text"]

    if settings.llm_provider == "openai":
        data = _post(
            "https://api.openai.com/v1/chat/completions",
            {"Authorization": f"Bearer {settings.openai_api_key}"},
            {"model": settings.llm_model, "messages": [message]},
        )
        return data["choices"][0]["message"]["content"]

    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider}")
