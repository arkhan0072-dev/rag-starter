"""Answer generation. Without an API key (LLM_PROVIDER=none) the best passage is returned."""

import httpx

from .config import settings
from .retrieval import Hit

PROMPT = """Answer the question using only the numbered sources below.
Cite the sources you use like [1] or [2]. If the sources do not contain the answer,
say that you don't know instead of guessing.

{sources}

Question: {question}"""

NO_RESULTS = "I couldn't find anything relevant in the indexed documents."


def build_prompt(question: str, hits: list[Hit]) -> str:
    sources = "\n\n".join(
        f"[{i}] {h.source} ({h.heading or 'no heading'})\n{h.content}"
        for i, h in enumerate(hits, start=1)
    )
    return PROMPT.format(sources=sources, question=question)


def answer(question: str, hits: list[Hit]) -> str:
    if not hits:
        return NO_RESULTS
    if settings.llm_provider == "none":
        return hits[0].content

    prompt = build_prompt(question, hits)
    if settings.llm_provider == "anthropic":
        resp = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": settings.anthropic_api_key,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": settings.llm_model,
                "max_tokens": 800,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["content"][0]["text"]

    if settings.llm_provider == "openai":
        resp = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {settings.openai_api_key}"},
            json={"model": settings.llm_model, "messages": [{"role": "user", "content": prompt}]},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider}")
