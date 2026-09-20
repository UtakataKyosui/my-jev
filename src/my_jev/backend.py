"""Ollama backend for next-token probability queries.

This is the only module that knows about Ollama's HTTP API. Swapping the
backend (a different local server, a hosted API, ...) should only require
changing this file.
"""

from __future__ import annotations

import httpx

OLLAMA_BASE_URL = "http://localhost:11434"
CHAT_PATH = "/api/chat"
TAGS_PATH = "/api/tags"

# Shared client set by the server's lifespan so concurrent gather()ed calls reuse
# connections. Falls back to a throwaway client (e.g. for scripts, tests) when unset.
_shared_client: httpx.AsyncClient | None = None


def init_client() -> httpx.AsyncClient:
    global _shared_client
    _shared_client = httpx.AsyncClient(timeout=60.0)
    return _shared_client


async def close_client() -> None:
    global _shared_client
    if _shared_client is not None:
        await _shared_client.aclose()
        _shared_client = None


async def next_token_logprobs(
    prompt: str,
    model: str,
    top_k: int = 20,
) -> tuple[dict[str, float], int, int]:
    """Ask the backend for the top-k log-probabilities of the next token.

    Returns (token -> logprob, prompt_eval_count, eval_count). Tokens are
    returned raw (no case-folding or stripping); callers normalize them.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "logprobs": True,
        "top_logprobs": top_k,
        "options": {"num_predict": 1, "temperature": 0},
    }

    owns_client = _shared_client is None
    http = (
        _shared_client
        if _shared_client is not None
        else httpx.AsyncClient(timeout=60.0)
    )
    try:
        response = await http.post(f"{OLLAMA_BASE_URL}{CHAT_PATH}", json=payload)
        response.raise_for_status()
        data = response.json()
    finally:
        if owns_client:
            await http.aclose()

    logprobs_field = data.get("logprobs") or []
    top_logprobs = logprobs_field[0]["top_logprobs"] if logprobs_field else []
    token_logprobs = {entry["token"]: entry["logprob"] for entry in top_logprobs}

    prompt_eval_count = int(data.get("prompt_eval_count", 0))
    eval_count = int(data.get("eval_count", 0))
    return token_logprobs, prompt_eval_count, eval_count


async def list_backend_models() -> list[str]:
    """Return the model names available on the backend."""
    owns_client = _shared_client is None
    http = (
        _shared_client
        if _shared_client is not None
        else httpx.AsyncClient(timeout=60.0)
    )
    try:
        response = await http.get(f"{OLLAMA_BASE_URL}{TAGS_PATH}")
        response.raise_for_status()
        data = response.json()
    finally:
        if owns_client:
            await http.aclose()
    return [entry["name"] for entry in data.get("models", [])]
