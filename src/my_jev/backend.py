"""Next-token probability queries against a local inference server.

This is the only module that knows about the backend's HTTP API. Two backends
are supported and selected with MY_JEV_BACKEND:

- "ollama" (default): Ollama's /api/chat. Convenient, but Ollama starts its
  llama-server child with -np 1, so concurrent requests are serialized.
- "llamacpp": a llama-server started directly, where -np sets the number of
  parallel slots. Same engine as Ollama uses underneath.
"""

from __future__ import annotations

import os

import httpx

OLLAMA_BASE_URL = os.environ.get("MY_JEV_OLLAMA_URL", "http://localhost:11434")
LLAMACPP_BASE_URL = os.environ.get("MY_JEV_LLAMA_URL", "http://127.0.0.1:8099")
BACKEND = os.environ.get("MY_JEV_BACKEND", "ollama")

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


class _Client:
    """Use the shared client when the server owns one, else a throwaway."""

    def __init__(self) -> None:
        self._owned: httpx.AsyncClient | None = None

    async def __aenter__(self) -> httpx.AsyncClient:
        if _shared_client is not None:
            return _shared_client
        self._owned = httpx.AsyncClient(timeout=60.0)
        return self._owned

    async def __aexit__(self, *exc: object) -> None:
        if self._owned is not None:
            await self._owned.aclose()
            self._owned = None


async def _post_ollama(
    prompt: str, model: str, top_k: int
) -> tuple[dict[str, float], int, int]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "logprobs": True,
        "top_logprobs": top_k,
        "options": {"num_predict": 1, "temperature": 0},
    }
    async with _Client() as http:
        response = await http.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()

    entries = data.get("logprobs") or []
    top = entries[0]["top_logprobs"] if entries else []
    return (
        {entry["token"]: entry["logprob"] for entry in top},
        int(data.get("prompt_eval_count", 0)),
        int(data.get("eval_count", 0)),
    )


async def _post_llamacpp(
    prompt: str, model: str, top_k: int
) -> tuple[dict[str, float], int, int]:
    # /v1/chat/completions applies the GGUF's chat template; the raw /completion
    # endpoint does not, and an instruction-tuned model without its template does
    # not answer with a bare label token.
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": top_k,
    }
    async with _Client() as http:
        response = await http.post(
            f"{LLAMACPP_BASE_URL}/v1/chat/completions", json=payload
        )
        response.raise_for_status()
        data = response.json()

    content = data["choices"][0]["logprobs"]["content"]
    top = content[0]["top_logprobs"] if content else []
    usage = data.get("usage") or {}
    return (
        {entry["token"]: entry["logprob"] for entry in top},
        int(usage.get("prompt_tokens", 0)),
        int(usage.get("completion_tokens", 0)),
    )


async def next_token_logprobs(
    prompt: str,
    model: str,
    top_k: int = 20,
) -> tuple[dict[str, float], int, int]:
    """Return (token -> logprob, prompt_eval_count, eval_count) for the next token.

    Tokens are returned raw; llama-server emits a leading space on most tokens
    while Ollama's chat endpoint does not, so callers must normalize them.
    """
    if BACKEND == "llamacpp":
        return await _post_llamacpp(prompt, model, top_k)
    return await _post_ollama(prompt, model, top_k)


async def list_backend_models() -> list[str]:
    """Return the model names available on the backend."""
    async with _Client() as http:
        if BACKEND == "llamacpp":
            response = await http.get(f"{LLAMACPP_BASE_URL}/v1/models")
            response.raise_for_status()
            return [entry["id"] for entry in response.json().get("data", [])]
        response = await http.get(f"{OLLAMA_BASE_URL}/api/tags")
        response.raise_for_status()
        return [entry["name"] for entry in response.json().get("models", [])]
