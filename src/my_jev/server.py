"""Jev-compatible System One server backed by a local Ollama model."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI
from pydantic import BaseModel

from my_jev import backend
from my_jev.primitives import ANSWERERS, validate_question

MODEL_CATALOG = [
    {
        "name": "gemma3:4b-it-qat",
        "description": "Local Ollama-backed System One model (gemma3:4b-it-qat).",
        "release_date": "2026-01-01",
    },
    {
        "name": "gemma4:e4b",
        "description": "Local Ollama-backed System One model (gemma4:e4b).",
        "release_date": "2026-01-01",
    },
    {
        "name": "dolphin-mistral:latest",
        "description": "Local Ollama-backed System One model (dolphin-mistral:latest).",
        "release_date": "2026-01-01",
    },
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    backend.init_client()
    try:
        yield
    finally:
        await backend.close_client()


app = FastAPI(title="my-jev", lifespan=lifespan)


class SystemOneRequest(BaseModel):
    state: str | dict[str, Any] | list[Any]
    model: str
    questions: dict[str, dict[str, Any]]


@app.post("/v1/systemone")
async def system_one(payload: SystemOneRequest) -> dict[str, Any]:
    for name, question in payload.questions.items():
        validate_question(name, question)

    async def run_one(name: str, question: dict[str, Any]):
        answerer = ANSWERERS[question["type"]]
        return name, await answerer(name, payload.state, question, payload.model)

    results = await asyncio.gather(
        *(run_one(name, question) for name, question in payload.questions.items())
    )

    answers: dict[str, Any] = {}
    total_input_tokens = 0
    total_output_tokens = 0
    for name, result in results:
        answers[name] = result.answer
        total_input_tokens += result.input_tokens
        total_output_tokens += result.output_tokens

    return {
        "model": payload.model,
        "usage": {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
        },
        "answers": answers,
    }


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    try:
        backend_names = set(await backend.list_backend_models())
    except httpx.HTTPError:
        backend_names = set()

    models = [
        entry
        for entry in MODEL_CATALOG
        if not backend_names or entry["name"] in backend_names
    ]
    if not models:
        models = MODEL_CATALOG
    return {"models": models}


def run() -> None:
    import uvicorn

    uvicorn.run("my_jev.server:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    run()
