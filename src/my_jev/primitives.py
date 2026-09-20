"""Prompt construction and probability math for the noul / choice / score primitives."""

from __future__ import annotations

import logging
import math
import string
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException

from my_jev.backend import next_token_logprobs

logger = logging.getLogger("my_jev.primitives")

MAX_CHOICE_LABELS = 26
MAX_SCORE_LEVELS = 10

_LETTERS = string.ascii_uppercase


def _render_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    import json

    return json.dumps(content, ensure_ascii=False, indent=2)


def _render_state(state: Any) -> str:
    if isinstance(state, str):
        return state
    import json

    return json.dumps(state, ensure_ascii=False, indent=2)


def _match_probability(
    token_logprobs: dict[str, float], candidates: list[str]
) -> tuple[dict[str, float], list[str]]:
    """Sum probability mass (via exp(logprob)) of tokens matching each candidate.

    Matching is case-insensitive and ignores surrounding whitespace. Multiple raw
    tokens (e.g. "B" and " B") can map to the same candidate; their mass is summed,
    not overwritten. Returns (candidate -> summed probability, candidates with zero mass).
    """
    normalized = {candidate: 0.0 for candidate in candidates}
    lookup = {candidate.strip().lower(): candidate for candidate in candidates}
    for raw_token, logprob in token_logprobs.items():
        key = raw_token.strip().lower()
        candidate = lookup.get(key)
        if candidate is not None:
            normalized[candidate] += math.exp(logprob)
    missing = [candidate for candidate, mass in normalized.items() if mass == 0.0]
    return normalized, missing


def _renormalize(masses: dict[str, float]) -> dict[str, float]:
    total = sum(masses.values())
    n = len(masses)
    if total <= 0.0:
        return {key: 1.0 / n for key in masses} if n else {}
    return {key: value / total for key, value in masses.items()}


@dataclass
class QuestionResult:
    answer: dict[str, Any]
    input_tokens: int
    output_tokens: int


def _base_prompt(state: Any, instructions: Any) -> str:
    lines = [
        "State:",
        _render_state(state),
        "",
    ]
    if instructions is not None:
        lines += ["Question:", _render_content(instructions), ""]
    return "\n".join(lines)


async def answer_noul(
    question_name: str, state: Any, question: dict[str, Any], model: str
) -> QuestionResult:
    instructions = question.get("instructions")
    criteria = question.get("criteria") or {}
    true_desc = criteria.get("true") if criteria else None
    false_desc = criteria.get("false") if criteria else None

    lines = [_base_prompt(state, instructions)]
    if true_desc is not None or false_desc is not None:
        lines.append("Meaning of each answer:")
        if true_desc is not None:
            lines.append(f"Yes = {_render_content(true_desc)}")
        if false_desc is not None:
            lines.append(f"No = {_render_content(false_desc)}")
        lines.append("")
    lines.append(
        'Respond with exactly one word, either "Yes" or "No", and nothing else. '
        "Output exactly one token. No explanation."
    )
    prompt = "\n".join(lines)

    token_logprobs, input_tokens, output_tokens = await next_token_logprobs(
        prompt, model
    )
    masses, missing = _match_probability(token_logprobs, ["Yes", "No"])
    if missing:
        logger.warning(
            "noul question %r: candidate token(s) %s not found in top_logprobs; falling back to uniform distribution",
            question_name,
            missing,
        )
        masses = {"Yes": 1.0, "No": 1.0}
    probabilities = _renormalize(masses)

    answer = {"type": "noul", "noul": float(probabilities["Yes"])}
    return QuestionResult(answer, input_tokens, output_tokens)


async def answer_choice(
    question_name: str, state: Any, question: dict[str, Any], model: str
) -> QuestionResult:
    instructions = question.get("instructions")
    criteria: dict[str, Any] = question.get("criteria") or {}
    labels = list(criteria.keys())
    # Count/emptiness are validated up front by validate_question() before dispatch.
    letters = [_LETTERS[i] for i in range(len(labels))]
    legend_lines = []
    for letter, label in zip(letters, labels):
        description = criteria[label]
        if description is None:
            legend_lines.append(f"{letter} = {label}")
        else:
            legend_lines.append(f"{letter} = {label}: {_render_content(description)}")

    lines = [_base_prompt(state, instructions)]
    lines.append("Options:")
    lines.extend(legend_lines)
    lines.append("")
    lines.append(
        f"Respond with exactly one letter from the options above ({', '.join(letters)}). "
        "Output exactly one character. No explanation."
    )
    prompt = "\n".join(lines)

    token_logprobs, input_tokens, output_tokens = await next_token_logprobs(
        prompt, model
    )
    masses, missing = _match_probability(token_logprobs, letters)
    if missing:
        logger.warning(
            "choice question %r: candidate token(s) %s not found in top_logprobs; falling back to uniform distribution",
            question_name,
            missing,
        )
        masses = {letter: 1.0 for letter in letters}
    probabilities = _renormalize(masses)

    letter_to_label = dict(zip(letters, labels))
    best_letter = max(probabilities, key=lambda letter: probabilities[letter])
    label_probabilities = {
        letter_to_label[letter]: float(probabilities[letter]) for letter in letters
    }

    answer = {
        "type": "choice",
        "choice": letter_to_label[best_letter],
        "confidence": float(probabilities[best_letter]),
        "probabilities": label_probabilities,
    }
    return QuestionResult(answer, input_tokens, output_tokens)


async def answer_score(
    question_name: str, state: Any, question: dict[str, Any], model: str
) -> QuestionResult:
    instructions = question.get("instructions")
    criteria: list[Any] = question.get("criteria") or []
    # Count/emptiness are validated up front by validate_question() before dispatch.
    digits = [str(i) for i in range(len(criteria))]
    legend_lines = [
        f"{digit} = {_render_content(desc)}" for digit, desc in zip(digits, criteria)
    ]

    lines = [_base_prompt(state, instructions)]
    lines.append("Rubric:")
    lines.extend(legend_lines)
    lines.append("")
    lines.append(
        f"Respond with exactly one digit from the rubric above ({', '.join(digits)}). "
        "Output exactly one character. No explanation."
    )
    prompt = "\n".join(lines)

    token_logprobs, input_tokens, output_tokens = await next_token_logprobs(
        prompt, model
    )
    masses, missing = _match_probability(token_logprobs, digits)
    if missing:
        logger.warning(
            "score question %r: candidate token(s) %s not found in top_logprobs; falling back to uniform distribution",
            question_name,
            missing,
        )
        masses = {digit: 1.0 for digit in digits}
    probabilities = _renormalize(masses)

    expected_score = sum(i * probabilities[str(i)] for i in range(len(criteria)))
    best_digit = max(probabilities, key=lambda digit: probabilities[digit])

    legend = {digit: criteria[int(digit)] for digit in digits}
    score_probabilities = {digit: float(probabilities[digit]) for digit in digits}

    answer = {
        "type": "score",
        "score": float(expected_score),
        "confidence": float(probabilities[best_digit]),
        "legend": legend,
        "probabilities": score_probabilities,
    }
    return QuestionResult(answer, input_tokens, output_tokens)


ANSWERERS = {
    "noul": answer_noul,
    "choice": answer_choice,
    "score": answer_score,
}


def validate_question(question_name: str, question: dict[str, Any]) -> None:
    """Raise HTTPException up front for questions that cannot be answered with a single token."""
    q_type = question.get("type")
    if q_type == "choice":
        labels = list((question.get("criteria") or {}).keys())
        if not labels:
            raise HTTPException(
                status_code=400,
                detail=f"choice question {question_name!r} has no criteria",
            )
        if len(labels) > MAX_CHOICE_LABELS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"choice question {question_name!r} has {len(labels)} criteria, "
                    f"exceeding the {MAX_CHOICE_LABELS}-label limit for single-token selection"
                ),
            )
    elif q_type == "score":
        criteria = question.get("criteria") or []
        if not criteria:
            raise HTTPException(
                status_code=400,
                detail=f"score question {question_name!r} has no criteria",
            )
        if len(criteria) > MAX_SCORE_LEVELS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"score question {question_name!r} has {len(criteria)} criteria, "
                    f"exceeding the {MAX_SCORE_LEVELS}-level limit for single-token selection"
                ),
            )
    elif q_type != "noul":
        raise HTTPException(
            status_code=400,
            detail=f"question {question_name!r} has unknown type {q_type!r}",
        )
