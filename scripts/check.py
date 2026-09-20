"""SDK の strict パースを含む end-to-end 回帰テスト。サーバーが起動している前提。"""

import sys

from typesafe_sdk import TypeSafeClient

BASE_URL = "http://localhost:8000"
MODEL = "gemma3:4b-it-qat"

QUESTIONS = {
    "billing": {"type": "noul", "instructions": "Is this message about billing?"},
    "tone": {
        "type": "choice",
        "instructions": "What is the tone of this message?",
        "criteria": {
            "angry": "upset or hostile",
            "calm": "neutral or polite",
            "excited": "enthusiastic",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this message?",
        "criteria": ["Can wait", "Needs attention this week", "Needs attention today"],
    },
}

STATE = {
    "subject": "Duplicate charge",
    "message": "I was charged twice. Please help ASAP.",
}


def main() -> int:
    client = TypeSafeClient(api_key="local", base_url=BASE_URL, model=MODEL)

    models = client.models.list()
    assert models.models, "GET /v1/models returned no models"
    print(f"models  : {len(models.models)} 件")

    result = client.system_one(state=STATE, questions=QUESTIONS)

    noul = result.nouls["billing"].noul
    choice = result.choices["tone"]
    score = result.scores["urgency"]

    # strict=True のパースを通った時点で型は保証されるが、int 混入の退行を明示的に見張る。
    assert isinstance(noul, float), f"noul is {type(noul)}"
    assert 0.0 <= noul <= 1.0, noul
    assert choice.choice in QUESTIONS["tone"]["criteria"], choice.choice
    assert abs(sum(choice.probabilities.values()) - 1.0) < 1e-6, choice.probabilities
    assert 0.0 <= score.score <= len(QUESTIONS["urgency"]["criteria"]) - 1, score.score
    assert set(score.legend) == set(range(3)), score.legend

    # 候補トークンが top_logprobs に無いと一様分布へフォールバックし、確率が潰れる。
    # 一様分布は上のアサーションを全て満たしてしまうため、明示的に弾く。
    uniform = 1.0 / 3
    assert abs(choice.confidence - uniform) > 1e-6, (
        f"choice fell back to a uniform distribution: {choice.probabilities}"
    )
    assert abs(score.confidence - uniform) > 1e-6, (
        f"score fell back to a uniform distribution: {score.probabilities}"
    )
    assert abs(noul - 0.5) > 1e-6, "noul fell back to a uniform distribution"
    assert result.usage.output_tokens == len(QUESTIONS), result.usage

    print(f"noul    : {noul}")
    print(f"choice  : {choice.choice} (confidence={choice.confidence})")
    print(f"score   : {score.score} (confidence={score.confidence})")
    print(
        f"usage   : input={result.usage.input_tokens} output={result.usage.output_tokens}"
    )
    print("OK: SDK の strict パースを含む全アサーションを通過")
    return 0


if __name__ == "__main__":
    sys.exit(main())
