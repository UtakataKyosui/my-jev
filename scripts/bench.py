"""逐次実行と asyncio.gather の所要時間を比較する。サーバーは不要で backend を直接叩く。"""

import asyncio
import statistics
import sys
import time

from my_jev.backend import close_client, init_client, next_token_logprobs

MODEL = "gemma3:4b-it-qat"
ROUNDS = 5

STATE = "Subject: Duplicate charge\nMessage: I was charged twice. Please help ASAP."
PROMPTS = [
    f"{STATE}\n\nIs this about billing? Answer Yes or No.\nAnswer:",
    f"{STATE}\n\nTone? A=angry B=calm C=excited. One letter.\nAnswer:",
    f"{STATE}\n\nUrgency? 0=can wait 1=this week 2=today. One digit.\nAnswer:",
]


async def sequential() -> None:
    for prompt in PROMPTS:
        await next_token_logprobs(prompt, MODEL)


async def parallel() -> None:
    await asyncio.gather(*(next_token_logprobs(prompt, MODEL) for prompt in PROMPTS))


async def measure(label: str, fn) -> float:
    await fn()  # ウォームアップ。初回はモデルのロードを含む。
    samples = []
    for _ in range(ROUNDS):
        started = time.perf_counter()
        await fn()
        samples.append(time.perf_counter() - started)
    median = statistics.median(samples)
    print(
        f"{label:10s}: median={median * 1000:7.1f}ms  min={min(samples) * 1000:7.1f}ms"
    )
    return median


async def main() -> int:
    init_client()
    try:
        print(f"{len(PROMPTS)} 問 × {ROUNDS} 回")
        seq = await measure("逐次", sequential)
        par = await measure("並行", parallel)
        print(f"短縮率    : {seq / par:.2f}x")
        print("注: 並行の効きは OLLAMA_NUM_PARALLEL に依存する。")
    finally:
        await close_client()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
