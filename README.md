# my-jev

TypeSafe SDK から接続できる、ローカル LLM を使った Jev 互換サーバー。

TypeSafe SDK (`typesafe_sdk`) はクライアント専用で、既定では `https://api.typesafe.ai` を向く。
本リポジトリは SDK が話すワイヤープロトコルをローカルに実装し、Ollama のモデルを推論バックエンドに
据える。SDK のコードは一切変更せず、`base_url` の差し替えだけで接続先が切り替わる。

## 仕組み

Jev が返すのは文章ではなく確率である。これはローカル LLM から次のように取り出せる。

1. 回答候補が単一トークンになるようプロンプトを組む（`Yes`/`No`、`A`/`B`/`C`、`0`/`1`/`2` など）
2. `num_predict: 1` で **1 トークンだけ**生成させる
3. そのトークンの `top_logprobs` を読み、候補トークンだけで再正規化する

文章を生成させないので、3 問に答えても出力トークンは合計 3 個で済む。応答を JSON として
パースする必要もなく、パース失敗という失敗モードが存在しない。

| 質問型 | 候補トークン | 返す値 |
|---|---|---|
| `noul` | `Yes` / `No` | `noul` = Yes の確率 |
| `choice` | `A`, `B`, `C`, ... | `choice` = 最尤ラベル、`confidence`、全ラベルの `probabilities` |
| `score` | `0`, `1`, ..., `N-1` | `score` = Σ i·pᵢ（期待値。整数に丸めない）、`legend`、`probabilities` |

## 構成

| ファイル | 役割 |
|---|---|
| `src/my_jev/backend.py` | `next_token_logprobs()`。Ollama 固有の知識をここだけに閉じ込める |
| `src/my_jev/primitives.py` | 3 プリミティブのプロンプト構築・確率計算・事前バリデーション |
| `src/my_jev/server.py` | FastAPI。`POST /v1/systemone`、`GET /v1/models` |
| `src/my_jev/main.mojo` | Mojo から TypeSafe SDK 経由で呼ぶサンプル |

推論バックエンドを Ollama 以外（MAX、llama.cpp など）へ差し替える場合、変更が必要なのは
`backend.py` だけである。他のモジュールは `next_token_logprobs()` の戻り値しか知らない。

## 使い方

Ollama を起動しておけば、あとはワンコマンドで動く。`just jev` は依存の同期、
Mojo CLI のビルド、サーバーの起動と停止まで面倒を見る。

```bash
ollama serve

just jev --state "I was charged twice. Please help ASAP." \
         --questions examples/support-triage.json
```

```text
model    : gemma3:4b-it-qat
billing (noul) : 0.9999845
tone (choice): angry confidence: 0.8648731
urgency (score) : 1.9999562 confidence: 0.9999563
usage    : input=259 output=3
```

`--json` を付けるとサーバーの生レスポンスをそのまま出力する。`--state-file` で
JSON またはプレーンテキストのファイルを渡せる。

| recipe | 役割 |
|---|---|
| `just jev <args>` | Mojo CLI を実行する。サーバーは自動で起動・停止 |
| `just run` | 同梱のサンプル質問で実行する |
| `just check` | SDK の strict パースを含む end-to-end 回帰テスト |
| `just bench` | 逐次実行と並行実行の所要時間を比較する |
| `just build` | Mojo CLI を `bin/jev` にビルドする |
| `just llama-serve` | llama-server を `-np 4` で起動する（下記） |
| `just jev-llama <args>` | llama.cpp バックエンドで Mojo CLI を実行する |
| `just check-llama` | llama.cpp バックエンドで回帰テストを実行する |
| `just bench-all` | 両バックエンドを比較する |

## Python から使う

```python
from typesafe_sdk import TypeSafeClient

client = TypeSafeClient(
    api_key="local",                      # 検証しないが、未指定だと SDK 側が例外を投げる
    base_url="http://localhost:8000",
    model="gemma3:4b-it-qat",
)

result = client.system_one(
    state={"subject": "Duplicate charge", "message": "I was charged twice. Please help ASAP."},
    questions={
        "billing": {"type": "noul", "instructions": "Is this message about billing?"},
        "tone": {
            "type": "choice",
            "instructions": "What is the tone?",
            "criteria": {"angry": "upset", "calm": "polite", "excited": "eager"},
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent?",
            "criteria": ["Can wait", "This week", "Today"],
        },
    },
)

result.nouls["billing"].noul        # 0.999987
result.choices["tone"].choice       # "calm"
result.scores["urgency"].score      # 1.9998
result.usage.output_tokens          # 3（3 問で 3 トークン）
```

## 速度について

1 問あたりの内訳を実測すると、時間のほぼ全部が推論エンジンの中にある。

| 区間 | 時間 | 全体比 |
|---|---|---|
| 1 問あたりの総時間 | 60.4ms | 100% |
| うち llama.cpp 内の forward pass | 54.3ms | 90% |
| うち HTTP + JSON | 約 6ms | 10% |
| Python 側の確率計算（exp と再正規化） | 0.57μs | 0.0009% |

したがって速度に効く手立ては、生成トークン数を減らすことと、問い合わせを並行化
することの 2 つしかない。前者は `num_predict: 1` で既に最小である。

### 並行化と Ollama の制約

Ollama は内部で llama.cpp（`llama-server`）を動かしているが、その子プロセスを
`-np 1` で起動する。スロットが 1 本しかないため、`asyncio.gather` で投げても
サーバー側で直列化される。同じ GGUF を `-np 4` で直接起動すると並行化が効く。

| 構成 | 3 問を逐次 | 3 問を並行 | 短縮率 |
|---|---|---|---|
| Ollama（`-np 1` 固定） | 302ms | 298ms | 1.01x |
| llama-server `-np 4` | 302ms | 141ms | 2.15x |

いずれもウォームアップ後の中央値である。初回はモデルのロードに約 3.3 秒かかる
ため、それを含めて計測すると逐次側が不当に遅く見える。

```bash
just llama-serve          # 別ターミナルで -np 4 の llama-server を起動
just bench-all            # 両構成を比較する
```

`just llama-serve` は Ollama が保持している GGUF blob をそのまま使うので、
モデルを二重に持つ必要はない。

### Mojo の位置づけ

**Mojo は推論のホットパスではない。** 上の表のとおり、Mojo に置き換えられる
Python の処理は全体の 0.0009% しかなく、そこを無限に速くしても体感は変わらない。
支配項は Metal 上で動く行列積で、別プロセスの中にある。

Mojo が担っているのは実行の入口である。`src/my_jev/cli.mojo` が引数解析を行い、
TypeSafe SDK の呼び出しだけを `std.python` 経由で Python に委ねる。`mojo build`
でネイティブバイナリ `bin/jev` になるため、サンプルだけでなく通常の実行も Mojo を
通る。

Mojo に推論そのものを担わせるなら、バックエンドを MAX へ移すことになる。MAX の
カーネルは Mojo で書かれており、26.6 は Apple Silicon の Metal に対応する
（`accelerator_api()` が `metal` を返す）。ただしこのモデルについては、実際に
試したところ二重に閉じていた。

| 障害 | 内容 |
|---|---|
| メモリ予約 | `gemma3multimodal/arch.py` が活性化メモリに 15 GiB を固定予約する。このマシンの GPU 予算は 11.84 GiB で、`--max-length` と `--max-batch-size` を下げても変わらない |
| GGUF 非対応 | GGUF の weight adapter を登録しているのは llama3・granite・olmo・phi3・qwen2 系のみで、gemma3 系はテキスト専用の `Gemma3ForCausalLM` を含めて GGUF を読めない |

テキスト専用の `gemma3` アーキテクチャ自体は活性化メモリを 0 しか予約しないため、
分類さえ変われば容量は足りる。しかしそちらは safetensors しか読めず、HF から
重みを落としても `config.json` が `Gemma3ForConditionalGeneration` を宣言する
ためマルチモーダル側へ戻り、再び 15 GiB の予約に当たる。

したがって MAX への移行は、モデルを llama3 系や qwen2 系へ変えるか、MAX 側が
gemma3 の GGUF に対応するのを待つかのどちらかになる。現時点では llama.cpp
`-np 4` が最速の構成である。

## 制約

単一トークンへの写像が破綻する入力は、推論を走らせる前に HTTP 400 で拒否する。

- `choice` の criteria は最大 26 件（`A`–`Z`）
- `score` の criteria は最大 10 件（`0`–`9`）
- criteria が空の `choice` / `score`、未知の `type`

候補トークンが `top_logprobs` に 1 つも現れなかった場合は一様分布にフォールバックし、
どの質問のどの候補が見つからなかったかを `logging.warning` に出す。一部だけ見つかった場合は、
見つかったものだけで再正規化する（正常動作として扱う）。

トークンの照合は前後の空白を除去し、大文字小文字を区別しない。複数の生トークンが同じ候補に
一致する場合は確率質量を合算する。
