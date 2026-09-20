# my-jev

TypeSafe SDK (typesafe_sdk) から接続できる、ローカル Jev 互換サーバーである。TypeSafe SDK はクライアント専用のため、SDK が話すワイヤープロトコル（`POST /v1/systemone`、`GET /v1/models`）をローカルの Ollama モデルで実装したサーバーを自作した。noul・choice・score の3種類の質問について、Ollama の `/api/chat` が返す次トークンの logprobs から確率分布を計算し、TypeSafe SDK の pydantic strict モデルが要求する形式で応答する。

## 構成

- `src/my_jev/backend.py`: Ollama 固有の知識をこのファイルだけに閉じ込める。次トークンの logprobs を取得する。
- `src/my_jev/primitives.py`: noul・choice・score それぞれのプロンプト構築と、logprobs から確率への変換を行う。
- `src/my_jev/server.py`: FastAPI サーバー。質問ごとの推論を `asyncio.gather` で並行実行する。
- `src/my_jev/main.mojo`: Mojo から TypeSafe SDK 経由でこのサーバーを呼ぶサンプル。

## 起動手順

Ollama をあらかじめ起動し、`gemma3:4b-it-qat` を pull しておく。

```bash
ollama serve
ollama pull gemma3:4b-it-qat
```

サーバーを起動する。

```bash
uv sync
uv run python -m my_jev.server
```

別ターミナルから Mojo のサンプルを実行する。

```bash
.venv/bin/mojo run src/my_jev/main.mojo
```

## Mojo と並行化が速度にどう効くか

Mojo はここではクライアントであり、推論の速度そのものには寄与しない。TypeSafe SDK の Python 実装を `std.python` 経由でそのまま呼んでいるだけで、ホットパスは Ollama へのネットワーク往復である。Mojo を使う利点は、静的型と Python 相互運用を両立させたアプリケーションコードを書けることにある。

速度に効くのはサーバー側の `asyncio.gather` である。1リクエストに含まれる3つの質問（noul・choice・score）を逐次実行すると Ollama への3回の往復を直列に待つが、`gather` で並行実行すると往復が重なる。実測では、同じ3問を逐次実行したときの所要時間は0.440秒、`gather` で並行実行したときは0.189秒で、約2.3倍の短縮になった。理論値の3倍には届いていない。Ollama がモデルの重みをロードした状態で同一モデルへの複数リクエストをどこまで並列に処理するかは `OLLAMA_NUM_PARALLEL` の設定に依存するため、環境によって短縮幅は変わる。
