set shell := ["bash", "-uc"]
set positional-arguments

model := "gemma3:4b-it-qat"
port := "8000"

llama_bin := "/Applications/Ollama.app/Contents/Resources/llama-server"
llama_port := "8099"
slots := "4"

default:
    @just --list

# 依存を同期し、モデルが未取得なら pull する
setup:
    uv sync
    @ollama list | grep -q '{{model}}' || ollama pull {{model}}

# Mojo CLI をネイティブバイナリ bin/jev にビルドする
build:
    @mkdir -p bin
    .venv/bin/mojo build src/my_jev/cli.mojo -o bin/jev

# Mojo CLI を実行する（サーバーは自動で起動・停止する）
#   just jev --state "..." --questions examples/support-triage.json
jev *args: setup build
    @scripts/with-server.sh ollama bin/jev "$@"

# llama.cpp バックエンドで Mojo CLI を実行する（just llama-serve を別ターミナルで）
jev-llama *args: setup build
    @scripts/with-server.sh llamacpp bin/jev "$@"

# サーバーを前景で起動する（Ctrl-C で停止）
serve: setup
    uv run python -m my_jev.server

# 同梱のサンプル質問で Mojo CLI を実行する
run: setup build
    @scripts/with-server.sh ollama bin/jev --state "I was charged twice. Please help ASAP." --questions examples/support-triage.json

# SDK の strict パースを含む end-to-end 回帰テスト
check: setup
    @scripts/with-server.sh ollama uv run python scripts/check.py

# llama.cpp バックエンドで回帰テストを実行する
check-llama: setup
    @scripts/with-server.sh llamacpp uv run python scripts/check.py

# 逐次実行と並行実行の所要時間を比較する（サーバー不要）
bench: setup
    uv run python scripts/bench.py

# --- llama.cpp 経路 ---------------------------------------------------------
# Ollama は内部の llama-server を -np 1 で起動するため並行リクエストが直列化する。
# 同じ GGUF を -np {{slots}} で直接起動すると並行化が効く。

# Ollama が保持する GGUF blob のパスを解決する
_gguf:
    @ollama show --modelfile {{model}} | sed -n 's/^FROM //p' | head -1

# llama-server を -np {{slots}} で前景起動する（Ctrl-C で停止）
llama-serve:
    #!/usr/bin/env bash
    set -euo pipefail
    gguf="$(just _gguf)"
    test -f "$gguf" || { echo "GGUF not found: $gguf" >&2; exit 1; }
    exec {{llama_bin}} --model "$gguf" --host 127.0.0.1 --port {{llama_port}} \
        --no-webui -c 8192 -np {{slots}} --flash-attn auto -b 512 -ub 512

# 両バックエンドの逐次/並行を比較する（llama-serve を起動しておく）
bench-all: setup
    #!/usr/bin/env bash
    set -uo pipefail
    echo "=== ollama (-np 1 固定) ==="
    uv run python scripts/bench.py
    if curl -sf -o /dev/null http://127.0.0.1:{{llama_port}}/health; then
        echo "=== llama.cpp (-np {{slots}}) ==="
        MY_JEV_BACKEND=llamacpp uv run python scripts/bench.py
    else
        echo "llama-server not running; start it with: just llama-serve" >&2
    fi

fmt:
    uvx ruff format src scripts
    uvx ruff check --fix src scripts

lint:
    uvx ruff format --check src scripts
    uvx ruff check src scripts
