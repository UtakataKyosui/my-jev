#!/usr/bin/env bash
# usage: with-server.sh <backend> <command> [args...]
# サーバーを <backend> で起動し、応答可能になってから <command> を実行して必ず停止する。
set -uo pipefail

backend="$1"
shift

port="${MY_JEV_PORT:-8000}"
log="${MY_JEV_LOG:-/tmp/my-jev-server.log}"

if lsof -ti "tcp:${port}" > /dev/null 2>&1; then
    echo "port ${port} is already in use" >&2
    exit 1
fi

MY_JEV_BACKEND="$backend" uv run python -m my_jev.server > "$log" 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null; wait "$server_pid" 2>/dev/null' EXIT

# モデルのコールドロードがあるため、実際に応答するまで待つ。
for _ in $(seq 1 60); do
    if curl -sf -o /dev/null "http://localhost:${port}/v1/models"; then
        break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
        echo "server exited during startup; see ${log}" >&2
        cat "$log" >&2
        exit 1
    fi
    sleep 0.5
done

"$@"
