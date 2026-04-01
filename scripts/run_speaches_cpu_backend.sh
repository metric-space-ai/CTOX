#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${CTOX_ENGINE_ENV_FILE:-$ROOT/runtime/engine.env}"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
fi

run_uvx() {
  if command -v uvx >/dev/null 2>&1; then
    uvx "$@"
  elif command -v uv >/dev/null 2>&1; then
    uv tool run "$@"
  else
    echo "uv or uvx is required for the CPU speech backend" >&2
    exit 1
  fi
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "required command not found: $1" >&2
    exit 1
  fi
}

wait_for_http() {
  local url="$1"
  local attempts="${2:-60}"
  local delay_s="${3:-1}"
  local i=0
  while (( i < attempts )); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep "$delay_s"
    i=$((i + 1))
  done
  return 1
}

ROLE="${CTOX_AUX_CPU_ROLE:-${CTOX_ENGINE_ROLE:-}}"
HOST="${CTOX_AUX_HOST:-127.0.0.1}"
PORT="${CTOX_AUX_PORT:-}"
MODEL="${CTOX_AUX_REQUEST_MODEL:-}"

case "$ROLE" in
  stt)
    : "${PORT:=1238}"
    : "${MODEL:=Systran/faster-whisper-small}"
    ;;
  tts)
    : "${PORT:=1239}"
    : "${MODEL:=speaches-ai/piper-en_US-lessac-medium}"
    ;;
  *)
    echo "unsupported CPU speech role: $ROLE" >&2
    exit 1
    ;;
esac

require_command curl

CACHE_ROOT="$ROOT/runtime/speaches"
mkdir -p "$CACHE_ROOT/hf" "$CACHE_ROOT/xdg"
export HF_HOME="$CACHE_ROOT/hf"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export HOME="${HOME:-$CACHE_ROOT/home}"
mkdir -p "$HOME"

SPEACHES_FROM="--from"
SPEACHES_SOURCE="git+https://github.com/speaches-ai/speaches.git"
SPEACHES_BASE_URL="http://$HOST:$PORT"
export SPEACHES_BASE_URL

server_pid=""
cleanup() {
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" >/dev/null 2>&1 || true
    wait "$server_pid" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

run_uvx "$SPEACHES_FROM" "$SPEACHES_SOURCE" \
  uvicorn --factory --host "$HOST" --port "$PORT" speaches.main:create_app &
server_pid="$!"

if ! wait_for_http "$SPEACHES_BASE_URL/v1/models" 120 1; then
  echo "speaches CPU backend did not become ready on $SPEACHES_BASE_URL" >&2
  exit 1
fi

run_uvx "$SPEACHES_FROM" "$SPEACHES_SOURCE" \
  speaches-cli model download "$MODEL"

trap - EXIT
wait "$server_pid"
