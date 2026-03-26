#!/usr/bin/env bash
set -euo pipefail

ROOT="${CTOX_ROOT:-$(pwd)}"
RUNTIME_DIR="$ROOT/runtime"
RESULTS_PATH="${1:-$RUNTIME_DIR/legacy_chat_loading_results.jsonl}"
shift || true

mkdir -p "$RUNTIME_DIR"
: > "$RESULTS_PATH"

if [[ "$#" -gt 0 ]]; then
  MODELS=("$@")
else
  MODELS=(
    "openai/gpt-oss-20b"
    "Qwen/Qwen3.5-4B"
    "Qwen/Qwen3.5-9B"
    "Qwen/Qwen3.5-27B"
    "Qwen/Qwen3.5-35B-A3B"
    "zai-org/GLM-4.7-Flash"
  )
fi

wait_secs_for_model() {
  case "$1" in
    "Qwen/Qwen3.5-27B"|"Qwen/Qwen3.5-35B-A3B")
      echo 600
      ;;
    "zai-org/GLM-4.7-Flash")
      echo 1200
      ;;
    *)
      echo 180
      ;;
  esac
}

cleanup_runtime() {
  pkill -f "ctox serve-responses-proxy" || true
  pkill -f "mistralrs serve" || true
  pkill -f "run_vllm_serve_backend.sh" || true
  fuser -k 1234/tcp 2>/dev/null || true
  fuser -k 1235/tcp 2>/dev/null || true
  fuser -k 1236/tcp 2>/dev/null || true
  sleep 2
}

append_result() {
  local model="$1"
  local status="$2"
  local response_status="$3"
  local response_body="$4"
  local log_path="$5"

  python3 - "$RESULTS_PATH" "$model" "$status" "$response_status" "$response_body" "$log_path" <<'PY'
import json
import pathlib
import subprocess
import sys
from datetime import datetime, timezone

results_path, model, status, response_status, response_body, log_path = sys.argv[1:]

def read_text(path: str, limit: int = 10000) -> str:
    p = pathlib.Path(path)
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8", errors="replace")[-limit:]

gpu_rows = []
try:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    for line in out.strip().splitlines():
        idx, name, total, used, util = [part.strip() for part in line.split(",", 4)]
        gpu_rows.append(
            {
                "index": int(idx),
                "name": name,
                "memory_total_mb": int(total),
                "memory_used_mb": int(used),
                "gpu_util_percent": int(util),
            }
        )
except Exception as exc:
    gpu_rows = [{"error": str(exc)}]

record = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "model": model,
    "status": status,
    "response_status": int(response_status),
    "response_body": response_body,
    "backend_log_tail": read_text(log_path),
    "gpus": gpu_rows,
}
with open(results_path, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(record) + "\n")
PY
}

run_legacy_model() {
  local model="$1"
  local slug
  slug="$(printf '%s' "$model" | tr '/:' '__')"
  local log_path="$RUNTIME_DIR/legacy_${slug}.log"
  local wait_secs
  wait_secs="$(wait_secs_for_model "$model")"
  local port=""
  local response_status=0
  local response_body=""
  local status="failed_start"

  cleanup_runtime
  : > "$log_path"

  (
    cd "$ROOT"
    "$ROOT/target/release/ctox" chat-runtime-apply-legacy "$model" >/dev/null
    set -a
    source "$ROOT/runtime/vllm_serve.env"
    set +a
    exec bash scripts/run_vllm_serve_backend.sh
  ) >"$log_path" 2>&1 &
  local launcher_pid=$!

  port="$(
    python3 - "$ROOT/runtime/vllm_serve.env" <<'PY'
import pathlib
import sys

env_path = pathlib.Path(sys.argv[1])
port = "1235"
if env_path.exists():
    for raw in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if raw.startswith("CTOX_VLLM_SERVE_PORT="):
            port = raw.split("=", 1)[1].strip().strip('"')
            break
print(port)
PY
  )"

  for ((i = 0; i < wait_secs; i++)); do
    if curl -fsS --max-time 2 "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      status="ready"
      break
    fi
    if ! kill -0 "$launcher_pid" 2>/dev/null; then
      status="failed_exit"
      break
    fi
    sleep 1
  done

  if [[ "$status" == "ready" ]]; then
    set +e
    response_body="$(curl -sS --max-time 90 -X POST "http://127.0.0.1:${port}/v1/chat/completions" \
      -H 'content-type: application/json' \
      -d '{"model":"legacy","messages":[{"role":"user","content":"Reply with CTOX_LEGACY_OK and nothing else."}],"max_tokens":24}')"
    response_status=$?
    set -e
    if [[ $response_status -eq 0 ]]; then
      status="ok"
    else
      status="failed_response"
    fi
  fi

  append_result "$model" "$status" "$response_status" "$response_body" "$log_path"
  cleanup_runtime
}

for model in "${MODELS[@]}"; do
  echo "LEGACY | $model"
  run_legacy_model "$model"
done

echo "$RESULTS_PATH"
