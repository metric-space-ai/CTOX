#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: remote_validate_one_chat_preset.sh <model> <preset>" >&2
  exit 2
fi

ROOT="${CTOX_ROOT:-$(pwd)}"
RUNTIME_DIR="$ROOT/runtime"
MODEL="$1"
PRESET="$2"
PROXY_URL="${CTOX_PROXY_URL:-http://127.0.0.1:12434}"

mkdir -p "$RUNTIME_DIR"

wait_proxy() {
  local retries=60
  for ((i = 0; i < retries; i++)); do
    if curl -fsS --max-time 2 "$PROXY_URL/ctox/telemetry" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

switch_timeout_secs() {
  case "$1" in
    "openai/gpt-oss-20b")
      echo 240
      ;;
    "Qwen/Qwen3.5-4B"|"Qwen/Qwen3.5-9B")
      echo 240
      ;;
    "Qwen/Qwen3.5-27B")
      echo 1200
      ;;
    "Qwen/Qwen3.5-35B-A3B")
      echo 1500
      ;;
    "zai-org/GLM-4.7-Flash")
      echo 900
      ;;
    *)
      echo 120
      ;;
  esac
}

response_timeout_secs() {
  case "$1" in
    "openai/gpt-oss-20b"|"Qwen/Qwen3.5-4B"|"Qwen/Qwen3.5-9B")
      echo 120
      ;;
    "Qwen/Qwen3.5-27B"|"Qwen/Qwen3.5-35B-A3B"|"zai-org/GLM-4.7-Flash")
      echo 180
      ;;
    *)
      echo 60
      ;;
  esac
}

pkill -f "ctox serve-responses-proxy" || true
pkill -f "mistralrs serve" || true
pkill -f "run_vllm_serve_backend.sh" || true
fuser -k 1235/tcp 2>/dev/null || true
: > "$RUNTIME_DIR/manual_proxy.log"
nohup "$ROOT/target/release/ctox" serve-responses-proxy >"$RUNTIME_DIR/manual_proxy.log" 2>&1 &
wait_proxy

"$ROOT/target/release/ctox" chat-runtime-apply "$MODEL" "$PRESET" >"$RUNTIME_DIR/last_plan.json"

switch_status=0
response_status=0
switch_body=""
response_body=""
switch_timeout="$(switch_timeout_secs "$MODEL")"
response_timeout="$(response_timeout_secs "$MODEL")"

set +e
switch_body="$(curl -sS --max-time "$switch_timeout" -X POST "$PROXY_URL/ctox/switch" \
  -H 'content-type: application/json' \
  -d "{\"model\":\"$MODEL\",\"preset\":\"$PRESET\"}")"
switch_status=$?
set -e

if [[ $switch_status -eq 0 ]]; then
  sleep 4
  set +e
  response_body="$(curl -sS --max-time "$response_timeout" -X POST "$PROXY_URL/v1/responses" \
    -H 'content-type: application/json' \
    -d '{"input":"Reply with CTOX_MATRIX_OK and nothing else.","max_output_tokens":24}')"
  response_status=$?
  set -e
fi

python3 - "$MODEL" "$PRESET" "$switch_status" "$response_status" "$switch_body" "$response_body" "$RUNTIME_DIR" <<'PY'
import json, subprocess, sys, pathlib

model, preset, switch_status, response_status, switch_body, response_body, runtime_dir = sys.argv[1:]
runtime = pathlib.Path(runtime_dir)

def read_json(path: pathlib.Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

def read_text(path: pathlib.Path, limit: int = 8000):
    if not path.exists():
        return ""
    data = path.read_text(encoding="utf-8", errors="replace")
    return data[-limit:]

gpu_rows = []
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

backend_log = ""
for candidate in sorted(runtime.glob("vllm_serve_*.log"), key=lambda p: p.stat().st_mtime, reverse=True):
    backend_log = read_text(candidate)
    if backend_log:
        break

result = {
    "model": model,
    "preset": preset,
    "switch_status": int(switch_status),
    "response_status": int(response_status),
    "switch_body": switch_body,
    "response_body": response_body,
    "plan": read_json(runtime / "chat_runtime_plan.json"),
    "gpus": gpu_rows,
    "proxy_log_tail": read_text(runtime / "manual_proxy.log"),
    "backend_log_tail": backend_log,
}
print(json.dumps(result))
PY
