#!/usr/bin/env bash
set -euo pipefail

ROOT="${CTOX_ROOT:-$(pwd)}"
RUNTIME_DIR="$ROOT/runtime"
RESULTS_PATH="${1:-$RUNTIME_DIR/chat_preset_validation_results.jsonl}"
PROXY_LOG="$RUNTIME_DIR/manual_proxy.log"
PROXY_PID_FILE="$RUNTIME_DIR/manual_proxy.pid"
PROXY_URL="${CTOX_PROXY_URL:-http://127.0.0.1:12434}"

mkdir -p "$RUNTIME_DIR"
: > "$RESULTS_PATH"

MODELS=(
  "openai/gpt-oss-20b"
  "Qwen/Qwen3.5-4B"
  "Qwen/Qwen3.5-9B"
  "Qwen/Qwen3.5-27B"
  "Qwen/Qwen3.5-35B-A3B"
  "nvidia/Nemotron-Cascade-2-30B-A3B"
  "zai-org/GLM-4.7-Flash"
)
PRESETS=(
  "quality"
  "max_context"
  "performance"
)

json_escape() {
  python3 - "$1" <<'PY'
import json, sys
print(json.dumps(sys.argv[1]))
PY
}

gpu_sample_json() {
  python3 - <<'PY'
import json, subprocess
out = subprocess.check_output([
    "nvidia-smi",
    "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
    "--format=csv,noheader,nounits",
], text=True)
rows = []
for line in out.strip().splitlines():
    idx, name, total, used, util = [part.strip() for part in line.split(",", 4)]
    rows.append({
        "index": int(idx),
        "name": name,
        "memory_total_mb": int(total),
        "memory_used_mb": int(used),
        "gpu_util_percent": int(util),
    })
print(json.dumps(rows))
PY
}

telemetry_json() {
  local payload
  if payload="$(curl -fsS --max-time 3 "$PROXY_URL/ctox/telemetry" 2>/dev/null)"; then
    printf '%s\n' "$payload"
  else
    printf 'null\n'
  fi
}

append_result() {
  local status="$1"
  local model="$2"
  local preset="$3"
  local switch_body="$4"
  local response_body="$5"
  local plan_path="$RUNTIME_DIR/chat_plan.json"
  local plan_json='null'
  local gpu_json
  local telemetry_payload
  gpu_json="$(gpu_sample_json)"
  telemetry_payload="$(telemetry_json)"
  if [[ -f "$plan_path" ]]; then
    plan_json="$(python3 - "$plan_path" <<'PY'
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    print(json.dumps(json.load(handle)))
PY
)"
  fi
  python3 - "$RESULTS_PATH" "$status" "$model" "$preset" "$switch_body" "$response_body" "$plan_json" "$gpu_json" "$telemetry_payload" <<'PY'
import json, sys, datetime
path, status, model, preset, switch_body, response_body, plan_json, gpu_json, telemetry_payload = sys.argv[1:]
telemetry = None if telemetry_payload == "null" else json.loads(telemetry_payload)
record = {
    "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    "status": status,
    "model": model,
    "preset": preset,
    "switch_body": switch_body,
    "response_body": response_body,
    "plan": json.loads(plan_json),
    "gpus": json.loads(gpu_json),
    "telemetry": telemetry,
    "load_observation": None if telemetry is None else telemetry.get("load_observation"),
}
with open(path, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(record) + "\n")
PY
}

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
    "nvidia/Nemotron-Cascade-2-30B-A3B")
      echo 1800
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
    "Qwen/Qwen3.5-27B"|"Qwen/Qwen3.5-35B-A3B"|"nvidia/Nemotron-Cascade-2-30B-A3B"|"zai-org/GLM-4.7-Flash")
      echo 180
      ;;
    *)
      echo 60
      ;;
  esac
}

restart_proxy() {
  pkill -f "ctox serve-responses-proxy" || true
  pkill -f "ctox-engine serve" || true
  pkill -f "run_engine.sh" || true
  fuser -k 1235/tcp 2>/dev/null || true
  sleep 2
  : > "$PROXY_LOG"
  nohup "$ROOT/target/release/ctox" serve-responses-proxy >"$PROXY_LOG" 2>&1 &
  echo $! > "$PROXY_PID_FILE"
  wait_proxy
}

switch_and_probe() {
  local model="$1"
  local preset="$2"
  local switch_body response_body err_file switch_timeout response_timeout
  err_file="$RUNTIME_DIR/last_validation_error.txt"
  switch_timeout="$(switch_timeout_secs "$model")"
  response_timeout="$(response_timeout_secs "$model")"

  if ! "$ROOT/target/release/ctox" chat-runtime-apply "$model" "$preset" >"$RUNTIME_DIR/last_plan.json" 2>"$err_file"; then
    append_result "failed_apply" "$model" "$preset" "" "$(cat "$err_file")"
    return 1
  fi
  if ! switch_body="$(curl -fsS --max-time "$switch_timeout" -X POST "$PROXY_URL/ctox/switch" \
    -H 'content-type: application/json' \
    -d "{\"model\":\"$model\",\"preset\":\"$preset\"}" 2>"$err_file")"; then
    append_result "failed_switch" "$model" "$preset" "" "$(cat "$err_file")"
    return 1
  fi
  sleep 4
  if ! response_body="$(curl -fsS --max-time "$response_timeout" -X POST "$PROXY_URL/v1/responses" \
    -H 'content-type: application/json' \
    -d "{\"input\":\"Reply with CTOX_MATRIX_OK and nothing else.\",\"max_output_tokens\":24}" \
    2>"$err_file")"; then
    append_result "failed_response" "$model" "$preset" "$switch_body" "$(cat "$err_file")"
    return 1
  fi
  append_result "ok" "$model" "$preset" "$switch_body" "$response_body"
}

trap 'append_result "aborted" "${CURRENT_MODEL:-}" "${CURRENT_PRESET:-}" "" "$(tail -n 80 "$PROXY_LOG" 2>/dev/null || true)"' INT TERM

restart_proxy

for model in "${MODELS[@]}"; do
  for preset in "${PRESETS[@]}"; do
    CURRENT_MODEL="$model"
    CURRENT_PRESET="$preset"
    echo "RUN | $model | $preset"
    if switch_and_probe "$model" "$preset"; then
      echo "OK | $model | $preset"
    else
      echo "FAIL | $model | $preset" >&2
      restart_proxy || true
    fi
    sleep 3
  done
done

echo "$RESULTS_PATH"
