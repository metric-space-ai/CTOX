#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$HOME/ctox-validation-20260325/work}"
TOPOLOGY_PATH="${2:?usage: remote_test_qwen35b_topology.sh <root> <topology-path>}"
PORT="${3:-1235}"
DEVICE_LAYERS_SPEC="${4:-}"
NM_DEVICE_ORDINAL="${5:-0}"
BASE_DEVICE_ORDINAL="${6:-}"
MOE_BACKEND_OVERRIDE="${7:-}"
RUN_PROMPT_TEST="${8:-0}"
LOG_PATH="$ROOT/runtime/qwen35b_topology_test.log"
JSON_PATH="$ROOT/runtime/qwen35b_topology_test.json"

cleanup() {
  pkill -f "$ROOT/target/release/ctox" 2>/dev/null || true
  pkill -f "$ROOT/engine/candle/target/release/ctox-engine" 2>/dev/null || true
  fuser -k "${PORT}/tcp" 2>/dev/null || true
  fuser -k 1234/tcp 2>/dev/null || true
  fuser -k 1236/tcp 2>/dev/null || true
}

cleanup
cd "$ROOT"

./target/release/ctox chat-runtime-apply "Qwen/Qwen3.5-35B-A3B" performance >/tmp/ctox_qwen35b_plan_apply.json
source "$ROOT/runtime/engine.env"

export CTOX_ENGINE_PORT="$PORT"
export CTOX_ENGINE_TOPOLOGY="$TOPOLOGY_PATH"
if [[ -n "$DEVICE_LAYERS_SPEC" ]]; then
  DEVICE_LAYERS_SPEC="${DEVICE_LAYERS_SPEC//,/;}"
  export CTOX_ENGINE_DEVICE_LAYERS="$DEVICE_LAYERS_SPEC"
  export CTOX_ENGINE_ALLOW_DEVICE_LAYERS_WITH_TOPOLOGY=1
else
  unset CTOX_ENGINE_DEVICE_LAYERS
  unset CTOX_ENGINE_DEVICE_LAYERS_CLI
  unset CTOX_ENGINE_ALLOW_DEVICE_LAYERS_WITH_TOPOLOGY
fi
export MISTRALRS_NM_DEVICE_ORDINAL="$NM_DEVICE_ORDINAL"
if [[ -n "$BASE_DEVICE_ORDINAL" ]]; then
  export MISTRALRS_BASE_DEVICE_ORDINAL="$BASE_DEVICE_ORDINAL"
else
  unset MISTRALRS_BASE_DEVICE_ORDINAL
fi
if [[ -n "$MOE_BACKEND_OVERRIDE" ]]; then
  export MISTRALRS_MOE_EXPERTS_BACKEND="$MOE_BACKEND_OVERRIDE"
else
  unset MISTRALRS_MOE_EXPERTS_BACKEND
fi

rm -f "$LOG_PATH" "$JSON_PATH"

(
  while true; do
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk 'BEGIN{ORS=","} {print $1}' || true
    echo
    sleep 1
  done
) >"$ROOT/runtime/qwen35b_topology_memtrace.csv" &
TRACE_PID=$!

set +e
timeout 900 ./scripts/engine/run_engine.sh >"$LOG_PATH" 2>&1 &
LAUNCH_PID=$!
set -e

status="failed_start"
prompt_status=""
prompt_body=""
for _ in $(seq 1 360); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    status="healthy"
    break
  fi
  if ! kill -0 "$LAUNCH_PID" 2>/dev/null; then
    wait "$LAUNCH_PID" || true
    status="launcher_exited"
    break
  fi
  sleep 2
done

if [[ "$status" == "healthy" && "$RUN_PROMPT_TEST" == "1" ]]; then
  set +e
  prompt_body="$(curl -sS --max-time 180 -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H 'content-type: application/json' \
    -d '{"model":"Qwen/Qwen3.5-35B-A3B","messages":[{"role":"user","content":"Reply with CTOX_MATRIX_OK and nothing else."}],"max_tokens":24}')"
  prompt_status=$?
  set -e
fi

kill "$TRACE_PID" 2>/dev/null || true
wait "$TRACE_PID" 2>/dev/null || true

python3 - "$ROOT/runtime/qwen35b_topology_memtrace.csv" "$LOG_PATH" "$JSON_PATH" "$status" "$prompt_status" "$prompt_body" <<'PY'
import json, sys
from pathlib import Path

mem_path = Path(sys.argv[1])
log_path = Path(sys.argv[2])
json_path = Path(sys.argv[3])
status = sys.argv[4]
prompt_status = sys.argv[5]
prompt_body = sys.argv[6]

peaks = []
if mem_path.exists():
    for line in mem_path.read_text().splitlines():
        vals = [v for v in line.strip().split(",") if v]
        if not vals:
            continue
        nums = [int(v) for v in vals]
        if not peaks:
            peaks = nums
        else:
            peaks = [max(a, b) for a, b in zip(peaks, nums)]

tail = ""
if log_path.exists():
    tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-120:])

payload = {
    "status": status,
    "prompt_status": int(prompt_status) if prompt_status else None,
    "prompt_body": prompt_body,
    "topology": str(sys.argv[1]),
    "peak_gpu_mem_mib": peaks,
    "log_tail": tail,
}
json_path.write_text(json.dumps(payload, indent=2))
print(json.dumps(payload, indent=2))
PY

if [[ "$status" == "healthy" && ( "$RUN_PROMPT_TEST" != "1" || "$prompt_status" == "0" ) ]]; then
  cleanup
  exit 0
fi

cleanup
exit 1
