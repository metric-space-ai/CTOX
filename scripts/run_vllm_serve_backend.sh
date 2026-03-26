#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${CTOX_VLLM_SERVE_ENV_FILE:-$ROOT/runtime/vllm_serve.env}"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
fi

configure_compatible_cuda_runtime() {
  if [[ "$(uname -s)" != "Linux" ]]; then
    return
  fi

  local driver_cuda_version
  local driver_cuda_major
  driver_cuda_version="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1)"
  driver_cuda_major="$(printf '%s\n' "$driver_cuda_version" | cut -d. -f1)"
  if [[ -z "$driver_cuda_major" ]]; then
    return
  fi

  local cuda_root=""
  local candidate=""
  for candidate in \
    "/usr/local/cuda-${driver_cuda_version}" \
    "/usr/local/cuda-${driver_cuda_major}" \
    "/usr/local/cuda-${driver_cuda_major}.0"
  do
    [[ -d "$candidate" ]] || continue
    cuda_root="$candidate"
    break
  done

  if [[ -z "$cuda_root" ]]; then
    return
  fi

  export CUDA_HOME="$cuda_root"
  export CUDA_PATH="$cuda_root"

  local cuda_bin="$cuda_root/bin"
  if [[ -d "$cuda_bin" ]]; then
    case ":${PATH:-}:" in
      *":$cuda_bin:"*) ;;
      *) export PATH="$cuda_bin${PATH:+:$PATH}" ;;
    esac
  fi

  local cuda_ld=""
  for candidate in \
    "$cuda_root/targets/x86_64-linux/lib" \
    "$cuda_root/lib64"
  do
    [[ -d "$candidate" ]] || continue
    cuda_ld="$candidate"
    break
  done

  if [[ -n "$cuda_ld" ]]; then
    case ":${LD_LIBRARY_PATH:-}:" in
      *":$cuda_ld:"*) ;;
      *) export LD_LIBRARY_PATH="$cuda_ld${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
    esac
  fi
}

write_runtime_env_value() {
  local key="$1"
  local value="$2"
  mkdir -p "$(dirname "$ENV_FILE")"
  if [[ -f "$ENV_FILE" ]]; then
    python3 - "$ENV_FILE" "$key" "$value" <<'PY'
import pathlib
import sys

env_path = pathlib.Path(sys.argv[1])
key = sys.argv[2]
value = sys.argv[3]
lines = []
found = False
if env_path.exists():
    lines = env_path.read_text().splitlines()
for idx, line in enumerate(lines):
    if line.startswith(f"{key}="):
        lines[idx] = f"{key}={value}"
        found = True
        break
if not found:
    lines.append(f"{key}={value}")
env_path.write_text("\n".join(lines) + "\n")
PY
  else
    printf '%s=%s\n' "$key" "$value" >"$ENV_FILE"
  fi
}

clear_runtime_env_value() {
  local key="$1"
  [[ -f "$ENV_FILE" ]] || return 0
  python3 - "$ENV_FILE" "$key" <<'PY'
import pathlib
import sys

env_path = pathlib.Path(sys.argv[1])
key = sys.argv[2]
lines = [
    line for line in env_path.read_text().splitlines()
    if not line.startswith(f"{key}=")
]
env_path.write_text("\n".join(lines) + ("\n" if lines else ""))
PY
}

detect_all_visible_nvidia_devices() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi
  nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null \
    | sed '/^[[:space:]]*$/d' \
    | paste -sd, -
}

detect_nvidia_devices_by_free_memory() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
    | sed '/^[[:space:]]*$/d' \
    | sort -t',' -k2,2nr \
    | awk -F',' '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1); print $1}' \
    | paste -sd, -
}

is_qwen35_moe_model_name() {
  case "${1:-}" in
    Qwen/Qwen3.5-35B-A3B)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

is_qwen35_4b_model_name() {
  case "${1:-}" in
    Qwen/Qwen3.5-4B)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

is_qwen35_27b_model_name() {
  case "${1:-}" in
    Qwen/Qwen3.5-27B)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

is_glm47_flash_model() {
  case "${CTOX_VLLM_SERVE_MODEL:-}" in
    zai-org/GLM-4.7-Flash)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

is_gpt_oss_model() {
  case "${CTOX_VLLM_SERVE_MODEL:-}" in
    openai/gpt-oss-20b|gpt-oss-20b|*gpt-oss*)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

csv_without_items() {
  local source_csv="${1:-}"
  local remove_csv="${2:-}"
  python3 - "$source_csv" "$remove_csv" <<'PY'
import sys

source = [item.strip() for item in (sys.argv[1] or "").split(",") if item.strip()]
remove = {item.strip() for item in (sys.argv[2] or "").split(",") if item.strip()}
print(",".join(item for item in source if item not in remove))
PY
}

csv_intersection() {
  local left_csv="${1:-}"
  local right_csv="${2:-}"
  python3 - "$left_csv" "$right_csv" <<'PY'
import sys

left = [item.strip() for item in (sys.argv[1] or "").split(",") if item.strip()]
right = {item.strip() for item in (sys.argv[2] or "").split(",") if item.strip()}
print(",".join(item for item in left if item in right))
PY
}

csv_union_items() {
  python3 - "$@" <<'PY'
import sys

seen = set()
merged = []
for raw in sys.argv[1:]:
    for item in (raw or "").split(","):
        item = item.strip()
        if not item or item in seen:
            continue
        seen.add(item)
        merged.append(item)
print(",".join(merged))
PY
}

count_csv_items() {
  local csv="${1:-}"
  [[ -n "$csv" ]] || {
    printf '0\n'
    return
  }
  awk -F',' '{print NF}' <<<"$csv"
}

resolve_role_cuda_visible_devices_setting() {
  local role="$1"
  case "$role" in
    embedding)
      printf '%s\n' "${CTOX_EMBEDDING_CUDA_VISIBLE_DEVICES:-${CTOX_AUXILIARY_CUDA_VISIBLE_DEVICES:-}}"
      ;;
    stt)
      printf '%s\n' "${CTOX_STT_CUDA_VISIBLE_DEVICES:-${CTOX_AUXILIARY_CUDA_VISIBLE_DEVICES:-}}"
      ;;
    tts)
      printf '%s\n' "${CTOX_TTS_CUDA_VISIBLE_DEVICES:-${CTOX_AUXILIARY_CUDA_VISIBLE_DEVICES:-}}"
      ;;
    *)
      printf '\n'
      ;;
  esac
}

rebalance_device_layers_for_auxiliary_loads() {
  local base_cli="${1:-}"
  local embedding_devices="${2:-}"
  local embedding_reservation="${3:-0}"
  local stt_devices="${4:-}"
  local stt_reservation="${5:-0}"
  local tts_devices="${6:-}"
  local tts_reservation="${7:-0}"
  local explicit_reservation_map="${8:-}"
  python3 - \
    "$base_cli" \
    "$embedding_devices" \
    "$embedding_reservation" \
    "$stt_devices" \
    "$stt_reservation" \
    "$tts_devices" \
    "$tts_reservation" \
    "$explicit_reservation_map" <<'PY'
import math
import sys

base_cli, embedding_devices, embedding_reservation, stt_devices, stt_reservation, tts_devices, tts_reservation, explicit_reservation_map = sys.argv[1:9]

def parse_cli(spec: str):
    items = []
    for raw in (spec or "").split(";"):
        raw = raw.strip()
        if not raw or ":" not in raw:
            continue
        left, right = raw.split(":", 1)
        try:
            items.append((left.strip(), int(right.strip())))
        except ValueError:
            pass
    return items

def parse_csv(csv: str):
    return [item.strip() for item in (csv or "").split(",") if item.strip()]

def parse_float(raw: str):
    try:
        return float(raw or "0")
    except ValueError:
        return 0.0

def parse_reservation_map(spec: str):
    parsed = {}
    for raw in (spec or "").split(";"):
        raw = raw.strip()
        if not raw or ":" not in raw:
            continue
        left, right = raw.split(":", 1)
        try:
            parsed[left.strip()] = max(0.0, parse_float(right.strip()))
        except ValueError:
            pass
    return parsed

base = parse_cli(base_cli)
if not base:
    print("")
    raise SystemExit(0)

loads = {ordinal: 0.0 for ordinal, _ in base}
explicit_loads = parse_reservation_map(explicit_reservation_map)

def apply_reservation(devices_csv: str, reservation_raw: str):
    reservation = max(0.0, parse_float(reservation_raw))
    devices = [ordinal for ordinal in parse_csv(devices_csv) if ordinal in loads]
    if reservation <= 0.0 or not devices:
        return
    per_device = reservation / len(devices)
    for ordinal in devices:
        loads[ordinal] = loads.get(ordinal, 0.0) + per_device

if explicit_loads:
    for ordinal in list(loads):
        loads[ordinal] = max(0.0, explicit_loads.get(ordinal, 0.0))
else:
    apply_reservation(embedding_devices, embedding_reservation)
    apply_reservation(stt_devices, stt_reservation)
    apply_reservation(tts_devices, tts_reservation)

scores = []
for ordinal, layers in base:
    available_weight = max(0.05, 1.0 - loads.get(ordinal, 0.0))
    scores.append((ordinal, layers, layers * available_weight))

score_sum = sum(score for _, _, score in scores)
total_layers = sum(layers for _, layers, _ in scores)
if score_sum <= 0 or total_layers <= 0:
    print(base_cli)
    raise SystemExit(0)

targets = []
remaining = total_layers
for ordinal, _, score in scores:
    exact = (score / score_sum) * total_layers
    floored = math.floor(exact)
    targets.append([ordinal, floored, exact - floored])
    remaining -= floored

targets.sort(key=lambda item: item[2], reverse=True)
for idx in range(max(0, remaining)):
    targets[idx % len(targets)][1] += 1

ordered = {ordinal: layers for ordinal, layers, _ in targets}
print(";".join(f"{ordinal}:{ordered[ordinal]}" for ordinal, _ in base if ordered.get(ordinal, 0) > 0))
PY
}

detect_live_auxiliary_gpu_layer_reservation_map() {
  local embedding_port="${CTOX_EMBEDDING_PORT:-1237}"
  local stt_port="${CTOX_STT_PORT:-1238}"
  local tts_port="${CTOX_TTS_PORT:-1239}"
  python3 - \
    "$embedding_port" \
    "$stt_port" \
    "$tts_port" <<'PY'
import math
import subprocess
import sys

embedding_port, stt_port, tts_port = sys.argv[1:4]

def run_command(cmd):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return ""

def find_pid_for_port(port: str):
    output = run_command(["pgrep", "-af", f"serve -p {port}"])
    for line in output.splitlines():
        if f"serve -p {port}" not in line or "pgrep -af" in line:
            continue
        try:
            return int(line.split(None, 1)[0])
        except Exception:
            continue
    return None

selected_pids = {
    pid for pid in (
        find_pid_for_port(embedding_port),
        find_pid_for_port(stt_port),
        find_pid_for_port(tts_port),
    )
    if pid is not None
}
if not selected_pids:
    print("")
    raise SystemExit(0)

gpu_output = run_command([
    "nvidia-smi",
    "--query-gpu=index,pci.bus_id,memory.total",
    "--format=csv,noheader,nounits",
])
if not gpu_output.strip():
    print("")
    raise SystemExit(0)

bus_to_index = {}
gpu_memory_totals = {}
for row in gpu_output.splitlines():
    parts = [part.strip() for part in row.split(",")]
    if len(parts) < 3:
        continue
    index, bus_id, total_raw = parts[:3]
    try:
        gpu_memory_totals[index] = float(total_raw)
    except ValueError:
        continue
    bus_to_index[bus_id] = index

app_output = run_command([
    "nvidia-smi",
    "--query-compute-apps=pid,gpu_bus_id,used_gpu_memory",
    "--format=csv,noheader,nounits",
])
if not app_output.strip():
    print("")
    raise SystemExit(0)

per_gpu_fraction = {}
for row in app_output.splitlines():
    parts = [part.strip() for part in row.split(",")]
    if len(parts) < 3:
        continue
    pid_raw, bus_id, used_raw = parts[:3]
    try:
        pid = int(pid_raw)
        used = float(used_raw)
    except ValueError:
        continue
    if pid not in selected_pids:
        continue
    gpu_index = bus_to_index.get(bus_id)
    total = gpu_memory_totals.get(gpu_index)
    if gpu_index is None or not total:
        continue
    per_gpu_fraction[gpu_index] = per_gpu_fraction.get(gpu_index, 0.0) + (used / total)

def round_up(value: float, digits: int = 3):
    factor = 10 ** digits
    return math.ceil(value * factor) / factor

items = []
for gpu_index in sorted(per_gpu_fraction, key=lambda item: int(item)):
    fraction = round_up(per_gpu_fraction[gpu_index])
    if fraction <= 0:
        continue
    items.append(f"{gpu_index}:{fraction:.3f}")

print(";".join(items))
PY
}

align_seq_len() {
  local value="${1:-0}"
  local minimum="${2:-1024}"
  local step="${3:-256}"
  if [[ "${value:-0}" -lt "$minimum" ]]; then
    printf '%s\n' "$minimum"
    return
  fi
  printf '%s\n' $(( (value / step) * step ))
}

compute_retry_seq_len() {
  local current="${1:-0}"
  local minimum="${2:-1024}"
  local reduced=$(( current * 3 / 4 ))
  align_seq_len "$reduced" "$minimum"
}

run_tuner_json() {
  local tune_mode="$1"
  local model="$2"
  shift 2
  local cmd=("$VLLM_SERVE_BIN" tune --json --profile balanced "$tune_mode" -m "$model")
  if [[ -n "${CTOX_VLLM_SERVE_ARCH:-}" && "$tune_mode" == "text" ]]; then
    cmd+=(-a "$CTOX_VLLM_SERVE_ARCH")
  fi
  if [[ -n "${CTOX_VLLM_SERVE_ISQ:-}" ]]; then
    cmd+=(--isq "$CTOX_VLLM_SERVE_ISQ")
  fi
  if [[ -n "${CTOX_VLLM_SERVE_MAX_BATCH_SIZE:-}" ]]; then
    cmd+=(--max-batch-size "$CTOX_VLLM_SERVE_MAX_BATCH_SIZE")
  fi
  if [[ -n "${CTOX_VLLM_SERVE_MAX_SEQ_LEN:-}" ]]; then
    cmd+=(--max-seq-len "$CTOX_VLLM_SERVE_MAX_SEQ_LEN")
  fi
  if [[ -n "${CTOX_VLLM_SERVE_TOPOLOGY:-}" ]]; then
    cmd+=(--topology "$CTOX_VLLM_SERVE_TOPOLOGY")
  fi
  RUST_LOG=error "${cmd[@]}" "$@"
}

extract_tuned_runtime_values() {
  local tune_json="$1"
  local requested_max="${2:-0}"
  local tune_json_file
  tune_json_file="$(mktemp)"
  printf '%s' "$tune_json" >"$tune_json_file"
  python3 - "$requested_max" "$tune_json_file" <<'PY'
import json
import sys

requested_max = int(sys.argv[1] or "0")
with open(sys.argv[2], "r", encoding="utf-8") as handle:
    data = json.load(handle)
recommended = None
for candidate in data.get("candidates", []):
    if candidate.get("recommended"):
        recommended = candidate
        break
if recommended is None:
    print("0")
    print("")
    sys.exit(0)
max_context = int(recommended.get("max_context_tokens") or 0)
if requested_max > 0 and max_context > 0:
    max_context = min(max_context, requested_max)
device_layers = data.get("device_layers_cli") or ""
print(max_context)
print(device_layers)
PY
  rm -f "$tune_json_file"
}

derive_runtime_budget() {
  local requested_max_seq_len="${CTOX_VLLM_SERVE_MAX_SEQ_LEN:-0}"
  if [[ "${CTOX_VLLM_SERVE_ROLE:-chat}" == "chat" && "${CTOX_CHAT_RUNTIME_PLAN_ACTIVE:-0}" == "1" ]]; then
    local planned_context="${CTOX_VLLM_SERVE_MAX_SEQ_LEN:-0}"
    local planned_layers="${CTOX_VLLM_SERVE_DEVICE_LAYERS:-}"
    if [[ "${planned_context:-0}" -gt 0 ]]; then
      planned_context="$(align_seq_len "$planned_context" 1024)"
    fi
    printf '%s\n%s\n' "${planned_context:-0}" "$planned_layers"
    return 0
  fi
  if is_embedding_role || is_tts_role; then
    if [[ "$requested_max_seq_len" -gt 0 ]]; then
      printf '%s\n\n' "$(align_seq_len "$requested_max_seq_len" 1024)"
    else
      printf '0\n\n'
    fi
    return 0
  fi

  if is_qwen35_moe_model; then
    local fixed_qwen35_context=2048
    if [[ "$requested_max_seq_len" -gt 0 && "$requested_max_seq_len" -lt "$fixed_qwen35_context" ]]; then
      fixed_qwen35_context="$requested_max_seq_len"
    fi
    fixed_qwen35_context="$(align_seq_len "$fixed_qwen35_context" 1024)"
    printf '%s\n%s\n' "$fixed_qwen35_context" "1:28;2:12"
    return 0
  fi

  if is_qwen35_4b_model_name "${CTOX_VLLM_SERVE_MODEL:-}"; then
    local fixed_qwen4b_context="${requested_max_seq_len:-0}"
    if [[ "${fixed_qwen4b_context:-0}" -le 0 ]]; then
      fixed_qwen4b_context=65536
    fi
    fixed_qwen4b_context="$(align_seq_len "$fixed_qwen4b_context" 1024)"
    printf '%s\n%s\n' "$fixed_qwen4b_context" "0:16;1:16"
    return 0
  fi

  if is_qwen35_27b_model_name "${CTOX_VLLM_SERVE_MODEL:-}"; then
    local fixed_qwen27b_context="${requested_max_seq_len:-0}"
    if [[ "${fixed_qwen27b_context:-0}" -le 0 || "${fixed_qwen27b_context:-0}" -gt 4096 ]]; then
      fixed_qwen27b_context=4096
    fi
    fixed_qwen27b_context="$(align_seq_len "$fixed_qwen27b_context" 1024)"
    printf '%s\n%s\n' "$fixed_qwen27b_context" "0:41;1:23"
    return 0
  fi

  if is_glm47_flash_model; then
    local fixed_glm_context="${requested_max_seq_len:-0}"
    if [[ "${fixed_glm_context:-0}" -le 0 || "${fixed_glm_context:-0}" -gt 2048 ]]; then
      fixed_glm_context=2048
    fi
    fixed_glm_context="$(align_seq_len "$fixed_glm_context" 1024)"
    if [[ -n "${CTOX_VLLM_SERVE_DEVICE_LAYERS:-}" ]]; then
      printf '%s\n%s\n' "$fixed_glm_context" "${CTOX_VLLM_SERVE_DEVICE_LAYERS}"
      return 0
    fi
    # GPU 0 also hosts auxiliary services, so keep it very light. GLM's
    # non-layer tensors still land on the base runtime device, so keep both
    # GPU 0 and the middle/base device minimal and bias almost all repeating
    # layers onto the freest card.
    printf '%s\n%s\n' "$fixed_glm_context" "0:2;1:2;2:43"
    return 0
  fi

  local tune_mode="text"
  if is_qwen35_vision_model || is_stt_role; then
    tune_mode="vision"
  fi

  local tune_json=""
  if tune_json="$(run_tuner_json "$tune_mode" "$CTOX_VLLM_SERVE_MODEL" 2>/dev/null)"; then
    mapfile -t tuned_values < <(extract_tuned_runtime_values "$tune_json" "$requested_max_seq_len")
    local tuned_context="${tuned_values[0]:-0}"
    local tuned_device_layers="${tuned_values[1]:-}"
    if [[ "${tuned_context:-0}" -gt 0 ]]; then
      local safety_context
      safety_context="$(align_seq_len $(( tuned_context * 97 / 100 )) 1024)"
      printf '%s\n%s\n' "$safety_context" "$tuned_device_layers"
      return 0
    fi
  fi

  local fallback_requested="${requested_max_seq_len:-0}"
  if [[ "$fallback_requested" -gt 0 ]]; then
    printf '%s\n\n' "$(align_seq_len "$fallback_requested" 1024)"
  else
    printf '0\n\n'
  fi
}

resolve_vllm_serve_binary() {
  if [[ -n "${CTOX_VLLM_SERVE_BINARY:-}" && -x "${CTOX_VLLM_SERVE_BINARY:-}" ]]; then
    printf '%s\n' "$CTOX_VLLM_SERVE_BINARY"
    return
  fi
  if [[ -x "$ROOT/ctox-vllm-serve/target/release/mistralrs" ]]; then
    printf '%s\n' "$ROOT/ctox-vllm-serve/target/release/mistralrs"
    return
  fi
  if [[ -x "$HOME/.local/bin/vllm-serve" ]]; then
    printf '%s\n' "$HOME/.local/bin/vllm-serve"
    return
  fi
  command -v vllm-serve || command -v mistralrs || true
}

: "${CTOX_VLLM_SERVE_ROLE:=chat}"

apply_role_defaults() {
  case "${CTOX_VLLM_SERVE_ROLE:-chat}" in
    embedding)
      : "${CTOX_EMBEDDING_MODEL:=Qwen/Qwen3-Embedding-0.6B}"
      : "${CTOX_EMBEDDING_PORT:=1237}"
      : "${CTOX_EMBEDDING_ISQ:=Q4K}"
      CTOX_VLLM_SERVE_MODEL="${CTOX_VLLM_SERVE_MODEL_OVERRIDE:-$CTOX_EMBEDDING_MODEL}"
      CTOX_VLLM_SERVE_PORT="${CTOX_EMBEDDING_PORT}"
      CTOX_VLLM_SERVE_ARCH=""
      CTOX_VLLM_SERVE_MAX_SEQS="${CTOX_EMBEDDING_MAX_SEQS:-8}"
      CTOX_VLLM_SERVE_MAX_BATCH_SIZE="${CTOX_EMBEDDING_MAX_BATCH_SIZE:-8}"
      CTOX_VLLM_SERVE_MAX_SEQ_LEN="${CTOX_EMBEDDING_MAX_SEQ_LEN:-32768}"
      CTOX_VLLM_SERVE_PAGED_ATTN="${CTOX_EMBEDDING_PAGED_ATTN:-auto}"
      CTOX_VLLM_SERVE_PA_CACHE_TYPE="${CTOX_EMBEDDING_PA_CACHE_TYPE:-f8e4m3}"
      CTOX_VLLM_SERVE_PA_MEMORY_FRACTION="${CTOX_EMBEDDING_PA_MEMORY_FRACTION:-0.30}"
      CTOX_VLLM_SERVE_DISABLE_NCCL="${CTOX_EMBEDDING_DISABLE_NCCL:-1}"
      CTOX_VLLM_SERVE_ISQ="${CTOX_EMBEDDING_ISQ}"
      local embedding_devices=""
      embedding_devices="$(resolve_role_cuda_visible_devices_setting embedding)"
      if [[ -z "${CTOX_VLLM_SERVE_CUDA_VISIBLE_DEVICES:-}" && -n "$embedding_devices" ]]; then
        CTOX_VLLM_SERVE_CUDA_VISIBLE_DEVICES="$embedding_devices"
      fi
      ;;
    stt)
      : "${CTOX_STT_MODEL:=mistralai/Voxtral-Mini-4B-Realtime-2602}"
      : "${CTOX_STT_PORT:=1238}"
      : "${CTOX_STT_ISQ:=Q4K}"
      CTOX_VLLM_SERVE_MODEL="${CTOX_VLLM_SERVE_MODEL_OVERRIDE:-$CTOX_STT_MODEL}"
      CTOX_VLLM_SERVE_PORT="${CTOX_STT_PORT}"
      CTOX_VLLM_SERVE_ARCH=""
      CTOX_VLLM_SERVE_MAX_SEQS="${CTOX_STT_MAX_SEQS:-2}"
      CTOX_VLLM_SERVE_MAX_BATCH_SIZE="${CTOX_STT_MAX_BATCH_SIZE:-2}"
      CTOX_VLLM_SERVE_MAX_SEQ_LEN="${CTOX_STT_MAX_SEQ_LEN:-32768}"
      CTOX_VLLM_SERVE_PAGED_ATTN="${CTOX_STT_PAGED_ATTN:-auto}"
      CTOX_VLLM_SERVE_PA_CACHE_TYPE="${CTOX_STT_PA_CACHE_TYPE:-f8e4m3}"
      CTOX_VLLM_SERVE_PA_MEMORY_FRACTION="${CTOX_STT_PA_MEMORY_FRACTION:-0.55}"
      CTOX_VLLM_SERVE_DISABLE_NCCL="${CTOX_STT_DISABLE_NCCL:-1}"
      CTOX_VLLM_SERVE_ISQ="${CTOX_STT_ISQ}"
      local stt_devices=""
      stt_devices="$(resolve_role_cuda_visible_devices_setting stt)"
      if [[ -z "${CTOX_VLLM_SERVE_CUDA_VISIBLE_DEVICES:-}" && -n "$stt_devices" ]]; then
        CTOX_VLLM_SERVE_CUDA_VISIBLE_DEVICES="$stt_devices"
      fi
      ;;
    tts)
      : "${CTOX_TTS_MODEL:=Qwen/Qwen3-TTS-12Hz-0.6B-Base}"
      : "${CTOX_TTS_PORT:=1239}"
      : "${CTOX_TTS_ISQ:=Q4K}"
      CTOX_VLLM_SERVE_MODEL="${CTOX_VLLM_SERVE_MODEL_OVERRIDE:-$CTOX_TTS_MODEL}"
      CTOX_VLLM_SERVE_PORT="${CTOX_TTS_PORT}"
      CTOX_VLLM_SERVE_ARCH=""
      CTOX_VLLM_SERVE_MAX_SEQS="${CTOX_TTS_MAX_SEQS:-1}"
      CTOX_VLLM_SERVE_MAX_BATCH_SIZE="${CTOX_TTS_MAX_BATCH_SIZE:-1}"
      CTOX_VLLM_SERVE_PAGED_ATTN="${CTOX_TTS_PAGED_ATTN:-off}"
      CTOX_VLLM_SERVE_DISABLE_NCCL="${CTOX_TTS_DISABLE_NCCL:-1}"
      CTOX_VLLM_SERVE_ISQ="${CTOX_TTS_ISQ:-}"
      local tts_devices=""
      tts_devices="$(resolve_role_cuda_visible_devices_setting tts)"
      if [[ -z "${CTOX_VLLM_SERVE_CUDA_VISIBLE_DEVICES:-}" && -n "$tts_devices" ]]; then
        CTOX_VLLM_SERVE_CUDA_VISIBLE_DEVICES="$tts_devices"
      fi
      ;;
    chat|*)
      if [[ -n "${CTOX_VLLM_SERVE_MODEL_OVERRIDE:-}" ]]; then
        CTOX_VLLM_SERVE_MODEL="${CTOX_VLLM_SERVE_MODEL_OVERRIDE}"
      elif [[ -n "${CTOX_CHAT_MODEL:-}" ]]; then
        CTOX_VLLM_SERVE_MODEL="${CTOX_CHAT_MODEL}"
      elif [[ -n "${CTOX_ACTIVE_MODEL:-}" ]]; then
        CTOX_VLLM_SERVE_MODEL="${CTOX_ACTIVE_MODEL}"
      fi
      : "${CTOX_VLLM_SERVE_MODEL:=openai/gpt-oss-20b}"
      : "${CTOX_VLLM_SERVE_PORT:=1234}"
      if [[ -z "${CTOX_VLLM_SERVE_ARCH:-}" ]]; then
        if is_gpt_oss_model; then
          CTOX_VLLM_SERVE_ARCH="gpt_oss"
        else
          CTOX_VLLM_SERVE_ARCH=""
        fi
      fi
      : "${CTOX_VLLM_SERVE_MAX_SEQS:=1}"
      : "${CTOX_VLLM_SERVE_MAX_BATCH_SIZE:=1}"
      : "${CTOX_CHAT_SHARE_AUXILIARY_GPUS:=1}"
      if [[ -z "${CTOX_VLLM_SERVE_CUDA_VISIBLE_DEVICES:-}" ]]; then
        local all_devices=""
        all_devices="$(detect_all_visible_nvidia_devices || true)"
        if [[ -n "$all_devices" ]]; then
          if [[ "${CTOX_CHAT_SHARE_AUXILIARY_GPUS:-1}" == "0" && -n "${CTOX_AUXILIARY_CUDA_VISIBLE_DEVICES:-}" ]]; then
            CTOX_VLLM_SERVE_CUDA_VISIBLE_DEVICES="$(csv_without_items "$all_devices" "${CTOX_AUXILIARY_CUDA_VISIBLE_DEVICES:-}")"
          else
            CTOX_VLLM_SERVE_CUDA_VISIBLE_DEVICES="$all_devices"
          fi
        fi
      fi
      ;;
  esac
}

configure_compatible_cuda_runtime
apply_role_defaults

is_qwen35_vision_model() {
  case "${CTOX_VLLM_SERVE_MODEL:-}" in
    Qwen/Qwen3.5-*|Qwen/Qwen3.5-*)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

is_embedding_role() {
  [[ "${CTOX_VLLM_SERVE_ROLE:-chat}" == "embedding" ]]
}

is_stt_role() {
  [[ "${CTOX_VLLM_SERVE_ROLE:-chat}" == "stt" ]]
}

is_tts_role() {
  [[ "${CTOX_VLLM_SERVE_ROLE:-chat}" == "tts" ]]
}

is_qwen35_moe_model() {
  case "${CTOX_VLLM_SERVE_MODEL:-}" in
    Qwen/Qwen3.5-35B-A3B)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

if [[ "${CTOX_VLLM_SERVE_COMPUTE_TARGET:-gpu}" == "cpu" ]]; then
  unset CTOX_VLLM_SERVE_CUDA_VISIBLE_DEVICES || true
fi

if [[ "${CTOX_VLLM_SERVE_COMPUTE_TARGET:-gpu}" == "cpu" ]]; then
  unset CUDA_VISIBLE_DEVICES || true
elif [[ -n "${CTOX_VLLM_SERVE_CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$CTOX_VLLM_SERVE_CUDA_VISIBLE_DEVICES"
else
  auto_cuda_visible_devices="$(detect_all_visible_nvidia_devices || true)"
  if [[ -n "${auto_cuda_visible_devices:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="$auto_cuda_visible_devices"
  else
    unset CUDA_VISIBLE_DEVICES || true
  fi
fi

# Do not block proxy health checks behind a synchronous warmup request.
# The first real request still exercises the full model path.
export MISTRALRS_SKIP_DUMMY_RUN="${MISTRALRS_SKIP_DUMMY_RUN:-1}"

# Honor explicit planner/runtime flags first. The legacy heuristics below are
# only for unmanaged startup paths without an active chat runtime plan.
if [[ "${CTOX_VLLM_SERVE_DISABLE_FLASH_ATTN:-0}" == "1" ]]; then
  export MISTRALRS_DISABLE_FLASH_ATTN="${MISTRALRS_DISABLE_FLASH_ATTN:-1}"
elif [[ "${CTOX_CHAT_RUNTIME_PLAN_ACTIVE:-0}" == "1" ]]; then
  unset MISTRALRS_DISABLE_FLASH_ATTN || true
  # Qwen 3.5 on the multi-GPU layer-map path currently hits illegal CUDA accesses
  # in the flash-attention kernel path. Force the safe attention path for that profile.
elif ( is_qwen35_vision_model || is_stt_role ) && [[ "${CTOX_VLLM_SERVE_DISABLE_NCCL:-0}" == "1" ]]; then
  export MISTRALRS_DISABLE_FLASH_ATTN="${MISTRALRS_DISABLE_FLASH_ATTN:-1}"
else
  unset MISTRALRS_DISABLE_FLASH_ATTN || true
fi

if [[ "${CTOX_VLLM_SERVE_ISQ_SINGLETHREAD:-0}" == "1" ]]; then
  export MISTRALRS_ISQ_SINGLETHREAD="${MISTRALRS_ISQ_SINGLETHREAD:-1}"
  unset MISTRALRS_ISQ_CPU_THREADS || true
elif [[ -n "${CTOX_VLLM_SERVE_ISQ_CPU_THREADS:-}" ]]; then
  unset MISTRALRS_ISQ_SINGLETHREAD || true
  export MISTRALRS_ISQ_CPU_THREADS="${CTOX_VLLM_SERVE_ISQ_CPU_THREADS}"
elif [[ "${CTOX_CHAT_RUNTIME_PLAN_ACTIVE:-0}" == "1" ]]; then
  unset MISTRALRS_ISQ_SINGLETHREAD || true
  unset MISTRALRS_ISQ_CPU_THREADS || true
  # Qwen 3.5 MoE models can briefly spike VRAM during immediate ISQ if multiple
  # tensors quantize concurrently while the vision/text weights are still mapping
  # in. Force serialized immediate ISQ for that profile.
elif is_qwen35_moe_model; then
  export MISTRALRS_ISQ_SINGLETHREAD="${MISTRALRS_ISQ_SINGLETHREAD:-1}"
  unset MISTRALRS_ISQ_CPU_THREADS || true
elif is_glm47_flash_model; then
  unset MISTRALRS_ISQ_SINGLETHREAD || true
  export MISTRALRS_ISQ_CPU_THREADS="${MISTRALRS_ISQ_CPU_THREADS:-4}"
else
  unset MISTRALRS_ISQ_SINGLETHREAD || true
  unset MISTRALRS_ISQ_CPU_THREADS || true
fi

if [[ "${CTOX_VLLM_SERVE_NO_MMAP:-0}" == "1" ]]; then
  export MISTRALRS_NO_MMAP="${MISTRALRS_NO_MMAP:-1}"
elif [[ "${CTOX_CHAT_RUNTIME_PLAN_ACTIVE:-0}" == "1" ]]; then
  unset MISTRALRS_NO_MMAP || true
elif is_glm47_flash_model; then
  : "${CTOX_VLLM_SERVE_CUDA_VISIBLE_DEVICES:=0,2,1}"
  export CUDA_VISIBLE_DEVICES="$CTOX_VLLM_SERVE_CUDA_VISIBLE_DEVICES"
  export MISTRALRS_NO_MMAP="${MISTRALRS_NO_MMAP:-1}"
  export MISTRALRS_DISABLE_FLASH_ATTN="${MISTRALRS_DISABLE_FLASH_ATTN:-1}"
  CTOX_VLLM_SERVE_PAGED_ATTN="off"
  unset CTOX_VLLM_SERVE_PA_CACHE_TYPE || true
  unset CTOX_VLLM_SERVE_PA_CONTEXT_LEN || true
  unset CTOX_VLLM_SERVE_PA_MEMORY_FRACTION || true
else
  unset MISTRALRS_NO_MMAP || true
fi

if [[ "${CTOX_VLLM_SERVE_LANGUAGE_MODEL_ONLY:-0}" == "1" ]]; then
  export MISTRALRS_LANGUAGE_MODEL_ONLY="${MISTRALRS_LANGUAGE_MODEL_ONLY:-1}"
elif [[ "${CTOX_CHAT_RUNTIME_PLAN_ACTIVE:-0}" == "1" ]]; then
  unset MISTRALRS_LANGUAGE_MODEL_ONLY || true
else
  unset MISTRALRS_LANGUAGE_MODEL_ONLY || true
fi

if [[ -n "${CTOX_VLLM_SERVE_NM_DEVICE_ORDINAL:-}" ]]; then
  export MISTRALRS_NM_DEVICE_ORDINAL="${CTOX_VLLM_SERVE_NM_DEVICE_ORDINAL}"
else
  unset MISTRALRS_NM_DEVICE_ORDINAL || true
fi

if [[ -n "${CTOX_VLLM_SERVE_BASE_DEVICE_ORDINAL:-}" ]]; then
  export MISTRALRS_BASE_DEVICE_ORDINAL="${CTOX_VLLM_SERVE_BASE_DEVICE_ORDINAL}"
else
  unset MISTRALRS_BASE_DEVICE_ORDINAL || true
fi

if [[ -n "${CTOX_VLLM_SERVE_MOE_EXPERTS_BACKEND:-}" ]]; then
  export MISTRALRS_MOE_EXPERTS_BACKEND="${CTOX_VLLM_SERVE_MOE_EXPERTS_BACKEND}"
else
  unset MISTRALRS_MOE_EXPERTS_BACKEND || true
fi

VLLM_SERVE_BIN="$(resolve_vllm_serve_binary)"
if [[ -z "$VLLM_SERVE_BIN" ]]; then
  echo "vllm-serve binary not found; run scripts/install_ctox.sh first" >&2
  exit 1
fi

mapfile -t derived_runtime_values < <(derive_runtime_budget)
CTOX_VLLM_SERVE_REALIZED_MAX_SEQ_LEN="${derived_runtime_values[0]:-0}"
CTOX_VLLM_SERVE_DEVICE_LAYERS_CLI="${derived_runtime_values[1]:-}"
# Uneven multi-GPU sharing cannot be expressed through NCCL world size alone.
# When auxiliary backends share a chat GPU, convert the tuner's even layer map
# into a weighted one and force the chat runtime onto the explicit layer-map path.
if [[ "${CTOX_VLLM_SERVE_ROLE:-chat}" == "chat" \
  && "${CTOX_CHAT_RUNTIME_PLAN_ACTIVE:-0}" != "1" \
  && ! is_qwen35_moe_model \
  && -z "${CTOX_VLLM_SERVE_NUM_DEVICE_LAYERS:-}" \
  && -n "${CTOX_VLLM_SERVE_DEVICE_LAYERS_CLI:-}" ]]; then
  embedding_devices="$(resolve_role_cuda_visible_devices_setting embedding)"
  stt_devices="$(resolve_role_cuda_visible_devices_setting stt)"
  tts_devices="$(resolve_role_cuda_visible_devices_setting tts)"
  shared_auxiliary_devices="$(csv_intersection \
    "${CUDA_VISIBLE_DEVICES:-}" \
    "$(csv_union_items "$embedding_devices" "$stt_devices" "$tts_devices")")"
  if [[ -n "$shared_auxiliary_devices" ]]; then
    auxiliary_reservation_map="${CTOX_AUXILIARY_GPU_LAYER_RESERVATION_MAP:-}"
    if [[ -z "$auxiliary_reservation_map" ]]; then
      auxiliary_reservation_map="$(detect_live_auxiliary_gpu_layer_reservation_map)"
    fi
    rebalanced_device_layers="$(rebalance_device_layers_for_auxiliary_loads \
      "$CTOX_VLLM_SERVE_DEVICE_LAYERS_CLI" \
      "$embedding_devices" "${CTOX_EMBEDDING_GPU_LAYER_RESERVATION:-0.30}" \
      "$stt_devices" "${CTOX_STT_GPU_LAYER_RESERVATION:-0.55}" \
      "$tts_devices" "${CTOX_TTS_GPU_LAYER_RESERVATION:-0.35}" \
      "$auxiliary_reservation_map")"
    if [[ -n "$rebalanced_device_layers" && "$rebalanced_device_layers" != "$CTOX_VLLM_SERVE_DEVICE_LAYERS_CLI" ]]; then
      if [[ -n "$auxiliary_reservation_map" ]]; then
        echo "detected live auxiliary GPU reservation map: $auxiliary_reservation_map" >&2
      fi
      echo "rebalance chat device layers for shared auxiliary GPUs: $CTOX_VLLM_SERVE_DEVICE_LAYERS_CLI -> $rebalanced_device_layers" >&2
      CTOX_VLLM_SERVE_DEVICE_LAYERS_CLI="$rebalanced_device_layers"
      CTOX_VLLM_SERVE_DISABLE_NCCL=1
    fi
  fi
fi

if [[ "${CTOX_VLLM_SERVE_DISABLE_NCCL:-0}" == "1" ]] \
  || [[ "${CTOX_VLLM_SERVE_TENSOR_PARALLEL_BACKEND:-}" == "disabled" ]]; then
  export MISTRALRS_NO_NCCL=1
else
  unset MISTRALRS_NO_NCCL || true
fi

if [[ -n "${CTOX_VLLM_SERVE_MN_LOCAL_WORLD_SIZE:-}" ]] \
  && [[ "${CTOX_VLLM_SERVE_DISABLE_NCCL:-0}" != "1" ]] \
  && [[ "${CTOX_VLLM_SERVE_TENSOR_PARALLEL_BACKEND:-}" != "disabled" ]]; then
  export MISTRALRS_MN_LOCAL_WORLD_SIZE="$CTOX_VLLM_SERVE_MN_LOCAL_WORLD_SIZE"
elif [[ "${CTOX_VLLM_SERVE_DISABLE_NCCL:-0}" != "1" ]] \
  && [[ "${CTOX_VLLM_SERVE_TENSOR_PARALLEL_BACKEND:-}" != "disabled" ]] \
  && [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  auto_world_size="$(count_csv_items "$CUDA_VISIBLE_DEVICES")"
  if [[ "${auto_world_size:-0}" -gt 1 ]]; then
    export MISTRALRS_MN_LOCAL_WORLD_SIZE="$auto_world_size"
  else
    unset MISTRALRS_MN_LOCAL_WORLD_SIZE || true
  fi
else
  unset MISTRALRS_MN_LOCAL_WORLD_SIZE || true
fi

if [[ "${CTOX_VLLM_SERVE_ROLE:-chat}" == "chat" && "${CTOX_VLLM_SERVE_REALIZED_MAX_SEQ_LEN:-0}" -gt 0 ]]; then
  write_runtime_env_value "CTOX_VLLM_SERVE_REALIZED_MAX_SEQ_LEN" "$CTOX_VLLM_SERVE_REALIZED_MAX_SEQ_LEN"
  write_runtime_env_value "CTOX_CHAT_MODEL_REALIZED_CONTEXT" "$CTOX_VLLM_SERVE_REALIZED_MAX_SEQ_LEN"
  write_runtime_env_value "CTOX_VLLM_SERVE_REALIZED_MODEL" "$CTOX_VLLM_SERVE_MODEL"
fi

build_serve_command() {
  local effective_seq_len="$1"
  set -- "$VLLM_SERVE_BIN" serve -p "$CTOX_VLLM_SERVE_PORT"
  if is_tts_role && [[ -n "${CTOX_VLLM_SERVE_ISQ:-}" ]]; then
    set -- "$@" --isq "$CTOX_VLLM_SERVE_ISQ"
  fi
  if is_tts_role && [[ -n "${CTOX_VLLM_SERVE_ISQ_ORGANIZATION:-}" ]]; then
    set -- "$@" --isq-organization "$CTOX_VLLM_SERVE_ISQ_ORGANIZATION"
  fi
  if ! is_embedding_role; then
    if [[ -n "${CTOX_VLLM_SERVE_MAX_SEQS:-}" ]]; then
      set -- "$@" --max-seqs "$CTOX_VLLM_SERVE_MAX_SEQS"
    fi
    if is_qwen35_vision_model; then
      set -- "$@" --prefix-cache-n 0
    fi
  fi
  if is_embedding_role; then
    set -- "$@" embedding \
      -m "$CTOX_VLLM_SERVE_MODEL"
  elif is_tts_role; then
    set -- "$@" speech \
      -m "$CTOX_VLLM_SERVE_MODEL"
  elif is_qwen35_vision_model || is_stt_role; then
    if [[ -n "${CTOX_VLLM_SERVE_MAX_BATCH_SIZE:-}" ]]; then
      set -- "$@" --max-batch-size "$CTOX_VLLM_SERVE_MAX_BATCH_SIZE"
    fi
    set -- "$@" vision \
      -m "$CTOX_VLLM_SERVE_MODEL"
  else
    set -- "$@" \
      -m "$CTOX_VLLM_SERVE_MODEL"
    if [[ -n "${CTOX_VLLM_SERVE_MAX_BATCH_SIZE:-}" ]]; then
      set -- "$@" --max-batch-size "$CTOX_VLLM_SERVE_MAX_BATCH_SIZE"
    fi
    if [[ -n "${CTOX_VLLM_SERVE_ARCH:-}" ]]; then
      set -- "$@" -a "$CTOX_VLLM_SERVE_ARCH"
    fi
  fi

  if ! is_tts_role && [[ -n "${CTOX_VLLM_SERVE_PAGED_ATTN:-}" ]]; then
    set -- "$@" --paged-attn "$CTOX_VLLM_SERVE_PAGED_ATTN"
  fi
  if ! is_tts_role && [[ -n "${CTOX_VLLM_SERVE_PA_CACHE_TYPE:-}" ]]; then
    set -- "$@" --pa-cache-type "$CTOX_VLLM_SERVE_PA_CACHE_TYPE"
  fi
  if ! is_tts_role && [[ "${CTOX_VLLM_SERVE_PAGED_ATTN:-}" != "off" ]] && [[ "$effective_seq_len" -gt 0 ]]; then
    set -- "$@" --pa-context-len "$effective_seq_len"
  elif ! is_tts_role && [[ "${CTOX_VLLM_SERVE_PAGED_ATTN:-}" != "off" ]] && [[ -n "${CTOX_VLLM_SERVE_PA_CONTEXT_LEN:-}" ]]; then
    set -- "$@" --pa-context-len "$CTOX_VLLM_SERVE_PA_CONTEXT_LEN"
  elif ! is_tts_role && [[ "${CTOX_VLLM_SERVE_PAGED_ATTN:-}" != "off" ]] && [[ -n "${CTOX_VLLM_SERVE_PA_MEMORY_FRACTION:-}" ]]; then
    set -- "$@" --pa-memory-fraction "$CTOX_VLLM_SERVE_PA_MEMORY_FRACTION"
  fi

  local allow_layers_with_topology="${CTOX_VLLM_SERVE_ALLOW_DEVICE_LAYERS_WITH_TOPOLOGY:-0}"
  if ! is_tts_role && [[ ( -z "${CTOX_VLLM_SERVE_TOPOLOGY:-}" || "$allow_layers_with_topology" == "1" ) && -n "${CTOX_VLLM_SERVE_NUM_DEVICE_LAYERS:-}" ]]; then
    set -- "$@" --device-layers "$CTOX_VLLM_SERVE_NUM_DEVICE_LAYERS"
  elif ! is_tts_role && [[ ( -z "${CTOX_VLLM_SERVE_TOPOLOGY:-}" || "$allow_layers_with_topology" == "1" ) && -n "${CTOX_VLLM_SERVE_DEVICE_LAYERS:-}" ]]; then
    set -- "$@" --device-layers "$CTOX_VLLM_SERVE_DEVICE_LAYERS"
  elif ! is_tts_role && [[ ( -z "${CTOX_VLLM_SERVE_TOPOLOGY:-}" || "$allow_layers_with_topology" == "1" ) && -n "${CTOX_VLLM_SERVE_DEVICE_LAYERS_CLI:-}" && "${CTOX_VLLM_SERVE_DISABLE_NCCL:-0}" == "1" ]]; then
    set -- "$@" --device-layers "$CTOX_VLLM_SERVE_DEVICE_LAYERS_CLI"
  fi

  if [[ -n "${CTOX_VLLM_SERVE_TOPOLOGY:-}" ]]; then
    set -- "$@" --topology "$CTOX_VLLM_SERVE_TOPOLOGY"
  fi
  if [[ "$effective_seq_len" -gt 0 ]]; then
    set -- "$@" --max-seq-len "$effective_seq_len"
  elif [[ -n "${CTOX_VLLM_SERVE_MAX_SEQ_LEN:-}" ]]; then
    set -- "$@" --max-seq-len "$CTOX_VLLM_SERVE_MAX_SEQ_LEN"
  fi
  if ! is_tts_role && [[ -n "${CTOX_VLLM_SERVE_ISQ:-}" ]]; then
    set -- "$@" --isq "$CTOX_VLLM_SERVE_ISQ"
  fi
  if ! is_tts_role && [[ -n "${CTOX_VLLM_SERVE_ISQ_ORGANIZATION:-}" ]]; then
    set -- "$@" --isq-organization "$CTOX_VLLM_SERVE_ISQ_ORGANIZATION"
  fi

  printf '%s\0' "$@"
}

run_serve_command() {
  local -a serve_cmd=("$@")
  if is_glm47_flash_model; then
    local public_hf_home="$ROOT/runtime/hf_public"
    mkdir -p "$public_hf_home"
    mkdir -p "$public_hf_home/home"
    mkdir -p "$public_hf_home/hub"
    mkdir -p "$public_hf_home/xdg"
    rm -rf "$public_hf_home/token" "$public_hf_home/stored_tokens"
    if HF_TOKEN= \
      HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
      HF_HOME="$public_hf_home" \
      HUGGINGFACE_HUB_CACHE="$public_hf_home/hub" \
      HF_TOKEN_PATH="$public_hf_home/token" \
      HF_STORED_TOKENS_PATH="$public_hf_home/stored_tokens" \
      XDG_CACHE_HOME="$public_hf_home/xdg" \
      HOME="$public_hf_home/home" \
      "${serve_cmd[@]}"; then
      exit_code=0
    else
      exit_code=$?
    fi
    return "$exit_code"
  else
    "${serve_cmd[@]}"
  fi
}

effective_seq_len="${CTOX_VLLM_SERVE_REALIZED_MAX_SEQ_LEN:-0}"
attempt=0
while :; do
  attempt=$((attempt + 1))
  if [[ "${CTOX_VLLM_SERVE_ROLE:-chat}" == "chat" && "${effective_seq_len:-0}" -gt 0 ]]; then
    write_runtime_env_value "CTOX_VLLM_SERVE_REALIZED_MAX_SEQ_LEN" "$effective_seq_len"
    write_runtime_env_value "CTOX_CHAT_MODEL_REALIZED_CONTEXT" "$effective_seq_len"
    write_runtime_env_value "CTOX_VLLM_SERVE_REALIZED_MODEL" "$CTOX_VLLM_SERVE_MODEL"
  elif [[ "${CTOX_VLLM_SERVE_ROLE:-chat}" == "chat" ]]; then
    clear_runtime_env_value "CTOX_VLLM_SERVE_REALIZED_MAX_SEQ_LEN"
    clear_runtime_env_value "CTOX_CHAT_MODEL_REALIZED_CONTEXT"
    clear_runtime_env_value "CTOX_VLLM_SERVE_REALIZED_MODEL"
  fi

  if [[ "${effective_seq_len:-0}" -gt 0 ]]; then
    export MISTRALRS_MAX_SEQ_LEN_OVERRIDE="$effective_seq_len"
  elif [[ -n "${CTOX_VLLM_SERVE_MAX_SEQ_LEN:-}" ]]; then
    export MISTRALRS_MAX_SEQ_LEN_OVERRIDE="$CTOX_VLLM_SERVE_MAX_SEQ_LEN"
  else
    unset MISTRALRS_MAX_SEQ_LEN_OVERRIDE || true
  fi

  mapfile -d '' -t serve_cmd < <(build_serve_command "${effective_seq_len:-0}")
  echo "ctox backend env: attempt=${attempt} role=${CTOX_VLLM_SERVE_ROLE:-chat} model=${CTOX_VLLM_SERVE_MODEL:-} port=${CTOX_VLLM_SERVE_PORT:-} arch=${CTOX_VLLM_SERVE_ARCH:-} cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-} paged_attn=${CTOX_VLLM_SERVE_PAGED_ATTN:-} pa_cache_type=${CTOX_VLLM_SERVE_PA_CACHE_TYPE:-} pa_memory_fraction=${CTOX_VLLM_SERVE_PA_MEMORY_FRACTION:-} max_seq_len=${CTOX_VLLM_SERVE_MAX_SEQ_LEN:-} realized_seq_len=${effective_seq_len:-0} runtime_seq_override=${MISTRALRS_MAX_SEQ_LEN_OVERRIDE:-} isq=${CTOX_VLLM_SERVE_ISQ:-} disable_nccl=${CTOX_VLLM_SERVE_DISABLE_NCCL:-} no_mmap=${MISTRALRS_NO_MMAP:-} language_model_only=${MISTRALRS_LANGUAGE_MODEL_ONLY:-} disable_flash_attn=${MISTRALRS_DISABLE_FLASH_ATTN:-} isq_singlethread=${MISTRALRS_ISQ_SINGLETHREAD:-} isq_cpu_threads=${MISTRALRS_ISQ_CPU_THREADS:-} nm_device=${MISTRALRS_NM_DEVICE_ORDINAL:-} base_device=${MISTRALRS_BASE_DEVICE_ORDINAL:-} moe_backend=${MISTRALRS_MOE_EXPERTS_BACKEND:-}" >&2
  printf 'ctox backend cmd:'
  printf ' %q' "${serve_cmd[@]}"
  printf '\n' >&2
  started_at="$(date +%s)"
  if run_serve_command "${serve_cmd[@]}"; then
    exit_code=0
  else
    exit_code=$?
  fi
  runtime_s=$(( $(date +%s) - started_at ))

  if [[ $exit_code -eq 0 ]]; then
    exit 0
  fi
  if [[ $attempt -ge 5 || "${effective_seq_len:-0}" -le 4096 || $runtime_s -gt 30 ]]; then
    exit "$exit_code"
  fi

  effective_seq_len="$(compute_retry_seq_len "$effective_seq_len" 4096)"
  echo "retrying ${CTOX_VLLM_SERVE_MODEL} with reduced realized context ${effective_seq_len}" >&2
done
