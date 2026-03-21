#!/bin/sh
set -eu

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$ROOT"

TOOLCHAIN_BIN="$HOME/.rustup/toolchains/stable-aarch64-apple-darwin/bin"

if [ -x "$TOOLCHAIN_BIN/cargo" ] && [ -x "$TOOLCHAIN_BIN/rustc" ]; then
  export PATH="$TOOLCHAIN_BIN:$PATH"
fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

if [ "$(uname -s)" = "Linux" ]; then
  export DEBIAN_FRONTEND="${DEBIAN_FRONTEND:-noninteractive}"
  export NEEDRESTART_MODE="${NEEDRESTART_MODE:-a}"
  export APT_LISTCHANGES_FRONTEND="${APT_LISTCHANGES_FRONTEND:-none}"
fi

resolve_kleinhirn_profile() {
  if [ -n "${CTO_AGENT_KLEINHIRN_PROFILE:-}" ]; then
    case "$(printf '%s' "$CTO_AGENT_KLEINHIRN_PROFILE" | tr '[:upper:]' '[:lower:]')" in
      qwen3|qwen|qwen3-30b-a3b|qwen/qwen3-30b-a3b|qwen35|qwen3.5|qwen3.5-35b-a3b|qwen35-35b-a3b)
        printf '%s\n' "qwen35"
        return
        ;;
      *)
        printf '%s\n' "gpt_oss"
        return
        ;;
    esac
  fi

  case "$(printf '%s' "${CTO_AGENT_KLEINHIRN_MODEL:-}" | tr '[:upper:]' '[:lower:]')" in
    *qwen3-30b-a3b*|qwen/qwen3-30b-a3b|*qwen3.5-35b-a3b*|qwen/qwen3.5-35b-a3b)
      printf '%s\n' "qwen35"
      ;;
    *)
      printf '%s\n' "gpt_oss"
      ;;
  esac
}

is_gpt_oss_family() {
  case "$(printf '%s\n%s' "${KLEINHIRN_POLICY_MODEL:-}" "${KLEINHIRN_RUNTIME_MODEL:-}" | tr '[:upper:]' '[:lower:]')" in
    *gpt-oss*) return 0 ;;
    *) return 1 ;;
  esac
}

REQUESTED_KLEINHIRN_PROFILE="$(resolve_kleinhirn_profile)"
KLEINHIRN_PROFILE="$REQUESTED_KLEINHIRN_PROFILE"
PROFILE_PINNED=1

selected_matches_profile() {
  case "$KLEINHIRN_PROFILE" in
    qwen35)
      case "$(printf '%s' "${SELECTED_POLICY_MODEL:-}" | tr '[:upper:]' '[:lower:]')" in
        *qwen*) return 0 ;;
      esac
      return 1
      ;;
    *)
      case "$(printf '%s' "${SELECTED_POLICY_MODEL:-}" | tr '[:upper:]' '[:lower:]')" in
        *gpt-oss*) return 0 ;;
      esac
      return 1
      ;;
  esac
}

case "$KLEINHIRN_PROFILE" in
  qwen35)
    KLEINHIRN_POLICY_MODEL="${CTO_AGENT_KLEINHIRN_POLICY_MODEL:-gpt-oss-20b}"
    KLEINHIRN_RUNTIME_MODEL="${CTO_AGENT_KLEINHIRN_RUNTIME_MODEL:-openai/gpt-oss-20b}"
    KLEINHIRN_OFFICIAL_LABEL="${CTO_AGENT_KLEINHIRN_OFFICIAL_LABEL:-GPT-OSS 20B}"
    KLEINHIRN_AGENTIC_ADAPTER="${CTO_AGENT_KLEINHIRN_AGENTIC_ADAPTER:-mistralrs_gpt_oss_harmony_completion}"
    KLEINHIRN_MAX_SEQ_LEN="${CTO_AGENT_KLEINHIRN_MAX_SEQ_LEN:-131072}"
    KLEINHIRN_DISABLE_PAGED_ATTN="${CTO_AGENT_KLEINHIRN_DISABLE_PAGED_ATTN:-0}"
    KLEINHIRN_PAGED_ATTN_MODE="${CTO_AGENT_KLEINHIRN_PAGED_ATTN_MODE:-off}"
    ;;
  *)
    KLEINHIRN_POLICY_MODEL="${CTO_AGENT_KLEINHIRN_POLICY_MODEL:-gpt-oss-20b}"
    KLEINHIRN_RUNTIME_MODEL="${CTO_AGENT_KLEINHIRN_RUNTIME_MODEL:-openai/gpt-oss-20b}"
    KLEINHIRN_OFFICIAL_LABEL="${CTO_AGENT_KLEINHIRN_OFFICIAL_LABEL:-GPT-OSS 20B}"
    KLEINHIRN_AGENTIC_ADAPTER="${CTO_AGENT_KLEINHIRN_AGENTIC_ADAPTER:-mistralrs_gpt_oss_harmony_completion}"
    KLEINHIRN_MAX_SEQ_LEN="${CTO_AGENT_KLEINHIRN_MAX_SEQ_LEN:-131072}"
    KLEINHIRN_DISABLE_PAGED_ATTN="${CTO_AGENT_KLEINHIRN_DISABLE_PAGED_ATTN:-0}"
    KLEINHIRN_PAGED_ATTN_MODE="${CTO_AGENT_KLEINHIRN_PAGED_ATTN_MODE:-off}"
    ;;
esac

if is_gpt_oss_family; then
  KLEINHIRN_ARCH="${CTO_AGENT_KLEINHIRN_ARCH:-gpt_oss}"
else
  KLEINHIRN_ARCH="${CTO_AGENT_KLEINHIRN_ARCH:-}"
fi
KLEINHIRN_NUM_DEVICE_LAYERS="${CTO_AGENT_KLEINHIRN_NUM_DEVICE_LAYERS:-}"
KLEINHIRN_MAX_SEQS="${CTO_AGENT_KLEINHIRN_MAX_SEQS:-}"
KLEINHIRN_MAX_BATCH_SIZE="${CTO_AGENT_KLEINHIRN_MAX_BATCH_SIZE:-}"
KLEINHIRN_PA_GPU_MEM="${CTO_AGENT_KLEINHIRN_PA_GPU_MEM:-}"
KLEINHIRN_PA_GPU_MEM_USAGE="${CTO_AGENT_KLEINHIRN_PA_GPU_MEM_USAGE:-}"
KLEINHIRN_PA_CTXT_LEN="${CTO_AGENT_KLEINHIRN_PA_CTXT_LEN:-${KLEINHIRN_PA_CTXT_LEN:-}}"
KLEINHIRN_PA_CACHE_TYPE="${CTO_AGENT_KLEINHIRN_PA_CACHE_TYPE:-${KLEINHIRN_PA_CACHE_TYPE:-}}"
KLEINHIRN_PAGED_ATTN_MODE="${CTO_AGENT_KLEINHIRN_PAGED_ATTN_MODE:-${KLEINHIRN_PAGED_ATTN_MODE:-}}"
KLEINHIRN_DEVICE_LAYERS="${CTO_AGENT_KLEINHIRN_DEVICE_LAYERS:-}"
KLEINHIRN_ISQ="${CTO_AGENT_KLEINHIRN_ISQ:-}"
KLEINHIRN_CHAT_TEMPLATE="${CTO_AGENT_KLEINHIRN_CHAT_TEMPLATE:-}"
KLEINHIRN_JINJA_EXPLICIT="${CTO_AGENT_KLEINHIRN_JINJA_EXPLICIT:-}"
KLEINHIRN_TOKENIZER_JSON="${CTO_AGENT_KLEINHIRN_TOKENIZER_JSON:-}"
KLEINHIRN_TOPOLOGY="${CTO_AGENT_KLEINHIRN_TOPOLOGY:-}"
KLEINHIRN_CUDA_VISIBLE_DEVICES="${CTO_AGENT_KLEINHIRN_CUDA_VISIBLE_DEVICES:-}"
KLEINHIRN_MULTI_GPU_MODE="${CTO_AGENT_KLEINHIRN_MULTI_GPU_MODE:-}"
KLEINHIRN_TENSOR_PARALLEL_BACKEND="${CTO_AGENT_KLEINHIRN_TENSOR_PARALLEL_BACKEND:-}"
KLEINHIRN_VISIBLE_GPU_POLICY="${CTO_AGENT_KLEINHIRN_VISIBLE_GPU_POLICY:-}"
KLEINHIRN_MN_LOCAL_WORLD_SIZE="${CTO_AGENT_KLEINHIRN_MN_LOCAL_WORLD_SIZE:-}"
KLEINHIRN_DISABLE_NCCL="${CTO_AGENT_KLEINHIRN_DISABLE_NCCL:-}"

KLEINHIRN_PORT="${CTO_AGENT_KLEINHIRN_PORT:-1234}"
KLEINHIRN_STARTUP_WAIT_SECS="${CTO_AGENT_KLEINHIRN_STARTUP_WAIT_SECS:-900}"
KLEINHIRN_BASE_URL="http://127.0.0.1:${KLEINHIRN_PORT}/v1"
CTO_AGENT_PORT="${CTO_AGENT_PORT:-8443}"
CTO_AGENT_BIND_HOST="${CTO_AGENT_BIND_HOST:-}"
CTO_AGENT_PUBLIC_BASE_URL="${CTO_AGENT_PUBLIC_BASE_URL:-}"
CTO_AGENT_TLS_ALT_NAMES="${CTO_AGENT_TLS_ALT_NAMES:-}"
CONTEXT_EMBEDDING_ENABLED="${CTO_AGENT_CONTEXT_EMBEDDING_ENABLED:-1}"
CONTEXT_EMBEDDING_MODEL="${CTO_AGENT_CONTEXT_EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-0.6B}"
CONTEXT_EMBEDDING_RUNTIME_MODEL="${CTO_AGENT_CONTEXT_EMBEDDING_RUNTIME_MODEL:-$CONTEXT_EMBEDDING_MODEL}"
CONTEXT_EMBEDDING_PORT="${CTO_AGENT_CONTEXT_EMBEDDING_PORT:-1235}"
CONTEXT_EMBEDDING_BASE_URL="${CTO_AGENT_CONTEXT_EMBEDDING_BASE_URL:-http://127.0.0.1:${CONTEXT_EMBEDDING_PORT}/v1}"
CONTEXT_EMBEDDING_API_KEY="${CTO_AGENT_CONTEXT_EMBEDDING_API_KEY:-local-context-embedding}"
CONTEXT_EMBEDDING_MAX_BATCH_SIZE="${CTO_AGENT_CONTEXT_EMBEDDING_MAX_BATCH_SIZE:-12}"
CONTEXT_EMBEDDING_CHUNK_CHARS="${CTO_AGENT_CONTEXT_EMBEDDING_CHUNK_CHARS:-900}"
CONTEXT_EMBEDDING_CHUNK_OVERLAP="${CTO_AGENT_CONTEXT_EMBEDDING_CHUNK_OVERLAP:-120}"
KLEINHIRN_LOG_DIR="$ROOT/runtime/logs"
KLEINHIRN_LOG="$KLEINHIRN_LOG_DIR/kleinhirn.log"
KLEINHIRN_PID_FILE="$ROOT/runtime/kleinhirn.pid"
ENV_FILE="$ROOT/runtime/kleinhirn.env"
HEALTH_URL="https://127.0.0.1:${CTO_AGENT_PORT}/healthz"
READY_URL="https://127.0.0.1:${CTO_AGENT_PORT}/readyz"
HEARTBEAT_FILE="$ROOT/runtime/state/agent_state.json"
CARGO_TARGET_DIR="${CTO_AGENT_MISTRALRS_TARGET_DIR:-$ROOT/runtime/build/mistralrs}"
CTO_EMAIL_ADDRESS="${CTO_EMAIL_ADDRESS:-}"
CTO_EMAIL_PASSWORD="${CTO_EMAIL_PASSWORD:-}"
CTO_EMAIL_IMAP_HOST="${CTO_EMAIL_IMAP_HOST:-}"
CTO_EMAIL_IMAP_PORT="${CTO_EMAIL_IMAP_PORT:-}"
CTO_EMAIL_SMTP_HOST="${CTO_EMAIL_SMTP_HOST:-}"
CTO_EMAIL_SMTP_PORT="${CTO_EMAIL_SMTP_PORT:-}"
CTO_AGENT_EMAIL_SYNC_LIMIT="${CTO_AGENT_EMAIL_SYNC_LIMIT:-}"
CTO_JAMI_ACCOUNT_ID="${CTO_JAMI_ACCOUNT_ID:-}"
CTO_JAMI_PROFILE_NAME="${CTO_JAMI_PROFILE_NAME:-}"
CTO_JAMI_INBOX_DIR="${CTO_JAMI_INBOX_DIR:-}"
CTO_JAMI_OUTBOX_DIR="${CTO_JAMI_OUTBOX_DIR:-}"
CTO_JAMI_ARCHIVE_DIR="${CTO_JAMI_ARCHIVE_DIR:-}"
CTO_JAMI_DAEMON_ARGS="${CTO_JAMI_DAEMON_ARGS:-}"
CTO_AGENT_INSTALL_JAMI_GUI="${CTO_AGENT_INSTALL_JAMI_GUI:-0}"
CTO_AGENT_GROSSHIRN_API_KEY="${CTO_AGENT_GROSSHIRN_API_KEY:-}"
CTO_AGENT_GROSSHIRN_MODEL="${CTO_AGENT_GROSSHIRN_MODEL:-}"
CTO_AGENT_GROSSHIRN_AGENTIC_ADAPTER="${CTO_AGENT_GROSSHIRN_AGENTIC_ADAPTER:-}"
CTO_AGENT_GROSSHIRN_BASE_URL="${CTO_AGENT_GROSSHIRN_BASE_URL:-}"
CTO_AGENT_GROSSHIRN_REASONING="${CTO_AGENT_GROSSHIRN_REASONING:-}"
CTO_AGENT_COMPACT_SIMPLE_MODEL="${CTO_AGENT_COMPACT_SIMPLE_MODEL:-}"
CTO_AGENT_COMPACT_MEDIUM_MODEL="${CTO_AGENT_COMPACT_MEDIUM_MODEL:-}"
CTO_AGENT_COMPACT_RED_MODEL="${CTO_AGENT_COMPACT_RED_MODEL:-}"

detect_public_control_plane_host() {
  if command -v tailscale >/dev/null 2>&1; then
    tailscale ip -4 2>/dev/null | awk 'NF { print; exit }'
    return
  fi
  if command -v hostname >/dev/null 2>&1; then
    hostname -I 2>/dev/null | tr ' ' '\n' | awk '
      NF && $1 ~ /^100\./ { print; exit }
      NF && $1 !~ /^127\./ && preferred == "" { preferred = $1 }
      END {
        if (preferred != "") {
          print preferred;
        }
      }'
    return
  fi
  if command -v ip >/dev/null 2>&1; then
    ip -o -4 addr show scope global 2>/dev/null | awk '{
      split($4, address, "/");
      if (address[1] ~ /^100\./) {
        print address[1];
        exit;
      }
      if (address[1] !~ /^127\./ && preferred == "") {
        preferred = address[1];
      }
    } END {
      if (preferred != "") {
        print preferred;
      }
    }'
    return
  fi
}

detect_gpu_count() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | awk 'NF {count += 1} END {print count + 0}')"
    if [ "${gpu_count:-0}" -gt 0 ]; then
      printf '%s\n' "$gpu_count"
      return
    fi
  fi
  if command -v lspci >/dev/null 2>&1; then
    lspci -nn 2>/dev/null | grep -Ei 'VGA compatible controller|3D controller|Display controller' | grep -ci 'NVIDIA'
    return
  fi
  printf '%s\n' "0"
}

nccl_packages_available() {
  if ! command -v apt-cache >/dev/null 2>&1; then
    return 1
  fi
  apt-cache policy libnccl2 2>/dev/null | grep -q 'Candidate:'
}

nccl_runtime_missing() {
  if command -v ldconfig >/dev/null 2>&1 && ldconfig -p 2>/dev/null | grep -q 'libnccl'; then
    return 1
  fi
  return 0
}

detect_mistralrs_features() {
  if [ -n "${CTO_AGENT_MISTRALRS_FEATURES:-}" ]; then
    printf '%s\n' "$CTO_AGENT_MISTRALRS_FEATURES"
    return
  fi

  features="cuda flash-attn"
  if command -v ldconfig >/dev/null 2>&1 && ldconfig -p 2>/dev/null | grep -q 'libnccl'; then
    features="$features nccl"
  fi
  if command -v ldconfig >/dev/null 2>&1 && ldconfig -p 2>/dev/null | grep -q 'libcudnn'; then
    features="$features cudnn"
  fi
  printf '%s\n' "$features"
}

MISTRALRS_FEATURES="$(detect_mistralrs_features)"

installed_mistralrs_features() {
  if [ -n "${CTO_AGENT_MISTRALRS_FEATURES:-}" ]; then
    printf '%s\n' "$CTO_AGENT_MISTRALRS_FEATURES"
    return
  fi

  cargo_home="${CARGO_HOME:-$HOME/.cargo}"
  crates_json="$cargo_home/.crates2.json"
  if [ ! -f "$crates_json" ]; then
    return
  fi

  CRATES_JSON_PATH="$crates_json" python3 - <<'PY'
import json
import os
import pathlib
import sys

path = pathlib.Path(os.environ["CRATES_JSON_PATH"])
try:
    installs = json.loads(path.read_text()).get("installs", {})
except Exception:
    raise SystemExit(0)

selected = None
for key, value in installs.items():
    if key.startswith("mistralrs-cli"):
        selected = value

if not isinstance(selected, dict):
    raise SystemExit(0)

features = selected.get("features")
if not isinstance(features, list):
    raise SystemExit(0)

items = []
for feature in features:
    if isinstance(feature, str):
        feature = feature.strip()
        if feature and feature not in items:
            items.append(feature)

if items:
    print(" ".join(items))
PY
}

mistralrs_features_satisfy_required() {
  required="$1"
  installed="$(installed_mistralrs_features || true)"
  if [ -z "$installed" ]; then
    return 1
  fi
  OLD_IFS="${IFS:- }"
  IFS=' '
  for feature in $required; do
    case "$installed" in
      *"$feature"*) ;;
      *)
        IFS="$OLD_IFS"
        return 1
        ;;
    esac
  done
  IFS="$OLD_IFS"
  return 0
}

apply_runtime_tune_defaults() {
  if ! command -v mistralrs >/dev/null 2>&1; then
    return
  fi

  TUNE_STDOUT="$(mistralrs tune -m "$KLEINHIRN_RUNTIME_MODEL" --json 2>"$ROOT/runtime/logs/mistralrs_tune.err" || true)"
  if [ -z "$TUNE_STDOUT" ]; then
    return
  fi

  TUNE_JSON="$TUNE_STDOUT" python3 - <<'PY'
import json
import os
import shlex
import sys

raw = os.environ.get("TUNE_JSON", "").strip()
if not raw:
    raise SystemExit(0)
try:
    data = json.loads(raw)
except Exception:
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or start >= end:
        raise SystemExit(0)
    try:
        data = json.loads(raw[start:end + 1])
    except Exception:
        raise SystemExit(0)

recommended_isq = data.get("recommended_isq") or ""
device_layers_cli = data.get("device_layers_cli") or ""
recommended_candidate = next(
    (candidate for candidate in data.get("candidates", []) if candidate.get("recommended")),
    None,
)
max_context_tokens = (
    (recommended_candidate or {}).get("max_context_tokens")
    or data.get("max_context_tokens")
    or ""
)

for key, value in [
    ("RECOMMENDED_ISQ", recommended_isq),
    ("RECOMMENDED_DEVICE_LAYERS", device_layers_cli),
    ("RECOMMENDED_MAX_CONTEXT_TOKENS", str(max_context_tokens or "")),
]:
    print(f"{key}={shlex.quote(value)}")
PY
}

select_recommended_kleinhirn_env() {
  SELECTED_JSON="$("$ROOT/target/release/cto-agent" recommend-kleinhirn 2>/dev/null || true)"
  if [ -z "$SELECTED_JSON" ]; then
    return
  fi

  SELECTED_JSON="$SELECTED_JSON" python3 - <<'PY'
import json
import os
import shlex

raw = os.environ.get("SELECTED_JSON", "").strip()
if not raw:
    raise SystemExit(0)

selected = json.loads(raw)

for key, value in [
    ("SELECTED_POLICY_MODEL", selected.get("modelId") or ""),
    ("SELECTED_RUNTIME_MODEL", selected.get("runtimeModelId") or selected.get("modelId") or ""),
    ("SELECTED_OFFICIAL_LABEL", selected.get("officialLabel") or ""),
    ("SELECTED_AGENTIC_ADAPTER", selected.get("agenticAdapter") or ""),
    ("SELECTED_MAX_SEQS", str(selected.get("startupMaxSeqs") or "")),
    ("SELECTED_MAX_BATCH_SIZE", str(selected.get("startupMaxBatchSize") or "")),
    ("SELECTED_MAX_SEQ_LEN", str(selected.get("startupMaxSeqLen") or "")),
    ("SELECTED_PA_CTXT_LEN", str(selected.get("startupPaContextLen") or "")),
    ("SELECTED_PA_CACHE_TYPE", selected.get("startupPaCacheType") or ""),
    ("SELECTED_PAGED_ATTN_MODE", selected.get("startupPagedAttnMode") or ""),
    ("SELECTED_CHAT_TEMPLATE", selected.get("startupChatTemplatePath") or ""),
    ("SELECTED_JINJA_EXPLICIT", selected.get("startupJinjaExplicitPath") or ""),
    ("SELECTED_TOKENIZER_JSON", selected.get("startupTokenizerJsonPath") or ""),
    ("SELECTED_TOPOLOGY", selected.get("startupTopologyPath") or ""),
    ("SELECTED_MULTI_GPU_MODE", selected.get("startupMultiGpuMode") or ""),
    ("SELECTED_TENSOR_PARALLEL_BACKEND", selected.get("startupTensorParallelBackend") or ""),
    ("SELECTED_VISIBLE_GPU_POLICY", selected.get("startupVisibleGpuPolicy") or ""),
    ("SELECTED_PREFER_AUTO_DEVICE_MAPPING", "1" if selected.get("preferAutoDeviceMapping") else "0"),
]:
    print(f"{key}={shlex.quote(value)}")
PY
}

load_install_census_hints() {
  if [ ! -f /tmp/cto_system_census.json ]; then
    return
  fi

  python3 - <<'PY'
import json
import pathlib
import shlex

path = pathlib.Path("/tmp/cto_system_census.json")
try:
    census = json.loads(path.read_text())
except Exception:
    raise SystemExit(0)

for key, value in [
    ("CENSUS_GPU_COUNT", str(census.get("gpuCount") or "")),
    ("CENSUS_TOTAL_GPU_MEMORY_GB", str(census.get("totalGpuMemoryGb") or "")),
    ("CENSUS_MAX_SINGLE_GPU_MEMORY_GB", str(census.get("maxSingleGpuMemoryGb") or "")),
]:
    print(f"{key}={shlex.quote(value)}")
PY
}

shell_quote() {
  printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"
}

normalize_compact_model_value() {
  value="$(printf '%s' "$1" | sed 's#^[[:space:]]*##; s#[[:space:]]*$##')"
  case "$(printf '%s' "$value" | tr '[:upper:]' '[:lower:]')" in
    gpt-oss-20b|openai/gpt-oss-20b)
      printf '%s\n' "openai/gpt-oss-20b"
      ;;
    gpt-5.4-nano|openai/gpt-5.4-nano)
      printf '%s\n' "openai/gpt-5.4-nano"
      ;;
    gpt-5.4-mini|openai/gpt-5.4-mini)
      printf '%s\n' "openai/gpt-5.4-mini"
      ;;
    gpt-5.4|openai/gpt-5.4)
      printf '%s\n' "openai/gpt-5.4"
      ;;
    *)
      printf '%s\n' "$value"
      ;;
  esac
}

read_runtime_env_file_value() {
  key="$1"
  if [ ! -f "$ENV_FILE" ]; then
    return 0
  fi
  ENV_FILE_PATH="$ENV_FILE" ENV_KEY="$key" sh -c '
    [ -f "$ENV_FILE_PATH" ] || exit 0
    # shellcheck disable=SC1090
    . "$ENV_FILE_PATH"
    eval "printf %s \"\${$ENV_KEY-}\""
  '
}

discover_local_jami_account_id() {
  config_file="${CTO_JAMI_DRING_PATH:-${XDG_CONFIG_HOME:-$HOME/.config}/jami/dring.yml}"
  if [ ! -f "$config_file" ]; then
    return 0
  fi
  awk '
    /^[[:space:]]*id:[[:space:]]*/ {
      value=$0
      sub(/^[[:space:]]*id:[[:space:]]*/, "", value)
      gsub(/["'"'"']/, "", value)
      gsub(/[[:space:]]+$/, "", value)
      if (length(value) >= 16) {
        print value
        exit
      }
    }
  ' "$config_file"
}

adopt_existing_runtime_communication_env() {
  existing_address="$(read_runtime_env_file_value CTO_EMAIL_ADDRESS || true)"
  existing_password="$(read_runtime_env_file_value CTO_EMAIL_PASSWORD || true)"
  existing_imap_host="$(read_runtime_env_file_value CTO_EMAIL_IMAP_HOST || true)"
  existing_imap_port="$(read_runtime_env_file_value CTO_EMAIL_IMAP_PORT || true)"
  existing_smtp_host="$(read_runtime_env_file_value CTO_EMAIL_SMTP_HOST || true)"
  existing_smtp_port="$(read_runtime_env_file_value CTO_EMAIL_SMTP_PORT || true)"
  existing_sync_limit="$(read_runtime_env_file_value CTO_AGENT_EMAIL_SYNC_LIMIT || true)"
  existing_jami_account_id="$(read_runtime_env_file_value CTO_JAMI_ACCOUNT_ID || true)"
  existing_jami_profile_name="$(read_runtime_env_file_value CTO_JAMI_PROFILE_NAME || true)"
  existing_jami_inbox_dir="$(read_runtime_env_file_value CTO_JAMI_INBOX_DIR || true)"
  existing_jami_outbox_dir="$(read_runtime_env_file_value CTO_JAMI_OUTBOX_DIR || true)"
  existing_jami_archive_dir="$(read_runtime_env_file_value CTO_JAMI_ARCHIVE_DIR || true)"
  existing_jami_daemon_args="$(read_runtime_env_file_value CTO_JAMI_DAEMON_ARGS || true)"
  existing_grosshirn_api_key="$(read_runtime_env_file_value CTO_AGENT_GROSSHIRN_API_KEY || true)"
  existing_grosshirn_model="$(read_runtime_env_file_value CTO_AGENT_GROSSHIRN_MODEL || true)"
  existing_grosshirn_adapter="$(read_runtime_env_file_value CTO_AGENT_GROSSHIRN_AGENTIC_ADAPTER || true)"
  existing_grosshirn_base_url="$(read_runtime_env_file_value CTO_AGENT_GROSSHIRN_BASE_URL || true)"
  existing_grosshirn_reasoning="$(read_runtime_env_file_value CTO_AGENT_GROSSHIRN_REASONING || true)"
  existing_agent_port="$(read_runtime_env_file_value CTO_AGENT_PORT || true)"
  existing_bind_host="$(read_runtime_env_file_value CTO_AGENT_BIND_HOST || true)"
  existing_public_base_url="$(read_runtime_env_file_value CTO_AGENT_PUBLIC_BASE_URL || true)"
  existing_tls_alt_names="$(read_runtime_env_file_value CTO_AGENT_TLS_ALT_NAMES || true)"
  detected_public_host="$(detect_public_control_plane_host || true)"

  CTO_EMAIL_ADDRESS="${CTO_EMAIL_ADDRESS:-$existing_address}"
  CTO_EMAIL_PASSWORD="${CTO_EMAIL_PASSWORD:-$existing_password}"
  CTO_EMAIL_IMAP_HOST="${CTO_EMAIL_IMAP_HOST:-${existing_imap_host:-imap.one.com}}"
  CTO_EMAIL_IMAP_PORT="${CTO_EMAIL_IMAP_PORT:-${existing_imap_port:-993}}"
  CTO_EMAIL_SMTP_HOST="${CTO_EMAIL_SMTP_HOST:-${existing_smtp_host:-send.one.com}}"
  CTO_EMAIL_SMTP_PORT="${CTO_EMAIL_SMTP_PORT:-${existing_smtp_port:-465}}"
  CTO_AGENT_EMAIL_SYNC_LIMIT="${CTO_AGENT_EMAIL_SYNC_LIMIT:-${existing_sync_limit:-20}}"
  discovered_jami_account_id="$(discover_local_jami_account_id || true)"
  CTO_JAMI_ACCOUNT_ID="${CTO_JAMI_ACCOUNT_ID:-${existing_jami_account_id:-$discovered_jami_account_id}}"
  if [ -n "${CTO_JAMI_ACCOUNT_ID:-}" ]; then
    CTO_JAMI_PROFILE_NAME="${CTO_JAMI_PROFILE_NAME:-${existing_jami_profile_name:-$CTO_JAMI_ACCOUNT_ID}}"
  else
    CTO_JAMI_PROFILE_NAME="${CTO_JAMI_PROFILE_NAME:-$existing_jami_profile_name}"
  fi
  jami_runtime_root="$ROOT/runtime/communication/jami"
  CTO_JAMI_INBOX_DIR="${CTO_JAMI_INBOX_DIR:-${existing_jami_inbox_dir:-$jami_runtime_root/inbox}}"
  CTO_JAMI_OUTBOX_DIR="${CTO_JAMI_OUTBOX_DIR:-${existing_jami_outbox_dir:-$jami_runtime_root/outbox}}"
  CTO_JAMI_ARCHIVE_DIR="${CTO_JAMI_ARCHIVE_DIR:-${existing_jami_archive_dir:-$jami_runtime_root/archive}}"
  CTO_JAMI_DAEMON_ARGS="${CTO_JAMI_DAEMON_ARGS:-${existing_jami_daemon_args:--p}}"
  CTO_AGENT_GROSSHIRN_API_KEY="${CTO_AGENT_GROSSHIRN_API_KEY:-$existing_grosshirn_api_key}"
  CTO_AGENT_GROSSHIRN_MODEL="${CTO_AGENT_GROSSHIRN_MODEL:-${existing_grosshirn_model:-openai/gpt-5.4}}"
  CTO_AGENT_GROSSHIRN_AGENTIC_ADAPTER="${CTO_AGENT_GROSSHIRN_AGENTIC_ADAPTER:-${existing_grosshirn_adapter:-openai_responses}}"
  CTO_AGENT_GROSSHIRN_BASE_URL="${CTO_AGENT_GROSSHIRN_BASE_URL:-${existing_grosshirn_base_url:-https://api.openai.com/v1}}"
  CTO_AGENT_GROSSHIRN_REASONING="${CTO_AGENT_GROSSHIRN_REASONING:-${existing_grosshirn_reasoning:-medium}}"
  CTO_AGENT_PORT="${CTO_AGENT_PORT:-${existing_agent_port:-8443}}"
  bind_host_candidate="${existing_bind_host:-}"
  if [ -z "$bind_host_candidate" ] || [ "$bind_host_candidate" = "127.0.0.1" ]; then
    if [ -n "$detected_public_host" ]; then
      bind_host_candidate="0.0.0.0"
    else
      bind_host_candidate="${bind_host_candidate:-127.0.0.1}"
    fi
  fi
  public_base_url_candidate="${existing_public_base_url:-}"
  if [ -z "$public_base_url_candidate" ] && [ -n "$detected_public_host" ]; then
    public_base_url_candidate="https://${detected_public_host}:${CTO_AGENT_PORT}"
  fi
  tls_alt_names_candidate="${existing_tls_alt_names:-}"
  if [ -z "$tls_alt_names_candidate" ] && [ -n "$detected_public_host" ]; then
    tls_alt_names_candidate="$detected_public_host"
  fi

  CTO_AGENT_BIND_HOST="${CTO_AGENT_BIND_HOST:-$bind_host_candidate}"
  CTO_AGENT_PUBLIC_BASE_URL="${CTO_AGENT_PUBLIC_BASE_URL:-$public_base_url_candidate}"
  CTO_AGENT_TLS_ALT_NAMES="${CTO_AGENT_TLS_ALT_NAMES:-$tls_alt_names_candidate}"
}

ensure_jami_runtime_directories() {
  mkdir -p \
    "$ROOT/runtime/communication/jami/raw" \
    "$CTO_JAMI_INBOX_DIR" \
    "$CTO_JAMI_OUTBOX_DIR" \
    "$CTO_JAMI_ARCHIVE_DIR"
}

ensure_mail_and_cli_bootstrap_assets() {
  for required in \
    "$ROOT/scripts/communication_mail_cli.mjs" \
    "$ROOT/scripts/communication_jami_cli.mjs" \
    "$ROOT/scripts/communication_schema.sql" \
    "$ROOT/scripts/run_jami_daemon.sh" \
    "$ROOT/.agents/skills/codex-command-exec-operations/SKILL.md" \
    "$ROOT/contracts/system/codex-command-exec-capability-policy.json" \
    "$ROOT/.agents/skills/workspace-execution-operations/SKILL.md" \
    "$ROOT/contracts/system/workspace-execution-capability-policy.json"
  do
    if [ ! -f "$required" ]; then
      echo "Required bootstrap asset is missing: $required" >&2
      exit 1
    fi
  done
}

initialize_communication_runtime_store() {
  sqlite3 "$ROOT/runtime/cto_agent.db" < "$ROOT/scripts/communication_schema.sql"
}

smoke_check_mail_bootstrap() {
  node "$ROOT/scripts/communication_mail_cli.mjs" \
    list \
    --db "$ROOT/runtime/cto_agent.db" \
    --limit 1 >/tmp/cto_mail_bootstrap_list.out

  node "$ROOT/scripts/communication_jami_cli.mjs" \
    list \
    --db "$ROOT/runtime/cto_agent.db" \
    --limit 1 >/tmp/cto_jami_bootstrap_list.out

  if [ -n "$CTO_EMAIL_ADDRESS" ] && [ -n "$CTO_EMAIL_PASSWORD" ]; then
    CTO_EMAIL_ADDRESS="$CTO_EMAIL_ADDRESS" \
      CTO_EMAIL_PASSWORD="$CTO_EMAIL_PASSWORD" \
      node "$ROOT/scripts/communication_mail_cli.mjs" \
        sync \
        --db "$ROOT/runtime/cto_agent.db" \
        --imap-host "$CTO_EMAIL_IMAP_HOST" \
        --imap-port "$CTO_EMAIL_IMAP_PORT" \
        --folder INBOX \
        --limit 1 \
        --emit-interrupts false >/tmp/cto_mail_bootstrap_sync.out
  fi
}

smoke_check_cli_tooling() {
  "$ROOT/target/release/cto-agent" command-exec-smoke >/tmp/cto_command_exec_smoke.out
}

write_kleinhirn_env_file() {
  {
    printf 'CTO_AGENT_KLEINHIRN_BASE_URL=%s\n' "$(shell_quote "$KLEINHIRN_BASE_URL")"
    printf 'CTO_AGENT_KLEINHIRN_API_KEY=%s\n' "$(shell_quote "local-kleinhirn")"
    printf 'CTO_AGENT_MISTRALRS_FEATURES=%s\n' "$(shell_quote "$MISTRALRS_FEATURES")"
    printf 'CTO_AGENT_KLEINHIRN_PROFILE=%s\n' "$(shell_quote "$KLEINHIRN_PROFILE")"
    printf 'CTO_AGENT_KLEINHIRN_MODEL=%s\n' "$(shell_quote "$KLEINHIRN_POLICY_MODEL")"
    printf 'CTO_AGENT_KLEINHIRN_RUNTIME_MODEL=%s\n' "$(shell_quote "$KLEINHIRN_RUNTIME_MODEL")"
    printf 'CTO_AGENT_KLEINHIRN_OFFICIAL_LABEL=%s\n' "$(shell_quote "$KLEINHIRN_OFFICIAL_LABEL")"
    printf 'CTO_AGENT_KLEINHIRN_AGENTIC_ADAPTER=%s\n' "$(shell_quote "$KLEINHIRN_AGENTIC_ADAPTER")"
    printf 'CTO_AGENT_KLEINHIRN_SERVER_IMPL=%s\n' "$(shell_quote "$KLEINHIRN_SERVER_IMPL")"
    printf 'CTO_AGENT_KLEINHIRN_ARCH=%s\n' "$(shell_quote "$KLEINHIRN_ARCH")"
    printf 'CTO_AGENT_KLEINHIRN_PORT=%s\n' "$(shell_quote "$KLEINHIRN_PORT")"
    printf 'CTO_AGENT_KLEINHIRN_STARTUP_WAIT_SECS=%s\n' "$(shell_quote "$KLEINHIRN_STARTUP_WAIT_SECS")"
    printf 'CTO_AGENT_KLEINHIRN_MAX_SEQ_LEN=%s\n' "$(shell_quote "$KLEINHIRN_MAX_SEQ_LEN")"
    printf 'CTO_AGENT_KLEINHIRN_DISABLE_PAGED_ATTN=%s\n' "$(shell_quote "$KLEINHIRN_DISABLE_PAGED_ATTN")"
    printf 'CTO_AGENT_KLEINHIRN_NUM_DEVICE_LAYERS=%s\n' "$(shell_quote "$KLEINHIRN_NUM_DEVICE_LAYERS")"
    printf 'CTO_AGENT_KLEINHIRN_MAX_SEQS=%s\n' "$(shell_quote "$KLEINHIRN_MAX_SEQS")"
    printf 'CTO_AGENT_KLEINHIRN_MAX_BATCH_SIZE=%s\n' "$(shell_quote "$KLEINHIRN_MAX_BATCH_SIZE")"
    printf 'CTO_AGENT_KLEINHIRN_PA_GPU_MEM=%s\n' "$(shell_quote "$KLEINHIRN_PA_GPU_MEM")"
    printf 'CTO_AGENT_KLEINHIRN_PA_GPU_MEM_USAGE=%s\n' "$(shell_quote "$KLEINHIRN_PA_GPU_MEM_USAGE")"
    printf 'CTO_AGENT_KLEINHIRN_PA_CTXT_LEN=%s\n' "$(shell_quote "$KLEINHIRN_PA_CTXT_LEN")"
    printf 'CTO_AGENT_KLEINHIRN_PA_CACHE_TYPE=%s\n' "$(shell_quote "$KLEINHIRN_PA_CACHE_TYPE")"
    printf 'CTO_AGENT_KLEINHIRN_PAGED_ATTN_MODE=%s\n' "$(shell_quote "$KLEINHIRN_PAGED_ATTN_MODE")"
    printf 'CTO_AGENT_KLEINHIRN_DEVICE_LAYERS=%s\n' "$(shell_quote "$KLEINHIRN_DEVICE_LAYERS")"
    printf 'CTO_AGENT_KLEINHIRN_DISABLE_NCCL=%s\n' "$(shell_quote "$KLEINHIRN_DISABLE_NCCL")"
    printf 'CTO_AGENT_KLEINHIRN_ISQ=%s\n' "$(shell_quote "$KLEINHIRN_ISQ")"
    printf 'CTO_AGENT_KLEINHIRN_CHAT_TEMPLATE=%s\n' "$(shell_quote "$KLEINHIRN_CHAT_TEMPLATE")"
    printf 'CTO_AGENT_KLEINHIRN_JINJA_EXPLICIT=%s\n' "$(shell_quote "$KLEINHIRN_JINJA_EXPLICIT")"
    printf 'CTO_AGENT_KLEINHIRN_TOKENIZER_JSON=%s\n' "$(shell_quote "$KLEINHIRN_TOKENIZER_JSON")"
    printf 'CTO_AGENT_KLEINHIRN_TOPOLOGY=%s\n' "$(shell_quote "$KLEINHIRN_TOPOLOGY")"
    printf 'CTO_AGENT_KLEINHIRN_CUDA_VISIBLE_DEVICES=%s\n' "$(shell_quote "$KLEINHIRN_CUDA_VISIBLE_DEVICES")"
    printf 'CTO_AGENT_KLEINHIRN_MULTI_GPU_MODE=%s\n' "$(shell_quote "$KLEINHIRN_MULTI_GPU_MODE")"
    printf 'CTO_AGENT_KLEINHIRN_TENSOR_PARALLEL_BACKEND=%s\n' "$(shell_quote "$KLEINHIRN_TENSOR_PARALLEL_BACKEND")"
    printf 'CTO_AGENT_KLEINHIRN_VISIBLE_GPU_POLICY=%s\n' "$(shell_quote "$KLEINHIRN_VISIBLE_GPU_POLICY")"
    printf 'CTO_AGENT_KLEINHIRN_MN_LOCAL_WORLD_SIZE=%s\n' "$(shell_quote "$KLEINHIRN_MN_LOCAL_WORLD_SIZE")"
    printf 'CTO_AGENT_CONTEXT_EMBEDDING_ENABLED=%s\n' "$(shell_quote "$CONTEXT_EMBEDDING_ENABLED")"
    printf 'CTO_AGENT_CONTEXT_EMBEDDING_MODEL=%s\n' "$(shell_quote "$CONTEXT_EMBEDDING_MODEL")"
    printf 'CTO_AGENT_CONTEXT_EMBEDDING_RUNTIME_MODEL=%s\n' "$(shell_quote "$CONTEXT_EMBEDDING_RUNTIME_MODEL")"
    printf 'CTO_AGENT_CONTEXT_EMBEDDING_PORT=%s\n' "$(shell_quote "$CONTEXT_EMBEDDING_PORT")"
    printf 'CTO_AGENT_CONTEXT_EMBEDDING_BASE_URL=%s\n' "$(shell_quote "$CONTEXT_EMBEDDING_BASE_URL")"
    printf 'CTO_AGENT_CONTEXT_EMBEDDING_API_KEY=%s\n' "$(shell_quote "$CONTEXT_EMBEDDING_API_KEY")"
    printf 'CTO_AGENT_CONTEXT_EMBEDDING_MAX_BATCH_SIZE=%s\n' "$(shell_quote "$CONTEXT_EMBEDDING_MAX_BATCH_SIZE")"
    printf 'CTO_AGENT_CONTEXT_EMBEDDING_CHUNK_CHARS=%s\n' "$(shell_quote "$CONTEXT_EMBEDDING_CHUNK_CHARS")"
    printf 'CTO_AGENT_CONTEXT_EMBEDDING_CHUNK_OVERLAP=%s\n' "$(shell_quote "$CONTEXT_EMBEDDING_CHUNK_OVERLAP")"
    printf 'CTO_EMAIL_ADDRESS=%s\n' "$(shell_quote "$CTO_EMAIL_ADDRESS")"
    printf 'CTO_EMAIL_PASSWORD=%s\n' "$(shell_quote "$CTO_EMAIL_PASSWORD")"
    printf 'CTO_EMAIL_IMAP_HOST=%s\n' "$(shell_quote "$CTO_EMAIL_IMAP_HOST")"
    printf 'CTO_EMAIL_IMAP_PORT=%s\n' "$(shell_quote "$CTO_EMAIL_IMAP_PORT")"
    printf 'CTO_EMAIL_SMTP_HOST=%s\n' "$(shell_quote "$CTO_EMAIL_SMTP_HOST")"
    printf 'CTO_EMAIL_SMTP_PORT=%s\n' "$(shell_quote "$CTO_EMAIL_SMTP_PORT")"
    printf 'CTO_AGENT_EMAIL_SYNC_LIMIT=%s\n' "$(shell_quote "$CTO_AGENT_EMAIL_SYNC_LIMIT")"
    printf 'CTO_JAMI_ACCOUNT_ID=%s\n' "$(shell_quote "$CTO_JAMI_ACCOUNT_ID")"
    printf 'CTO_JAMI_PROFILE_NAME=%s\n' "$(shell_quote "$CTO_JAMI_PROFILE_NAME")"
    printf 'CTO_JAMI_INBOX_DIR=%s\n' "$(shell_quote "$CTO_JAMI_INBOX_DIR")"
    printf 'CTO_JAMI_OUTBOX_DIR=%s\n' "$(shell_quote "$CTO_JAMI_OUTBOX_DIR")"
    printf 'CTO_JAMI_ARCHIVE_DIR=%s\n' "$(shell_quote "$CTO_JAMI_ARCHIVE_DIR")"
    printf 'CTO_JAMI_DAEMON_ARGS=%s\n' "$(shell_quote "$CTO_JAMI_DAEMON_ARGS")"
    printf 'CTO_AGENT_GROSSHIRN_API_KEY=%s\n' "$(shell_quote "$CTO_AGENT_GROSSHIRN_API_KEY")"
    printf 'CTO_AGENT_GROSSHIRN_MODEL=%s\n' "$(shell_quote "$CTO_AGENT_GROSSHIRN_MODEL")"
    printf 'CTO_AGENT_GROSSHIRN_AGENTIC_ADAPTER=%s\n' "$(shell_quote "$CTO_AGENT_GROSSHIRN_AGENTIC_ADAPTER")"
    printf 'CTO_AGENT_GROSSHIRN_BASE_URL=%s\n' "$(shell_quote "$CTO_AGENT_GROSSHIRN_BASE_URL")"
    printf 'CTO_AGENT_GROSSHIRN_REASONING=%s\n' "$(shell_quote "$CTO_AGENT_GROSSHIRN_REASONING")"
    printf 'CTO_AGENT_PORT=%s\n' "$(shell_quote "$CTO_AGENT_PORT")"
    printf 'CTO_AGENT_BIND_HOST=%s\n' "$(shell_quote "$CTO_AGENT_BIND_HOST")"
    printf 'CTO_AGENT_PUBLIC_BASE_URL=%s\n' "$(shell_quote "$CTO_AGENT_PUBLIC_BASE_URL")"
    printf 'CTO_AGENT_TLS_ALT_NAMES=%s\n' "$(shell_quote "$CTO_AGENT_TLS_ALT_NAMES")"
    printf 'CTO_AGENT_COMPACT_SIMPLE_MODEL=%s\n' "$(shell_quote "$CTO_AGENT_COMPACT_SIMPLE_MODEL")"
    printf 'CTO_AGENT_COMPACT_MEDIUM_MODEL=%s\n' "$(shell_quote "$CTO_AGENT_COMPACT_MEDIUM_MODEL")"
    printf 'CTO_AGENT_COMPACT_RED_MODEL=%s\n' "$(shell_quote "$CTO_AGENT_COMPACT_RED_MODEL")"
  } > "$ENV_FILE"
}

preferred_cuda_visible_devices_from_census() {
  python3 - <<'PY'
import json
import pathlib
import subprocess

path = pathlib.Path("/tmp/cto_system_census.json")
try:
    census = json.loads(path.read_text())
except Exception:
    raise SystemExit(0)

gpu_entries = census.get("gpus") or []
gpu_indices = []
for fallback_index, gpu in enumerate(gpu_entries):
    try:
        gpu_indices.append(int((gpu or {}).get("index", fallback_index)))
    except Exception:
        gpu_indices.append(fallback_index)

if not gpu_indices:
    try:
        gpu_count = int(census.get("gpuCount") or 0)
    except Exception:
        gpu_count = 0
    gpu_indices = list(range(gpu_count))

gpu_indices = sorted(set(gpu_indices))
gpu_count = len(gpu_indices)
if gpu_count <= 1 or gpu_count & (gpu_count - 1) == 0:
    raise SystemExit(0)

visible_gpu_count = 1
while visible_gpu_count * 2 <= gpu_count:
    visible_gpu_count *= 2

if visible_gpu_count < 2 or gpu_count < visible_gpu_count:
    raise SystemExit(0)

display_free_indices = []
try:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,display_active",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stderr=subprocess.DEVNULL,
    )
except Exception:
    output = ""

for line in output.splitlines():
    parts = [part.strip() for part in line.split(",", 1)]
    if len(parts) != 2:
        continue
    try:
        gpu_index = int(parts[0])
    except Exception:
        continue
    if gpu_index not in gpu_indices:
        continue
    if parts[1].lower() in {"disabled", "off", "no"}:
        display_free_indices.append(gpu_index)

display_free_indices = sorted(set(display_free_indices))
if len(display_free_indices) >= visible_gpu_count:
    subset = display_free_indices[:visible_gpu_count]
else:
    subset = gpu_indices[-visible_gpu_count:]

print(",".join(str(index) for index in subset))
PY
}

count_csv_items() {
  raw="${1:-}"
  if [ -z "$raw" ]; then
    printf '%s\n' "0"
    return
  fi
  printf '%s\n' "$raw" | awk -F',' '{
    count = 0
    for (i = 1; i <= NF; i += 1) {
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", $i)
      if ($i != "") {
        count += 1
      }
    }
    print count
  }'
}

resolve_multi_gpu_mode() {
  if [ -n "${KLEINHIRN_MULTI_GPU_MODE:-}" ]; then
    printf '%s\n' "$KLEINHIRN_MULTI_GPU_MODE"
    return
  fi
  if is_gpt_oss_family; then
    printf '%s\n' "auto_device_map"
    return
  fi
  if [ "${SELECTED_PREFER_AUTO_DEVICE_MAPPING:-0}" = "1" ]; then
    printf '%s\n' "auto_device_map"
    return
  fi
  printf '%s\n' "tensor_parallel"
}

resolve_tensor_parallel_backend() {
  if [ -n "${KLEINHIRN_TENSOR_PARALLEL_BACKEND:-}" ]; then
    printf '%s\n' "$KLEINHIRN_TENSOR_PARALLEL_BACKEND"
    return
  fi
  if [ "$(resolve_multi_gpu_mode)" = "tensor_parallel" ] && ! is_gpt_oss_family; then
    printf '%s\n' "nccl"
    return
  fi
  printf '%s\n' "disabled"
}

resolve_visible_gpu_policy() {
  if [ -n "${KLEINHIRN_VISIBLE_GPU_POLICY:-}" ]; then
    printf '%s\n' "$KLEINHIRN_VISIBLE_GPU_POLICY"
    return
  fi
  if [ "$(resolve_multi_gpu_mode)" = "tensor_parallel" ]; then
    printf '%s\n' "largest_power_of_two_prefer_display_free"
    return
  fi
  printf '%s\n' "all"
}

resolve_mn_local_world_size() {
  visible_devices="${1:-}"
  gpu_count="${2:-0}"

  if [ -n "$visible_devices" ]; then
    count_csv_items "$visible_devices"
    return
  fi

  case "$gpu_count" in
    ''|*[!0-9]*)
      printf '%s\n' ""
      ;;
    *)
      if [ "$gpu_count" -gt 1 ]; then
        printf '%s\n' "$gpu_count"
      else
        printf '%s\n' ""
      fi
      ;;
  esac
}

next_context_backoff_value() {
  current="${1:-}"
  case "$current" in
    ''|*[!0-9]*)
      return 1
      ;;
  esac
  if [ "$current" -le 4096 ]; then
    return 1
  fi
  reduction=$((current / 8))
  if [ "$reduction" -lt 2048 ]; then
    reduction=2048
  fi
  next=$((current - reduction))
  next=$((next / 2048 * 2048))
  if [ "$next" -lt 4096 ]; then
    next=4096
  fi
  printf '%s\n' "$next"
}

capture_kleinhirn_failure_detail() {
  {
    [ -f /tmp/cto_kleinhirn_ready.err ] && cat /tmp/cto_kleinhirn_ready.err
    [ -f /tmp/cto_kleinhirn_check.err ] && cat /tmp/cto_kleinhirn_check.err
    journalctl --user -u cto-kleinhirn.service -n 80 --no-pager 2>/dev/null || true
  } | tail -n 200
}

kleinhirn_failure_supports_context_backoff() {
  detail="$(capture_kleinhirn_failure_detail | tr '[:upper:]' '[:lower:]')"
  case "$detail" in
    *"cuda_error_out_of_memory"*|*"out of memory"*|*"illegal memory access"*|*"cuda_error_illegal_address"*|*"no response received from the model"*|*"channel closed"*|*"senderror"*|*"engine openai/gpt-oss-20b is dead"*)
      return 0
      ;;
  esac
  return 1
}

run_sudo() {
  if [ -n "${CTO_AGENT_SUDO_PASSWORD:-}" ]; then
    printf '%s\n' "$CTO_AGENT_SUDO_PASSWORD" | sudo -S env \
      DEBIAN_FRONTEND="${DEBIAN_FRONTEND:-}" \
      NEEDRESTART_MODE="${NEEDRESTART_MODE:-}" \
      APT_LISTCHANGES_FRONTEND="${APT_LISTCHANGES_FRONTEND:-}" \
      "$@"
  else
    sudo env \
      DEBIAN_FRONTEND="${DEBIAN_FRONTEND:-}" \
      NEEDRESTART_MODE="${NEEDRESTART_MODE:-}" \
      APT_LISTCHANGES_FRONTEND="${APT_LISTCHANGES_FRONTEND:-}" \
      "$@"
  fi
}

version_ge() {
  [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n 1)" = "$2" ]
}

ensure_modern_rust_toolchain() {
  MIN_RUST_VERSION="${CTO_AGENT_MIN_RUST_VERSION:-1.85.0}"
  CURRENT_RUST_VERSION=""

  if command -v rustc >/dev/null 2>&1; then
    CURRENT_RUST_VERSION="$(rustc --version 2>/dev/null | awk '{print $2}' | head -n 1)"
  fi

  if [ -n "$CURRENT_RUST_VERSION" ] && version_ge "$CURRENT_RUST_VERSION" "$MIN_RUST_VERSION"; then
    return
  fi

  echo "[prep] Install/update Rust stable toolchain (>= $MIN_RUST_VERSION)"
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  if [ -f "$HOME/.cargo/env" ]; then
    # shellcheck disable=SC1090
    . "$HOME/.cargo/env"
  fi
  rustup toolchain install stable >/dev/null 2>&1 || rustup toolchain install stable
  rustup default stable >/dev/null 2>&1 || rustup default stable
}

google_chrome_apt_lists_present() {
  ls /etc/apt/sources.list.d/google-chrome*.list >/dev/null 2>&1
}

disable_google_chrome_apt_lists() {
  for source_list in /etc/apt/sources.list.d/google-chrome*.list; do
    [ -f "$source_list" ] || continue
    echo "[prep] Disable stale Google Chrome apt source: $source_list"
    run_sudo mv "$source_list" "$source_list.disabled"
  done
}

apt_update_with_retry() {
  tmp_log="$(mktemp /tmp/cto-apt-update.XXXXXX)"
  if run_sudo apt-get update >"$tmp_log" 2>&1; then
    cat "$tmp_log"
    rm -f "$tmp_log"
    return 0
  fi

  cat "$tmp_log" >&2
  if google_chrome_apt_lists_present && grep -Eqi 'google|dl\.google\.com|chrome' "$tmp_log"; then
    echo "[prep] apt-get update failed against a stale Google Chrome source; disabling it and retrying." >&2
    disable_google_chrome_apt_lists
    rm -f "$tmp_log"
    run_sudo apt-get update
    return $?
  fi

  rm -f "$tmp_log"
  return 1
}

mistralrs_uses_cuda() {
  case " $MISTRALRS_FEATURES " in
    *" cuda "*) return 0 ;;
    *) return 1 ;;
  esac
}

mistralrs_uses_nccl() {
  case " $MISTRALRS_FEATURES " in
    *" nccl "*) return 0 ;;
    *) return 1 ;;
  esac
}

latest_apt_package_matching() {
  pattern="$1"
  apt-cache pkgnames 2>/dev/null | grep -E "$pattern" | sort -V | tail -n 1
}

cuda_linker_prereqs_ready() {
  command -v nvcc >/dev/null 2>&1 || return 1
  command -v ldconfig >/dev/null 2>&1 || return 1
  ldconfig -p 2>/dev/null | grep -q 'libnvrtc' || return 1
  ldconfig -p 2>/dev/null | grep -q 'libcurand' || return 1
  ldconfig -p 2>/dev/null | grep -q 'libcublasLt' || return 1
  ldconfig -p 2>/dev/null | grep -q 'libcublas' || return 1
  return 0
}

ensure_cuda_build_prereqs() {
  if [ "$(uname -s)" != "Linux" ] || ! command -v apt-get >/dev/null 2>&1 || ! command -v sudo >/dev/null 2>&1; then
    return
  fi
  if ! mistralrs_uses_cuda; then
    return
  fi
  if cuda_linker_prereqs_ready; then
    return
  fi

  cuda_packages=""
  for pattern in \
    '^cuda-driver-dev-[0-9]+-[0-9]+$' \
    '^cuda-cudart-dev-[0-9]+-[0-9]+$' \
    '^cuda-nvcc-[0-9]+-[0-9]+$' \
    '^cuda-nvrtc-dev-[0-9]+-[0-9]+$' \
    '^libcublas-dev-[0-9]+-[0-9]+$' \
    '^libcurand-dev-[0-9]+-[0-9]+$'
  do
    pkg="$(latest_apt_package_matching "$pattern" || true)"
    [ -n "$pkg" ] || continue
    cuda_packages="$cuda_packages $pkg"
  done

  if [ -n "$cuda_packages" ]; then
    echo "[prep] Install CUDA build prerequisites for mistral.rs"
    apt_update_with_retry
    # shellcheck disable=SC2086
    run_sudo apt-get install -y $cuda_packages
  fi
}

detect_cuda_home() {
  if [ -n "${CTO_AGENT_CUDA_HOME:-}" ] && [ -d "${CTO_AGENT_CUDA_HOME:-}" ]; then
    printf '%s\n' "$CTO_AGENT_CUDA_HOME"
    return
  fi
  if [ -d /usr/local/cuda ]; then
    printf '%s\n' "/usr/local/cuda"
    return
  fi
  for candidate in /usr/local/cuda-*; do
    [ -x "$candidate/bin/nvcc" ] || continue
    printf '%s\n' "$candidate"
    return
  done
  if command -v nvcc >/dev/null 2>&1; then
    nvcc_path="$(command -v nvcc)"
    case "$nvcc_path" in
      */bin/nvcc)
        printf '%s\n' "${nvcc_path%/bin/nvcc}"
        return
        ;;
    esac
  fi
}

detect_cuda_compute_cap() {
  if [ -n "${CTO_AGENT_CUDA_COMPUTE_CAP:-}" ]; then
    printf '%s\n' "$CTO_AGENT_CUDA_COMPUTE_CAP"
    return
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    compute_cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '.[:space:]')"
    case "$compute_cap" in
      [0-9][0-9]*)
        printf '%s\n' "$compute_cap"
        return
        ;;
    esac
  fi
  if command -v lspci >/dev/null 2>&1; then
    gpu_lines="$(lspci 2>/dev/null | grep -Ei 'NVIDIA|GA10|GA100|AD10|RTX A4500|RTX A5000|RTX A4000|RTX 4000 Ada|RTX 5000 Ada|RTX 6000 Ada|A100|A10|A30|A40' || true)"
    case "$gpu_lines" in
      *GA100*|*A100*)
        printf '%s\n' "80"
        return
        ;;
      *GA10*|*"RTX A4500"*|*"RTX A5000"*|*"RTX A4000"*|*A10*|*A30*|*A40*)
        printf '%s\n' "86"
        return
        ;;
      *AD10*|*"RTX 4000 Ada"*|*"RTX 5000 Ada"*|*"RTX 6000 Ada"*)
        printf '%s\n' "89"
        return
        ;;
    esac
  fi
}

configure_cuda_env() {
  if ! mistralrs_uses_cuda; then
    return
  fi

  cuda_home="$(detect_cuda_home || true)"
  if [ -n "$cuda_home" ]; then
    export CUDA_HOME="$cuda_home"
    export PATH="$cuda_home/bin:$PATH"
    if [ -d "$cuda_home/lib64" ]; then
      export LD_LIBRARY_PATH="$cuda_home/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
  fi

  compute_cap="$(detect_cuda_compute_cap || true)"
  if [ -n "$compute_cap" ]; then
    export CUDA_COMPUTE_CAP="$compute_cap"
  fi
}

cuda_runtime_driver_ready() {
  if ! mistralrs_uses_cuda; then
    return 0
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi

  tmp_log="$(mktemp /tmp/cto-nvidia-smi.XXXXXX)"
  if nvidia-smi >"$tmp_log" 2>&1; then
    rm -f "$tmp_log"
    return 0
  fi

  if grep -Eqi 'driver/library version mismatch|unsupported display driver / cuda driver combination' "$tmp_log"; then
    echo "Detected NVIDIA driver/library mismatch. CUDA Kleinhirn startup will fail until the host is rebooted or the driver stack is repaired." >&2
    cat "$tmp_log" >&2
    rm -f "$tmp_log"
    return 1
  fi

  rm -f "$tmp_log"
  return 0
}

linux_build_prereqs_missing() {
  command -v cc >/dev/null 2>&1 || return 0
  command -v make >/dev/null 2>&1 || return 0
  command -v pkg-config >/dev/null 2>&1 || return 0
  command -v cmake >/dev/null 2>&1 || return 0
  command -v python3 >/dev/null 2>&1 || return 0
  command -v curl >/dev/null 2>&1 || return 0
  command -v git >/dev/null 2>&1 || return 0
  command -v node >/dev/null 2>&1 || return 0
  command -v npm >/dev/null 2>&1 || return 0
  command -v sqlite3 >/dev/null 2>&1 || return 0
  command -v ninja >/dev/null 2>&1 || command -v ninja-build >/dev/null 2>&1 || return 0
  pkg-config --exists openssl >/dev/null 2>&1 || return 0
  return 1
}

resolve_jami_linux_repo_suffix() {
  if [ ! -f /etc/os-release ]; then
    return 1
  fi

  os_id=""
  version_id=""
  id_like=""
  # shellcheck disable=SC1091
  . /etc/os-release
  os_id="$(printf '%s' "${ID:-}" | tr '[:upper:]' '[:lower:]')"
  version_id="$(printf '%s' "${VERSION_ID:-}" | tr -d '"')"
  id_like="$(printf '%s' "${ID_LIKE:-}" | tr '[:upper:]' '[:lower:]')"

  case "$os_id" in
    ubuntu)
      case "$version_id" in
        20.04|22.04|24.04|24.10|25.04)
          printf 'ubuntu_%s\n' "$version_id"
          return 0
          ;;
      esac
      ;;
    debian)
      case "$version_id" in
        11|12|13)
          printf 'debian_%s\n' "$version_id"
          return 0
          ;;
      esac
      ;;
    linuxmint)
      case "$version_id" in
        21*)
          printf '%s\n' "ubuntu_22.04"
          return 0
          ;;
        22*)
          printf '%s\n' "ubuntu_24.04"
          return 0
          ;;
        6)
          printf '%s\n' "debian_12"
          return 0
          ;;
      esac
      ;;
  esac

  case "$id_like" in
    *ubuntu*)
      case "$version_id" in
        20.04|22.04|24.04|24.10|25.04)
          printf 'ubuntu_%s\n' "$version_id"
          return 0
          ;;
      esac
      ;;
    *debian*)
      case "$version_id" in
        11|12|13)
          printf 'debian_%s\n' "$version_id"
          return 0
          ;;
      esac
      ;;
  esac

  return 1
}

ensure_linux_jami_installed() {
  if [ "$(uname -s)" != "Linux" ] || ! command -v apt-get >/dev/null 2>&1 || ! command -v sudo >/dev/null 2>&1; then
    return
  fi

  repo_suffix="$(resolve_jami_linux_repo_suffix || true)"
  if [ -z "$repo_suffix" ]; then
    echo "[prep] Skip Jami package auto-install because this Linux distribution is not mapped to an official Jami apt repository." >&2
    echo "[prep] Configure Jami manually via https://jami.net/en/download-jami-linux if this host still needs the daemon." >&2
    return
  fi

  jami_repo_line="deb [signed-by=/usr/share/keyrings/jami-archive-keyring.gpg] https://dl.jami.net/stable/${repo_suffix}/ jami main"
  echo "[prep] Install official Jami daemon runtime ($repo_suffix)"
  run_sudo apt-get install -y gnupg dirmngr ca-certificates curl --no-install-recommends
  tmp_keyring="$(mktemp /tmp/cto-jami-keyring.XXXXXX)"
  curl -fsSL https://dl.jami.net/public-key.gpg -o "$tmp_keyring"
  run_sudo install -m 0644 "$tmp_keyring" /usr/share/keyrings/jami-archive-keyring.gpg
  rm -f "$tmp_keyring"
  tmp_list="$(mktemp /tmp/cto-jami-repo.XXXXXX)"
  printf '%s\n' "$jami_repo_line" >"$tmp_list"
  run_sudo install -m 0644 "$tmp_list" /etc/apt/sources.list.d/jami.list
  rm -f "$tmp_list"
  apt_update_with_retry
  run_sudo apt-get install -y jami-daemon dbus-x11
  if [ "$CTO_AGENT_INSTALL_JAMI_GUI" = "1" ]; then
    run_sudo apt-get install -y jami
  fi
}

if [ "$(uname -s)" = "Linux" ] && command -v apt-get >/dev/null 2>&1 && command -v sudo >/dev/null 2>&1; then
  GPU_COUNT="$(detect_gpu_count)"
  if linux_build_prereqs_missing; then
    echo "[prep] Install Linux build prerequisites"
    apt_update_with_retry
    run_sudo apt-get install -y \
      build-essential \
      pkg-config \
      libssl-dev \
      cmake \
      ninja-build \
      python3-venv \
      curl \
      git \
      nodejs \
      npm \
      sqlite3
  else
    echo "[prep] Linux build prerequisites already present"
  fi
  ensure_linux_jami_installed
  ensure_modern_rust_toolchain
  ensure_cuda_build_prereqs
  configure_cuda_env
  if [ "$GPU_COUNT" -gt 1 ] && nccl_packages_available && nccl_runtime_missing; then
    echo "[prep] Install NCCL for multi-GPU mistral.rs tensor parallelism"
    run_sudo apt-get install -y libnccl2 libnccl-dev
  fi
else
  ensure_modern_rust_toolchain
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required" >&2
  exit 1
fi
if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required" >&2
  exit 1
fi
if ! command -v pkg-config >/dev/null 2>&1; then
  echo "pkg-config is required for mistralrs-server builds" >&2
  exit 1
fi

mkdir -p "$ROOT/runtime" "$KLEINHIRN_LOG_DIR"
adopt_existing_runtime_communication_env
ensure_jami_runtime_directories
ensure_mail_and_cli_bootstrap_assets
chmod +x \
  "$ROOT/scripts/communication_mail_cli.mjs" \
  "$ROOT/scripts/communication_jami_cli.mjs" \
  "$ROOT/scripts/install_cto_agent.sh" \
  "$ROOT/scripts/install_browser_engine.sh" \
  "$ROOT/scripts/install_browser_agent_extension.sh" \
  "$ROOT/scripts/install_linux_user_services.sh" \
  "$ROOT/scripts/launch_browser_agent_chrome.sh" \
  "$ROOT/scripts/run_jami_daemon.sh" \
  "$ROOT/scripts/run_kleinhirn.sh" \
  "$ROOT/scripts/run_control_plane.sh" \
  "$ROOT/scripts/start_control_plane.sh" \
  >/dev/null 2>&1 || true

echo "[prep] Stop stale local CTO-Agent processes"
systemctl --user stop cto-agent.service cto-kleinhirn.service cto-jami-daemon.service >/dev/null 2>&1 || true
pkill -f "$ROOT/target/debug/cto-agent" >/dev/null 2>&1 || true
pkill -f "$ROOT/target/release/cto-agent" >/dev/null 2>&1 || true
pkill -f "$ROOT/scripts/run_control_plane.sh" >/dev/null 2>&1 || true
pkill -f "$ROOT/scripts/run_kleinhirn.sh" >/dev/null 2>&1 || true
rm -f "$HEARTBEAT_FILE"

echo "[1/10] Build CTO-Agent host"
cargo build --release

echo "[2/10] Initialize contracts, TLS and SQLite"
"$ROOT/target/release/cto-agent" --init-only
initialize_communication_runtime_store
smoke_check_cli_tooling
smoke_check_mail_bootstrap

if [ -t 0 ] && [ -t 1 ] && [ "${CTO_AGENT_SKIP_INSTALL_TUI:-0}" != "1" ]; then
  echo "[3/10] Run installation communication bootstrap TUI"
  "$ROOT/target/release/cto-agent" install-bootstrap-tui
else
  echo "[3/10] Skip installation communication bootstrap TUI (non-interactive or CTO_AGENT_SKIP_INSTALL_TUI=1)"
  echo "You can run it later with: $ROOT/target/release/cto-agent install-bootstrap-tui"
fi

python3 "$ROOT/scripts/configure_model_policy.py" \
  --policy "$ROOT/contracts/models/model-policy.json" \
  --profile "$KLEINHIRN_PROFILE"

echo "[4/10] Install selected Kleinhirn runtime"
REINSTALL_MISTRALRS=0
if ! command -v mistralrs >/dev/null 2>&1; then
  REINSTALL_MISTRALRS=1
elif ! mistralrs_features_satisfy_required "$MISTRALRS_FEATURES"; then
  REINSTALL_MISTRALRS=1
fi
if [ "$REINSTALL_MISTRALRS" = "1" ]; then
  echo "Using mistralrs features: $MISTRALRS_FEATURES"
  CARGO_TARGET_DIR="$CARGO_TARGET_DIR" cargo install --locked \
    --git https://github.com/EricLBuehler/mistral.rs.git \
    mistralrs-cli \
    --force \
    --features "$MISTRALRS_FEATURES"
fi
if ! command -v mistralrs >/dev/null 2>&1; then
  echo "mistralrs installation failed" >&2
  exit 1
fi
KLEINHIRN_SERVER_IMPL="mistralrs"

echo "[5/10] Persist initial system census, choose the exact Kleinhirn candidate and collect tune evidence"
"$ROOT/target/release/cto-agent" run-census >/tmp/cto_system_census.json

eval "$(load_install_census_hints || true)"
eval "$(select_recommended_kleinhirn_env || true)"
APPLY_SELECTED_MODEL=1
if [ "$PROFILE_PINNED" = "1" ] && [ -n "${SELECTED_POLICY_MODEL:-}" ] && ! selected_matches_profile; then
  APPLY_SELECTED_MODEL=0
  echo "[5/10] Ignore mismatched recommended model ${SELECTED_POLICY_MODEL} because CTO_AGENT_KLEINHIRN_PROFILE=${KLEINHIRN_PROFILE} is pinned"
fi

if [ "$APPLY_SELECTED_MODEL" = "1" ] && [ -n "${SELECTED_POLICY_MODEL:-}" ]; then
  KLEINHIRN_POLICY_MODEL="$SELECTED_POLICY_MODEL"
fi
if [ "$APPLY_SELECTED_MODEL" = "1" ] && [ -n "${SELECTED_RUNTIME_MODEL:-}" ]; then
  KLEINHIRN_RUNTIME_MODEL="$SELECTED_RUNTIME_MODEL"
fi
if [ "$APPLY_SELECTED_MODEL" = "1" ] && [ -n "${SELECTED_OFFICIAL_LABEL:-}" ]; then
  KLEINHIRN_OFFICIAL_LABEL="$SELECTED_OFFICIAL_LABEL"
fi
if [ "$APPLY_SELECTED_MODEL" = "1" ] && [ -n "${SELECTED_AGENTIC_ADAPTER:-}" ]; then
  KLEINHIRN_AGENTIC_ADAPTER="$SELECTED_AGENTIC_ADAPTER"
fi
if [ "$APPLY_SELECTED_MODEL" = "1" ] && [ -n "${SELECTED_MAX_SEQS:-}" ]; then
  KLEINHIRN_MAX_SEQS="$SELECTED_MAX_SEQS"
fi
if [ "$APPLY_SELECTED_MODEL" = "1" ] && [ -n "${SELECTED_MAX_BATCH_SIZE:-}" ]; then
  KLEINHIRN_MAX_BATCH_SIZE="$SELECTED_MAX_BATCH_SIZE"
fi
if [ "$APPLY_SELECTED_MODEL" = "1" ] && [ -n "${SELECTED_MAX_SEQ_LEN:-}" ]; then
  KLEINHIRN_MAX_SEQ_LEN="$SELECTED_MAX_SEQ_LEN"
fi
if [ "$APPLY_SELECTED_MODEL" = "1" ] && [ -n "${SELECTED_PA_CTXT_LEN:-}" ]; then
  KLEINHIRN_PA_CTXT_LEN="$SELECTED_PA_CTXT_LEN"
fi
if [ "$APPLY_SELECTED_MODEL" = "1" ] && [ -n "${SELECTED_PA_CACHE_TYPE:-}" ]; then
  KLEINHIRN_PA_CACHE_TYPE="$SELECTED_PA_CACHE_TYPE"
fi
if [ "$APPLY_SELECTED_MODEL" = "1" ] && [ -n "${SELECTED_PAGED_ATTN_MODE:-}" ]; then
  KLEINHIRN_PAGED_ATTN_MODE="$SELECTED_PAGED_ATTN_MODE"
fi
if [ "$APPLY_SELECTED_MODEL" = "1" ] && [ -n "${SELECTED_CHAT_TEMPLATE:-}" ]; then
  KLEINHIRN_CHAT_TEMPLATE="$SELECTED_CHAT_TEMPLATE"
fi
if [ "$APPLY_SELECTED_MODEL" = "1" ] && [ -n "${SELECTED_JINJA_EXPLICIT:-}" ]; then
  KLEINHIRN_JINJA_EXPLICIT="$SELECTED_JINJA_EXPLICIT"
fi
if [ "$APPLY_SELECTED_MODEL" = "1" ] && [ -n "${SELECTED_TOKENIZER_JSON:-}" ]; then
  KLEINHIRN_TOKENIZER_JSON="$SELECTED_TOKENIZER_JSON"
fi
if [ "$APPLY_SELECTED_MODEL" = "1" ] && [ -n "${SELECTED_TOPOLOGY:-}" ]; then
  KLEINHIRN_TOPOLOGY="$SELECTED_TOPOLOGY"
fi
if [ "$APPLY_SELECTED_MODEL" = "1" ] && [ -n "${SELECTED_MULTI_GPU_MODE:-}" ]; then
  KLEINHIRN_MULTI_GPU_MODE="$SELECTED_MULTI_GPU_MODE"
fi
if [ "$APPLY_SELECTED_MODEL" = "1" ] && [ -n "${SELECTED_TENSOR_PARALLEL_BACKEND:-}" ]; then
  KLEINHIRN_TENSOR_PARALLEL_BACKEND="$SELECTED_TENSOR_PARALLEL_BACKEND"
fi
if [ "$APPLY_SELECTED_MODEL" = "1" ] && [ -n "${SELECTED_VISIBLE_GPU_POLICY:-}" ]; then
  KLEINHIRN_VISIBLE_GPU_POLICY="$SELECTED_VISIBLE_GPU_POLICY"
fi

eval "$(apply_runtime_tune_defaults || true)"
KLEINHIRN_MULTI_GPU_MODE="$(resolve_multi_gpu_mode)"
KLEINHIRN_TENSOR_PARALLEL_BACKEND="$(resolve_tensor_parallel_backend)"
KLEINHIRN_VISIBLE_GPU_POLICY="$(resolve_visible_gpu_policy)"
if [ -z "$KLEINHIRN_ISQ" ] && [ -n "${RECOMMENDED_ISQ:-}" ]; then
  KLEINHIRN_ISQ="$(printf '%s' "$RECOMMENDED_ISQ" | tr '[:upper:]' '[:lower:]')"
fi
if [ "$KLEINHIRN_PROFILE" = "gpt_oss" ]; then
  KLEINHIRN_ISQ=""
  CONTEXT_EMBEDDING_ENABLED="0"
fi
if [ -z "$KLEINHIRN_DEVICE_LAYERS" ] && [ -n "${RECOMMENDED_DEVICE_LAYERS:-}" ] && [ "${CENSUS_GPU_COUNT:-0}" -le 1 ]; then
  KLEINHIRN_DEVICE_LAYERS="$RECOMMENDED_DEVICE_LAYERS"
fi
if [ -n "${RECOMMENDED_MAX_CONTEXT_TOKENS:-}" ]; then
  if [ -z "$KLEINHIRN_MAX_SEQ_LEN" ] || [ "$KLEINHIRN_MAX_SEQ_LEN" -lt "$RECOMMENDED_MAX_CONTEXT_TOKENS" ]; then
    KLEINHIRN_MAX_SEQ_LEN="$RECOMMENDED_MAX_CONTEXT_TOKENS"
  fi
  if [ "$KLEINHIRN_PAGED_ATTN_MODE" != "off" ] && { [ -z "$KLEINHIRN_PA_CTXT_LEN" ] || [ "$KLEINHIRN_PA_CTXT_LEN" -lt "$RECOMMENDED_MAX_CONTEXT_TOKENS" ]; }; then
    KLEINHIRN_PA_CTXT_LEN="$RECOMMENDED_MAX_CONTEXT_TOKENS"
  fi
fi
if [ -z "$KLEINHIRN_MAX_SEQS" ]; then
  KLEINHIRN_MAX_SEQS="1"
fi
if [ -z "$KLEINHIRN_MAX_BATCH_SIZE" ]; then
  KLEINHIRN_MAX_BATCH_SIZE="1"
fi

if [ "${CENSUS_GPU_COUNT:-0}" -gt 1 ] && [ "$KLEINHIRN_MULTI_GPU_MODE" = "auto_device_map" ]; then
  if [ -z "$KLEINHIRN_TOPOLOGY" ]; then
    KLEINHIRN_DEVICE_LAYERS=""
    KLEINHIRN_NUM_DEVICE_LAYERS=""
  fi
  if [ -z "$KLEINHIRN_MAX_SEQ_LEN" ] && [ -n "${RECOMMENDED_MAX_CONTEXT_TOKENS:-}" ]; then
    KLEINHIRN_MAX_SEQ_LEN="$RECOMMENDED_MAX_CONTEXT_TOKENS"
  fi
  if [ -z "$KLEINHIRN_PA_CTXT_LEN" ] && [ -n "${RECOMMENDED_MAX_CONTEXT_TOKENS:-}" ]; then
    KLEINHIRN_PA_CTXT_LEN="$RECOMMENDED_MAX_CONTEXT_TOKENS"
  fi
fi

if [ "${CENSUS_GPU_COUNT:-0}" -gt 1 ] && [ -z "$KLEINHIRN_TOPOLOGY" ] && [ "$KLEINHIRN_MULTI_GPU_MODE" != "manual_device_layers" ]; then
  KLEINHIRN_DEVICE_LAYERS=""
  KLEINHIRN_NUM_DEVICE_LAYERS=""
fi

KLEINHIRN_MN_LOCAL_WORLD_SIZE=""
if [ "${CENSUS_GPU_COUNT:-0}" -gt 1 ] \
  && [ "$KLEINHIRN_MULTI_GPU_MODE" = "tensor_parallel" ] \
  && [ "$KLEINHIRN_TENSOR_PARALLEL_BACKEND" = "nccl" ]; then
  if [ "$KLEINHIRN_VISIBLE_GPU_POLICY" = "largest_power_of_two_prefer_display_free" ] \
    && [ -z "$KLEINHIRN_TOPOLOGY" ] \
    && [ -z "$KLEINHIRN_CUDA_VISIBLE_DEVICES" ]; then
    KLEINHIRN_CUDA_VISIBLE_DEVICES="$(preferred_cuda_visible_devices_from_census || true)"
  fi
  KLEINHIRN_MN_LOCAL_WORLD_SIZE="$(resolve_mn_local_world_size "$KLEINHIRN_CUDA_VISIBLE_DEVICES" "${CENSUS_GPU_COUNT:-0}")"
fi

if [ "${CENSUS_GPU_COUNT:-0}" -gt 1 ] && [ "$KLEINHIRN_MULTI_GPU_MODE" = "auto_device_map" ]; then
  KLEINHIRN_DISABLE_NCCL="1"
  KLEINHIRN_MN_LOCAL_WORLD_SIZE=""
elif [ "${CENSUS_GPU_COUNT:-0}" -gt 1 ] \
  && [ "$KLEINHIRN_MULTI_GPU_MODE" = "tensor_parallel" ] \
  && [ "$KLEINHIRN_TENSOR_PARALLEL_BACKEND" = "nccl" ]; then
  KLEINHIRN_DISABLE_NCCL=""
elif [ -z "${CTO_AGENT_KLEINHIRN_DISABLE_NCCL:-}" ]; then
  KLEINHIRN_DISABLE_NCCL=""
fi

CTO_AGENT_COMPACT_SIMPLE_MODEL="$(normalize_compact_model_value "${CTO_AGENT_COMPACT_SIMPLE_MODEL:-$KLEINHIRN_RUNTIME_MODEL}")"
CTO_AGENT_COMPACT_MEDIUM_MODEL="$(normalize_compact_model_value "${CTO_AGENT_COMPACT_MEDIUM_MODEL:-openai/gpt-5.4-mini}")"
CTO_AGENT_COMPACT_RED_MODEL="$(normalize_compact_model_value "${CTO_AGENT_COMPACT_RED_MODEL:-${CTO_AGENT_GROSSHIRN_MODEL:-openai/gpt-5.4}}")"

echo "[6/10] Write Kleinhirn environment"
write_kleinhirn_env_file
if [ -z "$CTO_JAMI_ACCOUNT_ID" ]; then
  echo "[6/10] Jami daemon/runtime prepared, but no CTO_JAMI_ACCOUNT_ID was discovered yet."
  echo "       Configure the Jami account in the TUI settings or copy an existing ~/.config/jami and ~/.local/share/jami profile onto this host." >&2
fi

echo "[7/10] Install browser runtime (KDE/Chrome/extension)"
CTO_AGENT_SUDO_PASSWORD="${CTO_AGENT_SUDO_PASSWORD:-}" \
  CTO_AGENT_INSTALL_KDE_DESKTOP="${CTO_AGENT_INSTALL_KDE_DESKTOP:-0}" \
  sh "$ROOT/scripts/install_browser_engine.sh"

echo "[8/10] Install and start Linux user services"
sh "$ROOT/scripts/install_linux_user_services.sh"
if ! cuda_runtime_driver_ready; then
  echo "CTO-Agent installation finished provisioning, but the CUDA runtime is not currently usable. Reboot the host and rerun the installer to complete Kleinhirn startup." >&2
  exit 1
fi
systemctl --user restart cto-jami-daemon.service >/dev/null 2>&1 || true
systemctl --user restart cto-kleinhirn.service

echo "[9/10] Wait for selected Kleinhirn startup readiness (${KLEINHIRN_OFFICIAL_LABEL})"
while :; do
  ATTEMPTS=240
  i=1
  startup_ready=0
  while [ "$i" -le "$ATTEMPTS" ]; do
    if env CTO_AGENT_KLEINHIRN_BASE_URL="$KLEINHIRN_BASE_URL" CTO_AGENT_KLEINHIRN_API_KEY="local-kleinhirn" \
      "$ROOT/target/release/cto-agent" wait-kleinhirn-startup >/tmp/cto_kleinhirn_ready.out 2>/tmp/cto_kleinhirn_ready.err
    then
      startup_ready=1
      break
    fi
    sleep 5
    i=$((i + 1))
  done

  model_ready=0
  if [ "$startup_ready" = "1" ] && env CTO_AGENT_KLEINHIRN_BASE_URL="$KLEINHIRN_BASE_URL" CTO_AGENT_KLEINHIRN_API_KEY="local-kleinhirn" \
    "$ROOT/target/release/cto-agent" check-kleinhirn >/tmp/cto_kleinhirn_check.out 2>/tmp/cto_kleinhirn_check.err
  then
    model_ready=1
  fi

  if [ "$model_ready" = "1" ]; then
    echo "Kleinhirn startup ready:"
    cat /tmp/cto_kleinhirn_ready.out
    echo "Kleinhirn readiness probe passed:"
    cat /tmp/cto_kleinhirn_check.out
    break
  fi

  if kleinhirn_failure_supports_context_backoff; then
    NEXT_MAX_SEQ_LEN="$(next_context_backoff_value "$KLEINHIRN_MAX_SEQ_LEN" || true)"
    if [ -n "$NEXT_MAX_SEQ_LEN" ] && [ "$NEXT_MAX_SEQ_LEN" != "$KLEINHIRN_MAX_SEQ_LEN" ]; then
      PREVIOUS_MAX_SEQ_LEN="$KLEINHIRN_MAX_SEQ_LEN"
      KLEINHIRN_MAX_SEQ_LEN="$NEXT_MAX_SEQ_LEN"
      if [ "$KLEINHIRN_PAGED_ATTN_MODE" != "off" ] && [ -n "$KLEINHIRN_PA_CTXT_LEN" ] && [ "$KLEINHIRN_PA_CTXT_LEN" -gt "$KLEINHIRN_MAX_SEQ_LEN" ]; then
        KLEINHIRN_PA_CTXT_LEN="$KLEINHIRN_MAX_SEQ_LEN"
      fi
      echo "Kleinhirn readiness hit an OOM/CUDA/runtime crash signature; reducing max context from ${PREVIOUS_MAX_SEQ_LEN} to ${KLEINHIRN_MAX_SEQ_LEN} and retrying." >&2
      write_kleinhirn_env_file
      systemctl --user restart cto-kleinhirn.service
      continue
    fi
  fi

  echo "Selected Kleinhirn startup readiness failed (${KLEINHIRN_OFFICIAL_LABEL}). systemd logs:" >&2
  [ -f /tmp/cto_kleinhirn_ready.err ] && cat /tmp/cto_kleinhirn_ready.err >&2 || true
  [ -f /tmp/cto_kleinhirn_check.err ] && cat /tmp/cto_kleinhirn_check.err >&2 || true
  systemctl --user status cto-kleinhirn.service --no-pager >&2 || true
  journalctl --user -u cto-kleinhirn.service -n 120 --no-pager >&2 || true
  exit 1
done

systemctl --user restart cto-agent.service

echo "[10/10] Wait for always-on loop and heartbeat"
i=1
while [ "$i" -le 60 ]; do
  if curl -k -fsS "$READY_URL" >/tmp/cto_health.out 2>/dev/null; then
    "$HOME/.local/bin/cto-agent" status >/tmp/cto_cli_wrapper_status.out
    "$HOME/.local/bin/cto-mail" list --db "$ROOT/runtime/cto_agent.db" --limit 1 >/tmp/cto_mail_wrapper_status.out
    "$HOME/.local/bin/cto-jami" list --db "$ROOT/runtime/cto_agent.db" --limit 1 >/tmp/cto_jami_wrapper_status.out
    if [ "$(uname -s)" = "Linux" ] && [ "${CTO_AGENT_START_BROWSER_AGENT_CHROME:-1}" = "1" ]; then
      echo "Starting Chrome with Browser-Agent extension..."
      CTO_AGENT_WAIT_FOR_BROWSER_AGENT_BRIDGE=1 \
        CTO_AGENT_BROWSER_AGENT_BRIDGE_URL="${CTO_AGENT_BROWSER_AGENT_BRIDGE_URL:-http://127.0.0.1:8765}" \
        sh "$ROOT/scripts/launch_browser_agent_chrome.sh" || {
          echo "Browser-Agent Chrome launch failed." >&2
          exit 1
        }
    fi
    echo "Installation erfolgreich."
    echo "Health: $(cat /tmp/cto_health.out)"
    if [ -f "$HEARTBEAT_FILE" ]; then
      echo "Heartbeat:"
      cat "$HEARTBEAT_FILE"
    fi
    if [ -t 0 ] && [ -t 1 ] && [ "${CTO_AGENT_SKIP_ATTACH:-0}" != "1" ]; then
      echo "Wechsle jetzt in das Infinity-Loop-Terminal..."
      exec "$ROOT/target/release/cto-agent" attach
    fi
    exit 0
  fi
  sleep 2
  i=$((i + 1))
done

echo "Always-on heartbeat check failed." >&2
curl -k -fsS "$HEALTH_URL" >&2 || true
systemctl --user status cto-agent.service --no-pager >&2 || true
journalctl --user -u cto-agent.service -n 120 --no-pager >&2 || true
exit 1
