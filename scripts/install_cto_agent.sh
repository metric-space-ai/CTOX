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
KLEINHIRN_PROFILE="gpt_oss"
PROFILE_PINNED=1

if [ "$REQUESTED_KLEINHIRN_PROFILE" != "$KLEINHIRN_PROFILE" ]; then
  echo "[policy] Initial installation stays on GPT-OSS 20B; requested profile ${REQUESTED_KLEINHIRN_PROFILE} can only be applied later through a runtime self-upgrade."
fi

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
    KLEINHIRN_POLICY_MODEL="${CTO_AGENT_KLEINHIRN_POLICY_MODEL:-Qwen3.5-0.8B}"
    KLEINHIRN_RUNTIME_MODEL="${CTO_AGENT_KLEINHIRN_RUNTIME_MODEL:-Qwen/Qwen3.5-0.8B}"
    KLEINHIRN_OFFICIAL_LABEL="${CTO_AGENT_KLEINHIRN_OFFICIAL_LABEL:-Qwen3.5 0.8B}"
    KLEINHIRN_AGENTIC_ADAPTER="${CTO_AGENT_KLEINHIRN_AGENTIC_ADAPTER:-openai_compatible_chat}"
    KLEINHIRN_MAX_SEQ_LEN="${CTO_AGENT_KLEINHIRN_MAX_SEQ_LEN:-131072}"
    KLEINHIRN_DISABLE_PAGED_ATTN="${CTO_AGENT_KLEINHIRN_DISABLE_PAGED_ATTN:-0}"
    KLEINHIRN_PA_CTXT_LEN="${CTO_AGENT_KLEINHIRN_PA_CTXT_LEN:-131072}"
    KLEINHIRN_PAGED_ATTN_MODE="${CTO_AGENT_KLEINHIRN_PAGED_ATTN_MODE:-auto}"
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
KLEINHIRN_PA_CTXT_LEN="${CTO_AGENT_KLEINHIRN_PA_CTXT_LEN:-}"
KLEINHIRN_PA_CACHE_TYPE="${CTO_AGENT_KLEINHIRN_PA_CACHE_TYPE:-}"
KLEINHIRN_PAGED_ATTN_MODE="${CTO_AGENT_KLEINHIRN_PAGED_ATTN_MODE:-}"
KLEINHIRN_DEVICE_LAYERS="${CTO_AGENT_KLEINHIRN_DEVICE_LAYERS:-}"
KLEINHIRN_ISQ="${CTO_AGENT_KLEINHIRN_ISQ:-}"
KLEINHIRN_CHAT_TEMPLATE="${CTO_AGENT_KLEINHIRN_CHAT_TEMPLATE:-}"
KLEINHIRN_JINJA_EXPLICIT="${CTO_AGENT_KLEINHIRN_JINJA_EXPLICIT:-}"
KLEINHIRN_TOKENIZER_JSON="${CTO_AGENT_KLEINHIRN_TOKENIZER_JSON:-}"
KLEINHIRN_TOPOLOGY="${CTO_AGENT_KLEINHIRN_TOPOLOGY:-}"
KLEINHIRN_DISABLE_NCCL="${CTO_AGENT_KLEINHIRN_DISABLE_NCCL:-}"

KLEINHIRN_PORT="${CTO_AGENT_KLEINHIRN_PORT:-1234}"
KLEINHIRN_STARTUP_WAIT_SECS="${CTO_AGENT_KLEINHIRN_STARTUP_WAIT_SECS:-900}"
KLEINHIRN_BASE_URL="http://127.0.0.1:${KLEINHIRN_PORT}/v1"
KLEINHIRN_LOG_DIR="$ROOT/runtime/logs"
KLEINHIRN_LOG="$KLEINHIRN_LOG_DIR/kleinhirn.log"
KLEINHIRN_PID_FILE="$ROOT/runtime/kleinhirn.pid"
ENV_FILE="$ROOT/runtime/kleinhirn.env"
HEALTH_URL="https://127.0.0.1:8443/healthz"
READY_URL="https://127.0.0.1:8443/readyz"
HEARTBEAT_FILE="$ROOT/runtime/state/agent_state.json"
CARGO_TARGET_DIR="${CTO_AGENT_MISTRALRS_TARGET_DIR:-$ROOT/runtime/build/mistralrs}"

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
  } > "$ENV_FILE"
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
  awk -v lhs="$1" -v rhs="$2" '
    function part(value, index,    count, pieces) {
      count = split(value, pieces, /\./)
      if (index > count) {
        return 0
      }
      return pieces[index] + 0
    }
    BEGIN {
      for (i = 1; i <= 4; i += 1) {
        left = part(lhs, i)
        right = part(rhs, i)
        if (left > right) {
          exit 0
        }
        if (left < right) {
          exit 1
        }
      }
      exit 0
    }
  '
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
chmod +x \
  "$ROOT/scripts/install_cto_agent.sh" \
  "$ROOT/scripts/install_browser_engine.sh" \
  "$ROOT/scripts/install_browser_agent_extension.sh" \
  "$ROOT/scripts/install_linux_user_services.sh" \
  "$ROOT/scripts/launch_browser_agent_chrome.sh" \
  "$ROOT/scripts/run_kleinhirn.sh" \
  "$ROOT/scripts/run_control_plane.sh" \
  "$ROOT/scripts/start_control_plane.sh" \
  >/dev/null 2>&1 || true

echo "[prep] Stop stale local CTO-Agent processes"
systemctl --user stop cto-agent.service cto-kleinhirn.service >/dev/null 2>&1 || true
pkill -f "$ROOT/target/debug/cto-agent" >/dev/null 2>&1 || true
pkill -f "$ROOT/target/release/cto-agent" >/dev/null 2>&1 || true
pkill -f "$ROOT/scripts/run_control_plane.sh" >/dev/null 2>&1 || true
pkill -f "$ROOT/scripts/run_kleinhirn.sh" >/dev/null 2>&1 || true
rm -f "$HEARTBEAT_FILE"

echo "[1/10] Build CTO-Agent host"
cargo build --release

echo "[2/10] Initialize contracts, TLS and SQLite"
"$ROOT/target/release/cto-agent" --init-only

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
USE_MULTI_GPU_TUNED_DEVICE_LAYERS=0
if [ "${CENSUS_GPU_COUNT:-0}" -gt 1 ] \
  && is_gpt_oss_family \
  && ! mistralrs_uses_nccl \
  && [ -n "${RECOMMENDED_DEVICE_LAYERS:-}" ] \
  && [ -z "$KLEINHIRN_TOPOLOGY" ] \
  && [ "${SELECTED_PREFER_AUTO_DEVICE_MAPPING:-0}" != "1" ]; then
  USE_MULTI_GPU_TUNED_DEVICE_LAYERS=1
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

eval "$(apply_runtime_tune_defaults || true)"
if [ "${CENSUS_GPU_COUNT:-0}" -gt 1 ] \
  && is_gpt_oss_family \
  && ! mistralrs_uses_nccl \
  && [ -n "${RECOMMENDED_DEVICE_LAYERS:-}" ] \
  && [ -z "$KLEINHIRN_TOPOLOGY" ] \
  && [ "${SELECTED_PREFER_AUTO_DEVICE_MAPPING:-0}" != "1" ]; then
  USE_MULTI_GPU_TUNED_DEVICE_LAYERS=1
fi
if [ -z "$KLEINHIRN_ISQ" ] && [ -n "${RECOMMENDED_ISQ:-}" ]; then
  KLEINHIRN_ISQ="$(printf '%s' "$RECOMMENDED_ISQ" | tr '[:upper:]' '[:lower:]')"
fi
if [ -z "$KLEINHIRN_DEVICE_LAYERS" ] && [ -n "${RECOMMENDED_DEVICE_LAYERS:-}" ] && { [ "${CENSUS_GPU_COUNT:-0}" -le 1 ] || [ "$USE_MULTI_GPU_TUNED_DEVICE_LAYERS" = "1" ]; }; then
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
if [ "$KLEINHIRN_PROFILE" = "gpt_oss" ] && [ -z "$KLEINHIRN_ISQ" ]; then
  KLEINHIRN_ISQ="q6k"
fi
if [ -z "$KLEINHIRN_MAX_SEQS" ]; then
  KLEINHIRN_MAX_SEQS="1"
fi
if [ -z "$KLEINHIRN_MAX_BATCH_SIZE" ]; then
  KLEINHIRN_MAX_BATCH_SIZE="1"
fi

if [ "$APPLY_SELECTED_MODEL" = "1" ] && [ "${SELECTED_PREFER_AUTO_DEVICE_MAPPING:-0}" = "1" ] && [ "${CENSUS_GPU_COUNT:-0}" -gt 1 ]; then
  if [ -z "$KLEINHIRN_TOPOLOGY" ]; then
    KLEINHIRN_DEVICE_LAYERS=""
    KLEINHIRN_NUM_DEVICE_LAYERS=""
  fi
  KLEINHIRN_PAGED_ATTN_MODE="on"
  if [ -z "$KLEINHIRN_PA_CACHE_TYPE" ]; then
    KLEINHIRN_PA_CACHE_TYPE="f8e4m3"
  fi
  if [ -z "$KLEINHIRN_MAX_SEQ_LEN" ] && [ -n "${RECOMMENDED_MAX_CONTEXT_TOKENS:-}" ]; then
    KLEINHIRN_MAX_SEQ_LEN="$RECOMMENDED_MAX_CONTEXT_TOKENS"
  fi
  if [ -z "$KLEINHIRN_PA_CTXT_LEN" ] && [ -n "${RECOMMENDED_MAX_CONTEXT_TOKENS:-}" ]; then
    KLEINHIRN_PA_CTXT_LEN="$RECOMMENDED_MAX_CONTEXT_TOKENS"
  fi
fi

if [ "${CENSUS_GPU_COUNT:-0}" -gt 1 ] && [ -z "$KLEINHIRN_TOPOLOGY" ] && [ "$USE_MULTI_GPU_TUNED_DEVICE_LAYERS" != "1" ]; then
  KLEINHIRN_DEVICE_LAYERS=""
  KLEINHIRN_NUM_DEVICE_LAYERS=""
fi

if [ "${CENSUS_GPU_COUNT:-0}" -gt 1 ] && [ "$KLEINHIRN_POLICY_MODEL" = "gpt-oss-20b" ] && ! mistralrs_uses_nccl; then
  KLEINHIRN_DISABLE_NCCL="1"
elif [ -z "${CTO_AGENT_KLEINHIRN_DISABLE_NCCL:-}" ]; then
  KLEINHIRN_DISABLE_NCCL=""
fi

echo "[6/10] Write Kleinhirn environment"
write_kleinhirn_env_file

echo "[7/10] Install browser runtime (KDE/Chrome/extension)"
CTO_AGENT_SUDO_PASSWORD="${CTO_AGENT_SUDO_PASSWORD:-}" \
  CTO_AGENT_INSTALL_KDE_DESKTOP="${CTO_AGENT_INSTALL_KDE_DESKTOP:-1}" \
  sh "$ROOT/scripts/install_browser_engine.sh"

echo "[8/10] Install and start Linux user services"
sh "$ROOT/scripts/install_linux_user_services.sh"
if ! cuda_runtime_driver_ready; then
  echo "CTO-Agent installation finished provisioning, but the CUDA runtime is not currently usable. Reboot the host and rerun the installer to complete Kleinhirn startup." >&2
  exit 1
fi
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
