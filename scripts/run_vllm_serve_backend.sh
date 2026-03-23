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

detect_all_visible_nvidia_devices() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi
  nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null \
    | sed '/^[[:space:]]*$/d' \
    | paste -sd, -
}

count_csv_items() {
  local csv="${1:-}"
  [[ -n "$csv" ]] || {
    printf '0\n'
    return
  }
  awk -F',' '{print NF}' <<<"$csv"
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

: "${CTOX_VLLM_SERVE_MODEL:=openai/gpt-oss-20b}"
: "${CTOX_VLLM_SERVE_PORT:=1234}"
: "${CTOX_VLLM_SERVE_ARCH:=gpt_oss}"
: "${CTOX_VLLM_SERVE_MAX_SEQS:=1}"
: "${CTOX_VLLM_SERVE_MAX_BATCH_SIZE:=1}"

configure_compatible_cuda_runtime

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

if [[ -n "${CTOX_VLLM_SERVE_CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$CTOX_VLLM_SERVE_CUDA_VISIBLE_DEVICES"
else
  auto_cuda_visible_devices="$(detect_all_visible_nvidia_devices || true)"
  if [[ -n "${auto_cuda_visible_devices:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="$auto_cuda_visible_devices"
  else
    unset CUDA_VISIBLE_DEVICES || true
  fi
fi

if [[ "${CTOX_VLLM_SERVE_DISABLE_NCCL:-0}" == "1" ]] \
  || [[ "${CTOX_VLLM_SERVE_TENSOR_PARALLEL_BACKEND:-}" == "disabled" ]]; then
  export MISTRALRS_NO_NCCL=1
else
  unset MISTRALRS_NO_NCCL || true
fi

if [[ -n "${CTOX_VLLM_SERVE_MN_LOCAL_WORLD_SIZE:-}" ]]; then
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

VLLM_SERVE_BIN="$(resolve_vllm_serve_binary)"
if [[ -z "$VLLM_SERVE_BIN" ]]; then
  echo "vllm-serve binary not found; run scripts/install_ctox.sh first" >&2
  exit 1
fi

if is_qwen35_vision_model; then
  set -- "$VLLM_SERVE_BIN" serve \
    -p "$CTOX_VLLM_SERVE_PORT" \
    vision \
    -m "$CTOX_VLLM_SERVE_MODEL"
  if [[ -n "${CTOX_VLLM_SERVE_MAX_SEQ_LEN:-}" ]]; then
    set -- "$@" --max-seq-len "$CTOX_VLLM_SERVE_MAX_SEQ_LEN"
  fi
  if [[ -n "${CTOX_VLLM_SERVE_PAGED_ATTN:-}" ]]; then
    set -- "$@" --paged-attn "$CTOX_VLLM_SERVE_PAGED_ATTN"
  fi
  if [[ -n "${CTOX_VLLM_SERVE_PA_CACHE_TYPE:-}" ]]; then
    set -- "$@" --pa-cache-type "$CTOX_VLLM_SERVE_PA_CACHE_TYPE"
  fi
  if [[ -n "${CTOX_VLLM_SERVE_PA_MEMORY_FRACTION:-}" ]]; then
    set -- "$@" --pa-memory-fraction "$CTOX_VLLM_SERVE_PA_MEMORY_FRACTION"
  fi
  if [[ -n "${CTOX_VLLM_SERVE_PA_CONTEXT_LEN:-}" ]]; then
    set -- "$@" --pa-context-len "$CTOX_VLLM_SERVE_PA_CONTEXT_LEN"
  fi
  if [[ -n "${CTOX_VLLM_SERVE_NUM_DEVICE_LAYERS:-}" ]]; then
    set -- "$@" --num-device-layers "$CTOX_VLLM_SERVE_NUM_DEVICE_LAYERS"
  fi
  if [[ -n "${CTOX_VLLM_SERVE_TOPOLOGY:-}" ]]; then
    set -- "$@" --topology "$CTOX_VLLM_SERVE_TOPOLOGY"
  fi
else
  set -- "$VLLM_SERVE_BIN" serve \
    --port "$CTOX_VLLM_SERVE_PORT" \
    --max-seqs "$CTOX_VLLM_SERVE_MAX_SEQS" \
    --max-batch-size "$CTOX_VLLM_SERVE_MAX_BATCH_SIZE" \
    -m "$CTOX_VLLM_SERVE_MODEL"

  if [[ -n "${CTOX_VLLM_SERVE_ARCH:-}" ]]; then
    set -- "$@" -a "$CTOX_VLLM_SERVE_ARCH"
  fi

  if [[ -n "${CTOX_VLLM_SERVE_PAGED_ATTN:-}" ]]; then
    set -- "$@" --paged-attn "$CTOX_VLLM_SERVE_PAGED_ATTN"
  fi
  if [[ -n "${CTOX_VLLM_SERVE_PA_CACHE_TYPE:-}" ]]; then
    set -- "$@" --pa-cache-type "$CTOX_VLLM_SERVE_PA_CACHE_TYPE"
  fi
  if [[ -n "${CTOX_VLLM_SERVE_PA_MEMORY_FRACTION:-}" ]]; then
    set -- "$@" --pa-memory-fraction "$CTOX_VLLM_SERVE_PA_MEMORY_FRACTION"
  fi
  if [[ -n "${CTOX_VLLM_SERVE_PA_CONTEXT_LEN:-}" ]]; then
    set -- "$@" --pa-context-len "$CTOX_VLLM_SERVE_PA_CONTEXT_LEN"
  fi

  if [[ -n "${CTOX_VLLM_SERVE_NUM_DEVICE_LAYERS:-}" ]]; then
    set -- "$@" --num-device-layers "$CTOX_VLLM_SERVE_NUM_DEVICE_LAYERS"
  fi

  if [[ -n "${CTOX_VLLM_SERVE_TOPOLOGY:-}" ]]; then
    set -- "$@" --topology "$CTOX_VLLM_SERVE_TOPOLOGY"
  fi

  if [[ -n "${CTOX_VLLM_SERVE_MAX_SEQ_LEN:-}" ]]; then
    set -- "$@" --max-seq-len "$CTOX_VLLM_SERVE_MAX_SEQ_LEN"
  fi
fi

if [[ -n "${CTOX_VLLM_SERVE_ISQ:-}" ]]; then
  set -- "$@" --isq "$CTOX_VLLM_SERVE_ISQ"
fi

if [[ -n "${CTOX_VLLM_SERVE_ISQ_ORGANIZATION:-}" ]]; then
  set -- "$@" --isq-organization "$CTOX_VLLM_SERVE_ISQ_ORGANIZATION"
fi

exec "$@"
