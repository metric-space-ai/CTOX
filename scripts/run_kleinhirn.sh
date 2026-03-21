#!/bin/sh
set -eu

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$ROOT"

ENV_FILE="$ROOT/runtime/kleinhirn.env"
if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
fi

prepend_compatible_cuda_runtime() {
  if [ "$(uname -s)" != "Linux" ]; then
    return
  fi

  driver_cuda_major="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9][0-9]*\)\..*/\1/p' | head -n 1)"
  if [ -z "$driver_cuda_major" ]; then
    return
  fi

  for candidate in \
    "/usr/local/cuda-${driver_cuda_major}/targets/x86_64-linux/lib" \
    "/usr/local/cuda-${driver_cuda_major}/lib64"
  do
    if [ ! -d "$candidate" ]; then
      continue
    fi
    case ":${LD_LIBRARY_PATH:-}:" in
      *":$candidate:"*)
        return
        ;;
      *)
        export LD_LIBRARY_PATH="$candidate${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        return
        ;;
    esac
  done
}

: "${CTO_AGENT_KLEINHIRN_MODEL:?missing CTO_AGENT_KLEINHIRN_MODEL}"
: "${CTO_AGENT_KLEINHIRN_PORT:?missing CTO_AGENT_KLEINHIRN_PORT}"
: "${CTO_AGENT_KLEINHIRN_RUNTIME_MODEL:=$CTO_AGENT_KLEINHIRN_MODEL}"
: "${CTO_AGENT_KLEINHIRN_SERVER_IMPL:=mistralrs}"
: "${CTO_AGENT_KLEINHIRN_ARCH:=}"
: "${CTO_AGENT_CONTEXT_EMBEDDING_ENABLED:=0}"
: "${CTO_AGENT_CONTEXT_EMBEDDING_PORT:=1235}"
: "${CTO_AGENT_CONTEXT_EMBEDDING_MODEL:=Qwen/Qwen3-Embedding-0.6B}"
: "${CTO_AGENT_CONTEXT_EMBEDDING_RUNTIME_MODEL:=$CTO_AGENT_CONTEXT_EMBEDDING_MODEL}"
: "${CTO_AGENT_CONTEXT_EMBEDDING_MAX_BATCH_SIZE:=12}"

prepend_compatible_cuda_runtime

if [ -n "${CTO_AGENT_KLEINHIRN_CUDA_VISIBLE_DEVICES:-}" ]; then
  export CUDA_VISIBLE_DEVICES="$CTO_AGENT_KLEINHIRN_CUDA_VISIBLE_DEVICES"
else
  unset CUDA_VISIBLE_DEVICES
fi

if [ "${CTO_AGENT_KLEINHIRN_DISABLE_NCCL:-0}" = "1" ] \
  || [ "${CTO_AGENT_KLEINHIRN_ARCH:-}" = "gpt_oss" ] \
  || [ "${CTO_AGENT_KLEINHIRN_TENSOR_PARALLEL_BACKEND:-}" = "disabled" ]; then
  export MISTRALRS_NO_NCCL=1
else
  unset MISTRALRS_NO_NCCL
fi

if [ -n "${CTO_AGENT_KLEINHIRN_MN_LOCAL_WORLD_SIZE:-}" ]; then
  export MISTRALRS_MN_LOCAL_WORLD_SIZE="$CTO_AGENT_KLEINHIRN_MN_LOCAL_WORLD_SIZE"
else
  unset MISTRALRS_MN_LOCAL_WORLD_SIZE
fi

set -- "$HOME/.cargo/bin/mistralrs" serve \
  --port "$CTO_AGENT_KLEINHIRN_PORT"

if [ -n "${CTO_AGENT_KLEINHIRN_MAX_SEQS:-}" ]; then
  set -- "$@" --max-seqs "$CTO_AGENT_KLEINHIRN_MAX_SEQS"
fi

if [ -n "${CTO_AGENT_KLEINHIRN_MAX_BATCH_SIZE:-}" ]; then
  set -- "$@" --max-batch-size "$CTO_AGENT_KLEINHIRN_MAX_BATCH_SIZE"
fi

if [ -n "${CTO_AGENT_KLEINHIRN_NUM_DEVICE_LAYERS:-}" ]; then
  set -- "$@" --num-device-layers "$CTO_AGENT_KLEINHIRN_NUM_DEVICE_LAYERS"
fi

case "${CTO_AGENT_KLEINHIRN_PAGED_ATTN_MODE:-}" in
  on|off|auto)
    set -- "$@" --paged-attn "${CTO_AGENT_KLEINHIRN_PAGED_ATTN_MODE}"
    ;;
  *)
    if [ "${CTO_AGENT_KLEINHIRN_DISABLE_PAGED_ATTN:-0}" = "1" ]; then
      set -- "$@" --paged-attn off
    fi
    ;;
esac

if [ -n "${CTO_AGENT_KLEINHIRN_DEVICE_LAYERS:-}" ]; then
  set -- "$@" --device-layers "$CTO_AGENT_KLEINHIRN_DEVICE_LAYERS"
fi

if [ -n "${CTO_AGENT_KLEINHIRN_TOPOLOGY:-}" ]; then
  set -- "$@" --topology "$CTO_AGENT_KLEINHIRN_TOPOLOGY"
fi

if [ -n "${CTO_AGENT_KLEINHIRN_PA_GPU_MEM:-}" ]; then
  set -- "$@" --pa-memory-mb "$CTO_AGENT_KLEINHIRN_PA_GPU_MEM"
fi

if [ -n "${CTO_AGENT_KLEINHIRN_PA_GPU_MEM_USAGE:-}" ]; then
  set -- "$@" --pa-memory-fraction "$CTO_AGENT_KLEINHIRN_PA_GPU_MEM_USAGE"
fi

if [ -n "${CTO_AGENT_KLEINHIRN_PA_CTXT_LEN:-}" ]; then
  set -- "$@" --pa-context-len "$CTO_AGENT_KLEINHIRN_PA_CTXT_LEN"
fi

if [ -n "${CTO_AGENT_KLEINHIRN_PA_CACHE_TYPE:-}" ]; then
  set -- "$@" --pa-cache-type "$CTO_AGENT_KLEINHIRN_PA_CACHE_TYPE"
fi

set -- "$@" -m "$CTO_AGENT_KLEINHIRN_RUNTIME_MODEL"

if [ -n "$CTO_AGENT_KLEINHIRN_ARCH" ]; then
  set -- "$@" -a "$CTO_AGENT_KLEINHIRN_ARCH"
fi

if [ -n "${CTO_AGENT_KLEINHIRN_CHAT_TEMPLATE:-}" ]; then
  set -- "$@" --chat-template "$CTO_AGENT_KLEINHIRN_CHAT_TEMPLATE"
fi

if [ -n "${CTO_AGENT_KLEINHIRN_JINJA_EXPLICIT:-}" ]; then
  set -- "$@" --jinja-explicit "$CTO_AGENT_KLEINHIRN_JINJA_EXPLICIT"
fi

if [ -n "${CTO_AGENT_KLEINHIRN_TOKENIZER_JSON:-}" ]; then
  set -- "$@" -t "$CTO_AGENT_KLEINHIRN_TOKENIZER_JSON"
fi

if [ -n "${CTO_AGENT_KLEINHIRN_ISQ:-}" ]; then
  set -- "$@" --isq "$CTO_AGENT_KLEINHIRN_ISQ"
fi

if [ -n "${CTO_AGENT_KLEINHIRN_MAX_SEQ_LEN:-}" ]; then
  set -- "$@" --max-seq-len "$CTO_AGENT_KLEINHIRN_MAX_SEQ_LEN"
fi

EMBED_PID=""

main_runtime_uses_parallel_local_server() {
  case "${CTO_AGENT_KLEINHIRN_MN_LOCAL_WORLD_SIZE:-${MISTRALRS_MN_LOCAL_WORLD_SIZE:-}}" in
    ""|0|1)
      ;;
    *)
      return 0
      ;;
  esac

  if [ -n "${CTO_AGENT_KLEINHIRN_TOPOLOGY:-}" ]; then
    return 0
  fi

  return 1
}

start_context_embedding_server() {
  if [ "${CTO_AGENT_KLEINHIRN_ARCH:-}" = "gpt_oss" ]; then
    echo "Skipping context embedding sidecar for GPT-OSS runtime stability." >&2
    return 0
  fi
  case "$(printf '%s' "${CTO_AGENT_CONTEXT_EMBEDDING_ENABLED:-1}" | tr '[:upper:]' '[:lower:]')" in
    0|false|no|off)
      return 0
      ;;
  esac
  if [ -z "${CTO_AGENT_CONTEXT_EMBEDDING_RUNTIME_MODEL:-}" ]; then
    return 0
  fi
  if main_runtime_uses_parallel_local_server; then
    echo "Skipping context embedding sidecar because the active kleinhirn runtime uses parallel local startup." >&2
    return 0
  fi
  "$HOME/.cargo/bin/mistralrs" serve \
    --port "$CTO_AGENT_CONTEXT_EMBEDDING_PORT" \
    --max-seqs "$CTO_AGENT_CONTEXT_EMBEDDING_MAX_BATCH_SIZE" \
    --max-batch-size "$CTO_AGENT_CONTEXT_EMBEDDING_MAX_BATCH_SIZE" \
    -m "$CTO_AGENT_CONTEXT_EMBEDDING_RUNTIME_MODEL" &
  EMBED_PID=$!
}

cleanup_children() {
  if [ -n "${EMBED_PID:-}" ]; then
    kill "$EMBED_PID" >/dev/null 2>&1 || true
    wait "$EMBED_PID" >/dev/null 2>&1 || true
  fi
}

trap cleanup_children EXIT INT TERM

start_context_embedding_server

"$@" &
MAIN_PID=$!
wait "$MAIN_PID"
MAIN_STATUS=$?
cleanup_children
exit "$MAIN_STATUS"
