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

: "${CTO_AGENT_KLEINHIRN_MODEL:?missing CTO_AGENT_KLEINHIRN_MODEL}"
: "${CTO_AGENT_KLEINHIRN_PORT:?missing CTO_AGENT_KLEINHIRN_PORT}"
: "${CTO_AGENT_KLEINHIRN_RUNTIME_MODEL:=$CTO_AGENT_KLEINHIRN_MODEL}"
: "${CTO_AGENT_KLEINHIRN_SERVER_IMPL:=mistralrs}"
: "${CTO_AGENT_KLEINHIRN_ARCH:=}"

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

exec "$@"
