#!/usr/bin/env bash
set -euo pipefail

ROOT="${CTOX_ROOT:-$(pwd)}"
MODEL="${1:-}"
PROXY_HOST="${CTOX_PROXY_HOST:-127.0.0.1}"
PROXY_PORT="${CTOX_PROXY_PORT:-12434}"
WORK_DIR="${CTOX_LOOP_SMOKE_DIR:-$ROOT/runtime/ctox_loop_smoke}"
REQUEST_TIMEOUT="${CTOX_LOOP_SMOKE_TIMEOUT:-600}"

if [[ -z "$MODEL" ]]; then
  echo "usage: $0 <model-id>" >&2
  exit 64
fi

mkdir -p "$WORK_DIR"

slugify() {
  printf '%s' "$1" | tr '/ :.' '_' | tr -cd '[:alnum:]_-'
}

MODEL_SLUG="$(slugify "$MODEL")"
RUN_DIR="$WORK_DIR/$MODEL_SLUG"
mkdir -p "$RUN_DIR"

is_local_qwen_model() {
  case "$1" in
    Qwen/Qwen3.5-4B|Qwen/Qwen3.5-9B|Qwen/Qwen3.5-27B|Qwen/Qwen3.5-35B-A3B|nvidia/Nemotron-Cascade-2-30B-A3B)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

is_local_glm_model() {
  case "$1" in
    zai-org/GLM-4.7-Flash)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

qwen_context_window() {
  case "$1" in
    Qwen/Qwen3.5-27B)
      printf '4096'
      ;;
    Qwen/Qwen3.5-35B-A3B)
      printf '2048'
      ;;
    nvidia/Nemotron-Cascade-2-30B-A3B)
      printf '8192'
      ;;
    *)
      printf '4096'
      ;;
  esac
}

qwen_compact_limit() {
  case "$1" in
    Qwen/Qwen3.5-27B)
      printf '3584'
      ;;
    Qwen/Qwen3.5-35B-A3B)
      printf '1536'
      ;;
    nvidia/Nemotron-Cascade-2-30B-A3B)
      printf '1536'
      ;;
    *)
      printf '3072'
      ;;
  esac
}

toml_multiline_override() {
  local key="$1"
  local value="$2"
  printf '%s="""%s"""' "$key" "$value"
}

CODEX_EXEC_BIN="$ROOT/references/openai-codex/codex-rs/target/release/codex-exec"
CTOX_BIN="$ROOT/target/release/ctox"

if [[ ! -x "$CTOX_BIN" ]]; then
  CTOX_BIN="$ROOT/target/debug/ctox"
fi

if [[ ! -x "$CODEX_EXEC_BIN" ]]; then
  CODEX_EXEC_BIN="$HOME/.cargo/bin/cargo run --quiet --release -p codex-exec --bin codex-exec --manifest-path $ROOT/references/openai-codex/codex-rs/Cargo.toml --"
fi

if [[ ! -x "$CTOX_BIN" ]]; then
  "$HOME/.cargo/bin/cargo" build --release --quiet --manifest-path "$ROOT/Cargo.toml"
  CTOX_BIN="$ROOT/target/release/ctox"
fi

ensure_proxy_running() {
  if curl -fsS --max-time 5 "http://$PROXY_HOST:$PROXY_PORT/ctox/telemetry" >"$RUN_DIR/telemetry.json" 2>/dev/null; then
    return 0
  fi

  pkill -f 'ctox serve-responses-proxy' || true
  pkill -f 'target/debug/ctox serve-responses-proxy' || true
  fuser -k "${PROXY_PORT}/tcp" >/dev/null 2>&1 || true
  nohup setsid "$CTOX_BIN" serve-responses-proxy </dev/null >"$RUN_DIR/proxy.log" 2>&1 &

  for _ in $(seq 1 60); do
    if curl -fsS --max-time 5 "http://$PROXY_HOST:$PROXY_PORT/ctox/telemetry" >"$RUN_DIR/telemetry.json" 2>/dev/null; then
      return 0
    fi
    sleep 1
  done

  echo "proxy failed to become healthy on $PROXY_HOST:$PROXY_PORT" >&2
  tail -n 120 "$RUN_DIR/proxy.log" >&2 || true
  return 1
}

ensure_proxy_running

curl -fsS --max-time "$REQUEST_TIMEOUT" \
  -H 'content-type: application/json' \
  -d "{\"model\":\"$MODEL\"}" \
  "http://$PROXY_HOST:$PROXY_PORT/ctox/switch" >"$RUN_DIR/switch.json"

run_codex_exec() {
  local prompt="$1"
  local out_file="$2"
  local prompt_file="$3"
  printf '%s\n' "$prompt" >"$prompt_file"
  local -a extra_config=()
  local include_apply_patch_config='include_apply_patch_tool=true'
  local extra_config_string=""
  local bundled_catalog="$ROOT/references/openai-codex/codex-rs/core/models.json"

  if is_local_qwen_model "$MODEL"; then
    local qwen_window
    qwen_window="$(qwen_context_window "$MODEL")"
    local qwen_compact
    qwen_compact="$(qwen_compact_limit "$MODEL")"
    include_apply_patch_config='include_apply_patch_tool=false'
    extra_config+=(
      "-c" "$(toml_multiline_override "base_instructions" "You are Codex running through CTOX on a local Qwen model. Be concise and tool-accurate. Prefer exec_command for shell work and simple file edits. When the user asks for an exact marker, reply with only that marker after completing any required tool calls.")"
      "-c" "model_context_window=$qwen_window"
      "-c" "model_auto_compact_token_limit=$qwen_compact"
      "-c" "model_catalog_json=\"$bundled_catalog\""
    )
    extra_config_string="-c '$(toml_multiline_override "base_instructions" "You are Codex running through CTOX on a local Qwen model. Be concise and tool-accurate. Prefer exec_command for shell work and simple file edits. When the user asks for an exact marker, reply with only that marker after completing any required tool calls.")' -c 'model_context_window=$qwen_window' -c 'model_auto_compact_token_limit=$qwen_compact' -c 'model_catalog_json=\"$bundled_catalog\"'"
  elif is_local_glm_model "$MODEL"; then
    include_apply_patch_config='include_apply_patch_tool=false'
    extra_config+=(
      "-c" "$(toml_multiline_override "base_instructions" "You are Codex running through CTOX on a local GLM model. Be concise and tool-accurate. Prefer exec_command for shell work and simple file edits. When the user asks for an exact marker, reply with only that marker after completing any required tool calls.")"
      "-c" "model_context_window=2048"
      "-c" "model_auto_compact_token_limit=1536"
      "-c" "model_catalog_json=\"$bundled_catalog\""
    )
    extra_config_string="-c '$(toml_multiline_override "base_instructions" "You are Codex running through CTOX on a local GLM model. Be concise and tool-accurate. Prefer exec_command for shell work and simple file edits. When the user asks for an exact marker, reply with only that marker after completing any required tool calls.")' -c 'model_context_window=2048' -c 'model_auto_compact_token_limit=1536' -c 'model_catalog_json=\"$bundled_catalog\"'"
  fi

  if [[ "$CODEX_EXEC_BIN" == *"cargo run"* ]]; then
    timeout "$REQUEST_TIMEOUT" bash -lc \
      "$CODEX_EXEC_BIN -m \"\$0\" --skip-git-repo-check --json --dangerously-bypass-approvals-and-sandbox -c 'model_provider=\"cto_local\"' -c 'model_providers.cto_local={name=\"cto-local\",base_url=\"http://$PROXY_HOST:$PROXY_PORT/v1\",wire_api=\"responses\",requires_openai_auth=false}' $extra_config_string -c '$include_apply_patch_config' -c 'web_search=\"disabled\"' -- \"\$1\"" \
      "$MODEL" "$prompt" >"$out_file"
  else
    timeout "$REQUEST_TIMEOUT" "$CODEX_EXEC_BIN" \
      -m "$MODEL" \
      --skip-git-repo-check \
      --json \
      --dangerously-bypass-approvals-and-sandbox \
      -c 'model_provider="cto_local"' \
      -c "model_providers.cto_local={name=\"cto-local\",base_url=\"http://$PROXY_HOST:$PROXY_PORT/v1\",wire_api=\"responses\",requires_openai_auth=false}" \
      "${extra_config[@]}" \
      -c "$include_apply_patch_config" \
      -c 'web_search="disabled"' \
      -- "$prompt" >"$out_file"
  fi
}

step() {
  local step_id="$1"
  local expected="$2"
  local prompt="$3"
  local out_file="$RUN_DIR/${step_id}.jsonl"
  local prompt_file="$RUN_DIR/${step_id}.prompt.txt"

  run_codex_exec "$prompt" "$out_file" "$prompt_file"
  if ! grep -q "$expected" "$out_file"; then
    echo "step $step_id failed: expected marker $expected" >&2
    tail -n 40 "$out_file" >&2 || true
    return 1
  fi
}

step "01_reply" "LOOP_OK" \
  "Reply with LOOP_OK and nothing else."

step "02_shell" "EXISTS" \
  "Use exactly one shell command to check whether AGENTS.md exists at the repository root. Then answer with only EXISTS or MISSING."

TEMP_FILE="/tmp/ctox_loop_smoke_${MODEL_SLUG}.txt"
step "03_file" "FILE_OK" \
  "Use tools to write the exact text LOOP_FILE_OK to $TEMP_FILE. Then verify the file contents yourself and answer with only FILE_OK."

curl -fsS --max-time 5 "http://$PROXY_HOST:$PROXY_PORT/ctox/telemetry" >"$RUN_DIR/telemetry_after.json"

cat <<EOF
CTOX loop smoke passed for $MODEL
artifacts: $RUN_DIR
EOF
