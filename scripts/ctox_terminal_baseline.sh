#!/usr/bin/env bash
set -euo pipefail

ROOT="${CTOX_ROOT:-$(pwd)}"
MODEL="${1:-}"
PROXY_HOST="${CTOX_PROXY_HOST:-127.0.0.1}"
PROXY_PORT="${CTOX_PROXY_PORT:-12434}"
WORK_DIR="${CTOX_TERMINAL_BASELINE_DIR:-$ROOT/runtime/ctox_terminal_baseline}"
REQUEST_TIMEOUT="${CTOX_TERMINAL_BASELINE_TIMEOUT:-900}"

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
FIXTURE_DIR="/tmp/ctox_terminal_fixture_${MODEL_SLUG}"
mkdir -p "$RUN_DIR"

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

prepare_fixture() {
  rm -rf "$FIXTURE_DIR"
  mkdir -p "$FIXTURE_DIR/output"
  cat >"$FIXTURE_DIR/README.txt" <<'EOF'
Project: CTOX terminal baseline
Owner: Michael Welsch
TODO: verify loop
EOF
  cat >"$FIXTURE_DIR/tasks.txt" <<'EOF'
TODO: collect todo count
TODO: extract owner
NOTE: produce deterministic report
EOF
  cat >"$FIXTURE_DIR/notes.md" <<'EOF'
# Fixture Notes

Alpha line: ALPHA_READY
EOF
}

run_codex_exec() {
  local prompt="$1"
  local out_file="$2"
  local prompt_file="$3"
  printf '%s\n' "$prompt" >"$prompt_file"

  if [[ "$CODEX_EXEC_BIN" == *"cargo run"* ]]; then
    timeout "$REQUEST_TIMEOUT" bash -lc \
      "$CODEX_EXEC_BIN -m \"\$0\" --skip-git-repo-check --json --dangerously-bypass-approvals-and-sandbox -c 'model_provider=\"cto_local\"' -c 'model_providers.cto_local={name=\"cto-local\",base_url=\"http://$PROXY_HOST:$PROXY_PORT/v1\",wire_api=\"responses\",requires_openai_auth=false}' -c 'include_apply_patch_tool=true' -c 'web_search=\"disabled\"' -- \"\$1\"" \
      "$MODEL" "$prompt" >"$out_file"
  else
    timeout "$REQUEST_TIMEOUT" "$CODEX_EXEC_BIN" \
      -m "$MODEL" \
      --skip-git-repo-check \
      --json \
      --dangerously-bypass-approvals-and-sandbox \
      -c 'model_provider="cto_local"' \
      -c "model_providers.cto_local={name=\"cto-local\",base_url=\"http://$PROXY_HOST:$PROXY_PORT/v1\",wire_api=\"responses\",requires_openai_auth=false}" \
      -c 'include_apply_patch_tool=true' \
      -c 'web_search="disabled"' \
      -- "$prompt" >"$out_file"
  fi
}

validate_outputs() {
  local report="$FIXTURE_DIR/output/report.json"
  local checksum="$FIXTURE_DIR/output/report.sha256"
  [[ -f "$report" ]] || { echo "missing report.json" >&2; return 1; }
  [[ -f "$checksum" ]] || { echo "missing report.sha256" >&2; return 1; }
  grep -q '"owner":"Michael Welsch"' "$report" || { echo "owner mismatch" >&2; return 1; }
  grep -q '"todo_count":3' "$report" || { echo "todo_count mismatch" >&2; return 1; }
  grep -q '"alpha_line":"ALPHA_READY"' "$report" || { echo "alpha_line mismatch" >&2; return 1; }
  local actual
  actual="$(sha256sum "$report" | awk '{print $1}')"
  grep -q "$actual" "$checksum" || { echo "checksum mismatch" >&2; return 1; }
}

ensure_proxy_running
prepare_fixture

curl -fsS --max-time "$REQUEST_TIMEOUT" \
  -H 'content-type: application/json' \
  -d "{\"model\":\"$MODEL\"}" \
  "http://$PROXY_HOST:$PROXY_PORT/ctox/switch" >"$RUN_DIR/switch.json"

PROMPT="Work only inside $FIXTURE_DIR. Use tools to inspect the fixture files, then create $FIXTURE_DIR/output/report.json as one compact JSON object with exactly these keys and values: owner, todo_count, alpha_line. owner must come from the fixture. todo_count must be the total number of TODO lines across the fixture text files. alpha_line must be the value after 'Alpha line:'. Then create $FIXTURE_DIR/output/report.sha256 containing the sha256 of report.json plus two spaces plus the literal filename report.json. After creating both files, verify them yourself with tools and answer with only TERMINAL_BENCH_OK."

run_codex_exec "$PROMPT" "$RUN_DIR/terminal_task.jsonl" "$RUN_DIR/terminal_task.prompt.txt"
grep -q 'TERMINAL_BENCH_OK' "$RUN_DIR/terminal_task.jsonl" || {
  echo "missing TERMINAL_BENCH_OK marker" >&2
  tail -n 80 "$RUN_DIR/terminal_task.jsonl" >&2 || true
  exit 1
}

validate_outputs
curl -fsS --max-time 5 "http://$PROXY_HOST:$PROXY_PORT/ctox/telemetry" >"$RUN_DIR/telemetry_after.json"

cat <<EOF
CTOX terminal baseline passed for $MODEL
artifacts: $RUN_DIR
fixture: $FIXTURE_DIR
EOF
