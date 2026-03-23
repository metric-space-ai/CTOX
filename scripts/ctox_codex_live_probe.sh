#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/metricspace/ctox-public-20260322-full"
cd "$ROOT"

if [[ ! -x target/debug/ctox ]]; then
  "$HOME/.cargo/bin/cargo" build --quiet
fi

pkill -f 'target/debug/ctox serve-responses-proxy' || true
nohup target/debug/ctox serve-responses-proxy >/tmp/ctox_proxy.log 2>&1 &

for _ in $(seq 1 30); do
  if curl -s http://127.0.0.1:12434/ctox/telemetry >/tmp/ctox_proxy_telemetry.json 2>/dev/null; then
    break
  fi
  sleep 1
done

echo '--- PROXY TELEMETRY ---'
cat /tmp/ctox_proxy_telemetry.json || true
echo
echo '--- CODEX EXEC ---'
"$ROOT/references/openai-codex/codex-rs/target/release/codex-exec" \
  -m openai/gpt-oss-20b \
  --skip-git-repo-check \
  --json \
  -c 'model_provider="cto_local"' \
  -c 'model_providers.cto_local={name="cto-local",base_url="http://127.0.0.1:12434/v1",wire_api="responses",requires_openai_auth=false}' \
  -c 'include_apply_patch_tool=false' \
  -c 'web_search="disabled"' \
  'Reply with CTOX_OK and nothing else.' > /tmp/ctox_codex_exec_smoke.jsonl

cat /tmp/ctox_codex_exec_smoke.jsonl
echo
echo '--- PROXY LOG ---'
tail -n 120 /tmp/ctox_proxy.log || true
