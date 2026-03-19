#!/bin/sh
set -eu

ROOT="${CTO_AGENT_ROOT:-$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)}"
cd "$ROOT"

ENV_FILE="$ROOT/runtime/kleinhirn.env"
if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
fi

LOG_DIR="$ROOT/runtime/logs"
LOG_FILE="$LOG_DIR/watchdog.log"
HEALTH_URL="${CTO_AGENT_WATCHDOG_URL:-https://127.0.0.1:8443/readyz}"
mkdir -p "$LOG_DIR"

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log_line() {
  printf '[%s] %s\n' "$(timestamp)" "$1" >> "$LOG_FILE"
}

if curl -k -fsS "$HEALTH_URL" >/dev/null 2>&1; then
  exit 0
fi

if [ -x "$ROOT/target/release/cto-agent" ]; then
  "$ROOT/target/release/cto-agent" hard-reset-report "watchdog readyz failure before automatic restart" >/dev/null 2>&1 || true
fi

log_line "Health check failed for $HEALTH_URL; restarting cto-agent.service"
systemctl --user restart cto-agent.service || true
sleep 8

if curl -k -fsS "$HEALTH_URL" >/dev/null 2>&1; then
  log_line "CTO-Agent recovered after service restart"
  exit 0
fi

log_line "CTO-Agent still unhealthy; restarting kleinhirn and agent"
systemctl --user restart cto-kleinhirn.service || true
sleep 6
systemctl --user restart cto-agent.service || true
sleep 8

if curl -k -fsS "$HEALTH_URL" >/dev/null 2>&1; then
  log_line "CTO-Agent recovered after kleinhirn+agent restart"
  exit 0
fi

log_line "Watchdog could not restore healthy state"
exit 1
