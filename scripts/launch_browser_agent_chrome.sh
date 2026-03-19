#!/bin/sh
set -eu

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
EXTENSION_DIR="$ROOT/runtime/browser-agent-extension"
PROFILE_DIR="$ROOT/runtime/browser-agent-chrome-profile"
CFT_BINARY="$ROOT/runtime/chrome-for-testing/chrome-linux64/chrome"
LOG_DIR="$ROOT/runtime/logs"
LOG_PATH="$LOG_DIR/browser-agent-chrome.log"
PID_PATH="$ROOT/runtime/browser-agent-chrome.pid"
STATE_PATH="$ROOT/runtime/browser-agent-chrome.state"
START_URL="${CTO_AGENT_BROWSER_AGENT_START_URL:-about:blank}"
BRIDGE_URL="${CTO_AGENT_BROWSER_AGENT_BRIDGE_URL:-http://127.0.0.1:8765}"
WAIT_FOR_BRIDGE="${CTO_AGENT_WAIT_FOR_BROWSER_AGENT_BRIDGE:-0}"
DISABLE_FEATURES="ExtensionDisableUnsupportedDeveloper"
UNSUPPORTED_DEVELOPER_EXTENSION_REASON="16777216"

have_command() {
  command -v "$1" >/dev/null 2>&1
}

find_chrome_binary() {
  if [ -x "$CFT_BINARY" ]; then
    printf '%s\n' "$CFT_BINARY"
    return 0
  fi
  for candidate in \
    "/usr/bin/chromium" \
    "/usr/bin/chromium-browser"
  do
    if [ -x "$candidate" ]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  for name in chromium chromium-browser; do
    if have_command "$name"; then
      command -v "$name"
      return 0
    fi
  done
  for candidate in \
    "/usr/bin/google-chrome-stable" \
    "/usr/bin/google-chrome" \
    "/opt/google/chrome/google-chrome"
  do
    if [ -x "$candidate" ]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  for name in google-chrome-stable google-chrome; do
    if have_command "$name"; then
      command -v "$name"
      return 0
    fi
  done
  return 1
}

detect_runtime_dir() {
  if [ -n "${XDG_RUNTIME_DIR:-}" ] && [ -d "${XDG_RUNTIME_DIR:-}" ]; then
    printf '%s\n' "$XDG_RUNTIME_DIR"
    return 0
  fi
  candidate="/run/user/$(id -u)"
  if [ -d "$candidate" ]; then
    printf '%s\n' "$candidate"
    return 0
  fi
  return 1
}

process_env_value() {
  key="$1"
  ps eww -u "$(id -un)" 2>/dev/null | awk -v key="$key" '
    {
      for (i = 1; i <= NF; i += 1) {
        if (index($i, key "=") == 1) {
          print substr($i, length(key) + 2)
          exit
        }
      }
    }
  '
}

detect_display() {
  if [ -n "${DISPLAY:-}" ]; then
    printf '%s\n' "$DISPLAY"
    return 0
  fi
  value="$(process_env_value DISPLAY || true)"
  if [ -n "$value" ]; then
    printf '%s\n' "$value"
    return 0
  fi
  for socket in /tmp/.X11-unix/X*; do
    if [ ! -e "$socket" ]; then
      continue
    fi
    base="$(basename "$socket")"
    printf ':%s\n' "${base#X}"
    return 0
  done
  return 1
}

detect_wayland_display() {
  if [ -n "${WAYLAND_DISPLAY:-}" ]; then
    printf '%s\n' "$WAYLAND_DISPLAY"
    return 0
  fi
  value="$(process_env_value WAYLAND_DISPLAY || true)"
  if [ -n "$value" ]; then
    printf '%s\n' "$value"
    return 0
  fi
  runtime_dir="$(detect_runtime_dir || true)"
  if [ -n "$runtime_dir" ]; then
    for socket in "$runtime_dir"/wayland-*; do
      if [ -S "$socket" ]; then
        basename "$socket"
        return 0
      fi
    done
  fi
  return 1
}

detect_xauthority() {
  if [ -n "${XAUTHORITY:-}" ] && [ -f "${XAUTHORITY:-}" ]; then
    printf '%s\n' "$XAUTHORITY"
    return 0
  fi
  value="$(process_env_value XAUTHORITY || true)"
  if [ -n "$value" ] && [ -f "$value" ]; then
    printf '%s\n' "$value"
    return 0
  fi
  runtime_dir="$(detect_runtime_dir || true)"
  for candidate in \
    "$HOME/.Xauthority" \
    "$runtime_dir/gdm/Xauthority" \
    "$runtime_dir/.Xauthority"
  do
    if [ -n "$candidate" ] && [ -f "$candidate" ]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

detect_dbus_address() {
  if [ -n "${DBUS_SESSION_BUS_ADDRESS:-}" ]; then
    printf '%s\n' "$DBUS_SESSION_BUS_ADDRESS"
    return 0
  fi
  value="$(process_env_value DBUS_SESSION_BUS_ADDRESS || true)"
  if [ -n "$value" ]; then
    printf '%s\n' "$value"
    return 0
  fi
  runtime_dir="$(detect_runtime_dir || true)"
  if [ -n "$runtime_dir" ] && [ -S "$runtime_dir/bus" ]; then
    printf 'unix:path=%s/bus\n' "$runtime_dir"
    return 0
  fi
  return 1
}

wait_for_bridge_worker() {
  if [ "$WAIT_FOR_BRIDGE" = "0" ]; then
    return 0
  fi
  if ! have_command curl; then
    echo "curl missing; skipping Browser-Agent bridge wait." >&2
    return 0
  fi
  timeout_s="${CTO_AGENT_BROWSER_AGENT_BRIDGE_TIMEOUT_S:-90}"
  freshness_s="${CTO_AGENT_BROWSER_AGENT_WORKER_FRESHNESS_S:-180}"
  i=0
  while [ "$i" -lt "$timeout_s" ]; do
    payload="$(curl -fsS "$BRIDGE_URL/api/browser-agent/bridge-state" 2>/dev/null || true)"
    if [ -n "$payload" ] && printf '%s' "$payload" | python3 -c 'import datetime as dt,json,sys; freshness=int(sys.argv[1]); data=json.load(sys.stdin); bridge=data.get("bridge") or {}; workers=bridge.get("activeWorkers") or data.get("activeWorkers") or []; now=dt.datetime.now(dt.timezone.utc); fresh=[]; 
for worker in workers:
  updated_at=(worker or {}).get("updatedAt");
  if not updated_at:
    continue
  try:
    parsed=dt.datetime.fromisoformat(updated_at.replace("Z","+00:00"))
  except Exception:
    continue
  if (now - parsed).total_seconds() <= freshness:
    fresh.append(worker)
raise SystemExit(0 if fresh else 1)' "$freshness_s" >/dev/null 2>&1; then
      printf '%s\n' "$payload"
      return 0
    fi
    sleep 1
    i=$((i + 1))
  done
  echo "Browser-Agent bridge did not observe an active extension worker within ${timeout_s}s." >&2
  return 1
}

profile_has_unsupported_extension_disable_reason() {
  prefs_path="$PROFILE_DIR/Default/Preferences"
  if [ ! -f "$prefs_path" ]; then
    return 1
  fi
  python3 - "$prefs_path" "$EXTENSION_DIR" "$UNSUPPORTED_DEVELOPER_EXTENSION_REASON" <<'PY'
import json
import pathlib
import sys

prefs_path = pathlib.Path(sys.argv[1])
extension_dir = sys.argv[2]
target_reason = int(sys.argv[3])

try:
    data = json.loads(prefs_path.read_text())
except Exception:
    raise SystemExit(1)

settings = ((data.get("extensions") or {}).get("settings") or {})
for entry in settings.values():
    if str(entry.get("path") or "").strip() != extension_dir:
        continue
    reasons = entry.get("disable_reasons") or []
    raise SystemExit(0 if target_reason in reasons else 1)

raise SystemExit(1)
PY
}

kill_browser_process() {
  pid=""
  if [ -f "$PID_PATH" ]; then
    pid="$(cat "$PID_PATH" 2>/dev/null || true)"
  fi
  if [ -n "$pid" ] && kill -0 "$pid" >/dev/null 2>&1; then
    kill "$pid" >/dev/null 2>&1 || true
    sleep 2
  fi
}

reset_browser_profile() {
  rm -rf "$PROFILE_DIR"
  mkdir -p "$PROFILE_DIR"
  rm -f "$PID_PATH" "$STATE_PATH"
}

launch_browser_process() {
  nohup env \
    "$CHROME_BINARY" \
    "--user-data-dir=$PROFILE_DIR" \
    "--no-first-run" \
    "--no-default-browser-check" \
    "--no-sandbox" \
    "--disable-setuid-sandbox" \
    "--disable-session-crashed-bubble" \
    "--disable-features=$DISABLE_FEATURES" \
    "--disable-extensions-except=$EXTENSION_DIR" \
    "--load-extension=$EXTENSION_DIR" \
    "--new-window" \
    "$START_URL" \
    >>"$LOG_PATH" 2>&1 &
  browser_pid=$!

  printf '%s\n' "$browser_pid" > "$PID_PATH"
  {
    printf 'browser_pid=%s\n' "$browser_pid"
    printf 'chrome_binary=%s\n' "$CHROME_BINARY"
    printf 'extension_dir=%s\n' "$EXTENSION_DIR"
    printf 'profile_dir=%s\n' "$PROFILE_DIR"
    printf 'bridge_url=%s\n' "$BRIDGE_URL"
    printf 'display=%s\n' "$DISPLAY_VALUE"
    printf 'wayland_display=%s\n' "$WAYLAND_VALUE"
    printf 'dbus_session_bus_address=%s\n' "$DBUS_VALUE"
    printf 'xauthority=%s\n' "$XAUTHORITY_VALUE"
    printf 'disable_features=%s\n' "$DISABLE_FEATURES"
  } > "$STATE_PATH"

  printf 'Browser-Agent Chrome launch requested.\n'
  printf '  pid: %s\n' "$browser_pid"
  printf '  chrome: %s\n' "$CHROME_BINARY"
  printf '  extension: %s\n' "$EXTENSION_DIR"
  printf '  profile: %s\n' "$PROFILE_DIR"
  printf '  log: %s\n' "$LOG_PATH"
  if [ -n "$DISPLAY_VALUE" ]; then
    printf '  display: %s\n' "$DISPLAY_VALUE"
  fi
  if [ -n "$WAYLAND_VALUE" ]; then
    printf '  wayland: %s\n' "$WAYLAND_VALUE"
  fi
}

mkdir -p "$LOG_DIR" "$PROFILE_DIR"

if [ ! -f "$EXTENSION_DIR/manifest.json" ]; then
  sh "$ROOT/scripts/install_browser_agent_extension.sh"
fi

CHROME_BINARY="$(find_chrome_binary || true)"
if [ -z "$CHROME_BINARY" ]; then
  echo "No Chrome/Chromium binary found. Run scripts/install_browser_engine.sh first." >&2
  exit 1
fi

DISPLAY_VALUE="$(detect_display || true)"
WAYLAND_VALUE="$(detect_wayland_display || true)"
DBUS_VALUE="$(detect_dbus_address || true)"
XAUTHORITY_VALUE="$(detect_xauthority || true)"
RUNTIME_DIR="$(detect_runtime_dir || true)"

if [ -z "$DISPLAY_VALUE" ] && [ -z "$WAYLAND_VALUE" ]; then
  echo "No interactive desktop session detected for launching Chrome." >&2
  exit 1
fi

if [ -n "$DISPLAY_VALUE" ]; then
  DISPLAY="$DISPLAY_VALUE"
  export DISPLAY
fi
if [ -n "$WAYLAND_VALUE" ]; then
  WAYLAND_DISPLAY="$WAYLAND_VALUE"
  export WAYLAND_DISPLAY
fi
if [ -n "$DBUS_VALUE" ]; then
  DBUS_SESSION_BUS_ADDRESS="$DBUS_VALUE"
  export DBUS_SESSION_BUS_ADDRESS
fi
if [ -n "$XAUTHORITY_VALUE" ]; then
  XAUTHORITY="$XAUTHORITY_VALUE"
  export XAUTHORITY
fi
if [ -n "$RUNTIME_DIR" ]; then
  XDG_RUNTIME_DIR="$RUNTIME_DIR"
  export XDG_RUNTIME_DIR
fi

if [ -f "$PID_PATH" ]; then
  old_pid="$(cat "$PID_PATH" 2>/dev/null || true)"
  if [ -n "$old_pid" ] && kill -0 "$old_pid" >/dev/null 2>&1; then
    echo "Browser-Agent Chrome already running with pid $old_pid."
    wait_for_bridge_worker
    exit 0
  fi
fi

launch_browser_process

if wait_for_bridge_worker; then
  exit 0
fi

if profile_has_unsupported_extension_disable_reason; then
  echo "Detected unpacked Browser-Agent extension disabled by Chrome developer-mode gating; resetting the dedicated profile and retrying." >&2
  kill_browser_process
  reset_browser_profile
  launch_browser_process
  wait_for_bridge_worker
  exit 0
fi

exit 1
