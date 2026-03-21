#!/bin/sh
set -eu

find_jami_daemon_bin() {
  if [ -n "${CTO_JAMI_DAEMON_BIN:-}" ]; then
    if [ -x "${CTO_JAMI_DAEMON_BIN:-}" ]; then
      printf '%s\n' "$CTO_JAMI_DAEMON_BIN"
      return 0
    fi
    if command -v "$CTO_JAMI_DAEMON_BIN" >/dev/null 2>&1; then
      command -v "$CTO_JAMI_DAEMON_BIN"
      return 0
    fi
  fi

  for candidate in /usr/libexec/jamid jamid jami-daemon; do
    if [ -x "$candidate" ]; then
      printf '%s\n' "$candidate"
      return 0
    fi
    if command -v "$candidate" >/dev/null 2>&1; then
      command -v "$candidate"
      return 0
    fi
  done

  return 1
}

DBUS_ENV_FILE="${XDG_RUNTIME_DIR:-/tmp}/cto-jami-dbus.env"
JAMI_CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/jami"
JAMI_DATA_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/jami"
JAMI_CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/jami"

mkdir -p "$JAMI_CONFIG_DIR" "$JAMI_DATA_DIR" "$JAMI_CACHE_DIR"

JAMI_DAEMON_BIN="$(find_jami_daemon_bin || true)"
if [ -z "$JAMI_DAEMON_BIN" ]; then
  echo "No Jami daemon binary found. Install jami-daemon or set CTO_JAMI_DAEMON_BIN." >&2
  exit 1
fi

if command -v dbus-launch >/dev/null 2>&1; then
  dbus-launch --auto-syntax >"$DBUS_ENV_FILE"
  # shellcheck disable=SC1090
  . "$DBUS_ENV_FILE"
fi

set -- -p
if [ -n "${CTO_JAMI_DAEMON_ARGS:-}" ]; then
  # shellcheck disable=SC2086
  set -- $CTO_JAMI_DAEMON_ARGS
fi

exec "$JAMI_DAEMON_BIN" "$@"
