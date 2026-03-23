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

read_saved_dbus_bus_pid() {
  if [ ! -f "$DBUS_ENV_FILE" ]; then
    return 0
  fi
  awk -F= '
    /^DBUS_SESSION_BUS_PID=/ {
      gsub(/[^0-9]/, "", $2)
      if ($2 != "") {
        print $2
        exit
      }
    }
  ' "$DBUS_ENV_FILE"
}

cleanup_jami_bus_runtime() {
  saved_bus_pid="$(read_saved_dbus_bus_pid || true)"
  if [ -n "${saved_bus_pid:-}" ]; then
    kill "$saved_bus_pid" >/dev/null 2>&1 || true
  fi
  rm -f "$DBUS_ENV_FILE"
}

JAMI_DAEMON_BIN="$(find_jami_daemon_bin || true)"
if [ -z "$JAMI_DAEMON_BIN" ]; then
  echo "No Jami daemon binary found. Install jami-daemon or set CTO_JAMI_DAEMON_BIN." >&2
  exit 1
fi

cleanup_jami_bus_runtime
if command -v dbus-launch >/dev/null 2>&1; then
  tmp_dbus_env="$(mktemp "${DBUS_ENV_FILE}.XXXXXX")"
  dbus-launch --auto-syntax >"$tmp_dbus_env"
  chmod 600 "$tmp_dbus_env" >/dev/null 2>&1 || true
  mv "$tmp_dbus_env" "$DBUS_ENV_FILE"
  # shellcheck disable=SC1090
  . "$DBUS_ENV_FILE"
  export DBUS_SESSION_BUS_ADDRESS DBUS_SESSION_BUS_PID DBUS_SESSION_BUS_WINDOWID
  export CTO_JAMI_DBUS_ENV_FILE="$DBUS_ENV_FILE"
fi

set -- -p
if [ -n "${CTO_JAMI_DAEMON_ARGS:-}" ]; then
  # shellcheck disable=SC2086
  set -- $CTO_JAMI_DAEMON_ARGS
fi

daemon_pid=""
forward_and_wait() {
  signal="$1"
  if [ -n "${daemon_pid:-}" ]; then
    kill "-$signal" "$daemon_pid" >/dev/null 2>&1 || true
    wait "$daemon_pid" >/dev/null 2>&1 || true
  fi
  cleanup_jami_bus_runtime
  exit 0
}

trap 'forward_and_wait TERM' TERM
trap 'forward_and_wait INT' INT
trap 'forward_and_wait HUP' HUP

"$JAMI_DAEMON_BIN" "$@" &
daemon_pid=$!
if wait "$daemon_pid"; then
  status=0
else
  status=$?
fi
cleanup_jami_bus_runtime
exit "$status"
