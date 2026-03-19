#!/bin/sh
set -eu

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
SERVICE_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
BIN_DIR="$HOME/.local/bin"
mkdir -p "$SERVICE_DIR"
mkdir -p "$BIN_DIR"

run_sudo() {
  if [ -n "${CTO_AGENT_SUDO_PASSWORD:-}" ]; then
    printf '%s\n' "$CTO_AGENT_SUDO_PASSWORD" | sudo -S "$@"
  else
    sudo "$@"
  fi
}

if ! command -v systemctl >/dev/null 2>&1; then
  echo "systemctl not found; cannot guarantee always-on service management" >&2
  exit 1
fi

if ! systemctl --user --version >/dev/null 2>&1; then
  echo "systemd user services are unavailable; cannot guarantee always-on mode" >&2
  exit 1
fi

if command -v loginctl >/dev/null 2>&1; then
  LINGER="$(loginctl show-user "$USER" -p Linger --value 2>/dev/null || true)"
  if [ "$LINGER" != "yes" ]; then
    if command -v sudo >/dev/null 2>&1; then
      run_sudo loginctl enable-linger "$USER"
      LINGER="$(loginctl show-user "$USER" -p Linger --value 2>/dev/null || true)"
    fi
    if [ "$LINGER" != "yes" ]; then
      echo "loginctl linger is not enabled for $USER; cannot guarantee always-on across logout/reboot" >&2
      echo "Enable it with: sudo loginctl enable-linger $USER" >&2
      exit 1
    fi
  fi
fi

cat > "$SERVICE_DIR/cto-kleinhirn.service" <<EOF
[Unit]
Description=CTO-Agent Kleinhirn Local Model Server
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=0

[Service]
Type=simple
WorkingDirectory=$ROOT
EnvironmentFile=$ROOT/runtime/kleinhirn.env
ExecStart=$ROOT/scripts/run_kleinhirn.sh
Restart=always
RestartSec=5
KillMode=control-group
KillSignal=SIGTERM
SendSIGKILL=yes
FinalKillSignal=SIGKILL
TimeoutStopSec=120

[Install]
WantedBy=default.target
EOF

cat > "$SERVICE_DIR/cto-agent.service" <<EOF
[Unit]
Description=CTO-Agent Control Plane
After=cto-kleinhirn.service
Wants=cto-kleinhirn.service
StartLimitIntervalSec=0

[Service]
Type=simple
WorkingDirectory=$ROOT
EnvironmentFile=$ROOT/runtime/kleinhirn.env
ExecStart=$ROOT/scripts/run_control_plane.sh
Restart=always
RestartSec=5
TimeoutStopSec=20

[Install]
WantedBy=default.target
EOF

cat > "$SERVICE_DIR/cto-agent-watchdog.service" <<EOF
[Unit]
Description=CTO-Agent Watchdog Check
After=cto-agent.service
Requires=cto-agent.service

[Service]
Type=oneshot
WorkingDirectory=$ROOT
EnvironmentFile=$ROOT/runtime/kleinhirn.env
ExecStart=$ROOT/scripts/watchdog_check.sh
EOF

cat > "$SERVICE_DIR/cto-agent-watchdog.timer" <<EOF
[Unit]
Description=Run CTO-Agent watchdog every minute

[Timer]
OnBootSec=90
OnUnitActiveSec=60
AccuracySec=15
Persistent=true

[Install]
WantedBy=timers.target
EOF

cat > "$BIN_DIR/cto-agent" <<EOF
#!/bin/sh
exec "$ROOT/scripts/run_control_plane.sh" "\$@"
EOF
chmod +x "$BIN_DIR/cto-agent"

cat > "$BIN_DIR/cto" <<EOF
#!/bin/sh
exec "$ROOT/scripts/run_control_plane.sh" attach "\$@"
EOF
chmod +x "$BIN_DIR/cto"

systemctl --user daemon-reload
chmod +x "$ROOT/scripts/watchdog_check.sh" >/dev/null 2>&1 || true
systemctl --user enable cto-kleinhirn.service cto-agent.service cto-agent-watchdog.timer

case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *)
    echo "CLI wrappers installed in $BIN_DIR" >&2
    echo "Add it to PATH if needed: export PATH=\"$BIN_DIR:\$PATH\"" >&2
    ;;
esac
