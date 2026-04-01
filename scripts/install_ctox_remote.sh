#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${CTOX_REPO_URL:-https://github.com/metric-space-ai/CTOX.git}"
INSTALL_DIR="${CTOX_INSTALL_DIR:-$HOME/ctox}"
WIPE_EXISTING="${CTOX_REMOTE_WIPE_EXISTING:-1}"

run_sudo() {
  if command -v sudo >/dev/null 2>&1; then
    sudo "$@"
    return
  fi
  "$@"
}

apt_package_installed() {
  dpkg-query -W -f='${Status}' "$1" 2>/dev/null | grep -q "install ok installed"
}

ensure_apt_bootstrap_packages() {
  if [[ "$(uname -s)" != "Linux" ]] || ! command -v apt-get >/dev/null 2>&1; then
    return 0
  fi

  local packages=()
  local package=""
  for package in ca-certificates curl git build-essential pkg-config libssl-dev; do
    apt_package_installed "$package" || packages+=("$package")
  done

  if [[ "${#packages[@]}" -eq 0 ]]; then
    return 0
  fi

  echo "[prep] Install Ubuntu bootstrap packages: ${packages[*]}"
  run_sudo apt-get update
  run_sudo apt-get install -y "${packages[@]}"
}

ensure_rust_toolchain() {
  if [[ -x "$HOME/.cargo/bin/cargo" ]]; then
    export PATH="$HOME/.cargo/bin:$PATH"
    return 0
  fi

  if command -v cargo >/dev/null 2>&1; then
    return 0
  fi

  if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required to bootstrap Rust for CTOX" >&2
    exit 1
  fi

  echo "[prep] Install Rust toolchain via rustup"
  curl --proto '=https' --tlsv1.2 -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal
  # shellcheck disable=SC1090
  source "$HOME/.cargo/env"
}

stop_ctox_user_services() {
  if [[ "$(uname -s)" != "Linux" ]] || ! command -v systemctl >/dev/null 2>&1; then
    return 0
  fi

  systemctl --user stop ctox.service >/dev/null 2>&1 || true
  systemctl --user disable ctox.service >/dev/null 2>&1 || true
  systemctl --user stop cto-jami-daemon.service >/dev/null 2>&1 || true
  systemctl --user disable cto-jami-daemon.service >/dev/null 2>&1 || true
  rm -f "${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/ctox.service"
  rm -f "${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/cto-jami-daemon.service"
  systemctl --user daemon-reload >/dev/null 2>&1 || true
}

kill_residual_ctox_processes() {
  if ! command -v pkill >/dev/null 2>&1; then
    return 0
  fi

  pkill -x ctox >/dev/null 2>&1 || true
  pkill -x ctox-engine >/dev/null 2>&1 || true
  pkill -x codex-exec >/dev/null 2>&1 || true
  pkill -f "$INSTALL_DIR/scripts/engine/run_engine.sh" >/dev/null 2>&1 || true
  pkill -f "$INSTALL_DIR/scripts/run_jami_daemon.sh" >/dev/null 2>&1 || true
  pkill -f "$INSTALL_DIR/runtime/browser/interactive-reference" >/dev/null 2>&1 || true
}

wipe_existing_installation() {
  if [[ ! -e "$INSTALL_DIR" ]]; then
    return 0
  fi

  echo "[prep] Wipe existing CTOX install at $INSTALL_DIR"
  stop_ctox_user_services
  kill_residual_ctox_processes
  rm -rf "$INSTALL_DIR"
}

ensure_apt_bootstrap_packages
ensure_rust_toolchain

if [[ "$WIPE_EXISTING" != "0" && "$WIPE_EXISTING" != "false" && "$WIPE_EXISTING" != "FALSE" && "$WIPE_EXISTING" != "no" && "$WIPE_EXISTING" != "NO" ]]; then
  wipe_existing_installation
fi

if [[ -d "$INSTALL_DIR/.git" ]]; then
  git -C "$INSTALL_DIR" fetch --all --prune
  git -C "$INSTALL_DIR" reset --hard origin/main
else
  if [[ -e "$INSTALL_DIR" ]]; then
    wipe_existing_installation
  fi
  git clone "$REPO_URL" "$INSTALL_DIR"
fi

cd "$INSTALL_DIR"
./scripts/install_ctox.sh
