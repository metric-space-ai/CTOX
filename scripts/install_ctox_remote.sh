#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${CTOX_REPO_URL:-https://github.com/metric-space-ai/CTOX.git}"
INSTALL_DIR="${CTOX_INSTALL_DIR:-$HOME/ctox}"

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

ensure_apt_bootstrap_packages
ensure_rust_toolchain

if [[ -d "$INSTALL_DIR/.git" ]]; then
  git -C "$INSTALL_DIR" pull --ff-only
else
  if [[ -e "$INSTALL_DIR" ]]; then
    backup_dir="${INSTALL_DIR}-backup-$(date +%Y%m%d-%H%M%S)"
    mv "$INSTALL_DIR" "$backup_dir"
  fi
  git clone "$REPO_URL" "$INSTALL_DIR"
fi

cd "$INSTALL_DIR"
./scripts/install_ctox.sh
