#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${CTOX_REPO_URL:-https://github.com/metric-space-ai/CTOX.git}"
INSTALL_DIR="${CTOX_INSTALL_DIR:-$HOME/ctox}"

if ! command -v git >/dev/null 2>&1; then
  echo "git is required to install CTOX" >&2
  exit 1
fi

if [[ ! -x "$HOME/.cargo/bin/cargo" ]] && ! command -v cargo >/dev/null 2>&1; then
  echo "cargo is required to install CTOX" >&2
  exit 1
fi

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
