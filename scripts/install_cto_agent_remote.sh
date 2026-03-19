#!/bin/sh
set -eu

REPO_URL="${CTO_AGENT_REPO_URL:-https://github.com/metric-space-ai/CTO-Agent.git}"
GIT_REF="${CTO_AGENT_GIT_REF:-main}"
INSTALL_DIR="${CTO_AGENT_INSTALL_DIR:-$HOME/cto-agent}"

normalize_remote() {
  printf '%s' "$1" | sed 's#^[[:space:]]*##; s#[[:space:]]*$##; s#\.git$##'
}

run_sudo() {
  if [ -n "${CTO_AGENT_SUDO_PASSWORD:-}" ]; then
    printf '%s\n' "$CTO_AGENT_SUDO_PASSWORD" | sudo -S "$@"
  else
    sudo "$@"
  fi
}

ensure_git_available() {
  if command -v git >/dev/null 2>&1; then
    return
  fi

  if [ "$(uname -s)" != "Linux" ] || ! command -v apt-get >/dev/null 2>&1 || ! command -v sudo >/dev/null 2>&1; then
    echo "git is required, but no supported auto-install path is available on this host." >&2
    exit 1
  fi

  echo "[bootstrap] Install git"
  run_sudo apt-get update
  run_sudo apt-get install -y git
}

sync_repo() {
  normalized_repo_url="$(normalize_remote "$REPO_URL")"

  if [ -d "$INSTALL_DIR/.git" ]; then
    current_remote="$(git -C "$INSTALL_DIR" remote get-url origin 2>/dev/null || true)"
    if [ -n "$current_remote" ] && [ "$(normalize_remote "$current_remote")" != "$normalized_repo_url" ]; then
      echo "Install dir $INSTALL_DIR already points to a different git remote: $current_remote" >&2
      exit 1
    fi

    echo "[bootstrap] Update existing CTO-Agent checkout in $INSTALL_DIR"
    git -C "$INSTALL_DIR" fetch origin "$GIT_REF" --tags
    git -C "$INSTALL_DIR" checkout "$GIT_REF" >/dev/null 2>&1 || git -C "$INSTALL_DIR" checkout -B "$GIT_REF" "origin/$GIT_REF"
    git -C "$INSTALL_DIR" pull --ff-only origin "$GIT_REF"
    return
  fi

  if [ -e "$INSTALL_DIR" ] && [ -n "$(find "$INSTALL_DIR" -mindepth 1 -maxdepth 1 2>/dev/null | head -n 1)" ]; then
    echo "Install dir $INSTALL_DIR exists and is not an empty CTO-Agent checkout." >&2
    exit 1
  fi

  mkdir -p "$INSTALL_DIR"
  if [ -n "$(find "$INSTALL_DIR" -mindepth 1 -maxdepth 1 2>/dev/null | head -n 1)" ]; then
    echo "Install dir $INSTALL_DIR must be empty before cloning." >&2
    exit 1
  fi

  rmdir "$INSTALL_DIR" 2>/dev/null || true
  echo "[bootstrap] Clone CTO-Agent into $INSTALL_DIR"
  git clone --branch "$GIT_REF" "$REPO_URL" "$INSTALL_DIR"
}

ensure_git_available
sync_repo

cd "$INSTALL_DIR"
exec sh scripts/install_cto_agent.sh
