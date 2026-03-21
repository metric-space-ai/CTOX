#!/bin/sh
set -eu

REPO_URL="${CTO_AGENT_REPO_URL:-https://github.com/metric-space-ai/CTO-Agent.git}"
GIT_REF="${CTO_AGENT_GIT_REF:-main}"
INSTALL_DIR="${CTO_AGENT_INSTALL_DIR:-$HOME/cto-agent}"

normalize_remote() {
  printf '%s' "$1" | sed 's#^[[:space:]]*##; s#[[:space:]]*$##; s#\.git$##'
}

timestamp_suffix() {
  date -u +%Y%m%d-%H%M%S 2>/dev/null || date +%Y%m%d-%H%M%S
}

next_backup_dir() {
  stamp="$(timestamp_suffix)"
  candidate="${INSTALL_DIR}-backup-${stamp}"
  suffix=1
  while [ -e "$candidate" ]; do
    candidate="${INSTALL_DIR}-backup-${stamp}-${suffix}"
    suffix=$((suffix + 1))
  done
  printf '%s\n' "$candidate"
}

backup_install_dir() {
  reason="$1"
  backup_dir="$(next_backup_dir)"
  echo "[bootstrap] Move existing install dir to $backup_dir ($reason)"
  mv "$INSTALL_DIR" "$backup_dir"
}

stop_existing_install_services() {
  if ! command -v systemctl >/dev/null 2>&1; then
    return
  fi

  systemctl --user stop \
    cto-agent-watchdog.timer \
    cto-agent-watchdog.service \
    cto-agent.service \
    cto-kleinhirn.service >/dev/null 2>&1 || true
  systemctl --user reset-failed \
    cto-agent-watchdog.timer \
    cto-agent-watchdog.service \
    cto-agent.service \
    cto-kleinhirn.service >/dev/null 2>&1 || true
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

clone_repo() {
  attempts=0
  while [ "$attempts" -lt 3 ]; do
    stop_existing_install_services
    mkdir -p "$INSTALL_DIR"
    if [ -z "$(find "$INSTALL_DIR" -mindepth 1 -maxdepth 1 2>/dev/null | head -n 1)" ]; then
      rmdir "$INSTALL_DIR" 2>/dev/null || true
      echo "[bootstrap] Clone CTO-Agent into $INSTALL_DIR"
      git clone --branch "$GIT_REF" "$REPO_URL" "$INSTALL_DIR"
      return
    fi

    backup_install_dir "install dir was repopulated while preparing fresh clone"
    attempts=$((attempts + 1))
  done

  echo "Install dir $INSTALL_DIR could not be kept empty long enough for a fresh clone." >&2
  exit 1
}

prepare_fresh_clone() {
  stop_existing_install_services
  clone_repo
}

sync_repo() {
  normalized_repo_url="$(normalize_remote "$REPO_URL")"

  if [ -d "$INSTALL_DIR/.git" ]; then
    current_remote="$(git -C "$INSTALL_DIR" remote get-url origin 2>/dev/null || true)"
    if [ -n "$current_remote" ] && [ "$(normalize_remote "$current_remote")" != "$normalized_repo_url" ]; then
      backup_install_dir "existing checkout points to a different git remote: $current_remote"
      prepare_fresh_clone
      return
    fi

    if [ -n "$(git -C "$INSTALL_DIR" status --porcelain --untracked-files=all 2>/dev/null || true)" ]; then
      backup_install_dir "existing checkout has local or untracked changes"
      prepare_fresh_clone
      return
    fi

    echo "[bootstrap] Update existing CTO-Agent checkout in $INSTALL_DIR"
    git -C "$INSTALL_DIR" fetch origin "$GIT_REF" --tags
    if ! git -C "$INSTALL_DIR" checkout "$GIT_REF" >/dev/null 2>&1; then
      git -C "$INSTALL_DIR" checkout -B "$GIT_REF" "origin/$GIT_REF"
    fi
    if ! git -C "$INSTALL_DIR" pull --ff-only origin "$GIT_REF"; then
      backup_install_dir "fast-forward update failed"
      prepare_fresh_clone
    fi
    return
  fi

  if [ -e "$INSTALL_DIR" ] && [ -n "$(find "$INSTALL_DIR" -mindepth 1 -maxdepth 1 2>/dev/null | head -n 1)" ]; then
    backup_install_dir "existing install dir is not an empty CTO-Agent checkout"
    prepare_fresh_clone
    return
  fi

  prepare_fresh_clone
}

ensure_git_available
sync_repo

cd "$INSTALL_DIR"
exec sh scripts/install_cto_agent.sh
