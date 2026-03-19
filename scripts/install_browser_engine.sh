#!/bin/sh
set -eu

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
CFT_ROOT="$ROOT/runtime/chrome-for-testing"
CFT_BINARY="$CFT_ROOT/chrome-linux64/chrome"

have_command() {
  command -v "$1" >/dev/null 2>&1
}

run_sudo() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
    return
  fi
  if ! have_command sudo; then
    echo "sudo is required for browser engine installation." >&2
    exit 1
  fi
  if [ -n "${CTO_AGENT_SUDO_PASSWORD:-}" ]; then
    printf '%s\n' "$CTO_AGENT_SUDO_PASSWORD" | sudo -S "$@"
  else
    sudo "$@"
  fi
}

have_chrome() {
  if [ -x "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" ]; then
    return 0
  fi
  if have_command google-chrome-stable; then
    return 0
  fi
  if have_command google-chrome; then
    return 0
  fi
  if have_command chromium; then
    return 0
  fi
  if have_command chromium-browser; then
    return 0
  fi
  return 1
}

have_browser_agent_chrome() {
  [ -x "$CFT_BINARY" ] || have_command chromium || have_command chromium-browser
}

want_kde_desktop() {
  case "$(printf '%s' "${CTO_AGENT_INSTALL_KDE_DESKTOP:-1}" | tr '[:upper:]' '[:lower:]')" in
    0|false|no|off)
      return 1
      ;;
    *)
      return 0
      ;;
  esac
}

install_kde_desktop_linux() {
  if ! want_kde_desktop; then
    echo "Skipping KDE desktop installation (CTO_AGENT_INSTALL_KDE_DESKTOP=0)."
    return
  fi

  kde_packages=""
  if apt-cache show kde-plasma-desktop >/dev/null 2>&1; then
    kde_packages="kde-plasma-desktop"
  else
    kde_packages="plasma-desktop plasma-workspace"
  fi

  echo "Installing KDE desktop runtime for browser agent..."
  run_sudo apt-get install -y \
    $kde_packages \
    konsole \
    dbus-x11 \
    xdg-utils \
    xauth
}

stage_browser_agent_extension() {
  if [ -f "$ROOT/scripts/install_browser_agent_extension.sh" ]; then
    sh "$ROOT/scripts/install_browser_agent_extension.sh"
  fi
}

install_chrome_for_testing_linux() {
  if [ -x "$CFT_BINARY" ]; then
    echo "Chrome for Testing runtime already present."
    return
  fi
  run_sudo apt-get install -y unzip
  tmp_json="/tmp/cto-chrome-for-testing.json"
  tmp_zip="/tmp/cto-chrome-for-testing.zip"
  tmp_dir="$(mktemp -d /tmp/cto-cft.XXXXXX)"
  echo "Resolving Chrome for Testing stable download..."
  curl -fsSL -o "$tmp_json" \
    "https://googlechromelabs.github.io/chrome-for-testing/last-known-good-versions-with-downloads.json"
  cft_url="$(python3 - "$tmp_json" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    data = json.load(handle)

downloads = (
    data.get("channels", {})
    .get("Stable", {})
    .get("downloads", {})
    .get("chrome", [])
)
for entry in downloads:
    if entry.get("platform") == "linux64":
        print(entry.get("url", ""))
        raise SystemExit(0)
raise SystemExit(1)
PY
)"
  if [ -z "$cft_url" ]; then
    echo "Failed to resolve Chrome for Testing linux64 download." >&2
    rm -rf "$tmp_dir"
    exit 1
  fi
  echo "Downloading Chrome for Testing..."
  curl -fsSL -o "$tmp_zip" "$cft_url"
  unzip -q -o "$tmp_zip" -d "$tmp_dir"
  rm -rf "$CFT_ROOT"
  mkdir -p "$CFT_ROOT"
  mv "$tmp_dir/chrome-linux64" "$CFT_ROOT/"
  rm -rf "$tmp_dir"
  chmod +x "$CFT_BINARY" >/dev/null 2>&1 || true
  echo "Chrome for Testing installed at $CFT_BINARY"
}

os="$(uname -s)"
case "$os" in
  Darwin)
    if ! have_command brew; then
      echo "Homebrew is required to install Google Chrome on macOS." >&2
      exit 1
    fi
    if have_chrome; then
      echo "Browser engine runtime already present."
    else
      echo "Installing Google Chrome via Homebrew cask..."
      brew install --cask google-chrome
    fi
    stage_browser_agent_extension
    ;;
  Linux)
    if ! have_command apt-get; then
      echo "Linux installer currently supports apt-get based hosts only." >&2
      exit 1
    fi
    export DEBIAN_FRONTEND=noninteractive
    echo "Installing Linux prerequisites for browser engine..."
    run_sudo apt-get update
    run_sudo apt-get install -y \
      ca-certificates \
      curl \
      gnupg \
      python3 \
      unzip \
      xvfb
    install_kde_desktop_linux
    if have_chrome; then
      echo "Google Chrome runtime already present."
    else
      tmp_deb="/tmp/google-chrome-stable_current_amd64.deb"
      echo "Downloading Google Chrome stable package..."
      curl -fsSL -o "$tmp_deb" \
        "https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb"
      echo "Installing Google Chrome..."
      run_sudo apt-get install -y "$tmp_deb"
    fi
    install_chrome_for_testing_linux
    stage_browser_agent_extension
    if have_browser_agent_chrome; then
      echo "Browser-Agent launch runtime ready."
    fi
    echo "Chrome installed. If no real desktop session exists, the engine will run headless/Xvfb-only until the host gets a GUI session."
    ;;
  *)
    echo "Unsupported operating system for browser engine installer: $os" >&2
    exit 1
    ;;
esac

echo "Browser engine installation completed."
