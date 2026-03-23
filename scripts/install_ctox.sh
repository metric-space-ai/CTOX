#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

run_sudo() {
  sudo "$@"
}

apt_package_installed() {
  dpkg-query -W -f='${Status}' "$1" 2>/dev/null | grep -q "install ok installed"
}

apt_update_with_retry() {
  if ! command -v apt-get >/dev/null 2>&1; then
    return 0
  fi
  local attempt=1
  while [ "$attempt" -le 3 ]; do
    if run_sudo apt-get update; then
      return 0
    fi
    sleep $((attempt * 2))
    attempt=$((attempt + 1))
  done
  return 1
}

resolve_cargo_bin() {
  if [[ -n "${CARGO_BIN:-}" ]]; then
    printf '%s\n' "$CARGO_BIN"
    return
  fi
  if [[ -x "$HOME/.cargo/bin/cargo" ]]; then
    printf '%s\n' "$HOME/.cargo/bin/cargo"
    return
  fi
  command -v cargo || true
}

ensure_rust_build_toolchain() {
  if command -v cargo >/dev/null 2>&1 || [[ -x "$HOME/.cargo/bin/cargo" ]]; then
    return 0
  fi
  if [[ "$(uname -s)" != "Linux" ]] || ! command -v apt-get >/dev/null 2>&1 || ! command -v curl >/dev/null 2>&1; then
    return 0
  fi

  local packages=()
  local package=""
  for package in build-essential pkg-config libssl-dev ca-certificates; do
    apt_package_installed "$package" || packages+=("$package")
  done

  if [[ "${#packages[@]}" -gt 0 ]] && command -v sudo >/dev/null 2>&1; then
    echo "[prep] Install Rust build prerequisites"
    apt_update_with_retry
    run_sudo apt-get install -y "${packages[@]}"
  fi

  echo "[prep] Install Rust toolchain via rustup"
  curl --proto '=https' --tlsv1.2 -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal
  # shellcheck disable=SC1090
  source "$HOME/.cargo/env"
}

ensure_codex_linux_build_prereqs() {
  if [[ "$(uname -s)" != "Linux" ]] || ! command -v apt-get >/dev/null 2>&1 || ! command -v sudo >/dev/null 2>&1; then
    return 0
  fi

  local packages=()
  local package=""
  for package in libcap-dev; do
    apt_package_installed "$package" || packages+=("$package")
  done

  if [[ "${#packages[@]}" -eq 0 ]]; then
    return 0
  fi

  echo "[prep] Install Linux prerequisites for vendored Codex sandbox binaries"
  apt_update_with_retry
  run_sudo apt-get install -y "${packages[@]}"
}

ensure_project_references_present() {
  local missing=0
  local required_paths=(
    "$ROOT/references/openai-codex/codex-rs/Cargo.toml"
    "$ROOT/ctox-vllm-serve/Cargo.toml"
  )
  local path=""
  for path in "${required_paths[@]}"; do
    if [[ ! -f "$path" ]]; then
      echo "[install] Missing bundled project reference: $path" >&2
      missing=1
    fi
  done

  if [[ "$missing" -ne 0 ]]; then
    cat >&2 <<EOF
CTOX install now expects the vendored references to already be present inside the project tree.
The installer no longer clones openai/codex or the CTOX vllm-serve fork from upstream during install.
Make sure this checkout already contains:
  references/openai-codex
  ctox-vllm-serve
EOF
    exit 1
  fi
}

resolve_codex_home() {
  if [[ -n "${CODEX_HOME:-}" ]]; then
    printf '%s\n' "$CODEX_HOME"
    return
  fi
  printf '%s\n' "$HOME/.codex"
}

sync_repo_system_skills_into_vendored_codex() {
  local src_root dest_root skill_dir skill_name
  src_root="$ROOT/skills/.system"
  dest_root="$ROOT/references/openai-codex/codex-rs/skills/src/assets/samples"

  [[ -d "$src_root" ]] || return 0
  mkdir -p "$dest_root"
  find "$dest_root" -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} +

  for skill_dir in "$src_root"/*; do
    [[ -d "$skill_dir" ]] || continue
    skill_name="$(basename "$skill_dir")"
    cp -R "$skill_dir" "$dest_root/$skill_name"
  done
}

sync_repo_skills_into_codex_home() {
  local codex_home target_root system_target_root src_system_root src_curated_root
  local skill_dir skill_name installed_curated=0 skipped_curated=0

  codex_home="$(resolve_codex_home)"
  target_root="$codex_home/skills"
  system_target_root="$target_root/.system"
  src_system_root="$ROOT/skills/.system"
  src_curated_root="$ROOT/skills/.curated"

  mkdir -p "$target_root" "$system_target_root"

  if [[ -d "$src_system_root" ]]; then
    for skill_dir in "$src_system_root"/*; do
      [[ -d "$skill_dir" ]] || continue
      skill_name="$(basename "$skill_dir")"
      rm -rf "$system_target_root/$skill_name"
      cp -R "$skill_dir" "$system_target_root/$skill_name"
    done
  fi

  if [[ -d "$src_curated_root" ]]; then
    for skill_dir in "$src_curated_root"/*; do
      [[ -d "$skill_dir" ]] || continue
      skill_name="$(basename "$skill_dir")"
      if [[ -d "$src_system_root/$skill_name" ]] || [[ -e "$target_root/$skill_name" ]]; then
        skipped_curated=$((skipped_curated + 1))
        continue
      fi
      cp -R "$skill_dir" "$target_root/$skill_name"
      installed_curated=$((installed_curated + 1))
    done
  fi

  echo "[skills] Bundled skills synced to $target_root"
  echo "[skills] Curated skills installed: $installed_curated, skipped: $skipped_curated"
}

resolve_jami_linux_repo_suffix() {
  [ -r /etc/os-release ] || return 1
  # shellcheck disable=SC1091
  . /etc/os-release
  local distro_id="${ID:-}"
  local version_id="${VERSION_ID:-}"
  local id_like="${ID_LIKE:-}"
  case "$distro_id" in
    ubuntu)
      case "$version_id" in
        20.04|22.04|24.04|24.10|25.04)
          printf 'ubuntu_%s\n' "$version_id"
          return 0
          ;;
      esac
      ;;
    debian)
      case "$version_id" in
        11|12|13)
          printf 'debian_%s\n' "$version_id"
          return 0
          ;;
      esac
      ;;
  esac
  case "$id_like" in
    *ubuntu*)
      case "$version_id" in
        20.04|22.04|24.04|24.10|25.04)
          printf 'ubuntu_%s\n' "$version_id"
          return 0
          ;;
      esac
      ;;
    *debian*)
      case "$version_id" in
        11|12|13)
          printf 'debian_%s\n' "$version_id"
          return 0
          ;;
      esac
      ;;
  esac
  return 1
}

ensure_linux_jami_installed() {
  if [[ "$(uname -s)" != "Linux" ]] || ! command -v apt-get >/dev/null 2>&1 || ! command -v sudo >/dev/null 2>&1; then
    return 0
  fi

  local repo_suffix
  repo_suffix="$(resolve_jami_linux_repo_suffix || true)"
  if [[ -z "$repo_suffix" ]]; then
    echo "[prep] Skip Jami package auto-install because this Linux distribution is not mapped to an official Jami apt repository." >&2
    echo "[prep] Configure Jami manually via https://jami.net/en/download-jami-linux if this host still needs the daemon." >&2
    return 0
  fi

  local jami_repo_line tmp_keyring tmp_list
  jami_repo_line="deb [signed-by=/usr/share/keyrings/jami-archive-keyring.gpg] https://dl.jami.net/stable/${repo_suffix}/ jami main"
  echo "[prep] Install official Jami daemon runtime ($repo_suffix)"
  run_sudo apt-get install -y gnupg dirmngr ca-certificates curl --no-install-recommends
  tmp_keyring="$(mktemp /tmp/ctox-jami-keyring.XXXXXX)"
  curl -fsSL https://dl.jami.net/public-key.gpg -o "$tmp_keyring"
  run_sudo install -m 0644 "$tmp_keyring" /usr/share/keyrings/jami-archive-keyring.gpg
  rm -f "$tmp_keyring"
  tmp_list="$(mktemp /tmp/ctox-jami-repo.XXXXXX)"
  printf '%s\n' "$jami_repo_line" >"$tmp_list"
  run_sudo install -m 0644 "$tmp_list" /etc/apt/sources.list.d/jami.list
  rm -f "$tmp_list"
  apt_update_with_retry
  run_sudo apt-get install -y jami-daemon dbus-x11 libglib2.0-bin
}

jami_daemon_binary_present() {
  [ -x /usr/libexec/jamid ] && return 0
  command -v jamid >/dev/null 2>&1 && return 0
  command -v jami-daemon >/dev/null 2>&1 && return 0
  return 1
}

install_jami_user_service() {
  if [[ "$(uname -s)" != "Linux" ]] || ! command -v systemctl >/dev/null 2>&1; then
    return 0
  fi
  if ! jami_daemon_binary_present; then
    return 0
  fi

  local service_dir
  service_dir="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
  mkdir -p "$service_dir"
  cat > "$service_dir/cto-jami-daemon.service" <<EOF
[Unit]
Description=CTOX Jami Daemon
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=0

[Service]
Type=simple
WorkingDirectory=$ROOT
ExecStart=$ROOT/scripts/run_jami_daemon.sh
Restart=always
RestartSec=5
KillMode=control-group
TimeoutStopSec=20

[Install]
WantedBy=default.target
EOF
  systemctl --user daemon-reload
  systemctl --user enable cto-jami-daemon.service >/dev/null 2>&1 || true
  systemctl --user restart cto-jami-daemon.service >/dev/null 2>&1 || true
}

wait_for_jami_dbus_runtime() {
  if [[ "$(uname -s)" != "Linux" ]] || ! command -v systemctl >/dev/null 2>&1; then
    return 0
  fi
  if ! jami_daemon_binary_present; then
    return 0
  fi
  if ! command -v gdbus >/dev/null 2>&1; then
    echo "Jami daemon was installed but gdbus is unavailable for runtime verification." >&2
    return 1
  fi

  local jami_dbus_env_file attempt
  jami_dbus_env_file="${CTO_JAMI_DBUS_ENV_FILE:-${XDG_RUNTIME_DIR:-/tmp}/cto-jami-dbus.env}"
  systemctl --user restart cto-jami-daemon.service >/dev/null 2>&1 || true

  attempt=1
  while [ "$attempt" -le 30 ]; do
    if systemctl --user is-active --quiet cto-jami-daemon.service && [ -s "$jami_dbus_env_file" ]; then
      if CTO_INSTALL_JAMI_DBUS_ENV_FILE="$jami_dbus_env_file" sh -eu -c '
        DBUS_ENV_FILE="$CTO_INSTALL_JAMI_DBUS_ENV_FILE"
        set -a
        # shellcheck disable=SC1090
        . "$DBUS_ENV_FILE"
        set +a
        gdbus call --session \
          --dest cx.ring.Ring \
          --object-path /cx/ring/Ring/ConfigurationManager \
          --method cx.ring.Ring.ConfigurationManager.getAccountList >/dev/null 2>&1
      '; then
        return 0
      fi
    fi
    sleep 1
    attempt=$((attempt + 1))
  done

  echo "Jami daemon runtime failed post-install health verification." >&2
  echo "Expected a live DBus session via $jami_dbus_env_file and a reachable cx.ring.Ring service." >&2
  systemctl --user status cto-jami-daemon.service --no-pager -n 40 >&2 || true
  return 1
}

latest_apt_package_matching() {
  local pattern="$1"
  apt-cache pkgnames 2>/dev/null | grep -E "$pattern" | sort -V | tail -n 1
}

cuda_toolchain_ready() {
  command -v nvcc >/dev/null 2>&1 || return 1
  command -v ldconfig >/dev/null 2>&1 || return 1
  ldconfig -p 2>/dev/null | grep -q 'libnvrtc' || return 1
  ldconfig -p 2>/dev/null | grep -q 'libcurand' || return 1
  ldconfig -p 2>/dev/null | grep -q 'libcublasLt' || return 1
  ldconfig -p 2>/dev/null | grep -q 'libcublas' || return 1
  return 0
}

vllm_serve_uses_cuda() {
  case " ${MISTRALRS_FEATURES:-} " in
    *" cuda "*) return 0 ;;
    *) return 1 ;;
  esac
}

nccl_packages_available() {
  command -v apt-cache >/dev/null 2>&1 || return 1
  apt-cache policy libnccl2 2>/dev/null | grep -q 'Candidate:'
}

nccl_runtime_missing() {
  if command -v ldconfig >/dev/null 2>&1 && ldconfig -p 2>/dev/null | grep -q 'libnccl'; then
    return 1
  fi
  return 0
}

detect_vllm_serve_features() {
  if [[ -n "${CTOX_VLLM_SERVE_FEATURES:-}" ]]; then
    printf '%s\n' "$CTOX_VLLM_SERVE_FEATURES"
    return
  fi

  if ! cuda_toolchain_ready; then
    printf '%s\n' ""
    return
  fi

  local features="cuda flash-attn"
  if command -v ldconfig >/dev/null 2>&1 && ldconfig -p 2>/dev/null | grep -q 'libnccl'; then
    features="$features nccl"
  fi
  if command -v ldconfig >/dev/null 2>&1 && ldconfig -p 2>/dev/null | grep -q 'libcudnn'; then
    features="$features cudnn"
  fi
  printf '%s\n' "$features"
}

cuda_linker_prereqs_ready() {
  cuda_toolchain_ready
}

ensure_cuda_build_prereqs() {
  if [[ "$(uname -s)" != "Linux" ]] || ! command -v apt-get >/dev/null 2>&1 || ! command -v sudo >/dev/null 2>&1; then
    return
  fi
  if ! vllm_serve_uses_cuda; then
    return
  fi
  if cuda_linker_prereqs_ready; then
    return
  fi

  local cuda_packages=""
  local pkg=""
  local pattern=""
  for pattern in \
    '^cuda-driver-dev-[0-9]+-[0-9]+$' \
    '^cuda-cudart-dev-[0-9]+-[0-9]+$' \
    '^cuda-nvcc-[0-9]+-[0-9]+$' \
    '^cuda-nvrtc-dev-[0-9]+-[0-9]+$' \
    '^libcublas-dev-[0-9]+-[0-9]+$' \
    '^libcurand-dev-[0-9]+-[0-9]+$'
  do
    pkg="$(latest_apt_package_matching "$pattern" || true)"
    [[ -n "$pkg" ]] || continue
    cuda_packages="$cuda_packages $pkg"
  done

  if [[ -n "$cuda_packages" ]]; then
    echo "[prep] Install CUDA build prerequisites for CTOX vllm-serve"
    sudo apt-get update
    # shellcheck disable=SC2086
    sudo apt-get install -y $cuda_packages
  fi
}

detect_cuda_home() {
  if [[ -n "${CTOX_CUDA_HOME:-}" && -d "${CTOX_CUDA_HOME:-}" ]]; then
    printf '%s\n' "$CTOX_CUDA_HOME"
    return
  fi
  local preferred=""
  for preferred in /usr/local/cuda-12.6 /usr/local/cuda-12.5 /usr/local/cuda-12.4 /usr/local/cuda-12.3 /usr/local/cuda-12.2 /usr/local/cuda-12.1 /usr/local/cuda-12; do
    [[ -d "$preferred" ]] || continue
    printf '%s\n' "$preferred"
    return
  done
  if [[ -d /usr/local/cuda ]]; then
    printf '%s\n' "/usr/local/cuda"
    return
  fi
  local candidate=""
  for candidate in /usr/local/cuda-*; do
    [[ -x "$candidate/bin/nvcc" ]] || continue
    printf '%s\n' "$candidate"
    return
  done
  if command -v nvcc >/dev/null 2>&1; then
    local nvcc_path
    nvcc_path="$(command -v nvcc)"
    case "$nvcc_path" in
      */bin/nvcc)
        printf '%s\n' "${nvcc_path%/bin/nvcc}"
        return
        ;;
    esac
  fi
}

detect_cuda_compute_cap() {
  if [[ -n "${CTOX_CUDA_COMPUTE_CAP:-}" ]]; then
    printf '%s\n' "$CTOX_CUDA_COMPUTE_CAP"
    return
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    local compute_cap
    compute_cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '.[:space:]')"
    case "$compute_cap" in
      [0-9][0-9]*) printf '%s\n' "$compute_cap"; return ;;
    esac
  fi
}

configure_cuda_env() {
  if ! vllm_serve_uses_cuda; then
    return
  fi

  local cuda_home
  cuda_home="$(detect_cuda_home || true)"
  if [[ -n "$cuda_home" ]]; then
    export CUDA_HOME="$cuda_home"
    export PATH="$cuda_home/bin:$PATH"
    if [[ -d "$cuda_home/lib64" ]]; then
      export LD_LIBRARY_PATH="$cuda_home/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
  fi

  local compute_cap
  compute_cap="$(detect_cuda_compute_cap || true)"
  if [[ -n "$compute_cap" ]]; then
    export CUDA_COMPUTE_CAP="$compute_cap"
  fi
}

write_vllm_serve_env_file() {
  mkdir -p "$ROOT/runtime"
  local model
  model="${CTOX_VLLM_SERVE_MODEL:-openai/gpt-oss-20b}"
  local is_qwen35=0
  case "$model" in
    Qwen/Qwen3.5-*) is_qwen35=1 ;;
  esac

  local default_arch="gpt_oss"
  local default_max_seq_len="131072"
  local default_paged_attn="auto"
  local default_tp_backend="disabled"
  local default_isq=""
  local default_pa_cache_type="f8e4m3"
  local default_pa_memory_fraction="0.80"
  local default_disable_nccl="1"

  if [[ "$is_qwen35" == "1" ]]; then
    default_arch=""
    default_max_seq_len="32768"
    default_paged_attn="auto"
    default_tp_backend="nccl"
    default_isq="Q6K"
    default_pa_cache_type="f8e4m3"
    default_pa_memory_fraction="0.80"
    default_disable_nccl=""
  fi

  cat > "$ROOT/runtime/vllm_serve.env" <<EOF
CTOX_VLLM_SERVE_MODEL=${model}
CTOX_VLLM_SERVE_PORT=${CTOX_VLLM_SERVE_PORT:-1234}
CTOX_VLLM_SERVE_ARCH=${CTOX_VLLM_SERVE_ARCH:-$default_arch}
CTOX_VLLM_SERVE_MAX_SEQS=${CTOX_VLLM_SERVE_MAX_SEQS:-1}
CTOX_VLLM_SERVE_MAX_BATCH_SIZE=${CTOX_VLLM_SERVE_MAX_BATCH_SIZE:-1}
CTOX_VLLM_SERVE_MAX_SEQ_LEN=${CTOX_VLLM_SERVE_MAX_SEQ_LEN:-$default_max_seq_len}
CTOX_VLLM_SERVE_PAGED_ATTN=${CTOX_VLLM_SERVE_PAGED_ATTN:-$default_paged_attn}
CTOX_VLLM_SERVE_TENSOR_PARALLEL_BACKEND=${CTOX_VLLM_SERVE_TENSOR_PARALLEL_BACKEND:-$default_tp_backend}
CTOX_VLLM_SERVE_ISQ=${CTOX_VLLM_SERVE_ISQ:-$default_isq}
CTOX_VLLM_SERVE_PA_CACHE_TYPE=${CTOX_VLLM_SERVE_PA_CACHE_TYPE:-$default_pa_cache_type}
CTOX_VLLM_SERVE_PA_MEMORY_FRACTION=${CTOX_VLLM_SERVE_PA_MEMORY_FRACTION:-$default_pa_memory_fraction}
CTOX_VLLM_SERVE_PA_CONTEXT_LEN=${CTOX_VLLM_SERVE_PA_CONTEXT_LEN:-}
CTOX_VLLM_SERVE_CUDA_VISIBLE_DEVICES=${CTOX_VLLM_SERVE_CUDA_VISIBLE_DEVICES:-}
CTOX_VLLM_SERVE_DISABLE_NCCL=${CTOX_VLLM_SERVE_DISABLE_NCCL:-$default_disable_nccl}
CTOX_VLLM_SERVE_MN_LOCAL_WORLD_SIZE=${CTOX_VLLM_SERVE_MN_LOCAL_WORLD_SIZE:-}
CTOX_VLLM_SERVE_TOPOLOGY=${CTOX_VLLM_SERVE_TOPOLOGY:-}
CTOX_VLLM_SERVE_NUM_DEVICE_LAYERS=${CTOX_VLLM_SERVE_NUM_DEVICE_LAYERS:-}
EOF
}

ensure_rust_build_toolchain

CARGO_BIN="$(resolve_cargo_bin)"
if [[ -z "$CARGO_BIN" ]]; then
  echo "cargo is required to build CTOX" >&2
  exit 1
fi

ensure_linux_jami_installed

cd "$ROOT"
ensure_project_references_present
sync_repo_system_skills_into_vendored_codex
"$CARGO_BIN" build --release

mkdir -p "$HOME/.local/bin"
cat > "$HOME/.local/bin/ctox" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export CTOX_ROOT="$ROOT"
exec "$ROOT/target/release/ctox" "\$@"
EOF
chmod +x "$HOME/.local/bin/ctox"

MISTRALRS_FEATURES="$(detect_vllm_serve_features)"
ensure_cuda_build_prereqs
configure_cuda_env
if command -v sudo >/dev/null 2>&1 && nccl_packages_available && nccl_runtime_missing && vllm_serve_uses_cuda; then
  echo "[prep] Install NCCL runtime packages for multi-GPU CTOX vllm-serve hosts"
  sudo apt-get update
  sudo apt-get install -y libnccl2 libnccl-dev
fi

echo "[build] Build vendored vllm-serve engine with features: $MISTRALRS_FEATURES"
(
  cd "$ROOT/ctox-vllm-serve"
  if [[ -n "$MISTRALRS_FEATURES" ]]; then
    "$CARGO_BIN" build --release -p mistralrs-cli --bin mistralrs --features "$MISTRALRS_FEATURES"
  else
    echo "[build] No CUDA toolchain detected; build mistralrs in CPU-only mode"
    "$CARGO_BIN" build --release -p mistralrs-cli --bin mistralrs
  fi
)

echo "[build] Build vendored codex binaries"
(
  cd "$ROOT/references/openai-codex/codex-rs"
  ensure_codex_linux_build_prereqs
  "$CARGO_BIN" build --release -p codex-exec --bin codex-exec
  "$CARGO_BIN" build --release -p codex-cli --bin codex
)

ln -sf "$ROOT/ctox-vllm-serve/target/release/mistralrs" "$HOME/.local/bin/vllm-serve"
ln -sf "$ROOT/references/openai-codex/codex-rs/target/release/codex" "$HOME/.local/bin/codex-ctox"
write_vllm_serve_env_file
sync_repo_skills_into_codex_home
chmod +x "$ROOT/scripts/run_vllm_serve_backend.sh"
chmod +x "$ROOT/scripts/communication_mail_cli.mjs" "$ROOT/scripts/communication_jami_cli.mjs" "$ROOT/scripts/run_jami_daemon.sh"
install_jami_user_service
wait_for_jami_dbus_runtime

cat <<EOF
CTOX installed.

Binary:
  $HOME/.local/bin/ctox

Baseline binaries:
  $HOME/.local/bin/vllm-serve -> $ROOT/ctox-vllm-serve/target/release/mistralrs
  $ROOT/references/openai-codex/codex-rs/target/release/codex-exec
  $HOME/.local/bin/codex-ctox -> $ROOT/references/openai-codex/codex-rs/target/release/codex

Runtime launcher:
  $ROOT/scripts/run_vllm_serve_backend.sh

Bundled skills:
  $(resolve_codex_home)/skills
  $(resolve_codex_home)/skills/.system

Jami runtime:
  $ROOT/scripts/run_jami_daemon.sh
  systemctl --user status cto-jami-daemon.service

Try:
  ctox clean-room-baseline-plan gpt_oss "Reply with CTOX_OK and nothing else."
  $ROOT/scripts/run_vllm_serve_backend.sh
EOF
