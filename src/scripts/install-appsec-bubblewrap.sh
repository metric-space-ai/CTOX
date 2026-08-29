#!/usr/bin/env bash
set -euo pipefail

# CTOX scanner sandboxes require bwrap's sized tmpfs support. Ubuntu 22.04's
# repository package predates that feature, so CI builds the current upstream
# security release from a checksum-pinned release archive before installing a
# non-setuid binary. `debugoptimized` keeps upstream's optimized O2 build while
# avoiding a GCC 13 O3-only format-overflow false positive in bubblewrap 0.11.2.
version="0.11.2"
sha256="69abc30005d2186baf7737feacd8da35633b93cf5af38838ecff17c5f8e924f6"
install_path="/usr/local/bin/bwrap"

ensure_apparmor_userns_profile() {
  local restriction="/proc/sys/kernel/apparmor_restrict_unprivileged_userns"
  local profile_path="/etc/apparmor.d/ctox-appsec-bwrap"
  if [[ ! -r "$restriction" ]] || [[ "$(<"$restriction")" != "1" ]]; then
    return 0
  fi
  if ! command -v apparmor_parser >/dev/null 2>&1; then
    printf '%s\n' \
      "AppArmor restricts unprivileged user namespaces, but apparmor_parser is unavailable." \
      "Install the apparmor package before installing the CTOX Bubblewrap backend." >&2
    return 1
  fi
  printf '%s\n' \
    'include <tunables/global>' \
    '' \
    'profile ctox-appsec-bwrap /usr/local/bin/bwrap flags=(unconfined) {' \
    '  userns,' \
    '}' | sudo tee "$profile_path" >/dev/null
  sudo apparmor_parser --replace "$profile_path"
}

ensure_apparmor_userns_profile

build_root="$(mktemp -d)"
trap 'rm -rf -- "$build_root"' EXIT
archive="$build_root/bubblewrap-${version}.tar.xz"
source_url="https://github.com/containers/bubblewrap/releases/download/v${version}/bubblewrap-${version}.tar.xz"

curl --fail --location --silent --show-error "$source_url" --output "$archive"
printf '%s  %s\n' "$sha256" "$archive" | sha256sum --check --strict
tar --extract --xz --file "$archive" --directory "$build_root"

meson setup "$build_root/build" "$build_root/bubblewrap-${version}" \
  --buildtype=debugoptimized \
  -Dtests=false \
  -Dman=disabled \
  -Dbash_completion=disabled \
  -Dzsh_completion=disabled \
  -Dselinux=disabled \
  -Dsupport_setuid=false
ninja -C "$build_root/build" bwrap
sudo install -o root -g root -m 0755 "$build_root/build/bwrap" "$install_path"

"$install_path" --help 2>&1 | grep -q -- "--size"
[[ ! -u "$install_path" ]]
"$install_path" --version | grep -Fqx "bubblewrap ${version}"
