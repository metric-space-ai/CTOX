#!/bin/bash
set -eu

test "$#" -gt 0 || {
    echo "usage: run_cargo_test_gate.sh <cargo-test-args...>" >&2
    exit 2
}

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
guard="$script_dir/run_with_macos_cargo_metadata_guard.sh"
if [ "$(uname -s)" != Darwin ]; then
    exec cargo test "$@"
fi

# macOS can attach non-removable com.apple.provenance metadata to executables
# linked on an external volume. Cargo build scripts are guarded during compile;
# the final test binary is copied without xattrs to the local temporary volume
# and executed there, preserving the exact compiled test artifact.
"$guard" cargo test "$@" --no-run
target_dir=${CARGO_TARGET_DIR:-target}
case "$target_dir" in
    /*) ;;
    *) target_dir="$(pwd)/$target_dir" ;;
esac
binary=
for candidate in "$target_dir"/debug/deps/ctox_cliproxyapi-*; do
    test -f "$candidate" && test -x "$candidate" || continue
    if [ -z "$binary" ] || [ "$candidate" -nt "$binary" ]; then
        binary=$candidate
    fi
done
test -n "$binary" || {
    echo "compiled CLIProxyAPI test executable was not found" >&2
    exit 1
}

runner=$(mktemp /private/tmp/ctox-cliproxyapi-test.XXXXXX)
trap 'rm -f "$runner"' EXIT HUP INT TERM
cp -X "$binary" "$runner"
chmod +x "$runner"
xattr -c "$runner" 2>/dev/null || true
codesign --force --sign - "$runner"
codesign --verify "$runner"
echo "     Running copied unittests $runner (source $binary)"
"$runner" --test-threads=1
