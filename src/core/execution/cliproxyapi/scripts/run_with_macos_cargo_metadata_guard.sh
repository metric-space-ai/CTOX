#!/bin/bash
set -eu

test "$#" -gt 0 || {
    echo "usage: run_with_macos_cargo_metadata_guard.sh <command> [args...]" >&2
    exit 2
}

if [ "$(uname -s)" != Darwin ]; then
    exec "$@"
fi

target_dir=${CARGO_TARGET_DIR:?CARGO_TARGET_DIR is required for the macOS Cargo metadata guard}
case "$target_dir" in
    /*) ;;
    *) target_dir="$(pwd)/$target_dir" ;;
esac
mkdir -p "$target_dir/debug/build" "$target_dir/debug/deps"

clear_launch_metadata() {
    for binary in "$target_dir"/debug/build/*/build-script-*; do
        test -f "$binary" || continue
        xattr -d com.apple.provenance "$binary" 2>/dev/null || true
    done
}

guard_pid=
stop_guard() {
    test -z "$guard_pid" || kill "$guard_pid" 2>/dev/null || true
    test -z "$guard_pid" || wait "$guard_pid" 2>/dev/null || true
}
trap stop_guard EXIT
trap 'exit 130' HUP INT TERM

(
    while :; do
        clear_launch_metadata
        sleep 0.05
    done
) &
guard_pid=$!

"$@"
