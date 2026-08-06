#!/bin/bash
set -eu

crate_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
repo_dir=$(CDPATH= cd -- "$crate_dir/../../../.." && pwd)
upstream_dir=${1:-"$repo_dir/runtime/cliproxyapi-upstream"}
package_dir="$upstream_dir/internal/translator/claude/openai/responses"
probe="$package_dir/ctox_differential_test.go"
scratch=$(mktemp -d)
trap 'rm -f "$probe"; rm -rf "$scratch"' EXIT

cp "$crate_dir/tests/differential/probe.go.txt" "$probe"
CTOX_DIFF_FIXTURES="$crate_dir/tests/differential/fixtures.json" \
CTOX_DIFF_OUTPUT="$scratch/go.json" \
    go -C "$upstream_dir" test ./internal/translator/claude/openai/responses \
    -run '^TestCtoxDifferential$' -count=1 -timeout=30s -v

CARGO_TARGET_DIR="$repo_dir/runtime/build/cliproxyapi-target" \
    cargo run -q -p ctox-cliproxyapi --bin cliproxy-differential -- \
    "$crate_dir/tests/differential/fixtures.json" "$scratch/rust.json"

delta_name=stream-provider-error-ctox-delta
jq -S --arg name "$delta_name" '[.[] | select(.name != $name)]' "$scratch/go.json" > "$scratch/go.sorted.json"
jq -S --arg name "$delta_name" '[.[] | select(.name != $name)]' "$scratch/rust.json" > "$scratch/rust.sorted.json"
if ! cmp -s "$scratch/go.sorted.json" "$scratch/rust.sorted.json"; then
    diff -u "$scratch/go.sorted.json" "$scratch/rust.sorted.json"
    exit 1
fi

go_delta_events=$(jq -c --arg name "$delta_name" '[.[] | select(.name == $name).output[].event]' "$scratch/go.json")
rust_delta_events=$(jq -c --arg name "$delta_name" '[.[] | select(.name == $name).output[].event]' "$scratch/rust.json")
if [ "$go_delta_events" != '["response.created","response.in_progress"]' ]; then
    echo "unexpected upstream error behavior: $go_delta_events" >&2
    exit 1
fi
if [ "$rust_delta_events" != '["response.created","response.in_progress","response.failed"]' ]; then
    echo "unexpected CTOX error behavior: $rust_delta_events" >&2
    exit 1
fi

fixture_count=$(jq length "$scratch/rust.json")
echo "differential ok: $((fixture_count - 1)) parity fixtures + 1 verified CTOX delta"
