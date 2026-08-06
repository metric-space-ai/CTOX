#!/bin/bash
set -eu

crate_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
repo_dir=$(CDPATH= cd -- "$crate_dir/../../../.." && pwd)
upstream_dir=${1:-"$repo_dir/runtime/cliproxyapi-upstream"}
package_dir="$upstream_dir/internal/translator/openai/interactions/responses"
probe="$package_dir/ctox_interactions_responses_differential_test.go"
scratch=$(mktemp -d)
trap 'rm -f "$probe"; rm -rf "$scratch"' EXIT

cp "$crate_dir/tests/differential/interactions_responses_probe.go.txt" "$probe"
CTOX_DIFF_FIXTURES="$crate_dir/tests/differential/interactions_responses_fixtures.json" \
CTOX_DIFF_OUTPUT="$scratch/go.json" \
    go -C "$upstream_dir" test ./internal/translator/openai/interactions/responses \
    -run '^TestCtoxInteractionsResponsesDifferential$' -count=1 -timeout=30s -v

CARGO_TARGET_DIR="$repo_dir/runtime/build/cliproxyapi-target" \
    cargo run -q -p ctox-cliproxyapi --bin cliproxy-differential -- \
    "$crate_dir/tests/differential/interactions_responses_fixtures.json" "$scratch/rust.json"

jq -S 'walk(if type == "object" then del(.created, .updated) else . end)' "$scratch/go.json" > "$scratch/go.sorted.json"
jq -S 'walk(if type == "object" then del(.created, .updated) else . end)' "$scratch/rust.json" > "$scratch/rust.sorted.json"
if ! cmp -s "$scratch/go.sorted.json" "$scratch/rust.sorted.json"; then
    diff -u "$scratch/go.sorted.json" "$scratch/rust.sorted.json"
    exit 1
fi

fixture_count=$(jq length "$scratch/rust.json")
echo "interactions/responses differential ok: $fixture_count parity fixtures"
