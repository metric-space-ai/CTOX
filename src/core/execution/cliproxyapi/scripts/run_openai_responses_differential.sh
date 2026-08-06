#!/bin/bash
set -eu
crate_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
repo_dir=$(CDPATH= cd -- "$crate_dir/../../../.." && pwd)
upstream_dir=${1:-"$repo_dir/runtime/cliproxyapi-upstream"}
package_dir="$upstream_dir/internal/translator/openai/openai/responses"
probe="$package_dir/ctox_openai_responses_differential_test.go"
fixtures="$crate_dir/tests/differential/openai_responses_fixtures.json"
scratch=$(mktemp -d)
trap 'rm -f "$probe"; rm -rf "$scratch"' EXIT
cp "$crate_dir/tests/differential/openai_responses_probe.go.txt" "$probe"
CTOX_DIFF_FIXTURES="$fixtures" CTOX_DIFF_OUTPUT="$scratch/go.json" go -C "$upstream_dir" test ./internal/translator/openai/openai/responses -run '^TestCtoxOpenAIResponsesDifferential$' -count=1 -timeout=30s
CARGO_TARGET_DIR="$repo_dir/runtime/build/cliproxyapi-target" cargo run -q -p ctox-cliproxyapi --bin cliproxy-differential -- "$fixtures" "$scratch/rust.json"
jq -S . "$scratch/go.json" > "$scratch/go.sorted.json"
jq -S . "$scratch/rust.json" > "$scratch/rust.sorted.json"
if ! cmp -s "$scratch/go.sorted.json" "$scratch/rust.sorted.json"; then diff -u "$scratch/go.sorted.json" "$scratch/rust.sorted.json"; exit 1; fi
echo "OpenAI Responses differential ok: $(jq length "$scratch/rust.json") parity fixtures"
