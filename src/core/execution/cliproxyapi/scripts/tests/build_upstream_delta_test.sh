#!/bin/bash
set -eu

crate_dir=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
builder="$crate_dir/scripts/build_upstream_delta.sh"
review_builder="$crate_dir/scripts/init_upstream_review.sh"
scratch=$(mktemp -d "${TMPDIR:-/tmp}/cliproxyapi-delta-test.XXXXXX")
cleanup() {
    test ! -d "$scratch" || find "$scratch" -depth -delete
}
trap cleanup EXIT HUP INT TERM

repo="$scratch/upstream"
mkdir -p "$repo"
git -C "$repo" init -q
git -C "$repo" config user.name "CLIProxyAPI delta test"
git -C "$repo" config user.email "delta-test@invalid.example"
printf 'package base\n' >"$repo/modified.go"
printf 'package removed\n' >"$repo/deleted.go"
printf 'package renamed\n' >"$repo/old_name.go"
git -C "$repo" add .
git -C "$repo" commit -qm base
base=$(git -C "$repo" rev-parse HEAD)

printf 'package base\n// changed\n' >"$repo/modified.go"
git -C "$repo" mv old_name.go renamed.go
git -C "$repo" rm -q deleted.go
mkdir -p "$repo/internal/new" "$repo/examples/plugin/new/go" \
    "$repo/.github/workflows" "$repo/docs"
printf 'package new\n' >"$repo/internal/new/new.go"
printf 'package new\n' >"$repo/internal/new/new_test.go"
printf 'module invalid.example/new\n\ngo 1.26\n' >"$repo/examples/plugin/new/go/go.mod"
printf 'name: test\n' >"$repo/.github/workflows/test.yml"
printf '# Guide\n' >"$repo/docs/guide.md"
printf 'enabled: true\n' >"$repo/config.yaml"
printf 'opaque\n' >"$repo/data.bin"
git -C "$repo" add .
git -C "$repo" commit -qm candidate
candidate=$(git -C "$repo" rev-parse HEAD)

jq -n --arg base "$base" '{repository: "invalid.example/upstream", base_commit: $base}' \
    >"$scratch/lock.json"
jq -n --arg base "$base" '{upstream_commit: $base, files: []}' \
    >"$scratch/port-map.json"

UPSTREAM_CHECKOUT="$repo" \
UPSTREAM_LOCK_FILE="$scratch/lock.json" \
UPSTREAM_PORT_MAP="$scratch/port-map.json" \
    "$builder" "$candidate" "$scratch/delta.json" >/dev/null

raw_count=$(git -C "$repo" diff --name-status -M "$base" "$candidate" -- . | wc -l | tr -d ' ')
jq -e --argjson raw_count "$raw_count" '
  (.changes | length) == $raw_count and
  .summary.changed_files == $raw_count and
  .summary.added == 7 and
  .summary.modified == 1 and
  .summary.deleted == 1 and
  .summary.renamed == 1 and
  ([.changes[].upstream] | unique | length) == $raw_count and
  ([.changes[] | select(.kind == "added") | select(.old_upstream != null)] | length) == 0 and
  ([.changes[] | select(.upstream == "internal/new/new.go" and
      .source_kind == "go_production" and .test == false and
      .current_port_status == "unmapped" and .required_action == "port_new_file")] | length) == 1 and
  ([.changes[] | select(.upstream == "internal/new/new_test.go" and
      .source_kind == "go_test" and .test == true and
      .current_port_status == "unmapped" and .required_action == "port_new_file")] | length) == 1 and
  ([.changes[] | select(.upstream == "examples/plugin/new/go/go.mod" and
      .source_kind == "dependency_manifest" and
      .required_action == "revalidate_dependencies_and_full_build")] | length) == 1 and
  ([.changes[] | select(.upstream == ".github/workflows/test.yml" and
      .source_kind == "build_release")] | length) == 1 and
  ([.changes[] | select(.upstream == "docs/guide.md" and
      .source_kind == "documentation_license")] | length) == 1 and
  ([.changes[] | select(.upstream == "config.yaml" and
      .source_kind == "runtime_asset")] | length) == 1 and
  ([.changes[] | select(.upstream == "data.bin" and .source_kind == "other")] | length) == 1
' "$scratch/delta.json" >/dev/null

"$review_builder" "$scratch/delta.json" "$scratch/review.json" >/dev/null
jq -e --slurpfile delta "$scratch/delta.json" '
  (.changes | length) == $delta[0].summary.changed_files and
  ([.changes[].upstream] | sort) == ([$delta[0].changes[].upstream] | sort) and
  ([.changes[] | select(.review_status != "pending")] | length) == 0
' "$scratch/review.json" >/dev/null

echo "build_upstream_delta conservation tests passed"
