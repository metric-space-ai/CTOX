#!/bin/bash
set -eu

crate_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
upstream_dir=${1:?usage: run_strict_umbrella_gate.sh <pinned-upstream-checkout> <new-output-directory> [cargo-target-directory]}
output_dir=${2:?usage: run_strict_umbrella_gate.sh <pinned-upstream-checkout> <new-output-directory> [cargo-target-directory]}
pin=$(jq -r .base_commit "$crate_dir/upstream-lock.json")

test ! -e "$output_dir" || {
    echo "strict umbrella output already exists: $output_dir" >&2
    exit 1
}
mkdir -p "$output_dir"
output_dir=$(CDPATH= cd -- "$output_dir" && pwd)
entries="$output_dir/.entries.jsonl"
tmp_root=$(CDPATH= cd -- "${TMPDIR:-/tmp}" && pwd -P)
sandbox=$(mktemp -d "$tmp_root/cliproxyapi-strict-umbrella.XXXXXX")
cargo_target_root=$tmp_root
test ! -d /Volumes/tmp || test ! -w /Volumes/tmp || cargo_target_root=/Volumes/tmp
cargo_target_owned=true
if test "$#" -ge 3; then
    mkdir -p "$3"
    cargo_target=$(CDPATH= cd -- "$3" && pwd)
    cargo_target_owned=false
else
    cargo_target=$(mktemp -d "$cargo_target_root/cliproxyapi-strict-cargo.XXXXXX")
fi
cleanup_sandbox() {
    test ! -d "$sandbox" || find "$sandbox" -depth -delete
    test "$cargo_target_owned" = false || test ! -d "$cargo_target" || find "$cargo_target" -depth -delete
}
trap cleanup_sandbox EXIT HUP INT TERM
mkdir -p "$sandbox/home" "$sandbox/tmp"
source_snapshot="$sandbox/source"
mkdir -p "$source_snapshot"

head=$(git -C "$upstream_dir" rev-parse HEAD)
test "$head" = "$pin" || {
    echo "upstream checkout is not at accepted pin: $head != $pin" >&2
    exit 1
}
test -z "$(git -C "$upstream_dir" status --porcelain=v1 --untracked-files=all)" || {
    echo "upstream checkout is dirty before strict umbrella gate" >&2
    exit 1
}
git -C "$upstream_dir" archive "$pin" | tar -x -C "$source_snapshot"

started_at=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
go_version=$(go version)
rustc_version=$(rustc --version)
cargo_version=$(cargo --version)
go_cache=$(go env GOCACHE)
go_mod_cache=$(go env GOMODCACHE)

hash_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

run_gate() {
    kind=$1
    name=$2
    cwd=$3
    subject=$4
    shift 4
    log="gate-$name.log"
    command_json=$(jq -cn --args '$ARGS.positional' -- "$@")
    echo "==> $name"
    set +e
    (CDPATH= cd -- "$cwd" && "$@") >"$output_dir/$log" 2>&1
    status=$?
    set -e
    tail -n 40 "$output_dir/$log"
    test "$status" -eq 0 || {
        echo "strict umbrella gate failed: $name (exit $status)" >&2
        exit "$status"
    }
    output_hash=$(hash_file "$output_dir/$log")
    jq -cn \
        --arg kind "$kind" \
        --arg name "$name" \
        --arg subject "$subject" \
        --arg cwd "$cwd" \
        --arg log "$log" \
        --arg output_sha256 "$output_hash" \
        --argjson command "$command_json" \
        '{kind: $kind, name: $name, subject: $subject, cwd: $cwd, status: "passed", command: $command,
          log: $log, output_sha256: $output_sha256}' >>"$entries"
}

module_count=0
git -C "$upstream_dir" ls-tree -r --name-only "$pin" |
    awk '$0 == "go.mod" || $0 ~ /\/go\.mod$/ { print }' | LC_ALL=C sort |
while IFS= read -r go_mod; do
    module=$(dirname "$go_mod")
    module_dir="$source_snapshot"
    test "$module" = "." || module_dir="$source_snapshot/$module"
    safe_name=$(printf '%s' "$module" | tr '/.' '__')
    run_gate go_hydration "hydrate-$safe_name" "$module_dir" "$module" \
        env HOME="$sandbox/home" TMPDIR="$sandbox/tmp" \
        GOCACHE="$go_cache" GOMODCACHE="$go_mod_cache" GOTOOLCHAIN=local \
        GOWORK=off GOENV=off \
        go test -mod=mod -run '^$' -count=1 -p=1 -timeout=20m ./...
    run_gate go_module "go-$safe_name" "$module_dir" "$module" \
        env HOME="$sandbox/home" TMPDIR="$sandbox/tmp" \
        GOCACHE="$go_cache" GOMODCACHE="$go_mod_cache" GOTOOLCHAIN=local \
        GOWORK=off GOENV=off GOPROXY=off \
        go test -mod=readonly -count=1 -p=1 -timeout=20m ./...
done

module_count=$(jq -s '[.[] | select(.kind == "go_module")] | length' "$entries")
test "$module_count" -eq 23 || {
    echo "strict umbrella expected 23 Go modules, got $module_count" >&2
    exit 1
}
hydration_count=$(jq -s '[.[] | select(.kind == "go_hydration")] | length' "$entries")
test "$hydration_count" -eq 23 || {
    echo "strict umbrella expected 23 Go hydration gates, got $hydration_count" >&2
    exit 1
}

run_gate rust_gate rust_no_default "$crate_dir" rust_no_default env CARGO_TARGET_DIR="$cargo_target" "$crate_dir/scripts/run_cargo_test_gate.sh" --frozen --lib --no-default-features
run_gate rust_gate rust_all_features "$crate_dir" rust_all_features env CARGO_TARGET_DIR="$cargo_target" "$crate_dir/scripts/run_cargo_test_gate.sh" --frozen --lib --all-features
run_gate rust_gate clippy_no_default "$crate_dir" clippy_no_default env CARGO_TARGET_DIR="$cargo_target" "$crate_dir/scripts/run_with_macos_cargo_metadata_guard.sh" cargo clippy --frozen --all-targets --no-default-features -- -D warnings
run_gate rust_gate clippy_all_features "$crate_dir" clippy_all_features env CARGO_TARGET_DIR="$cargo_target" "$crate_dir/scripts/run_with_macos_cargo_metadata_guard.sh" cargo clippy --frozen --all-targets --all-features -- -D warnings
run_gate rust_gate formatting "$crate_dir" formatting cargo fmt --check
run_gate rust_gate tracking "$crate_dir" tracking "$crate_dir/scripts/check_tracking.sh"
run_gate rust_gate dashboard "$crate_dir" dashboard \
    sh -c '"$1/scripts/build_dashboard.sh" "$2/dashboard.html" && node "$1/scripts/tests/strict_credit_dashboard_test.mjs" "$2/dashboard.html" "$1/strict-credit-audit.json"' \
    sh "$crate_dir" "$output_dir"

test "$(git -C "$upstream_dir" rev-parse HEAD)" = "$pin" || {
    echo "upstream pin changed during strict umbrella gate" >&2
    exit 1
}
test -z "$(git -C "$upstream_dir" status --porcelain=v1 --untracked-files=all)" || {
    echo "upstream checkout became dirty during strict umbrella gate" >&2
    exit 1
}

completed_at=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
receipt="$output_dir/strict-umbrella-receipt.json"
hydrated_manifest="$output_dir/hydrated-go-manifest.tsv"
find "$source_snapshot" -type f \( -name 'go.mod' -o -name 'go.sum' \) -print |
    LC_ALL=C sort | while IFS= read -r source; do
        relative=${source#"$source_snapshot"/}
        printf '%s\t%s\n' "$(hash_file "$source")" "$relative"
    done >"$hydrated_manifest"
hydrated_manifest_hash=$(hash_file "$hydrated_manifest")
manifest="$output_dir/source-manifest.tsv"
find "$crate_dir" -type f \( -name '*.rs' -o -name 'Cargo.toml' -o -name 'Cargo.lock' \) -print |
LC_ALL=C sort | while IFS= read -r source; do
    printf '%s\t%s\n' "$(hash_file "$source")" "$source"
done >"$manifest"
manifest_hash=$(hash_file "$manifest")
jq -s \
    --arg pin "$pin" \
    --arg started_at "$started_at" \
    --arg completed_at "$completed_at" \
    --arg go_version "$go_version" \
    --arg rustc_version "$rustc_version" \
    --arg cargo_version "$cargo_version" \
    --arg manifest_hash "$manifest_hash" \
    --arg hydrated_manifest_hash "$hydrated_manifest_hash" '
    {
      schema: "ctox.cliproxyapi.strict-umbrella-receipt.v2",
      upstream_commit: $pin,
      upstream_clean_before: true,
      upstream_clean_after: true,
      started_at: $started_at,
      completed_at: $completed_at,
      go_version: $go_version,
      rustc_version: $rustc_version,
      cargo_version: $cargo_version,
      source_manifest: "source-manifest.tsv",
      source_manifest_sha256: $manifest_hash,
      hydrated_go_manifest: "hydrated-go-manifest.tsv",
      hydrated_go_manifest_sha256: $hydrated_manifest_hash,
      go_hydration: [.[] | select(.kind == "go_hydration") | {module: .subject, status, command, log, output_sha256}],
      go_modules: [.[] | select(.kind == "go_module") | {module: .subject, status, command, log, output_sha256}],
      gates: [.[] | select(.kind == "rust_gate") | {name, status, command, log, output_sha256}]
    }
' "$entries" >"$receipt"
rm -f "$entries"
cleanup_sandbox
trap - EXIT HUP INT TERM
"$crate_dir/scripts/check_strict_umbrella_receipt.sh" "$receipt"
echo "$receipt"
