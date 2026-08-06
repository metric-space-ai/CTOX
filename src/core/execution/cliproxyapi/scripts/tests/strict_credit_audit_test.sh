#!/bin/bash
set -eu

crate_dir=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
validator="$crate_dir/scripts/check_strict_credit_audit.sh"
umbrella_validator="$crate_dir/scripts/check_strict_umbrella_receipt.sh"
scratch=$(mktemp -d "${TMPDIR:-/tmp}/cliproxyapi-strict-audit-test.XXXXXX")
trap 'rm -rf "$scratch"' EXIT HUP INT TERM

expect_failure() {
    if "$@" >"$scratch/unexpected.stdout" 2>"$scratch/expected.stderr"; then
        echo "expected strict-credit validation failure: $*" >&2
        exit 1
    fi
}

expect_failure_containing() {
    expected=$1
    shift
    expect_failure "$@"
    grep -F "$expected" "$scratch/expected.stderr" >/dev/null || {
        echo "expected validation error containing: $expected" >&2
        cat "$scratch/expected.stderr" >&2
        exit 1
    }
}

"$validator" \
    "$crate_dir/strict-credit-audit.json" \
    "$crate_dir/project-state.json" \
    "$crate_dir/port-map.json" >/dev/null

jq '.strict_closure.production -= 1' \
    "$crate_dir/strict-credit-audit.json" > "$scratch/bad-count.json"
expect_failure "$validator" "$scratch/bad-count.json" \
    "$crate_dir/project-state.json" "$crate_dir/port-map.json"

jq '.closed_reconstructed[0].production[0] = "internal/missing.go"' \
    "$crate_dir/strict-credit-audit.json" > "$scratch/missing-path.json"
expect_failure "$validator" "$scratch/missing-path.json" \
    "$crate_dir/project-state.json" "$crate_dir/port-map.json"

jq '.closed_reconstructed[0].tests += [.closed_reconstructed[0].production[0]]
    | .closed_reconstructed[0].production = .closed_reconstructed[0].production[1:]' \
    "$crate_dir/strict-credit-audit.json" > "$scratch/wrong-test-flag.json"
expect_failure "$validator" "$scratch/wrong-test-flag.json" \
    "$crate_dir/project-state.json" "$crate_dir/port-map.json"

jq '.files |= map(if .upstream == "internal/browser/browser.go" then .status = "scaffold" else . end)' \
    "$crate_dir/port-map.json" > "$scratch/open-map.json"
expect_failure "$validator" "$crate_dir/strict-credit-audit.json" \
    "$crate_dir/project-state.json" "$scratch/open-map.json"

audit_pin=$(jq -r .upstream_commit "$crate_dir/strict-credit-audit.json")
receipt_relative=$(jq -r .umbrella_receipt.path "$crate_dir/strict-credit-audit.json")
receipt_dir=$(CDPATH= cd -- "$crate_dir/$(dirname -- "$receipt_relative")" && pwd)
receipt="$receipt_dir/$(basename -- "$receipt_relative")"

expect_failure_containing "upstream pin mismatch" \
    "$umbrella_validator" "$receipt" "0000000000000000000000000000000000000000"

jq '.umbrella_receipt.path = "../../../../runtime/not-a-strict-receipt/strict-umbrella-receipt.json"' \
    "$crate_dir/strict-credit-audit.json" >"$scratch/outside-receipt.json"
expect_failure_containing "outside expected pin subtree" \
    "$validator" "$scratch/outside-receipt.json" \
    "$crate_dir/project-state.json" "$crate_dir/port-map.json"

dashboard="$scratch/dashboard.html"
"$crate_dir/scripts/build_dashboard.sh" "$dashboard" >/dev/null
node "$crate_dir/scripts/tests/strict_credit_dashboard_test.mjs" \
    "$dashboard" "$crate_dir/strict-credit-audit.json"

echo "strict-credit audit tests passed"
