#!/bin/bash
set -eu

crate_dir=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
repo_dir=$(CDPATH= cd -- "$crate_dir/../../../.." && pwd)
builder="$crate_dir/scripts/build_dashboard.sh"
integration_source="$repo_dir/src/core/execution/cliproxyapi_integration/provider-integration.json"
scratch=$(mktemp -d)
trap 'find "$scratch" -depth -delete' EXIT

cp "$integration_source" "$scratch/integration.json"
CTOX_INTEGRATION_FILE="$scratch/integration.json" \
  "$builder" "$scratch/dashboard.html" >/dev/null

hash_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

baseline_hash=$(hash_file "$scratch/dashboard.html")

expect_rejection() {
    filter=$1
    jq "$filter" "$integration_source" > "$scratch/integration.invalid.json"
    if CTOX_INTEGRATION_FILE="$scratch/integration.invalid.json" \
       "$builder" "$scratch/dashboard.html" >/dev/null 2>&1; then
        echo "dashboard accepted invalid CTOX integration ledger: $filter" >&2
        exit 1
    fi
    test "$(hash_file "$scratch/dashboard.html")" = "$baseline_hash" || {
        echo "dashboard changed after rejecting invalid CTOX integration ledger" >&2
        exit 1
    }
}

expect_rejection 'del(.provider_modes[0].gates.runtime_route)'
expect_rejection '.lane = "portable_port"'
expect_rejection '.updated_at = "x"'
expect_rejection '.accepted_upstream_commit = "0000000000000000000000000000000000000000"'
expect_rejection '.provider_modes[0].gates.runtime_route = "done"'
expect_rejection '.provider_modes += [.provider_modes[0]]'
expect_rejection '.provider_modes = [.provider_modes[0]] | .required_provider_modes = [.provider_modes[0].id]'
expect_rejection '.provider_modes[0].implementation_paths = []'
expect_rejection '.provider_modes[0].gate_evidence.portable_capability = []'
expect_rejection '.evidence_registry = {}'
expect_rejection '.evidence_registry.accepted_pin_full_gate.sha256 = "0000000000000000000000000000000000000000000000000000000000000000"'
expect_rejection 'del(.evidence_registry.accepted_pin_full_gate.receipt_schema)'
expect_rejection '.evidence_registry.accepted_pin_full_gate.receipt_schema = "forged.v1"'
expect_rejection '.completion_rule.complete = true'
expect_rejection '.gate_definitions += [.gate_definitions[0]]'

track_b_id=$(jq -r '
  .evidence_registry | to_entries[] |
  select(.value.receipt_schema == "ctox.cliproxyapi.track-b-receipt.v1") |
  .key
' "$integration_source" | head -n 1)
if [ -n "$track_b_id" ]; then
    expect_rejection ".evidence_registry[\"$track_b_id\"].mode_gates.claude_subscription = []"
    expect_rejection ".evidence_registry[\"$track_b_id\"].valid_gates = []"
fi

printf 'dashboard CTOX integration validation tests passed\n'
