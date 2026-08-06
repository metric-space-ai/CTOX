#!/bin/bash
set -eu

crate_dir=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
test_dir=$(mktemp -d)
trap 'rm -rf "$test_dir"' EXIT HUP INT TERM
receipt="$test_dir/receipt.json"
base=1111111111111111111111111111111111111111
candidate=2222222222222222222222222222222222222222

jq -n \
  --arg base "$base" \
  --arg candidate "$candidate" '
  def change: {
    upstream: "internal/example.go",
    module: "internal/example",
    kind: "modified",
    source_kind: "go_production",
    required_action: "port_and_revalidate"
  };
  def gates: {
    non_go_impact_review: true,
    dependency_audit: true,
    rust_no_default: true,
    rust_default: true,
    integrations: true,
    clippy_no_default: true,
    clippy_all_features: true,
    formatting: true,
    tracking: true,
    dashboard: true
  };
  def gate_evidence:
    (gates | keys | map({
      key: .,
      value: {
        status: "passed",
        command: ["true"],
        completed_at: "2026-08-04T00:00:00Z",
        output_sha256: ("a" * 64),
        log: ("gate-" + . + ".log")
      }
    }) | from_entries);
  {
    schema: "ctox.cliproxyapi.upstream-promotion-receipt.v1",
    repository: "https://github.com/router-for-me/CLIProxyAPI.git",
    previous_commit: $base,
    accepted_commit: $candidate,
    promoted_at: "2026-08-04T00:00:00Z",
    delta_sha256: ("b" * 64),
    review_sha256: ("c" * 64),
    delta: {
      schema: "ctox.cliproxyapi.upstream-delta.v2",
      base_commit: $base,
      candidate_commit: $candidate,
      changes: [change]
    },
    review: {
      schema: "ctox.cliproxyapi.upstream-review.v3",
      base_commit: $base,
      candidate_commit: $candidate,
      status: "ready_for_promotion",
      changes: [change + {
        review_status: "complete",
        disposition: "ported",
        evidence: ["module gate"],
        rust_evidence: ["cargo test"],
        upstream_evidence: ["go test"]
      }],
      gates: gates,
      gate_evidence: gate_evidence
    }
  }
' > "$receipt"

"$crate_dir/scripts/check_upstream_receipt.sh" "$receipt" >/dev/null

jq '.delta' "$receipt" > "$test_dir/delta.json"
jq '.review' "$receipt" > "$test_dir/review.json"
jq '{repository, base_commit: .previous_commit}' \
  "$receipt" > "$test_dir/lock.json"
generated="$test_dir/generated-receipt.json"
"$crate_dir/scripts/write_upstream_receipt.sh" \
  "$test_dir/lock.json" \
  "$test_dir/delta.json" \
  "$test_dir/review.json" \
  "$base" \
  "$candidate" \
  "$generated" >/dev/null
"$crate_dir/scripts/check_upstream_receipt.sh" "$generated" >/dev/null
if "$crate_dir/scripts/write_upstream_receipt.sh" \
  "$test_dir/lock.json" \
  "$test_dir/delta.json" \
  "$test_dir/review.json" \
  "$base" \
  "$candidate" \
  "$generated" >/dev/null 2>&1; then
    echo "existing receipt was overwritten" >&2
    exit 1
fi

jq '.accepted_commit = "3333333333333333333333333333333333333333"' \
  "$receipt" > "$test_dir/tampered-identity.json"
if "$crate_dir/scripts/check_upstream_receipt.sh" \
  "$test_dir/tampered-identity.json" >/dev/null 2>&1; then
    echo "receipt identity tampering was accepted" >&2
    exit 1
fi

jq '.review.gates.dashboard = false' \
  "$receipt" > "$test_dir/tampered-gate.json"
if "$crate_dir/scripts/check_upstream_receipt.sh" \
  "$test_dir/tampered-gate.json" >/dev/null 2>&1; then
    echo "receipt gate tampering was accepted" >&2
    exit 1
fi

echo "upstream promotion receipt tests passed"
