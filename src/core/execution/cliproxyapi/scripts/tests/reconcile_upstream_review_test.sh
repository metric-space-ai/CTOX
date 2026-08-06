#!/bin/bash
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
reconcile="$script_dir/reconcile_upstream_review.sh"
init_review="$script_dir/init_upstream_review.sh"
scratch=$(mktemp -d "${TMPDIR:-/tmp}/cliproxy-reconcile-test.XXXXXX")
trap 'rm -rf "$scratch"' EXIT

delta="$scratch/corrected-delta.json"
fresh="$scratch/fresh-review.json"
old="$scratch/old-review.json"
remap="$scratch/remap.json"
output="$scratch/reconciled-review.json"

jq -n '
  def change($path; $module; $kind; $source_kind; $action): {
    upstream: $path,
    module: $module,
    kind: $kind,
    source_kind: $source_kind,
    required_action: $action
  };
  {
    schema: "ctox.cliproxyapi.upstream-delta.v2",
    repository: "https://example.invalid/upstream.git",
    base_commit: "1111111111111111111111111111111111111111",
    candidate_commit: "2222222222222222222222222222222222222222",
    clean: false,
    changes: [
      change("a.go"; "a"; "modified"; "go_production"; "revalidate_and_port_delta"),
      change("b.go"; "b"; "modified"; "go_production"; "revalidate_and_port_delta"),
      change("c_test.go"; "c"; "modified"; "go_test"; "revalidate_and_port_delta"),
      change("added.go"; "added"; "added"; "go_production"; "port_new_file")
    ]
  }
' > "$delta"

"$init_review" "$delta" "$fresh" >/dev/null

# Simulate the historical positional corruption: b.go evidence was written
# into the a.go slot, while c_test.go was correctly keyed. The corrected
# inventory also contains one newly recovered Added file.
jq '
  .changes |= map(
    if .upstream == "a.go" then
      .review_status = "complete" |
      .disposition = "completion-intended-for-b" |
      .evidence = ["b behavior reviewed"] |
      .rust_evidence = ["b.rs focused test"] |
      .upstream_evidence = ["go test b"]
    elif .upstream == "c_test.go" then
      .review_status = "complete" |
      .disposition = "direct-c-completion" |
      .evidence = ["c behavior reviewed"] |
      .rust_evidence = ["c_test.rs focused test"] |
      .upstream_evidence = ["go test c"]
    else . end
  ) |
  .changes |= map(select(.upstream != "added.go")) |
  .gates.tracking = true |
  .gate_evidence.tracking = {
    status: "passed",
    command: ["false-old-gate"],
    completed_at: "2026-08-04T00:00:00Z",
    output_sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    log: "old.log"
  }
' "$fresh" > "$old"

jq -n '{
  schema: "ctox.cliproxyapi.review-reconciliation-remap.v1",
  mappings: [{
    from_upstream: "a.go",
    to_upstream: "b.go",
    reason: "forensic evidence proves the old positional completion belongs to b.go"
  }]
}' > "$remap"

"$reconcile" "$old" "$delta" "$fresh" "$output" "$remap" >/dev/null

jq -e '
  (.changes | length) == 4 and
  (.changes | map(.upstream) | unique | length) == 4 and
  (.changes[] | select(.upstream == "a.go") | .review_status) == "pending" and
  (.changes[] | select(.upstream == "b.go") |
    .review_status == "complete" and .disposition == "completion-intended-for-b") and
  (.changes[] | select(.upstream == "c_test.go") |
    .review_status == "complete" and .disposition == "direct-c-completion") and
  (.changes[] | select(.upstream == "added.go") | .review_status) == "pending" and
  ([.gates[] | select(. != false)] | length) == 0 and
  ([.gate_evidence[] | select(
    .status != "pending" or .command != [] or .completed_at != "" or
    .output_sha256 != "" or .log != ""
  )] | length) == 0 and
  .status == "in_progress" and
  .reconciliation.schema == "ctox.cliproxyapi.review-reconciliation.v1" and
  .reconciliation.old_inventory_count == 3 and
  .reconciliation.corrected_inventory_count == 4 and
  .reconciliation.replayed_direct == 1 and
  .reconciliation.replayed_remapped == 1 and
  .reconciliation.pending == 2 and
  (.reconciliation.remappings | length) == 1
' "$output" >/dev/null

# Duplicate targets are ambiguous and must fail before an output is published.
bad_remap="$scratch/bad-remap.json"
bad_output="$scratch/bad-output.json"
jq '.mappings += [{
  from_upstream: "c_test.go",
  to_upstream: "b.go",
  reason: "ambiguous duplicate target"
}]' "$remap" > "$bad_remap"
if "$reconcile" "$old" "$delta" "$fresh" "$bad_output" "$bad_remap" \
    >"$scratch/unexpected.stdout" 2>"$scratch/expected.stderr"; then
    echo "duplicate remap target was accepted" >&2
    exit 1
fi
if [ -e "$bad_output" ]; then
    echo "failed reconciliation published an output" >&2
    exit 1
fi

echo "upstream review reconciliation tests passed"
