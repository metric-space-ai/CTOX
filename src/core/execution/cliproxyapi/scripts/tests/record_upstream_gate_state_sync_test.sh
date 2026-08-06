#!/bin/bash
set -eu

crate_dir=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
repo_dir=$(CDPATH= cd -- "$crate_dir/../../../.." && pwd)
state_source="$crate_dir/project-state.json"
candidate=$(jq -r '.upstream_candidate.commit' "$state_source")
review_root="$repo_dir/runtime/cliproxyapi-upstream-reviews/$candidate"
scratch=$(mktemp -d)
trap 'find "$scratch" -depth -delete' EXIT

cp "$state_source" "$scratch/state.json"
cp "$review_root/upstream-review.json" "$scratch/review.json"
cp "$review_root/upstream-delta.json" "$scratch/delta.json"

# Reopen one already-passed gate in the isolated ledger. Recording it again
# must atomically restore both the ledger and the derived project-state count.
jq '
  .gates.dependency_audit = false |
  .gate_evidence.dependency_audit = {
    status: "pending", command: [], completed_at: null,
    output_sha256: null, log: null
  }
' "$scratch/review.json" > "$scratch/review.next.json"
mv "$scratch/review.next.json" "$scratch/review.json"

PROJECT_STATE_FILE="$scratch/state.json" \
  "$crate_dir/scripts/record_upstream_gate.sh" \
  "$scratch/delta.json" "$scratch/review.json" dependency_audit -- true >/dev/null

ledger_gates=$(jq '[.gates[] | select(. == true)] | length' "$scratch/review.json")
jq -e --argjson ledger_gates "$ledger_gates" '
  .upstream_candidate.gates_passed == $ledger_gates and
  .upstream_candidate.reviewed == .upstream_candidate.inventory_total and
  .upstream_candidate.pending == 0
' "$scratch/state.json" >/dev/null

# A gate for another candidate must fail before running its command or touching
# the state, preventing cross-candidate progress corruption.
jq '.candidate_commit = "0000000000000000000000000000000000000000"' \
  "$scratch/review.json" > "$scratch/review.other.json"
if PROJECT_STATE_FILE="$scratch/state.json" \
   "$crate_dir/scripts/record_upstream_gate.sh" \
   "$scratch/delta.json" "$scratch/review.other.json" dependency_audit -- true \
   >/dev/null 2>&1; then
    echo "gate recorder accepted a review for a different state candidate" >&2
    exit 1
fi

printf 'upstream gate recorder state synchronization tests passed\n'
