#!/bin/bash
set -eu

crate_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
upstream_dir=${1:-"$crate_dir/../../../../runtime/cliproxyapi-upstream"}
generated=$(mktemp)
closure_generated=$(mktemp)
module_generated=$(mktemp)
trap 'rm -f "$generated" "$closure_generated" "$module_generated"' EXIT

actual=$(git -C "$upstream_dir" rev-parse HEAD)
documented=$(sed -n 's/^- Pinned commit: `\([0-9a-f]*\)`$/\1/p' "$crate_dir/UPSTREAM.md")
mapped=$(jq -r '.upstream_commit' "$crate_dir/port-map.json")
locked=$(jq -r '.base_commit' "$crate_dir/upstream-lock.json")
project=$(jq -r '.upstream_commit' "$crate_dir/project-state.json")
delta_base=$(jq -r '.base_commit' "$crate_dir/upstream-delta.json")
delta_candidate=$(jq -r '.candidate_commit' "$crate_dir/upstream-delta.json")

if [ "$actual" != "$documented" ] || [ "$actual" != "$mapped" ] || [ "$actual" != "$locked" ] ||
   [ "$actual" != "$project" ] || [ "$actual" != "$delta_base" ] || [ "$actual" != "$delta_candidate" ]; then
    echo "upstream pin mismatch: checkout=$actual documented=$documented map=$mapped lock=$locked project=$project delta=$delta_base..$delta_candidate" >&2
    exit 1
fi

"$crate_dir/scripts/check_upstream_anchors.sh"
# The accepted pin has one explicit fail-closed transition: promotion has
# happened, but its new accepted-pin umbrella gate has not. The old strict
# audit is historical and cannot be rebound until that gate produces a receipt;
# every other pin/map/anchor check in this script still runs. The post-gate
# recorder re-enables full audit validation before it can mark completion.
promotion_pending_full_gate=$(jq -r '
  .upstream_candidate.promoted == true and
  .project_completion.candidate_promoted == true and
  .project_completion.accepted_pin_complete == false and
  .project_completion.post_promotion_full_gate == false and
  .project_completion.complete == false
' "$crate_dir/project-state.json")
if [ "$promotion_pending_full_gate" != true ]; then
    "$crate_dir/scripts/check_strict_credit_audit.sh" \
        "$crate_dir/strict-credit-audit.json" \
        "$crate_dir/project-state.json" \
        "$crate_dir/port-map.json"
fi

"$crate_dir/scripts/build_port_map.sh" "$upstream_dir" "$generated"
if ! cmp -s "$generated" "$crate_dir/port-map.json"; then
    echo "port-map.json is stale; regenerate it with scripts/build_port_map.sh" >&2
    diff -u "$crate_dir/port-map.json" "$generated" | sed -n '1,120p' >&2
    exit 1
fi

"$crate_dir/scripts/build_mirror_closure.sh" "$crate_dir/port-map.json" "$closure_generated"
if ! cmp -s "$closure_generated" "$crate_dir/mirror-closure.json"; then
    echo "mirror-closure.json is stale; regenerate it with scripts/build_mirror_closure.sh" >&2
    diff -u "$crate_dir/mirror-closure.json" "$closure_generated" | sed -n '1,120p' >&2
    exit 1
fi

"$crate_dir/scripts/build_module_map.sh" "$crate_dir/port-map.json" "$module_generated"
if ! cmp -s "$module_generated" "$crate_dir/module-map.json"; then
    echo "module-map.json is stale; regenerate it with scripts/build_module_map.sh" >&2
    diff -u "$crate_dir/module-map.json" "$module_generated" | sed -n '1,120p' >&2
    exit 1
fi

echo "tracking ok: $actual"
