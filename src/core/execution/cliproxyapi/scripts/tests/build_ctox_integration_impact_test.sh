#!/bin/bash
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
repo_dir=$(CDPATH= cd -- "$script_dir/../../../../.." && pwd)
scratch=$(mktemp -d "${TMPDIR:-/tmp}/ctox-integration-impact-test.XXXXXX")
cleanup() { rm -rf "$scratch"; }
trap cleanup EXIT HUP INT TERM

integration="$repo_dir/src/core/execution/cliproxyapi_integration/provider-integration.json"
base=$(jq -r .accepted_upstream_commit "$integration")
candidate=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
jq -n --arg base "$base" --arg candidate "$candidate" '{
  schema: "ctox.cliproxyapi.upstream-delta.v3",
  base_commit: $base,
  candidate_commit: $candidate,
  changes: [{upstream: "internal/runtime/executor/kimi_executor.go"}]
}' >"$scratch/delta.json"

"$script_dir/build_ctox_integration_impact.sh" \
  "$scratch/delta.json" "$integration" "$scratch/impact.json" >/dev/null
jq -e --argjson expected "$(jq '.required_provider_modes | length' "$integration")" '
  .schema == "ctox.cliproxyapi.integration-impact.v1" and
  .changed_files == 1 and
  .completion_allowed == false and
  (.provider_modes | length) == $expected and
  ([.provider_modes[] | select(
    .disposition != "pending_impact_review" or
    (.gates_requiring_review | length) != 8
  )] | length) == 0
' "$scratch/impact.json" >/dev/null

jq '.changes = []' "$scratch/delta.json" >"$scratch/empty-delta.json"
"$script_dir/build_ctox_integration_impact.sh" \
  "$scratch/empty-delta.json" "$integration" "$scratch/empty-impact.json" >/dev/null
jq -e '
  .completion_allowed == true and
  ([.provider_modes[] | select(
    .disposition != "unaffected" or (.gates_requiring_review | length) != 0
  )] | length) == 0
' "$scratch/empty-impact.json" >/dev/null

echo "CTOX integration impact tests passed"
