#!/bin/bash
set -eu

delta=${1:?usage: build_ctox_integration_impact.sh <upstream-delta.json> <provider-integration.json> <output.json>}
integration=${2:?usage: build_ctox_integration_impact.sh <upstream-delta.json> <provider-integration.json> <output.json>}
output=${3:?usage: build_ctox_integration_impact.sh <upstream-delta.json> <provider-integration.json> <output.json>}

base=$(jq -r .base_commit "$delta")
accepted=$(jq -r .accepted_upstream_commit "$integration")
test "$base" = "$accepted" || {
    echo "CTOX integration ledger is not bound to the delta base commit" >&2
    exit 1
}

mkdir -p "$(dirname -- "$output")"
tmp=$(mktemp "$(dirname -- "$output")/.ctox-integration-impact.XXXXXX")
cleanup() { rm -f "$tmp"; }
trap cleanup EXIT HUP INT TERM

jq -n \
    --slurpfile delta "$delta" \
    --slurpfile integration "$integration" '
  ($delta[0].changes | length) as $changed |
  {
    schema: "ctox.cliproxyapi.integration-impact.v1",
    base_commit: $delta[0].base_commit,
    candidate_commit: $delta[0].candidate_commit,
    changed_files: $changed,
    integration_schema: $integration[0].schema,
    provider_modes: [
      $integration[0].provider_modes[] | {
        id,
        disposition: (if $changed == 0 then "unaffected" else "pending_impact_review" end),
        gates_requiring_review: (
          if $changed == 0 then []
          else ($integration[0].gate_definitions | map(.id))
          end
        )
      }
    ],
    completion_allowed: ($changed == 0),
    note: (
      if $changed == 0 then
        "No upstream delta; the accepted Track-B evidence remains bound to the same pin."
      else
        "Fail-closed conservative impact: every provider mode must be reviewed against this candidate before Track-B evidence can be rebound."
      end
    )
  }
' >"$tmp"

mv "$tmp" "$output"
trap - EXIT HUP INT TERM
printf '%s\n' "$output"
