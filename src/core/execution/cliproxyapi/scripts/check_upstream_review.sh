#!/bin/bash
set -eu

crate_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
delta=${1:-"$crate_dir/upstream-delta.json"}
review=${2:-"$crate_dir/upstream-review.json"}

jq -e --slurpfile delta "$delta" '
  $delta[0].schema == "ctox.cliproxyapi.upstream-delta.v2" and
  .schema == "ctox.cliproxyapi.upstream-review.v3" and
  .base_commit == $delta[0].base_commit and
  .candidate_commit == $delta[0].candidate_commit and
  ([.changes[] | {
      upstream, module, kind, source_kind, required_action
    }] | sort_by(.upstream)) ==
    ([$delta[0].changes[] | {
      upstream, module, kind, source_kind, required_action
    }] | sort_by(.upstream)) and
  ([.changes[] | select(
    .review_status != "complete" or
    (.disposition | length) == 0 or
    (.evidence | length) == 0 or
    (.upstream_evidence | length) == 0 or
    ((.source_kind == "go_production" or .source_kind == "go_test") and
      (.rust_evidence | length) == 0)
  )] | length) == 0 and
  (.gates | keys | sort) == ([
    "non_go_impact_review",
    "dependency_audit",
    "rust_no_default",
    "rust_default",
    "integrations",
    "clippy_no_default",
    "clippy_all_features",
    "formatting",
    "tracking",
    "dashboard"
  ] | sort) and
  ([.gates[] | select(. != true)] | length) == 0 and
  (.gate_evidence | keys | sort) == (.gates | keys | sort) and
  ([.gate_evidence[] | select(
    .status != "passed" or
    (.command | length) == 0 or
    (.completed_at | test("^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$") | not) or
    (.output_sha256 | test("^[0-9a-f]{64}$") | not) or
    (.log | length) == 0
  )] | length) == 0 and
  .status == "ready_for_promotion"
' "$review" >/dev/null || {
    echo "upstream review is incomplete or does not match the candidate delta" >&2
    exit 1
}

review_dir=$(CDPATH= cd -- "$(dirname -- "$review")" && pwd)
for gate in $(jq -r '.gates | keys[]' "$review"); do
    expected_log="gate-$gate.log"
    recorded_log=$(jq -r --arg gate "$gate" '.gate_evidence[$gate].log' "$review")
    if [ "$recorded_log" != "$expected_log" ] || [ ! -f "$review_dir/$recorded_log" ]; then
        echo "upstream gate evidence log missing or unsafe: $gate ($recorded_log)" >&2
        exit 1
    fi
    recorded_hash=$(jq -r --arg gate "$gate" '.gate_evidence[$gate].output_sha256' "$review")
    if command -v sha256sum >/dev/null 2>&1; then
        actual_hash=$(sha256sum "$review_dir/$recorded_log" | awk '{print $1}')
    else
        actual_hash=$(shasum -a 256 "$review_dir/$recorded_log" | awk '{print $1}')
    fi
    if [ "$actual_hash" != "$recorded_hash" ]; then
        echo "upstream gate evidence hash mismatch: $gate" >&2
        exit 1
    fi
done

echo "upstream review complete: $(jq -r '.candidate_commit' "$review")"
