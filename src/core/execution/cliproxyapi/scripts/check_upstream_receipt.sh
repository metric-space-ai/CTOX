#!/bin/bash
set -eu

receipt=${1:?usage: check_upstream_receipt.sh <promotion-receipt.json>}

jq -e '
  .schema == "ctox.cliproxyapi.upstream-promotion-receipt.v1" and
  (.repository | type) == "string" and
  (.previous_commit | test("^[0-9a-f]{40}$")) and
  (.accepted_commit | test("^[0-9a-f]{40}$")) and
  .previous_commit != .accepted_commit and
  (.promoted_at | test("^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")) and
  (.delta_sha256 | test("^[0-9a-f]{64}$")) and
  (.review_sha256 | test("^[0-9a-f]{64}$")) and
  .delta.schema == "ctox.cliproxyapi.upstream-delta.v2" and
  .delta.base_commit == .previous_commit and
  .delta.candidate_commit == .accepted_commit and
  .review.schema == "ctox.cliproxyapi.upstream-review.v3" and
  .review.base_commit == .previous_commit and
  .review.candidate_commit == .accepted_commit and
  .review.status == "ready_for_promotion" and
  ([.review.changes[] | select(
    .review_status != "complete" or
    (.disposition | length) == 0 or
    (.evidence | length) == 0 or
    (.upstream_evidence | length) == 0 or
    ((.source_kind == "go_production" or .source_kind == "go_test") and
      (.rust_evidence | length) == 0)
  )] | length) == 0 and
  (.review.gates | keys | sort) == ([
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
  ([.review.gates[] | select(. != true)] | length) == 0 and
  (.review.gate_evidence | keys | sort) == (.review.gates | keys | sort) and
  ([.review.gate_evidence[] | select(
    .status != "passed" or
    (.command | length) == 0 or
    (.completed_at | length) == 0 or
    (.output_sha256 | test("^[0-9a-f]{64}$") | not) or
    (.log | length) == 0
  )] | length) == 0 and
  ([.review.changes[] | {
      upstream, module, kind, source_kind, required_action
    }] | sort_by(.upstream)) ==
  ([.delta.changes[] | {
      upstream, module, kind, source_kind, required_action
    }] | sort_by(.upstream))
' "$receipt" >/dev/null || {
    echo "invalid upstream promotion receipt: $receipt" >&2
    exit 1
}

echo "upstream promotion receipt valid: $(jq -r .accepted_commit "$receipt")"
