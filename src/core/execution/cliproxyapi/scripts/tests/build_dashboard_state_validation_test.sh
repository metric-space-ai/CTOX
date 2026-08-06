#!/bin/bash
set -eu

crate_dir=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
repo_dir=$(CDPATH= cd -- "$crate_dir/../../../.." && pwd)
builder="$crate_dir/scripts/build_dashboard.sh"
state_source="$crate_dir/project-state.json"
candidate=$(jq -r '.upstream_candidate.commit' "$state_source")
review_source="$repo_dir/runtime/cliproxyapi-upstream-reviews/$candidate/upstream-review.json"
scratch=$(mktemp -d)
trap 'find "$scratch" -depth -delete' EXIT
review_root="$scratch/reviews"
promoted_review_root="$scratch/reviews-promoted"
mkdir -p "$review_root/$candidate"
mkdir -p "$promoted_review_root/$candidate"
cp "$state_source" "$scratch/state.json"
cp "$review_source" "$review_root/$candidate/upstream-review.json"
jq '.changes |= map(.review_status = "complete") | .gates |= with_entries(.value = true) | .status = "ready_for_promotion"' \
  "$review_source" > "$promoted_review_root/$candidate/upstream-review.json"

hash_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

PROJECT_STATE_FILE="$scratch/state.json" \
CANDIDATE_REVIEW_ROOT="$review_root" \
  "$builder" "$scratch/dashboard.html" >/dev/null
baseline_hash=$(hash_file "$scratch/dashboard.html")

# The same completed review remains the immutable authority after promotion.
# In that state its candidate (rather than its old base) is the accepted pin.
jq --arg candidate "$candidate" '
  .upstream_commit = $candidate |
  .upstream_candidate.promoted = true |
  .upstream_candidate.reviewed = .upstream_candidate.inventory_total |
  .upstream_candidate.pending = 0 |
  .upstream_candidate.gates_passed = .upstream_candidate.gates_total |
  .upstream_candidate.status = "promoted" |
  .project_completion.candidate_promoted = true |
  .project_completion.accepted_pin_complete = false |
  .project_completion.post_promotion_full_gate = false |
  .project_completion.complete = false |
  .checkpoint.status = "in_progress" |
  .mirror_verification.strict_production_files = 0 |
  .mirror_verification.strict_test_files = 0 |
  (.upstream_candidate.inventory_total | tostring) as $total |
  (.upstream_candidate.gates_total | tostring) as $gates |
  .work_items |= map(
    if .id == "upstream-candidate-promotion" then
      .status = "in_progress" |
      .detail = "\($total)/\($total) Candidate-Reviews vollständig, \($gates)/\($gates) Promotion-Gates, Promotion JA."
    else . end)
' "$state_source" > "$scratch/state.promoted.json"
jq --arg candidate "$candidate" '.upstream_commit = $candidate' \
  "$crate_dir/port-map.json" > "$scratch/map.promoted.json"
jq --arg base "$(jq -r .base_commit "$review_source")" '.upstream_commit = $base' \
  "$crate_dir/strict-credit-audit.json" > "$scratch/audit.promoted.json"
PROJECT_STATE_FILE="$scratch/state.promoted.json" \
CANDIDATE_REVIEW_ROOT="$promoted_review_root" \
PORT_MAP_FILE="$scratch/map.promoted.json" \
STRICT_AUDIT_FILE="$scratch/audit.promoted.json" \
  "$builder" "$scratch/dashboard-promoted.html" >/dev/null
node "$crate_dir/scripts/tests/strict_credit_dashboard_test.mjs" \
  "$scratch/dashboard-promoted.html" "$scratch/audit.promoted.json" >/dev/null

expect_promoted_rejection() {
    filter=$1
    jq "$filter" "$scratch/state.promoted.json" > "$scratch/state.invalid-promoted.json"
    if PROJECT_STATE_FILE="$scratch/state.invalid-promoted.json" \
       CANDIDATE_REVIEW_ROOT="$promoted_review_root" \
       PORT_MAP_FILE="$scratch/map.promoted.json" \
       STRICT_AUDIT_FILE="$scratch/audit.promoted.json" \
       "$builder" "$scratch/dashboard-invalid-promoted.html" >/dev/null 2>&1; then
        echo "dashboard accepted inconsistent promoted state: $filter" >&2
        exit 1
    fi
}
expect_promoted_rejection ".upstream_commit = \"$(jq -r .base_commit "$review_source")\""
expect_promoted_rejection '.upstream_candidate.promoted = false | .project_completion.candidate_promoted = false'
expect_promoted_rejection '.project_completion.candidate_promoted = false'
expect_promoted_rejection '.mirror_verification.strict_production_files = 1'
expect_promoted_rejection '.mirror_verification.verified_test_files -= 1'

expect_state_rejection() {
    filter=$1
    jq "$filter" "$state_source" > "$scratch/state.next.json"
    mv "$scratch/state.next.json" "$scratch/state.json"
    if PROJECT_STATE_FILE="$scratch/state.json" \
       CANDIDATE_REVIEW_ROOT="$review_root" \
       "$builder" "$scratch/dashboard.html" >/dev/null 2>&1; then
        echo "dashboard accepted inconsistent project state: $filter" >&2
        exit 1
    fi
    test "$(hash_file "$scratch/dashboard.html")" = "$baseline_hash" || {
        echo "dashboard output changed after rejected state: $filter" >&2
        exit 1
    }
}

expect_state_rejection '.upstream_candidate.reviewed += 1'
expect_state_rejection '.upstream_candidate.pending -= 1'
expect_state_rejection '.upstream_candidate.gates_passed += 1'
expect_state_rejection '.upstream_candidate.promoted = false'
expect_state_rejection '.project_completion.complete = false'

cp "$state_source" "$scratch/state.json"
jq '.changes[0].review_status = "working"' "$review_source" > "$scratch/review.next.json"
mv "$scratch/review.next.json" "$review_root/$candidate/upstream-review.json"
if PROJECT_STATE_FILE="$scratch/state.json" \
   CANDIDATE_REVIEW_ROOT="$review_root" \
   "$builder" "$scratch/dashboard.html" >/dev/null 2>&1; then
    echo "dashboard accepted unsupported review status" >&2
    exit 1
fi
test "$(hash_file "$scratch/dashboard.html")" = "$baseline_hash"

printf 'dashboard state validation tests passed\n'
