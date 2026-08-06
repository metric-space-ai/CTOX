#!/bin/bash
set -eu

crate_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
repo_dir=$(CDPATH= cd -- "$crate_dir/../../../.." && pwd)
upstream_dir=${UPSTREAM_CHECKOUT:-"$repo_dir/runtime/cliproxyapi-upstream"}
candidate_ref=${1:?usage: promote_upstream_pin.sh <candidate-ref> <review.json>}
review=${2:?usage: promote_upstream_pin.sh <candidate-ref> <review.json>}
lock="$crate_dir/upstream-lock.json"
project_state="$crate_dir/project-state.json"
delta=$(mktemp)
lock_tmp=$(mktemp)
project_tmp=$(mktemp)
backup_dir=$(mktemp -d)
backup_manifest="$backup_dir/manifest"
absent_manifest="$backup_dir/absent"
promotion_complete=false

restore_on_failure() {
    status=$?
    if [ "$promotion_complete" != true ] && [ -s "$backup_manifest" ]; then
        while IFS= read -r path; do
            relative=${path#"$repo_dir"/}
            cp "$backup_dir/files/$relative" "$path"
        done < "$backup_manifest"
        if [ -s "$absent_manifest" ]; then
            while IFS= read -r path; do
                rm -f "$path"
            done < "$absent_manifest"
        fi
        echo "upstream pin promotion failed; restored the previous accepted baseline" >&2
    fi
    rm -f "$delta" "$lock_tmp" "$project_tmp"
    rm -rf "$backup_dir"
    exit "$status"
}
trap restore_on_failure EXIT HUP INT TERM
trap 'exit 1' HUP INT TERM

old=$(jq -r '.base_commit' "$lock")
candidate=$(git -C "$upstream_dir" rev-parse "$candidate_ref^{commit}")
head=$(git -C "$upstream_dir" rev-parse HEAD)
if [ "$head" != "$candidate" ]; then
    echo "candidate checkout must be at $candidate before promotion (HEAD=$head)" >&2
    exit 1
fi
receipt="$repo_dir/docs/cliproxyapi-upstream-history/$candidate.json"
if [ -e "$receipt" ]; then
    echo "upstream promotion receipt already exists: $receipt" >&2
    exit 1
fi

"$crate_dir/scripts/build_upstream_delta.sh" "$candidate" "$delta" >/dev/null
"$crate_dir/scripts/check_upstream_review.sh" "$delta" "$review"

# The immutable review is promotion authority, while project-state is the
# dashboard projection. Refuse stale counters or a partially promoted state
# before taking the first snapshot/mutation.
jq -e --slurpfile review "$review" --arg old "$old" --arg candidate "$candidate" '
  $review[0] as $review |
  ($review.changes | length) as $total |
  ([$review.changes[] | select(.review_status == "complete")] | length) as $reviewed |
  ($review.gates | length) as $gates_total |
  ([$review.gates[] | select(. == true)] | length) as $gates_passed |
  .upstream_commit == $old and
  .upstream_candidate.commit == $candidate and
  .upstream_candidate.inventory_total == $total and
  .upstream_candidate.reviewed == $reviewed and
  .upstream_candidate.pending == 0 and
  .upstream_candidate.gates_total == 10 and
  .upstream_candidate.gates_total == $gates_total and
  .upstream_candidate.gates_passed == $gates_passed and
  .upstream_candidate.promoted == false and
  .project_completion.candidate_promoted == false and
  .project_completion.post_promotion_full_gate == false and
  .project_completion.complete == false
' "$project_state" >/dev/null || {
    echo "project-state is not consistent with the complete ten-gate candidate review" >&2
    exit 1
}

# Promotion changes several tracked source and generated files. Keep a complete
# pre-mutation snapshot so any failed generator or gate restores the accepted
# pin instead of leaving a mixed-baseline tree.
candidate_manifest="$backup_dir/candidates"
{
    find "$crate_dir" -type f -name '*.rs' -exec grep -l -F "@ $old" {} \;
    find "$crate_dir" -type f -name '*.rs' -exec grep -l -F "@ $candidate" {} \;
    printf '%s\n' \
        "$lock" \
        "$project_state" \
        "$crate_dir/UPSTREAM.md" \
        "$crate_dir/port-map.json" \
        "$crate_dir/module-map.json" \
        "$crate_dir/mirror-closure.json" \
        "$crate_dir/strict-credit-audit.json" \
        "$crate_dir/upstream-delta.json" \
        "$repo_dir/runtime/cliproxyapi-porting-dashboard.html" \
        "$receipt"
} | sort -u > "$candidate_manifest"

while IFS= read -r path; do
    if [ -f "$path" ]; then
        printf '%s\n' "$path" >> "$backup_manifest"
    else
        printf '%s\n' "$path" >> "$absent_manifest"
    fi
done < "$candidate_manifest"

while IFS= read -r path; do
    relative=${path#"$repo_dir"/}
    mkdir -p "$backup_dir/files/$(dirname "$relative")"
    cp "$path" "$backup_dir/files/$relative"
done < "$backup_manifest"

# Normalize staged candidate anchors/statuses into the accepted header form,
# remove mirrors for reviewed upstream deletions, and advance unchanged refs.
"$crate_dir/scripts/promote_candidate_headers.sh" \
    "$crate_dir" "$old" "$candidate" "$review"

commit_date=$(git -C "$upstream_dir" show -s --format=%cs "$candidate")
jq --arg candidate "$candidate" --arg date "$commit_date" \
    '.base_commit = $candidate | .base_commit_date = $date' "$lock" > "$lock_tmp"
mv "$lock_tmp" "$lock"
jq --arg candidate "$candidate" '
  .upstream_commit = $candidate |
  .project_completion.complete = false |
  .project_completion.accepted_pin_complete = false |
  .project_completion.candidate_promoted = true |
  .project_completion.post_promotion_full_gate = false |
  .upstream_candidate.reviewed = .upstream_candidate.inventory_total |
  .upstream_candidate.pending = 0 |
  .upstream_candidate.gates_passed = .upstream_candidate.gates_total |
  .upstream_candidate.promoted = true |
  .upstream_candidate.status = "promoted" |
  (.upstream_candidate.inventory_total | tostring) as $total |
  (.upstream_candidate.gates_total | tostring) as $gates |
  .work_items |= map(
    if .id == "upstream-candidate-promotion" then
      .status = "in_progress" |
      .detail = "\($total)/\($total) Candidate-Reviews vollständig, 0 pending, \($gates)/\($gates) Promotion-Gates, Promotion JA. Accepted-Pin-Full-Gate ausstehend."
    else . end)
' "$project_state" > "$project_tmp"
mv "$project_tmp" "$project_state"
OLD_PIN="$old" NEW_PIN="$candidate" COMMIT_DATE="$commit_date" perl -pi -e '
    s/Pinned commit: `\Q$ENV{OLD_PIN}\E`/Pinned commit: `$ENV{NEW_PIN}`/;
    s/Commit date: [0-9-]+/Commit date: $ENV{COMMIT_DATE}/;
' "$crate_dir/UPSTREAM.md"

"$crate_dir/scripts/build_port_map.sh" "$upstream_dir" "$crate_dir/port-map.json"
"$crate_dir/scripts/build_module_map.sh" "$crate_dir/port-map.json" "$crate_dir/module-map.json"
"$crate_dir/scripts/build_mirror_closure.sh" "$crate_dir/port-map.json" "$crate_dir/mirror-closure.json"
"$crate_dir/scripts/build_upstream_delta.sh" HEAD "$crate_dir/upstream-delta.json" >/dev/null

# The promoted tree is mechanically closed by its completed file review, but
# strict accepted-pin credit is deliberately zero until the fresh umbrella
# receipt is recorded. Keep the transitional dashboard counters explicit.
jq --slurpfile map "$crate_dir/port-map.json" '
  .mirror_verification.verified_classified_production_files =
    ($map[0].summary.production_go_files - $map[0].summary.production_open_files) |
  .mirror_verification.verified_test_files =
    ($map[0].summary.test_go_files - $map[0].summary.test_open_files) |
  .mirror_verification.strict_production_files = 0 |
  .mirror_verification.strict_test_files = 0
' "$project_state" > "$project_tmp"
mv "$project_tmp" "$project_state"
"$crate_dir/scripts/check_upstream_anchors.sh"
"$crate_dir/scripts/build_dashboard.sh" "$repo_dir/runtime/cliproxyapi-porting-dashboard.html" >/dev/null
node "$crate_dir/scripts/tests/strict_credit_dashboard_test.mjs" \
    "$repo_dir/runtime/cliproxyapi-porting-dashboard.html" \
    "$crate_dir/strict-credit-audit.json" >/dev/null
"$crate_dir/scripts/check_tracking.sh" "$upstream_dir"

"$crate_dir/scripts/write_upstream_receipt.sh" \
    "$lock" "$delta" "$review" "$old" "$candidate" "$receipt"

promotion_complete=true
echo "upstream pin promoted: $old -> $candidate"
