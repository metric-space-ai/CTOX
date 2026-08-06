#!/bin/bash
set -eu

crate_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
repo_dir=$(CDPATH= cd -- "$crate_dir/../../../.." && pwd)
receipt=${1:?usage: record_post_promotion_full_gate.sh <strict-umbrella-receipt.json>}
state=${PROJECT_STATE_FILE:-"$crate_dir/project-state.json"}
audit=${STRICT_AUDIT_FILE:-"$crate_dir/strict-credit-audit.json"}
map=${PORT_MAP_FILE:-"$crate_dir/port-map.json"}
dashboard=${DASHBOARD_OUTPUT:-"$repo_dir/runtime/cliproxyapi-porting-dashboard.html"}
artifact_validator=${POST_PROMOTION_ARTIFACT_VALIDATOR:-}
pin=$(jq -r .base_commit "$crate_dir/upstream-lock.json")
promotion_receipt=${PROMOTION_RECEIPT:-"$repo_dir/docs/cliproxyapi-upstream-history/$pin.json"}
strict_root=${STRICT_RECEIPT_ROOT:-"$repo_dir/runtime/cliproxyapi-strict-receipts/$pin"}

"$crate_dir/scripts/check_upstream_receipt.sh" "$promotion_receipt" >/dev/null
"$crate_dir/scripts/check_strict_umbrella_receipt.sh" "$receipt" "$pin" >/dev/null
test "$(jq -r .accepted_commit "$promotion_receipt")" = "$pin" || {
    echo "promotion receipt does not accept the current pin" >&2
    exit 1
}
if [ "$(jq -r .started_at "$receipt")" \< "$(jq -r .promoted_at "$promotion_receipt")" ]; then
    echo "full gate started before the recorded promotion" >&2
    exit 1
fi

strict_root=$(CDPATH= cd -- "$strict_root" && pwd -P)
receipt_dir=$(CDPATH= cd -- "$(dirname -- "$receipt")" && pwd -P)
receipt="$receipt_dir/$(basename -- "$receipt")"
case "$receipt" in
    "$strict_root"/*) ;;
    *)
        echo "post-promotion receipt is outside the current-pin strict receipt root" >&2
        exit 1
        ;;
esac
receipt_tail=${receipt#"$strict_root"/}
case "$receipt_tail" in
    ""|/*|*..*)
        echo "unsafe post-promotion receipt path" >&2
        exit 1
        ;;
esac

jq -e --arg pin "$pin" '
  .upstream_commit == $pin and
  .upstream_candidate.commit == $pin and
  .upstream_candidate.inventory_total == .upstream_candidate.reviewed and
  .upstream_candidate.pending == 0 and
  .upstream_candidate.gates_total == 10 and
  .upstream_candidate.gates_passed == 10 and
  .upstream_candidate.promoted == true and
  .project_completion.candidate_promoted == true and
  .project_completion.post_promotion_full_gate == false and
  .project_completion.complete == false and
  .project_completion.accepted_pin_complete == false
' "$state" >/dev/null || {
    echo "project-state is not in the promoted, post-gate-pending state" >&2
    exit 1
}
jq -e --arg pin "$pin" '
  .upstream_commit == $pin and
  .summary.production_open_files == 0 and
  .summary.test_open_files == 0
' "$map" >/dev/null || {
    echo "port map is not strictly closed at the promoted pin" >&2
    exit 1
}

if command -v sha256sum >/dev/null 2>&1; then
    receipt_hash=$(sha256sum "$receipt" | awk '{print $1}')
else
    receipt_hash=$(shasum -a 256 "$receipt" | awk '{print $1}')
fi
audit_receipt="../../../../runtime/cliproxyapi-strict-receipts/$pin/$receipt_tail"
transaction=$(mktemp -d "$crate_dir/.post-promotion-full-gate.XXXXXX")
audit_tmp="$transaction/audit.json"
state_tmp="$transaction/state.json"
completed=false

restore_on_failure() {
    status=$?
    if [ "$completed" != true ]; then
        cp "$transaction/audit.before.json" "$audit"
        cp "$transaction/state.before.json" "$state"
        if [ -f "$transaction/dashboard.before.html" ]; then
            cp "$transaction/dashboard.before.html" "$dashboard"
        else
            rm -f "$dashboard"
        fi
    fi
    rm -rf "$transaction"
    exit "$status"
}
cp "$audit" "$transaction/audit.before.json"
cp "$state" "$transaction/state.before.json"
test ! -f "$dashboard" || cp "$dashboard" "$transaction/dashboard.before.html"
trap restore_on_failure EXIT HUP INT TERM
trap 'exit 1' HUP INT TERM

# Rebind the strict-credit audit only after the accepted-pin umbrella receipt
# validates. Historical clusters are retained only for paths that still exist.
jq --slurpfile map "$map" \
  --arg pin "$pin" --arg receipt "$audit_receipt" --arg hash "$receipt_hash" '
  ($map[0].files | map(.upstream)) as $paths |
  ($map[0].summary.production_go_files) as $production |
  ($map[0].summary.test_go_files) as $tests |
  .upstream_commit = $pin |
  .mechanical_closure = {production:$production,production_total:$production,tests:$tests,tests_total:$tests} |
  .strict_closure = {production:$production,production_total:$production,tests:$tests,tests_total:$tests} |
  .gap = {production:0,tests:0,path_reconstructed_production:0,path_reconstructed_tests:0,
          ledger_membership_unresolved_production:0,ledger_membership_unresolved_tests:0} |
  .unresolved_ledger_membership.production = 0 |
  .unresolved_ledger_membership.tests = 0 |
  .reconstructed = [] |
  .closed_reconstructed |= map(
    .production |= map(select(. as $path | $paths | index($path))) |
    .tests |= map(select(. as $path | $paths | index($path)))) |
  .umbrella_receipt = {path:$receipt,sha256:$hash,credited_production:9,credited_tests:15}
' "$audit" > "$audit_tmp"

jq --slurpfile map "$map" '
  ($map[0].summary.production_go_files) as $production |
  ($map[0].summary.test_go_files) as $tests |
  .mirror_verification.verified_classified_production_files = $production |
  .mirror_verification.verified_test_files = $tests |
  .mirror_verification.strict_production_files = $production |
  .mirror_verification.strict_test_files = $tests |
  .project_completion.accepted_pin_complete = true |
  .project_completion.post_promotion_full_gate = true |
  .project_completion.complete = true |
  .checkpoint.status = "complete" |
  (.upstream_candidate.inventory_total | tostring) as $total |
  (.upstream_candidate.gates_total | tostring) as $gates |
  .work_items |= map(
    if .id == "upstream-candidate-promotion" then
      .status = "complete" |
      .detail = "\($total)/\($total) Candidate-Reviews, \($gates)/\($gates) Promotion-Gates, Promotion und Accepted-Pin-Full-Gate vollständig."
    else . end)
' "$state" > "$state_tmp"

mv "$audit_tmp" "$audit"
mv "$state_tmp" "$state"
"$crate_dir/scripts/check_strict_credit_audit.sh" "$audit" "$state" "$map" >/dev/null
if [ -n "$artifact_validator" ]; then
    "$artifact_validator" "$state" "$audit" "$map" "$dashboard"
else
    "$crate_dir/scripts/check_tracking.sh" >/dev/null
    "$crate_dir/scripts/build_dashboard.sh" "$dashboard" >/dev/null
    node "$crate_dir/scripts/tests/strict_credit_dashboard_test.mjs" \
        "$dashboard" "$audit" >/dev/null
fi

completed=true
rm -rf "$transaction"
trap - EXIT HUP INT TERM
echo "post-promotion full gate recorded: $pin $receipt_hash"
