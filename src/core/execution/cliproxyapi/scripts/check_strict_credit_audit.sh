#!/bin/bash
set -eu

crate_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
audit_file=${1:-"$crate_dir/strict-credit-audit.json"}
state_file=${2:-"$crate_dir/project-state.json"}
map_file=${3:-"$crate_dir/port-map.json"}

jq -e --slurpfile state "$state_file" --slurpfile map "$map_file" '
  def require($condition; $message):
    if $condition then . else error($message) end;
  def closed_status:
    . == "ported" or . == "adapted_to_ctox" or . == "replaced_by_ctox";

  . as $audit
  | $state[0] as $state
  | $map[0] as $map
  | ($map.files | map({key: .upstream, value: .}) | from_entries) as $files
  | ([
      $audit.reconstructed[] as $cluster
      | ($cluster.production[] | {path: ., test: false, cluster: $cluster.cluster}),
        ($cluster.tests[] | {path: ., test: true, cluster: $cluster.cluster})
    ]) as $reconstructed
  | ([
      $audit.closed_reconstructed[] as $cluster
      | ($cluster.production[] | {path: ., test: false, cluster: $cluster.cluster}),
        ($cluster.tests[] | {path: ., test: true, cluster: $cluster.cluster})
    ]) as $closed_reconstructed
  | ($reconstructed + $closed_reconstructed) as $audited_paths
  | require($audit.schema == "ctox.cliproxyapi.strict-credit-audit.v1";
      "strict-credit audit schema mismatch")
  | require($audit.upstream_commit == $state.upstream_commit and
            $audit.upstream_commit == $map.upstream_commit;
      "strict-credit audit upstream pin mismatch")
  | require(($audit.reconstructed | type) == "array" and
            ($audit.reconstructed | all(
              (.cluster | type) == "string" and (.cluster | length) > 0 and
              (.reason | type) == "string" and (.reason | length) > 0 and
              (.production | type) == "array" and (.tests | type) == "array"));
      "strict-credit reconstructed clusters are malformed")
  | require(($audit.closed_reconstructed | type) == "array" and
            ($audit.closed_reconstructed | all(
              (.cluster | type) == "string" and (.cluster | length) > 0 and
              (.reason | type) == "string" and (.reason | length) > 0 and
              (.production | type) == "array" and (.tests | type) == "array"));
      "strict-credit closed reconstructed clusters are malformed")
  | require(([$audit.reconstructed[], $audit.closed_reconstructed[]]
              | map(.cluster) | unique | length) ==
            (($audit.reconstructed | length) + ($audit.closed_reconstructed | length));
      "strict-credit cluster names are not unique")
  | require(($audited_paths | map(.path) | unique | length) == ($audited_paths | length);
      "strict-credit audited paths are duplicated")
  | require($audit.mechanical_closure.production_total == $map.summary.production_go_files and
            $audit.mechanical_closure.tests_total == $map.summary.test_go_files and
            $audit.mechanical_closure.production ==
              ($map.summary.production_go_files - $map.summary.production_open_files) and
            $audit.mechanical_closure.tests ==
              ($map.summary.test_go_files - $map.summary.test_open_files) and
            $audit.mechanical_closure.production ==
              $state.mirror_verification.verified_classified_production_files and
            $audit.mechanical_closure.tests ==
              $state.mirror_verification.verified_test_files;
      "strict-credit mechanical closure does not match project-state/port-map")
  | require($audit.strict_closure.production_total == $map.summary.production_go_files and
            $audit.strict_closure.tests_total == $map.summary.test_go_files and
            $audit.strict_closure.production ==
              $state.mirror_verification.strict_production_files and
            $audit.strict_closure.tests ==
              $state.mirror_verification.strict_test_files;
      "strict-credit strict closure does not match project-state/port-map")
  | require($audit.gap.production ==
              ($audit.mechanical_closure.production - $audit.strict_closure.production) and
            $audit.gap.tests ==
              ($audit.mechanical_closure.tests - $audit.strict_closure.tests);
      "strict-credit gap arithmetic is inconsistent")
  | require($audit.gap.path_reconstructed_production ==
              ($reconstructed | map(select(.test == false)) | length) and
            $audit.gap.path_reconstructed_tests ==
              ($reconstructed | map(select(.test == true)) | length);
      "strict-credit reconstructed path counts are inconsistent")
  | require($audit.gap.ledger_membership_unresolved_production ==
              ($audit.gap.production - $audit.gap.path_reconstructed_production) and
            $audit.gap.ledger_membership_unresolved_tests ==
              ($audit.gap.tests - $audit.gap.path_reconstructed_tests) and
            $audit.unresolved_ledger_membership.production ==
              $audit.gap.ledger_membership_unresolved_production and
            $audit.unresolved_ledger_membership.tests ==
              $audit.gap.ledger_membership_unresolved_tests;
      "strict-credit unresolved ledger arithmetic is inconsistent")
  | require(
      if $audit.gap.production == 0 and $audit.gap.tests == 0 then
        ($audit.umbrella_receipt.path | type) == "string" and
        ($audit.umbrella_receipt.path | length) > 0 and
        ($audit.umbrella_receipt.sha256 | test("^[0-9a-f]{64}$")) and
        $audit.umbrella_receipt.credited_production == 9 and
        $audit.umbrella_receipt.credited_tests == 15
      else true end;
      "strict-credit closed historical gap lacks its umbrella receipt")
  | reduce $audited_paths[] as $entry (.;
      require($files[$entry.path] != null;
        "strict-credit audited path is absent from port-map: \($entry.path)")
      | require($files[$entry.path].test == $entry.test;
        "strict-credit audited path has wrong test flag: \($entry.path)")
      | require(($files[$entry.path].status | closed_status);
        "strict-credit audited path is not mechanically closed: \($entry.path)"))
' "$audit_file" >/dev/null

if jq -e '.gap.production == 0 and .gap.tests == 0' "$audit_file" >/dev/null; then
    audit_pin=$(jq -r .upstream_commit "$audit_file")
    receipt_relative=$(jq -r .umbrella_receipt.path "$audit_file")
    expected_prefix="../../../../runtime/cliproxyapi-strict-receipts/$audit_pin/"
    case "$receipt_relative" in
        "$expected_prefix"*)
            receipt_tail=${receipt_relative#"$expected_prefix"}
            ;;
        *)
            echo "strict-credit umbrella receipt path is outside expected pin subtree: $receipt_relative" >&2
            exit 1
            ;;
    esac
    case "$receipt_tail" in
        ""|/*|*..*)
            echo "strict-credit umbrella receipt path is unsafe: $receipt_relative" >&2
            exit 1
            ;;
    esac
    expected_root=$(CDPATH= cd -- "$crate_dir/../../../../runtime/cliproxyapi-strict-receipts/$audit_pin" && pwd -P)
    receipt_dir=$(CDPATH= cd -- "$crate_dir/$(dirname -- "$receipt_relative")" && pwd -P)
    receipt="$receipt_dir/$(basename -- "$receipt_relative")"
    case "$receipt" in
        "$expected_root"/*) ;;
        *)
            echo "strict-credit umbrella receipt resolves outside expected pin subtree: $receipt" >&2
            exit 1
            ;;
    esac
    test -f "$receipt" || {
        echo "strict-credit umbrella receipt is missing: $receipt" >&2
        exit 1
    }
    test ! -L "$receipt" || {
        echo "strict-credit umbrella receipt must not be a symlink: $receipt" >&2
        exit 1
    }
    if command -v sha256sum >/dev/null 2>&1; then
        receipt_hash=$(sha256sum "$receipt" | awk '{print $1}')
    else
        receipt_hash=$(shasum -a 256 "$receipt" | awk '{print $1}')
    fi
    test "$receipt_hash" = "$(jq -r .umbrella_receipt.sha256 "$audit_file")" || {
        echo "strict-credit umbrella receipt hash mismatch: $receipt" >&2
        exit 1
    }
    "$crate_dir/scripts/check_strict_umbrella_receipt.sh" "$receipt" "$audit_pin" >/dev/null
fi

echo "strict-credit audit ok: $audit_file"
