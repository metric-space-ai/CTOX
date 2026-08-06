#!/bin/bash
set -eu

old_review=${1:?usage: reconcile_upstream_review.sh <old-review.json> <corrected-delta.json> <fresh-review.json> <output.json> [remap.json]}
corrected_delta=${2:?usage: reconcile_upstream_review.sh <old-review.json> <corrected-delta.json> <fresh-review.json> <output.json> [remap.json]}
fresh_review=${3:?usage: reconcile_upstream_review.sh <old-review.json> <corrected-delta.json> <fresh-review.json> <output.json> [remap.json]}
output=${4:?usage: reconcile_upstream_review.sh <old-review.json> <corrected-delta.json> <fresh-review.json> <output.json> [remap.json]}
remap_file=${5:-}

if [ -e "$output" ]; then
    echo "reconciled review already exists: $output" >&2
    exit 1
fi

hash_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

if [ -n "$remap_file" ]; then
    remap=$(jq -c '.' "$remap_file")
else
    remap='{"schema":"ctox.cliproxyapi.review-reconciliation-remap.v1","mappings":[]}'
fi

old_hash=$(hash_file "$old_review")
delta_hash=$(hash_file "$corrected_delta")
fresh_hash=$(hash_file "$fresh_review")
output_dir=$(CDPATH= cd -- "$(dirname -- "$output")" && pwd)
temporary=$(mktemp "$output_dir/reconciled-review.XXXXXX")
trap 'rm -f "$temporary"' EXIT

jq -n \
    --slurpfile old "$old_review" \
    --slurpfile delta "$corrected_delta" \
    --slurpfile fresh "$fresh_review" \
    --argjson remap "$remap" \
    --arg old_hash "$old_hash" \
    --arg delta_hash "$delta_hash" \
    --arg fresh_hash "$fresh_hash" '
  def require($condition; $message):
    if $condition then . else error($message) end;
  def inventory_fields:
    {upstream, module, kind, source_kind, required_action};
  def progress_fields:
    {
      review_status,
      disposition,
      evidence,
      rust_evidence,
      upstream_evidence
    };
  def valid_complete:
    .review_status == "complete" and
    (.disposition | type) == "string" and (.disposition | length) > 0 and
    (.evidence | type) == "array" and (.evidence | length) > 0 and
    (.upstream_evidence | type) == "array" and (.upstream_evidence | length) > 0 and
    ((.source_kind != "go_production" and .source_kind != "go_test") or
      ((.rust_evidence | type) == "array" and (.rust_evidence | length) > 0));

  $old[0] as $old_review |
  $delta[0] as $delta |
  $fresh[0] as $fresh |
  $remap as $remap |
  ($old_review.changes | map({key: .upstream, value: .}) | from_entries) as $old_by_path |
  ($fresh.changes | map({key: .upstream, value: .}) | from_entries) as $fresh_by_path |
  ($remap.mappings | map({key: .to_upstream, value: .}) | from_entries) as $remap_by_target |
  ($remap.mappings | map(.from_upstream)) as $remap_sources |
  ($remap.mappings | map(.to_upstream)) as $remap_targets |
  ($old_review.changes | map(select(.review_status == "complete"))) as $completed |

  require($old_review.schema == "ctox.cliproxyapi.upstream-review.v3";
    "old review schema mismatch") |
  require($delta.schema == "ctox.cliproxyapi.upstream-delta.v2";
    "corrected delta schema mismatch") |
  require($fresh.schema == "ctox.cliproxyapi.upstream-review.v3";
    "fresh review schema mismatch") |
  require($remap.schema == "ctox.cliproxyapi.review-reconciliation-remap.v1" and
          ($remap.mappings | type) == "array";
    "review reconciliation remap schema mismatch") |
  require($old_review.base_commit == $delta.base_commit and
          $old_review.candidate_commit == $delta.candidate_commit and
          $fresh.base_commit == $delta.base_commit and
          $fresh.candidate_commit == $delta.candidate_commit;
    "review reconciliation identity mismatch") |
  require(($old_review.changes | map(.upstream) | length) ==
          ($old_review.changes | map(.upstream) | unique | length);
    "old review contains duplicate upstream keys") |
  require(($delta.changes | map(.upstream) | length) ==
          ($delta.changes | map(.upstream) | unique | length);
    "corrected delta contains duplicate upstream keys") |
  require(($fresh.changes | map(.upstream) | length) ==
          ($fresh.changes | map(.upstream) | unique | length);
    "fresh review contains duplicate upstream keys") |
  require(([$fresh.changes[] | inventory_fields] | sort_by(.upstream)) ==
          ([$delta.changes[] | inventory_fields] | sort_by(.upstream));
    "fresh review inventory does not match corrected delta") |
  require(($fresh.changes | all(
            .review_status == "pending" and .disposition == "" and
            .evidence == [] and .rust_evidence == [] and .upstream_evidence == []));
    "fresh review is not fail-closed") |
  require(($fresh.gates | all(. == false)) and
          ($fresh.gate_evidence | all(
            .status == "pending" and .command == [] and .completed_at == "" and
            .output_sha256 == "" and .log == ""));
    "fresh review gate state is not fail-closed") |
  require(($completed | all(valid_complete));
    "old review contains an invalid completed disposition") |
  require(($remap.mappings | all(
            (.from_upstream | type) == "string" and (.from_upstream | length) > 0 and
            (.to_upstream | type) == "string" and (.to_upstream | length) > 0 and
            .from_upstream != .to_upstream and
            (.reason | type) == "string" and (.reason | length) > 0));
    "remap entries are malformed") |
  require(($remap_sources | length) == ($remap_sources | unique | length) and
          ($remap_targets | length) == ($remap_targets | unique | length);
    "remap sources and targets must be unique") |
  require(($remap.mappings | all(
            $old_by_path[.from_upstream] != null and
            ($old_by_path[.from_upstream] | valid_complete) and
            $fresh_by_path[.to_upstream] != null));
    "remap source must be completed and target must exist in fresh review") |
  require(($remap.mappings | all(
            . as $mapping |
            ($old_by_path[$mapping.to_upstream] == null) or
            ($old_by_path[$mapping.to_upstream].review_status != "complete") or
            ($remap_sources | index($mapping.to_upstream)) != null));
    "remap target conflicts with a directly replayable completion") |
  require(($completed | all(
            (. as $completed_entry | .upstream as $path |
              if ($remap_sources | index($path)) != null then true
              else
                $fresh_by_path[$path] != null and
                ($fresh_by_path[$path] | inventory_fields) ==
                  ($completed_entry | inventory_fields)
              end)));
    "a completed old review entry is not conserved by direct replay or remap") |

  ($fresh.changes | map(
    . as $fresh_entry |
    if $remap_by_target[$fresh_entry.upstream] != null then
      $remap_by_target[$fresh_entry.upstream] as $mapping |
      . + ($old_by_path[$mapping.from_upstream] | progress_fields)
    elif ($old_by_path[$fresh_entry.upstream] != null and
          $old_by_path[$fresh_entry.upstream].review_status == "complete" and
          ($remap_sources | index($fresh_entry.upstream)) == null) then
      . + ($old_by_path[$fresh_entry.upstream] | progress_fields)
    else . end
  )) as $reconciled_changes |
  ($completed | map(select(.upstream as $path |
    ($remap_sources | index($path)) == null)) | length) as $direct_count |
  ($remap.mappings | length) as $remap_count |
  ($reconciled_changes | map(select(.review_status == "complete")) | length) as $complete_count |

  require(($reconciled_changes | map(.upstream) | length) ==
          ($reconciled_changes | map(.upstream) | unique | length);
    "reconciled review contains duplicate upstream keys") |
  require(($reconciled_changes | length) == ($fresh.changes | length) and
          $complete_count == ($direct_count + $remap_count) and
          $complete_count == ($completed | length);
    "reconciliation conservation check failed") |

  $fresh + {
    status: "in_progress",
    changes: $reconciled_changes,
    gates: ($fresh.gates | with_entries(.value = false)),
    gate_evidence: ($fresh.gate_evidence | with_entries(.value = {
      status: "pending",
      command: [],
      completed_at: "",
      output_sha256: "",
      log: ""
    })),
    reconciliation: {
      schema: "ctox.cliproxyapi.review-reconciliation.v1",
      policy: "Replay completed dispositions by unique upstream key only; positional replay is forbidden and gates always reset fail-closed.",
      source_review_sha256: $old_hash,
      corrected_delta_sha256: $delta_hash,
      fresh_review_sha256: $fresh_hash,
      old_inventory_count: ($old_review.changes | length),
      corrected_inventory_count: ($fresh.changes | length),
      replayed_direct: $direct_count,
      replayed_remapped: $remap_count,
      pending: (($fresh.changes | length) - $complete_count),
      remappings: $remap.mappings
    }
  }
' > "$temporary"

if ! ln "$temporary" "$output" 2>/dev/null; then
    echo "reconciled review appeared concurrently: $output" >&2
    exit 1
fi

echo "$output"
