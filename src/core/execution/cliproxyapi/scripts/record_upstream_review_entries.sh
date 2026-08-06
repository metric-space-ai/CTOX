#!/bin/bash
set -eu

review=${1:?usage: record_upstream_review_entries.sh <review.json> <evidence-bundle.json>}
bundle=${2:?usage: record_upstream_review_entries.sh <review.json> <evidence-bundle.json>}
review_dir=$(CDPATH= cd -- "$(dirname -- "$review")" && pwd)
temporary=$(mktemp "$review_dir/.upstream-review.XXXXXX")
trap 'rm -f "$temporary"' EXIT

jq --slurpfile bundle "$bundle" '
  def require($condition; $message):
    if $condition then . else error($message) end;
  def valid_evidence($entry; $source_kind):
    ($entry.disposition | type) == "string" and ($entry.disposition | length) > 0 and
    ($entry.evidence | type) == "array" and ($entry.evidence | length) > 0 and
    ($entry.upstream_evidence | type) == "array" and ($entry.upstream_evidence | length) > 0 and
    (($source_kind != "go_production" and $source_kind != "go_test") or
      (($entry.rust_evidence | type) == "array" and ($entry.rust_evidence | length) > 0));

  . as $review |
  $bundle[0] as $bundle |
  ($bundle.entries | map(.upstream)) as $paths |
  ($review.changes | map({key: .upstream, value: .}) | from_entries) as $by_path |
  require($review.schema == "ctox.cliproxyapi.upstream-review.v3";
    "upstream review schema mismatch") |
  require($bundle.schema == "ctox.cliproxyapi.review-evidence-bundle.v1" and
          ($bundle.entries | type) == "array" and ($bundle.entries | length) > 0;
    "review evidence bundle schema mismatch") |
  require(($paths | length) == ($paths | unique | length);
    "review evidence bundle paths are duplicated") |
  reduce $bundle.entries[] as $entry (.;
    require($by_path[$entry.upstream] != null;
      "review evidence path is absent from candidate inventory: \($entry.upstream)") |
    require($by_path[$entry.upstream].review_status == "pending";
      "review evidence path is not pending: \($entry.upstream)") |
    require(valid_evidence($entry; $by_path[$entry.upstream].source_kind);
      "review evidence is incomplete: \($entry.upstream)") |
    .changes |= map(
      if .upstream == $entry.upstream then
        .review_status = "complete" |
        .disposition = $entry.disposition |
        .evidence = $entry.evidence |
        .rust_evidence = $entry.rust_evidence |
        .upstream_evidence = $entry.upstream_evidence
      else . end)) |
  require(([.changes[] | select(.upstream as $path | ($paths | index($path)) != null and
              .review_status == "complete")] | length) == ($paths | length);
    "review evidence update failed conservation")
' "$review" > "$temporary"

mv "$temporary" "$review"
trap - EXIT
jq '{total:(.changes|length), complete:([.changes[]|select(.review_status=="complete")]|length), pending:([.changes[]|select(.review_status=="pending")]|length)}' "$review"
