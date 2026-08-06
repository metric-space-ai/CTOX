#!/bin/bash
set -eu

crate_dir=${CLIPROXYAPI_CRATE_DIR:-$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)}
upstream_dir=${1:-"$crate_dir/../../../../../runtime/cliproxyapi-upstream"}
output=${2:-"$crate_dir/port-map.json"}
rows=$(mktemp)
sources=$(mktemp)
statuses=$(mktemp)
supplemental=$(mktemp)
candidate=$(mktemp)
trap 'rm -f "$rows" "$sources" "$statuses" "$supplemental" "$candidate"' EXIT

commit=$(git -C "$upstream_dir" rev-parse HEAD)
find "$upstream_dir" -type f -name '*.go' | sort | while IFS= read -r source; do
    relative=${source#"$upstream_dir"/}
    target=${relative%.go}.rs
    is_test=false
    case "$relative" in *_test.go) is_test=true ;; esac
    printf '%s\t%s\t%s\n' "$relative" "$target" "$is_test" >> "$sources"
done

find "$crate_dir" -type f -name '*.rs' -print0 |
    xargs -0 awk -v prefix="$crate_dir/" '
        /^\/\/ Port-Status: / {
            status = $0
            sub(/^\/\/ Port-Status: /, "", status)
            file = FILENAME
            sub("^" prefix, "", file)
            print file "\t" status "\taccepted"
            nextfile
        }
        /^\/\/ Candidate-Port-Status: / {
            status = $0
            sub(/^\/\/ Candidate-Port-Status: /, "", status)
            file = FILENAME
            sub("^" prefix, "", file)
            print file "\t" status "\tcandidate"
            nextfile
        }
    ' > "$statuses"

# Status values are part of the evidence contract. An unknown spelling must
# never disappear from both the open and closed counters, because that makes a
# file look complete without assigning it a reviewable disposition.
invalid_statuses=$(awk -F '\t' '
    $2 != "ported" &&
    $2 != "partial" &&
    $2 != "adapted_to_ctox" &&
    $2 != "replaced_by_ctox" &&
    $2 != "scaffold" &&
    $2 != "supplemental" { print }
' "$statuses")
if [ -n "$invalid_statuses" ]; then
    echo "unsupported Port-Status values:" >&2
    printf '%s\n' "$invalid_statuses" >&2
    exit 1
fi

awk -F '\t' 'NR == FNR { status[$1] = $2; next }
    { print $1 "\t" $2 "\t" (($2 in status) ? status[$2] : "missing") "\t" $3 }
' "$statuses" "$sources" > "$rows"

awk -F '\t' 'NR == FNR { upstream[$2] = 1; next }
    $3 == "accepted" && !($1 in upstream) { print $1 "\t" $2 }
' "$sources" "$statuses" > "$supplemental"

# Candidate-only mirrors are source-visible and fail-closed without changing
# accepted-pin closure counters. Once their upstream path is promoted into the
# accepted checkout, the same status participates in the ordinary file row and
# disappears from this overlay automatically.
awk -F '\t' 'NR == FNR { upstream[$2] = 1; next }
    $3 == "candidate" && !($1 in upstream) { print $1 "\t" $2 }
' "$sources" "$statuses" > "$candidate"

jq -Rn --arg commit "$commit" --rawfile supplemental "$supplemental" --rawfile candidate "$candidate" '
  [inputs | split("\t") | {
    upstream: .[0], rust: .[1], status: .[2], test: (.[3] == "true")
  }] as $files |
  ($supplemental | split("\n") | map(select(length > 0) | split("\t") | {
    rust: .[0], status: .[1]
  })) as $supplemental_files |
  ($candidate | split("\n") | map(select(length > 0) | split("\t") | {
    rust: .[0], status: .[1]
  })) as $candidate_files |
  {
    schema: "ctox.cliproxyapi.port-map.v1",
    upstream_commit: $commit,
    policy: "Scaffolds and tests contribute zero semantic port points.",
    summary: {
      go_files: ($files | length),
      production_go_files: ($files | map(select(.test == false)) | length),
      test_go_files: ($files | map(select(.test == true)) | length),
      ported_files: ($files | map(select(.status == "ported")) | length),
      partial_files: ($files | map(select(.status == "partial")) | length),
      production_partial_files: ($files | map(select(.status == "partial" and .test == false)) | length),
      test_partial_files: ($files | map(select(.status == "partial" and .test == true)) | length),
      adapted_to_ctox_files: ([$files[], $supplemental_files[]] | map(select(.status == "adapted_to_ctox")) | length),
      replaced_by_ctox_files: ([$files[], $supplemental_files[]] | map(select(.status == "replaced_by_ctox")) | length),
      supplemental_files: ($supplemental_files | length),
      candidate_staged_files: ($candidate_files | length),
      scaffold_files: ($files | map(select(.status == "scaffold")) | length),
      production_scaffold_files: ($files | map(select(.status == "scaffold" and .test == false)) | length),
      test_scaffold_files: ($files | map(select(.status == "scaffold" and .test == true)) | length),
      production_open_files: ($files | map(select((.status == "scaffold" or .status == "partial" or .status == "missing") and .test == false)) | length),
      test_open_files: ($files | map(select((.status == "scaffold" or .status == "partial" or .status == "missing") and .test == true)) | length)
    },
    files: $files,
    supplemental_files: $supplemental_files,
    candidate_files: $candidate_files
  }' < "$rows" > "$output"
