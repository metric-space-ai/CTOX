#!/bin/bash
set -eu

crate_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
repo_dir=$(CDPATH= cd -- "$crate_dir/../../../.." && pwd)
lock_file="$crate_dir/upstream-lock.json"
port_map="$crate_dir/port-map.json"
project_state="$crate_dir/project-state.json"
pin=$(jq -r '.base_commit' "$lock_file")
candidate=$(jq -r '.upstream_candidate.commit' "$project_state")
candidate_review="$repo_dir/runtime/cliproxyapi-upstream-reviews/$candidate/upstream-review.json"
failures=$(mktemp)
trap 'rm -f "$failures"' EXIT

jq -r '.files[] | [.upstream, .rust, .status] | @tsv' "$port_map" |
while IFS=$'\t' read -r upstream rust status; do
    file="$crate_dir/$rust"
    if [ ! -f "$file" ]; then
        # Missing mirrors remain ordinary open port work and are reported by
        # port-map/module-map. There is no stale anchor to validate yet.
        continue
    fi
    anchor=$(sed -n '1p' "$file")
    case "$anchor" in
        "// ref: $upstream @ $pin"|"// ref: $upstream:"*" @ $pin") ;;
        "// candidate-ref: $upstream deleted @ $candidate")
            # A reviewed deletion remains as an uncompiled tombstone until
            # promotion removes it atomically with the pin transition. Accept
            # no free-form marker: candidate identity and the single complete
            # deletion disposition must both match the active review ledger.
            if [ "$candidate" = "$pin" ] || [ ! -f "$candidate_review" ] ||
               ! jq -e --arg upstream "$upstream" --arg candidate "$candidate" '
                 .candidate_commit == $candidate and
                 ([.changes[] | select(
                   .upstream == $upstream and
                   .kind == "deleted" and
                   .review_status == "complete" and
                   (.disposition | length) > 0 and
                   (.evidence | length) > 0 and
                   (.rust_evidence | length) > 0 and
                   (.upstream_evidence | length) > 0
                 )] | length) == 1
               ' "$candidate_review" >/dev/null; then
                printf '%s\t%s\t%s\n' "$rust" "$status" "$anchor" >> "$failures"
            fi
            ;;
        *) printf '%s\t%s\t%s\n' "$rust" "$status" "$anchor" >> "$failures" ;;
    esac
done

if [ -s "$failures" ]; then
    echo "stale or missing upstream anchors:" >&2
    sed -n '1,80p' "$failures" >&2
    exit 1
fi

echo "upstream anchors ok: $pin"
