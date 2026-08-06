#!/bin/bash
set -eu

crate_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
repo_dir=$(CDPATH= cd -- "$crate_dir/../../../.." && pwd)
upstream_dir=${1:-"$repo_dir/runtime/cliproxyapi-upstream"}
state="$crate_dir/project-state.json"
accepted=$(jq -r '.upstream_commit' "$state")
candidate=$(jq -r '.upstream_candidate.commit' "$state")
head=$(git -C "$upstream_dir" rev-parse HEAD)

if [ "$head" != "$candidate" ]; then
    echo "candidate tracking requires checkout at candidate: $head != $candidate" >&2
    exit 1
fi
if [ -n "$(git -C "$upstream_dir" status --porcelain)" ]; then
    echo "candidate tracking requires a clean upstream checkout" >&2
    exit 1
fi

restore_candidate() {
    status=$?
    trap - EXIT HUP INT TERM
    current=$(git -C "$upstream_dir" rev-parse HEAD)
    if [ "$current" != "$candidate" ]; then
        git -C "$upstream_dir" -c advice.detachedHead=false checkout --detach "$candidate" >/dev/null
    fi
    exit "$status"
}
trap restore_candidate EXIT HUP INT TERM

git -C "$upstream_dir" -c advice.detachedHead=false checkout --detach "$accepted" >/dev/null
"$crate_dir/scripts/check_tracking.sh" "$upstream_dir"
git -C "$upstream_dir" -c advice.detachedHead=false checkout --detach "$candidate" >/dev/null
trap - EXIT HUP INT TERM

echo "candidate transition tracking ok: accepted=$accepted candidate=$candidate"
