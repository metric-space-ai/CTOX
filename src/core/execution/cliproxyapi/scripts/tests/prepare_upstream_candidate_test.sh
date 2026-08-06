#!/bin/bash
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
crate_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
repo_dir=$(CDPATH= cd -- "$crate_dir/../../../.." && pwd)
prepare="$script_dir/prepare_upstream_candidate.sh"
upstream_dir=${UPSTREAM_CHECKOUT:-"$repo_dir/runtime/cliproxyapi-upstream"}

scratch=$(mktemp -d "${TMPDIR:-/tmp}/cliproxyapi-prepare-test.XXXXXX")
trap 'rm -rf "$scratch"' EXIT

hash_file() {
    git hash-object "$1"
}

assert_unchanged() {
    file=$1
    expected=$2
    actual=$(hash_file "$file")
    if [ "$actual" != "$expected" ]; then
        printf 'expected %s to remain unchanged: %s != %s\n' "$file" "$actual" "$expected" >&2
        exit 1
    fi
}

expect_failure() {
    if "$@" >"$scratch/unexpected.stdout" 2>"$scratch/expected.stderr"; then
        printf 'expected command to fail: %s\n' "$*" >&2
        exit 1
    fi
    if ! grep -q 'candidate preparation refused' "$scratch/expected.stderr"; then
        printf 'missing fail-closed diagnostic for: %s\n' "$*" >&2
        cat "$scratch/expected.stderr" >&2
        exit 1
    fi
}

candidate=$(git -C "$upstream_dir" rev-parse 'HEAD^{commit}')
previous=$(git -C "$upstream_dir" rev-parse 'HEAD~1^{commit}')

# A matching review is resume state, not generated output. Preserve arbitrary
# operator progress and gate evidence byte-for-byte across reruns.
resume_dir="$scratch/resume"
UPSTREAM_CHECKOUT="$upstream_dir" "$prepare" "$candidate" "$resume_dir" >/dev/null
review="$resume_dir/upstream-review.json"
review_next="$scratch/review-next.json"
jq '.status = "in_progress" | .operator_note = "preserve me" | .gates.tracking = true' \
    "$review" > "$review_next"
mv "$review_next" "$review"
review_hash=$(hash_file "$review")
UPSTREAM_CHECKOUT="$upstream_dir" "$prepare" "$candidate" "$resume_dir" >/dev/null
assert_unchanged "$review" "$review_hash"

# Reusing an output directory for another immutable candidate must not mutate
# any of the first candidate's artifacts.
delta_hash=$(hash_file "$resume_dir/upstream-delta.json")
summary_hash=$(hash_file "$resume_dir/impact-summary.json")
expect_failure env UPSTREAM_CHECKOUT="$upstream_dir" "$prepare" "$previous" "$resume_dir"
assert_unchanged "$resume_dir/upstream-delta.json" "$delta_hash"
assert_unchanged "$review" "$review_hash"
assert_unchanged "$resume_dir/impact-summary.json" "$summary_hash"

# A same-candidate delta with altered content is also an identity mismatch.
delta_dir="$scratch/delta-mismatch"
UPSTREAM_CHECKOUT="$upstream_dir" "$prepare" "$candidate" "$delta_dir" >/dev/null
delta="$delta_dir/upstream-delta.json"
delta_next="$scratch/delta-next.json"
jq '.summary.changed_files += 1' "$delta" > "$delta_next"
mv "$delta_next" "$delta"
delta_hash=$(hash_file "$delta")
review_hash=$(hash_file "$delta_dir/upstream-review.json")
expect_failure env UPSTREAM_CHECKOUT="$upstream_dir" "$prepare" "$candidate" "$delta_dir"
assert_unchanged "$delta" "$delta_hash"
assert_unchanged "$delta_dir/upstream-review.json" "$review_hash"

# The impact summary is derived from the immutable delta. Treat edits as a
# mismatch rather than silently replacing the operator-visible review input.
summary_dir="$scratch/summary-mismatch"
UPSTREAM_CHECKOUT="$upstream_dir" "$prepare" "$candidate" "$summary_dir" >/dev/null
summary="$summary_dir/impact-summary.json"
summary_next="$scratch/summary-next.json"
jq '.summary.changed_files += 1' "$summary" > "$summary_next"
mv "$summary_next" "$summary"
summary_hash=$(hash_file "$summary")
review_hash=$(hash_file "$summary_dir/upstream-review.json")
expect_failure env UPSTREAM_CHECKOUT="$upstream_dir" "$prepare" "$candidate" "$summary_dir"
assert_unchanged "$summary" "$summary_hash"
assert_unchanged "$summary_dir/upstream-review.json" "$review_hash"

# A review whose candidate identity was edited must never be regenerated over.
review_dir="$scratch/review-mismatch"
UPSTREAM_CHECKOUT="$upstream_dir" "$prepare" "$candidate" "$review_dir" >/dev/null
review="$review_dir/upstream-review.json"
review_next="$scratch/review-mismatch-next.json"
jq '.candidate_commit = "0000000000000000000000000000000000000000"' "$review" > "$review_next"
mv "$review_next" "$review"
review_hash=$(hash_file "$review")
expect_failure env UPSTREAM_CHECKOUT="$upstream_dir" "$prepare" "$candidate" "$review_dir"
assert_unchanged "$review" "$review_hash"

printf 'prepare_upstream_candidate resume tests passed\n'
