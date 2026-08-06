#!/bin/bash
set -eu

lock=${1:?usage: write_upstream_receipt.sh <lock.json> <delta.json> <review.json> <previous-commit> <accepted-commit> <receipt.json>}
delta=${2:?usage: write_upstream_receipt.sh <lock.json> <delta.json> <review.json> <previous-commit> <accepted-commit> <receipt.json>}
review=${3:?usage: write_upstream_receipt.sh <lock.json> <delta.json> <review.json> <previous-commit> <accepted-commit> <receipt.json>}
previous_commit=${4:?usage: write_upstream_receipt.sh <lock.json> <delta.json> <review.json> <previous-commit> <accepted-commit> <receipt.json>}
accepted_commit=${5:?usage: write_upstream_receipt.sh <lock.json> <delta.json> <review.json> <previous-commit> <accepted-commit> <receipt.json>}
receipt=${6:?usage: write_upstream_receipt.sh <lock.json> <delta.json> <review.json> <previous-commit> <accepted-commit> <receipt.json>}
crate_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

if [ -e "$receipt" ]; then
    echo "upstream promotion receipt already exists: $receipt" >&2
    exit 1
fi

jq -e \
  --arg previous_commit "$previous_commit" \
  --arg accepted_commit "$accepted_commit" '
  .base_commit == $previous_commit and
  .candidate_commit == $accepted_commit
' "$delta" >/dev/null || {
    echo "receipt delta identity does not match promotion" >&2
    exit 1
}
jq -e \
  --arg previous_commit "$previous_commit" \
  --arg accepted_commit "$accepted_commit" '
  .base_commit == $previous_commit and
  .candidate_commit == $accepted_commit and
  .status == "ready_for_promotion"
' "$review" >/dev/null || {
    echo "receipt review identity does not match promotion" >&2
    exit 1
}

if command -v sha256sum >/dev/null 2>&1; then
    delta_hash=$(sha256sum "$delta" | awk '{print $1}')
    review_hash=$(sha256sum "$review" | awk '{print $1}')
else
    delta_hash=$(shasum -a 256 "$delta" | awk '{print $1}')
    review_hash=$(shasum -a 256 "$review" | awk '{print $1}')
fi
promoted_at=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
mkdir -p "$(dirname "$receipt")"
temporary=$(mktemp "$(dirname "$receipt")/promotion-receipt.XXXXXX")
trap 'rm -f "$temporary"' EXIT HUP INT TERM

jq -n \
    --arg repository "$(jq -r .repository "$lock")" \
    --arg previous_commit "$previous_commit" \
    --arg accepted_commit "$accepted_commit" \
    --arg promoted_at "$promoted_at" \
    --arg delta_hash "$delta_hash" \
    --arg review_hash "$review_hash" \
    --slurpfile delta_doc "$delta" \
    --slurpfile review_doc "$review" '
    {
      schema: "ctox.cliproxyapi.upstream-promotion-receipt.v1",
      repository: $repository,
      previous_commit: $previous_commit,
      accepted_commit: $accepted_commit,
      promoted_at: $promoted_at,
      delta_sha256: $delta_hash,
      review_sha256: $review_hash,
      delta: $delta_doc[0],
      review: $review_doc[0]
    }
' > "$temporary"
"$crate_dir/scripts/check_upstream_receipt.sh" "$temporary" >/dev/null
ln "$temporary" "$receipt" || {
    echo "upstream promotion receipt appeared concurrently: $receipt" >&2
    exit 1
}
rm -f "$temporary"
trap - EXIT HUP INT TERM
"$crate_dir/scripts/check_upstream_receipt.sh" "$receipt"
