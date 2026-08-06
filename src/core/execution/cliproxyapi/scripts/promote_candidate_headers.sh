#!/bin/bash
set -euo pipefail

crate_dir=${1:?usage: promote_candidate_headers.sh <crate-dir> <old-ref> <candidate-ref> <review.json>}
old_ref=${2:?usage: promote_candidate_headers.sh <crate-dir> <old-ref> <candidate-ref> <review.json>}
candidate_ref=${3:?usage: promote_candidate_headers.sh <crate-dir> <old-ref> <candidate-ref> <review.json>}
review=${4:?usage: promote_candidate_headers.sh <crate-dir> <old-ref> <candidate-ref> <review.json>}

printf '%s\n%s\n' "$old_ref" "$candidate_ref" |
    grep -Eqv '^[0-9a-f]{40}$' && {
    echo "promotion refs must be distinct 40-character lowercase commit hashes" >&2
    exit 2
}
test "${#old_ref}" -eq 40 && test "${#candidate_ref}" -eq 40 &&
    test "$old_ref" != "$candidate_ref" || {
    echo "promotion refs must be distinct 40-character lowercase commit hashes" >&2
    exit 2
}

# Header promotion is a semantic transaction, not a global search/replace. The
# review is the authority for which candidate paths may be accepted or deleted.
jq -e --arg old "$old_ref" --arg candidate "$candidate_ref" '
  .schema == "ctox.cliproxyapi.upstream-review.v3" and
  .base_commit == $old and .candidate_commit == $candidate and
  .status == "ready_for_promotion" and
  (.changes | length) > 0 and
  ([.changes[] | select(.review_status != "complete")] | length) == 0 and
  (.gates | length) == 10 and
  ([.gates[] | select(. != true)] | length) == 0
' "$review" >/dev/null || {
    echo "candidate header promotion requires the complete matching review and all ten gates" >&2
    exit 1
}

transaction=$(mktemp -d "$crate_dir/.candidate-header-promotion.XXXXXX")
manifest="$transaction/manifest"
deleted="$transaction/deleted"
committing=false
committed=false

restore_on_failure() {
    status=$?
    if [ "$committing" = true ] && [ "$committed" != true ]; then
        while IFS= read -r file; do
            [ -n "$file" ] || continue
            relative=${file#"$crate_dir"/}
            mkdir -p "$(dirname -- "$file")"
            cp "$transaction/original/$relative" "$file"
        done < "$manifest"
    fi
    rm -rf "$transaction"
    exit "$status"
}
trap restore_on_failure EXIT HUP INT TERM
trap 'exit 1' HUP INT TERM

# Every file that can be changed is declared before staging. This same set is
# suitable for the outer promotion snapshot and prevents late marker discovery.
find "$crate_dir" -type f -name '*.rs' \
    ! -path "$transaction/*" -print0 |
while IFS= read -r -d '' file; do
    if grep -q -E '^// (candidate-ref|Candidate-Port-Status):' "$file"; then
        test "$(grep -c -E '^// candidate-ref:' "$file" || true)" -eq 1 &&
            test "$(grep -c -E "^// candidate-ref: .* @ $candidate_ref$" "$file" || true)" -eq 1 &&
            test "$(grep -c -E '^// Candidate-Port-Status:' "$file" || true)" -eq 1 &&
            test "$(grep -c -E '^// Candidate-Port-Status: (ported|adapted_to_ctox|replaced_by_ctox|partial)$' "$file" || true)" -eq 1 || {
            echo "foreign, orphaned, or malformed candidate marker: ${file#"$crate_dir"/}" >&2
            exit 1
        }
    fi
    if grep -q -E "^// ref: .* @ $old_ref$|^// candidate-ref: .* @ $candidate_ref$" "$file"; then
        printf '%s\n' "$file"
    fi
done | LC_ALL=C sort > "$manifest"

test -s "$manifest" || {
    echo "no Rust mirrors are eligible for candidate header promotion" >&2
    exit 1
}

while IFS= read -r file; do
    relative=${file#"$crate_dir"/}
    mkdir -p "$transaction/original/$(dirname -- "$relative")"
    mkdir -p "$transaction/staged/$(dirname -- "$relative")"
    cp "$file" "$transaction/original/$relative"
    cp "$file" "$transaction/staged/$relative"

    staged="$transaction/staged/$relative"
    candidate_anchor_count=$(grep -c -E "^// candidate-ref: .* @ $candidate_ref$" "$staged" || true)
    candidate_status_count=$(grep -c -E '^// Candidate-Port-Status: (ported|adapted_to_ctox|replaced_by_ctox|partial)$' "$staged" || true)

    if [ "$candidate_anchor_count" -eq 0 ]; then
        # Unchanged mirrors may intentionally consolidate multiple upstream
        # sources, but must have at least one accepted anchor and exactly one
        # status. Every old anchor advances to the candidate commit.
        accepted_anchor_count=$(grep -c -E "^// ref: .* @ $old_ref$" "$staged" || true)
        test "$accepted_anchor_count" -ge 1 &&
            test "$(grep -c -E '^// Port-Status: (ported|adapted_to_ctox|replaced_by_ctox|partial|supplemental)$' "$staged" || true)" -eq 1 || {
            echo "malformed unchanged accepted header: $relative" >&2
            exit 1
        }
        OLD_REF="$old_ref" CANDIDATE_REF="$candidate_ref" perl -pi -e \
            's/\Q@ $ENV{OLD_REF}\E$/@ $ENV{CANDIDATE_REF}/ if m{^// ref:}' "$staged"
        continue
    fi

    test "$candidate_anchor_count" -eq 1 && test "$candidate_status_count" -eq 1 || {
        echo "candidate header pair is missing, duplicated, or invalid: $relative" >&2
        exit 1
    }
    marker=$(grep -m1 -E "^// candidate-ref: .* @ $candidate_ref$" "$staged")
    upstream=${marker#// candidate-ref: }
    upstream=${upstream% @ "$candidate_ref"}
    deleted_marker=false
    case "$upstream" in
        *" deleted")
            deleted_marker=true
            upstream=${upstream% deleted}
            ;;
    esac

    jq -e --arg upstream "$upstream" --argjson deleted "$deleted_marker" '
      ([.changes[] | select(.upstream == $upstream)] | length) == 1 and
      any(.changes[];
        .upstream == $upstream and .review_status == "complete" and
        (if $deleted then .kind == "deleted" else .kind != "deleted" end))
    ' "$review" >/dev/null || {
        echo "candidate marker lacks one matching completed review disposition: $upstream" >&2
        exit 1
    }

    if [ "$deleted_marker" = true ]; then
        printf '%s\n' "$file" >> "$deleted"
        continue
    fi

    CANDIDATE_REF="$candidate_ref" OLD_REF="$old_ref" perl -0777 -pi -e '
      my @lines = split(/(?<=\n)/, $_, -1);
      my ($candidate_anchor, $candidate_status);
      my @kept;
      for my $line (@lines) {
        if ($line =~ m{^// candidate-ref: .* \@ \Q$ENV{CANDIDATE_REF}\E\r?\n?$}) {
          die "duplicate candidate-ref in $ARGV\n" if defined $candidate_anchor;
          ($candidate_anchor = $line) =~ s{^// candidate-ref:}{// ref:};
          next;
        }
        if ($line =~ m{^// Candidate-Port-Status: .*\r?\n?$}) {
          die "duplicate Candidate-Port-Status in $ARGV\n" if defined $candidate_status;
          ($candidate_status = $line) =~ s{^// Candidate-Port-Status:}{// Port-Status:};
          next;
        }
        next if $line =~ m{^// ref: .* \@ \Q$ENV{OLD_REF}\E\r?\n?$};
        next if $line =~ m{^// Port-Status: .*\r?\n?$};
        push @kept, $line;
      }
      die "missing candidate-ref in $ARGV\n" unless defined $candidate_anchor;
      die "missing Candidate-Port-Status in $ARGV\n" unless defined $candidate_status;
      $_ = $candidate_anchor . $candidate_status . join("", @kept);
    ' "$staged"
done < "$manifest"

# Validate the complete staged result before the first live-tree mutation.
while IFS= read -r file; do
    relative=${file#"$crate_dir"/}
    staged="$transaction/staged/$relative"
    grep -Fxq "$file" "$deleted" 2>/dev/null && continue
    test "$(grep -c -E "^// ref: .* @ $candidate_ref$" "$staged" || true)" -ge 1 &&
        test "$(grep -c -E '^// Port-Status: (ported|adapted_to_ctox|replaced_by_ctox|partial|supplemental)$' "$staged" || true)" -eq 1 &&
        ! grep -q -E '^// (candidate-ref|Candidate-Port-Status):' "$staged" || {
        echo "staged accepted header validation failed: $relative" >&2
        exit 1
    }
done < "$manifest"

committing=true
while IFS= read -r file; do
    relative=${file#"$crate_dir"/}
    cmp -s "$file" "$transaction/original/$relative" || {
        echo "Rust mirror changed concurrently during promotion: $relative" >&2
        exit 1
    }
    if grep -Fxq "$file" "$deleted" 2>/dev/null; then
        rm -f "$file"
    else
        mv "$transaction/staged/$relative" "$file"
    fi
done < "$manifest"

committed=true
promoted_count=$(wc -l < "$manifest" | tr -d ' ')
rm -rf "$transaction"
trap - EXIT HUP INT TERM
echo "candidate headers promoted: $promoted_count mirrors"
