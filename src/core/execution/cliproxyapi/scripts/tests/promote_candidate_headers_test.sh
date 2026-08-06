#!/bin/bash
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
crate_dir=$(CDPATH= cd -- "$script_dir/../.." && pwd)
fixture=$(mktemp -d)
trap 'rm -rf "$fixture"' EXIT

old=1111111111111111111111111111111111111111
candidate=2222222222222222222222222222222222222222
mkdir -p "$fixture/pkg"
cat > "$fixture/pkg/modified.rs" <<EOF
// ref: pkg/modified.go @ $old
// Port-Status: partial
// candidate-ref: pkg/modified.go @ $candidate
// Candidate-Port-Status: ported
// retained
EOF
cat > "$fixture/pkg/added.rs" <<EOF
// staging note
// candidate-ref: pkg/added.go @ $candidate
// Candidate-Port-Status: adapted_to_ctox
// retained
EOF
cat > "$fixture/pkg/deleted.rs" <<EOF
// ref: pkg/deleted.go @ $old
// Port-Status: ported
// candidate-ref: pkg/deleted.go deleted @ $candidate
// Candidate-Port-Status: adapted_to_ctox
EOF
cat > "$fixture/pkg/unchanged.rs" <<EOF
// ref: pkg/unchanged.go:4-9 @ $old
// Port-Status: ported
EOF
jq -n --arg old "$old" --arg candidate "$candidate" '
  {
    schema: "ctox.cliproxyapi.upstream-review.v3",
    base_commit: $old,
    candidate_commit: $candidate,
    status: "ready_for_promotion",
    changes: [
      {upstream:"pkg/modified.go",kind:"modified",review_status:"complete"},
      {upstream:"pkg/added.go",kind:"added",review_status:"complete"},
      {upstream:"pkg/deleted.go",kind:"deleted",review_status:"complete"}
    ],
    gates: {
      non_go_impact_review:true, dependency_audit:true,
      rust_no_default:true, rust_default:true, integrations:true,
      clippy_no_default:true, clippy_all_features:true,
      formatting:true, tracking:true, dashboard:true
    }
  }
' > "$fixture/review.json"

"$crate_dir/scripts/promote_candidate_headers.sh" \
    "$fixture" "$old" "$candidate" "$fixture/review.json"

test "$(sed -n '1p' "$fixture/pkg/modified.rs")" = "// ref: pkg/modified.go @ $candidate"
test "$(sed -n '2p' "$fixture/pkg/modified.rs")" = "// Port-Status: ported"
test "$(sed -n '1p' "$fixture/pkg/added.rs")" = "// ref: pkg/added.go @ $candidate"
test "$(sed -n '2p' "$fixture/pkg/added.rs")" = "// Port-Status: adapted_to_ctox"
grep -q '^// staging note$' "$fixture/pkg/added.rs"
test ! -e "$fixture/pkg/deleted.rs"
test "$(sed -n '1p' "$fixture/pkg/unchanged.rs")" = "// ref: pkg/unchanged.go:4-9 @ $candidate"
! grep -R -E '^// (candidate-ref|Candidate-Port-Status):' "$fixture/pkg"

# Invalid staged input fails before touching any mirror.
rollback_fixture="$fixture/rollback"
mkdir -p "$rollback_fixture/pkg"
cp "$fixture/review.json" "$rollback_fixture/review.json"
cat > "$rollback_fixture/pkg/good.rs" <<EOF
// ref: pkg/modified.go @ $old
// Port-Status: partial
// candidate-ref: pkg/modified.go @ $candidate
// Candidate-Port-Status: ported
EOF
cat > "$rollback_fixture/pkg/bad.rs" <<EOF
// ref: pkg/unchanged.go @ $old
// missing accepted status makes the staged transaction invalid
EOF
cp "$rollback_fixture/pkg/good.rs" "$rollback_fixture/good.before"
cp "$rollback_fixture/pkg/bad.rs" "$rollback_fixture/bad.before"
if "$crate_dir/scripts/promote_candidate_headers.sh" \
    "$rollback_fixture" "$old" "$candidate" "$rollback_fixture/review.json" >/dev/null 2>&1; then
    echo "malformed staged promotion unexpectedly succeeded" >&2
    exit 1
fi
cmp -s "$rollback_fixture/good.before" "$rollback_fixture/pkg/good.rs"
cmp -s "$rollback_fixture/bad.before" "$rollback_fixture/pkg/bad.rs"

echo "candidate header promotion ok"
