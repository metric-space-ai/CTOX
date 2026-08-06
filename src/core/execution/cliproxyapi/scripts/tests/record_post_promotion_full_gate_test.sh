#!/bin/bash
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
crate_dir=$(CDPATH= cd -- "$script_dir/../.." && pwd)
repo_dir=$(CDPATH= cd -- "$crate_dir/../../../.." && pwd)
pin=$(jq -r .base_commit "$crate_dir/upstream-lock.json")
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT

receipt_relative=$(jq -r .umbrella_receipt.path "$crate_dir/strict-credit-audit.json")
receipt=$(CDPATH= cd -- "$crate_dir/$(dirname -- "$receipt_relative")" && pwd -P)/$(basename -- "$receipt_relative")
strict_root="$repo_dir/runtime/cliproxyapi-strict-receipts/$pin"
started_at=$(jq -r .started_at "$receipt")

jq --arg pin "$pin" '
  .upstream_commit = $pin |
  .upstream_candidate.commit = $pin |
  .upstream_candidate.reviewed = .upstream_candidate.inventory_total |
  .upstream_candidate.pending = 0 |
  .upstream_candidate.gates_total = 10 |
  .upstream_candidate.gates_passed = 10 |
  .upstream_candidate.promoted = true |
  .upstream_candidate.status = "promoted" |
  .project_completion.complete = false |
  .project_completion.accepted_pin_complete = false |
  .project_completion.candidate_promoted = true |
  .project_completion.post_promotion_full_gate = false
' "$crate_dir/project-state.json" > "$scratch/state.json"
cp "$crate_dir/strict-credit-audit.json" "$scratch/audit.json"
cp "$crate_dir/port-map.json" "$scratch/map.json"
cp "$scratch/state.json" "$scratch/state.before.json"
cp "$scratch/audit.json" "$scratch/audit.before.json"

previous=0000000000000000000000000000000000000000
jq -n --arg previous "$previous" --arg pin "$pin" --arg promoted_at "2026-01-01T00:00:00Z" '
  def gates: {
    non_go_impact_review:true, dependency_audit:true,
    rust_no_default:true, rust_default:true, integrations:true,
    clippy_no_default:true, clippy_all_features:true,
    formatting:true, tracking:true, dashboard:true
  };
  def evidence: (gates | with_entries(.value={status:"passed",command:["true"],completed_at:"2026-01-01T00:00:00Z",output_sha256:("a"*64),log:("gate-"+.key+".log")}));
  {
    schema:"ctox.cliproxyapi.upstream-promotion-receipt.v1", repository:"fixture",
    previous_commit:$previous, accepted_commit:$pin, promoted_at:$promoted_at,
    delta_sha256:("b"*64), review_sha256:("c"*64),
    delta:{schema:"ctox.cliproxyapi.upstream-delta.v2",base_commit:$previous,candidate_commit:$pin,
      changes:[{upstream:"fixture.go",module:"fixture",kind:"modified",source_kind:"go_production",required_action:"port"}]},
    review:{schema:"ctox.cliproxyapi.upstream-review.v3",base_commit:$previous,candidate_commit:$pin,status:"ready_for_promotion",
      changes:[{upstream:"fixture.go",module:"fixture",kind:"modified",source_kind:"go_production",required_action:"port",
        review_status:"complete",disposition:"ported",evidence:"fixture",upstream_evidence:"fixture",rust_evidence:"fixture"}],
      gates:gates,gate_evidence:evidence}
  }
' > "$scratch/promotion.json"

cat > "$scratch/validate.sh" <<'EOF'
#!/bin/bash
set -eu
state=$1
audit=$2
map=$3
dashboard=$4
jq -e '.project_completion.complete and .project_completion.accepted_pin_complete and .project_completion.post_promotion_full_gate' "$state" >/dev/null
jq -e --slurpfile map "$map" '.upstream_commit == $map[0].upstream_commit and .gap.production == 0 and .gap.tests == 0' "$audit" >/dev/null
printf 'validated\n' > "$dashboard"
EOF
chmod +x "$scratch/validate.sh"

PROJECT_STATE_FILE="$scratch/state.json" \
STRICT_AUDIT_FILE="$scratch/audit.json" \
PORT_MAP_FILE="$scratch/map.json" \
DASHBOARD_OUTPUT="$scratch/dashboard.html" \
PROMOTION_RECEIPT="$scratch/promotion.json" \
STRICT_RECEIPT_ROOT="$strict_root" \
POST_PROMOTION_ARTIFACT_VALIDATOR="$scratch/validate.sh" \
  "$crate_dir/scripts/record_post_promotion_full_gate.sh" "$receipt" >/dev/null

jq -e '
  .project_completion.complete == true and
  .project_completion.post_promotion_full_gate == true and
  any(.work_items[]; .id == "upstream-candidate-promotion" and .status == "complete")
' "$scratch/state.json" >/dev/null
test -s "$scratch/dashboard.html"

# Replaying an already-recorded receipt is fail-closed.
if PROJECT_STATE_FILE="$scratch/state.json" STRICT_AUDIT_FILE="$scratch/audit.json" \
   PORT_MAP_FILE="$scratch/map.json" PROMOTION_RECEIPT="$scratch/promotion.json" \
   STRICT_RECEIPT_ROOT="$strict_root" POST_PROMOTION_ARTIFACT_VALIDATOR="$scratch/validate.sh" \
   "$crate_dir/scripts/record_post_promotion_full_gate.sh" "$receipt" >/dev/null 2>&1; then
    echo "post-promotion full gate replay unexpectedly succeeded" >&2
    exit 1
fi

# An artifact-validation failure restores both ledgers and an existing dashboard.
cp "$scratch/state.before.json" "$scratch/state.json"
cp "$scratch/audit.before.json" "$scratch/audit.json"
printf 'before\n' > "$scratch/dashboard.html"
cat > "$scratch/fail.sh" <<'EOF'
#!/bin/bash
exit 23
EOF
chmod +x "$scratch/fail.sh"
if PROJECT_STATE_FILE="$scratch/state.json" STRICT_AUDIT_FILE="$scratch/audit.json" \
   PORT_MAP_FILE="$scratch/map.json" DASHBOARD_OUTPUT="$scratch/dashboard.html" \
   PROMOTION_RECEIPT="$scratch/promotion.json" STRICT_RECEIPT_ROOT="$strict_root" \
   POST_PROMOTION_ARTIFACT_VALIDATOR="$scratch/fail.sh" \
   "$crate_dir/scripts/record_post_promotion_full_gate.sh" "$receipt" >/dev/null 2>&1; then
    echo "failing post-promotion artifact validation unexpectedly succeeded" >&2
    exit 1
fi
cmp -s "$scratch/state.before.json" "$scratch/state.json"
cmp -s "$scratch/audit.before.json" "$scratch/audit.json"
test "$(sed -n '1p' "$scratch/dashboard.html")" = before

printf 'post-promotion full gate tests passed (%s)\n' "$started_at"
