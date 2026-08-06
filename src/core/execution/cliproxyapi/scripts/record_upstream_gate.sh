#!/bin/bash
set -eu

delta=${1:?usage: record_upstream_gate.sh <delta.json> <review.json> <gate> -- <command> [args...]}
review=${2:?usage: record_upstream_gate.sh <delta.json> <review.json> <gate> -- <command> [args...]}
gate=${3:?usage: record_upstream_gate.sh <delta.json> <review.json> <gate> -- <command> [args...]}
shift 3
if [ "${1:-}" != "--" ]; then
    echo "expected -- before gate command" >&2
    exit 2
fi
shift
if [ "$#" -eq 0 ]; then
    echo "gate command is required" >&2
    exit 2
fi

jq -e --arg gate "$gate" '
  .schema == "ctox.cliproxyapi.upstream-review.v3" and
  (.gates | has($gate)) and
  (.gate_evidence | has($gate))
' "$review" >/dev/null || {
    echo "unknown gate or unsupported review schema: $gate" >&2
    exit 2
}

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project_state=${PROJECT_STATE_FILE:-$script_dir/../project-state.json}
candidate_commit=$(jq -r '.candidate_commit' "$review")
if [ -f "$project_state" ]; then
    state_candidate=$(jq -r '.upstream_candidate.commit' "$project_state")
    if [ "$state_candidate" != "$candidate_commit" ]; then
        echo "project state candidate does not match review candidate: $state_candidate != $candidate_commit" >&2
        exit 2
    fi
fi

review_dir=$(CDPATH= cd -- "$(dirname -- "$review")" && pwd)
log_name="gate-$gate.log"
log_path="$review_dir/$log_name"

set +e
"$@" > "$log_path" 2>&1
status=$?
set -e
cat "$log_path"
if [ "$status" -ne 0 ]; then
    echo "upstream gate failed: $gate (exit $status)" >&2
    exit "$status"
fi

if command -v sha256sum >/dev/null 2>&1; then
    output_hash=$(sha256sum "$log_path" | awk '{print $1}')
else
    output_hash=$(shasum -a 256 "$log_path" | awk '{print $1}')
fi
completed_at=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
command_json=$(jq -cn --args '$ARGS.positional' -- "$@")
temporary=$(mktemp "$review_dir/upstream-review.XXXXXX")

jq \
  --arg gate "$gate" \
  --arg completed_at "$completed_at" \
  --arg output_hash "$output_hash" \
  --arg log "$log_name" \
  --argjson command "$command_json" '
  .gates[$gate] = true |
  .gate_evidence[$gate] = {
    status: "passed",
    command: $command,
    completed_at: $completed_at,
    output_sha256: $output_hash,
    log: $log
  } |
  .status = (
    if
      ([.changes[] | select(
        .review_status != "complete" or
        (.disposition | length) == 0 or
        (.evidence | length) == 0 or
        (.upstream_evidence | length) == 0 or
        ((.source_kind == "go_production" or .source_kind == "go_test") and
          (.rust_evidence | length) == 0)
      )] | length) == 0 and
      ([.gates[] | select(. != true)] | length) == 0
    then "ready_for_promotion"
    else "in_progress"
    end
  )
' "$review" > "$temporary"
mv "$temporary" "$review"

# The review ledger is the authority for candidate progress. Synchronize its
# derived counters atomically so dashboard state cannot lag behind a recorded
# gate and accidentally present a stale completion percentage.
if [ -f "$project_state" ]; then
    reviewed=$(jq '[.changes[] | select(.review_status == "complete")] | length' "$review")
    total=$(jq '.changes | length' "$review")
    pending=$((total - reviewed))
    gates_passed=$(jq '[.gates[] | select(. == true)] | length' "$review")
    gates_total=$(jq '.gates | length' "$review")
    review_status=$(jq -r '.status' "$review")
    state_temporary=$(mktemp "$(dirname -- "$project_state")/project-state.XXXXXX")
    jq \
      --arg candidate "$candidate_commit" \
      --arg review_status "$review_status" \
      --argjson reviewed "$reviewed" \
      --argjson total "$total" \
      --argjson pending "$pending" \
      --argjson gates_passed "$gates_passed" \
      --argjson gates_total "$gates_total" '
      if .upstream_candidate.commit != $candidate then
        error("project state candidate changed while recording gate")
      else
        .upstream_candidate.inventory_total = $total |
        .upstream_candidate.reviewed = $reviewed |
        .upstream_candidate.pending = $pending |
        .upstream_candidate.gates_passed = $gates_passed |
        .upstream_candidate.gates_total = $gates_total |
        .upstream_candidate.status = (
          if $review_status == "ready_for_promotion" then "ready_for_promotion"
          elif $pending == 0 then "ready_for_gates"
          else "in_progress"
          end
        ) |
        (.upstream_candidate.promoted | if . then 1 else 0 end) as $promotion_unit |
        (.project_completion.post_promotion_full_gate | if . then 1 else 0 end) as $post_unit |
        ($total + $gates_total + 2) as $completion_total |
        ($reviewed + $gates_passed + $promotion_unit + $post_unit) as $completion_done |
        .work_items |= map(
          if .id == "upstream-candidate-promotion" then
            .detail = (
              "\($reviewed)/\($total) Candidate-Reviews vollständig, " +
              "\($pending) pending, \($gates_passed)/\($gates_total) Promotion-Gates, " +
              "Promotion " + (if $promotion_unit == 1 then "JA" else "NEIN" end) + ". " +
              "Gesamtport-Freigabefortschritt: \($completion_done)/\($completion_total) Abschlussaktionen. " +
              "Abschluss erst nach 10/10 Gates, expliziter Promotion und erneutem Accepted-Pin-Full-Gate."
            )
          else . end
        )
      end
    ' "$project_state" > "$state_temporary"
    mv "$state_temporary" "$project_state"
fi

echo "upstream gate recorded: $gate $output_hash"
