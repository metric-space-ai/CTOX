#!/bin/bash
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
source_crate=$(CDPATH= cd -- "$script_dir/../.." && pwd)
fixture=$(mktemp -d)
trap 'rm -rf "$fixture"' EXIT
crate="$fixture/crate"
upstream="$fixture/upstream"
mkdir -p "$crate/scripts" "$upstream"
cp "$source_crate/scripts/check_tracking.sh" "$crate/scripts/check_tracking.sh"

git -C "$upstream" init -q
git -C "$upstream" -c user.name=fixture -c user.email=fixture@example.invalid \
  commit --allow-empty -q -m fixture
pin=$(git -C "$upstream" rev-parse HEAD)

cat > "$crate/UPSTREAM.md" <<EOF
- Pinned commit: \`$pin\`
EOF
cat > "$crate/port-map.json" <<EOF
{"upstream_commit":"$pin","files":[],"summary":{}}
EOF
printf '{"upstream_commit":"%s"}\n' "$pin" > "$crate/module-map.json"
printf '{"upstream_commit":"%s"}\n' "$pin" > "$crate/mirror-closure.json"
printf '{"base_commit":"%s"}\n' "$pin" > "$crate/upstream-lock.json"
printf '{"base_commit":"%s","candidate_commit":"%s"}\n' "$pin" "$pin" > "$crate/upstream-delta.json"
printf '{}\n' > "$crate/strict-credit-audit.json"

write_state() {
    promoted=$1
    candidate_promoted=$2
    accepted=$3
    post=$4
    complete=$5
    jq -n --arg pin "$pin" --argjson promoted "$promoted" \
      --argjson candidate_promoted "$candidate_promoted" --argjson accepted "$accepted" \
      --argjson post "$post" --argjson complete "$complete" \
      '{upstream_commit:$pin,upstream_candidate:{promoted:$promoted},project_completion:{candidate_promoted:$candidate_promoted,accepted_pin_complete:$accepted,post_promotion_full_gate:$post,complete:$complete}}' \
      > "$crate/project-state.json"
}

cat > "$crate/scripts/check_upstream_anchors.sh" <<'EOF'
#!/bin/bash
exit 0
EOF
cat > "$crate/scripts/check_strict_credit_audit.sh" <<EOF
#!/bin/bash
printf 'called\n' >> "$fixture/audit-called"
exit 42
EOF
for generator in build_port_map build_mirror_closure build_module_map; do
cat > "$crate/scripts/$generator.sh" <<EOF
#!/bin/bash
case "$generator" in
  build_port_map) cp "$crate/port-map.json" "\${2}" ;;
  build_mirror_closure) cp "$crate/mirror-closure.json" "\${2}" ;;
  build_module_map) cp "$crate/module-map.json" "\${2}" ;;
esac
EOF
done
chmod +x "$crate/scripts/"*.sh

# Only the exact promoted/post-gate-pending state may use the historical audit.
write_state true true false false false
"$crate/scripts/check_tracking.sh" "$upstream" >/dev/null
test ! -e "$fixture/audit-called"

expect_audit_fail_closed() {
    label=$1
    shift
    write_state "$@"
    if "$crate/scripts/check_tracking.sh" "$upstream" >/dev/null 2>&1; then
        echo "tracking accepted a non-transitional state: $label" >&2
        exit 1
    fi
    test -s "$fixture/audit-called"
    : > "$fixture/audit-called"
}
expect_audit_fail_closed 'promoted=false' false true false false false
expect_audit_fail_closed 'accepted=true' true true true false false
expect_audit_fail_closed 'post=true' true true false true false
expect_audit_fail_closed 'candidate_promoted=false' true false false false false
expect_audit_fail_closed 'complete=true' true true false false true

# The exception does not weaken independent pin identity checks.
write_state true true false false false
jq '.upstream_commit = ("f" * 40)' "$crate/port-map.json" > "$crate/port-map.next.json"
mv "$crate/port-map.next.json" "$crate/port-map.json"
if "$crate/scripts/check_tracking.sh" "$upstream" >/dev/null 2>&1; then
    echo "tracking transition bypassed an independent pin mismatch" >&2
    exit 1
fi

echo "tracking promotion-transition tests passed"
