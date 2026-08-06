#!/bin/bash
set -eu

crate_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
repo_dir=$(CDPATH= cd -- "$crate_dir/../../../.." && pwd)
lock_file=${UPSTREAM_LOCK_FILE:-"$crate_dir/upstream-lock.json"}
port_map=${UPSTREAM_PORT_MAP:-"$crate_dir/port-map.json"}
upstream_dir=${UPSTREAM_CHECKOUT:-"$repo_dir/runtime/cliproxyapi-upstream"}
candidate_ref=${1:-HEAD}
output=${2:-"$crate_dir/upstream-delta.json"}

base=$(jq -r '.base_commit' "$lock_file")
candidate=$(git -C "$upstream_dir" rev-parse "$candidate_ref^{commit}")
git -C "$upstream_dir" cat-file -e "$base^{commit}"

raw=$(mktemp)
rows=$(mktemp)
trap 'rm -f "$raw" "$rows"' EXIT

git -C "$upstream_dir" diff --name-status -M "$base" "$candidate" -- . > "$raw"

while IFS=$'\t' read -r code first second; do
    [ -n "$code" ] || continue
    case "$code" in
        R*) old_path=$first; path=$second; kind=renamed ;;
        A) old_path=""; path=$first; kind=added ;;
        D) old_path=$first; path=$first; kind=deleted ;;
        M) old_path=$first; path=$first; kind=modified ;;
        *) old_path=$first; path=${second:-$first}; kind=changed ;;
    esac
    lookup=$path
    [ "$kind" != "deleted" ] || lookup=$old_path
    case "$lookup" in
        *_test.go) source_kind=go_test; is_test=true ;;
        *.go) source_kind=go_production; is_test=false ;;
        go.mod|*/go.mod|go.sum|*/go.sum|go.work|*/go.work|go.work.sum|*/go.work.sum) source_kind=dependency_manifest; is_test=false ;;
        .github/*|Dockerfile*|*/Dockerfile*|Makefile|*/Makefile|*.mk|*.sh|.goreleaser*|*/.goreleaser*) source_kind=build_release; is_test=false ;;
        LICENSE*|*/LICENSE*|NOTICE*|*/NOTICE*|README*|*/README*|docs/*|*.md) source_kind=documentation_license; is_test=false ;;
        *.yaml|*.yml|*.json|*.toml|*.html|*.tmpl|*.tpl) source_kind=runtime_asset; is_test=false ;;
        *) source_kind=other; is_test=false ;;
    esac
    if [ "$source_kind" = go_production ] || [ "$source_kind" = go_test ]; then
        rust=$(jq -r --arg path "$lookup" '.files[] | select(.upstream == $path) | .rust' "$port_map" | head -1)
        status=$(jq -r --arg path "$lookup" '.files[] | select(.upstream == $path) | .status' "$port_map" | head -1)
        [ -n "$rust" ] || rust=${path%.go}.rs
        [ -n "$status" ] || status=unmapped
        case "$kind" in
            added) action=port_new_file ;;
            modified) action=revalidate_and_port_delta ;;
            deleted) action=remove_or_document_replacement ;;
            renamed) action=remap_and_revalidate ;;
            *) action=manual_review ;;
        esac
    else
        rust=""
        status=not_applicable
        case "$source_kind" in
            dependency_manifest) action=revalidate_dependencies_and_full_build ;;
            build_release) action=review_build_release_impact ;;
            documentation_license) action=review_documentation_license_impact ;;
            runtime_asset) action=review_runtime_asset_impact ;;
            *) action=review_repository_impact ;;
        esac
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$kind" "$code" "$old_path" "$path" "$rust" "$status" "$is_test" "$action" "$source_kind" >> "$rows"
done < "$raw"

jq -Rn \
    --arg base "$base" \
    --arg candidate "$candidate" \
    --arg repository "$(jq -r '.repository' "$lock_file")" '
  def module_id($path):
    ($path | split("/")) as $p |
    ($p[-1]) as $file |
    if $p[0] == "internal" and $p[1] == "runtime" and $p[2] == "executor" then
      if $p[3] == "helps" then $p[0:4] | join("/")
      elif ($file | test("^(aistudio_|gemini_)")) then "internal/runtime/executor/gemini"
      elif ($file | test("^antigravity_")) then "internal/runtime/executor/antigravity"
      elif ($file | test("^claude_")) then "internal/runtime/executor/claude"
      elif ($file | test("^(codex_|home_codex_)")) then "internal/runtime/executor/codex"
      elif ($file | test("^kimi_")) then "internal/runtime/executor/kimi"
      elif ($file | test("^openai_")) then "internal/runtime/executor/openai"
      elif ($file | test("^xai_")) then "internal/runtime/executor/xai"
      else "internal/runtime/executor/core"
      end
    elif $p[0] == "internal" and $p[1] == "runtime" then $p[0:3] | join("/")
    elif $p[0] == "internal" and $p[1] == "auth" then
      if ($p | length) > 3 then $p[0:3] | join("/") else "internal/auth/core" end
    elif $p[0] == "internal" and $p[1] == "translator" then $p[0:3] | join("/")
    elif $p[0] == "internal" then $p[0:2] | join("/")
    elif $p[0] == "sdk" and $p[1] == "auth" then
      if ($file | test("^(antigravity|claude|codex|kimi|xai)(_|\\.)"))
      then "sdk/auth/" + ($file | capture("^(?<provider>antigravity|claude|codex|kimi|xai)").provider)
      else "sdk/auth/core"
      end
    elif $p[0] == "sdk" and ($p[1] == "cliproxy" or $p[1] == "api") then $p[0:3] | join("/")
    elif $p[0] == "sdk" then $p[0:2] | join("/")
    elif $p[0] == "examples" or $p[0] == "cmd" then $p[0:2] | join("/")
    elif $path == "go.mod" or $path == "go.sum" or $path == "go.work" or $path == "go.work.sum"
      then "repository/dependencies"
    elif $p[0] == ".github" or ($file | test("^(Dockerfile|Makefile|\\.goreleaser)"))
      then "repository/build-release"
    elif $p[0] == "docs" or ($file | test("^(LICENSE|NOTICE|README)")) or ($file | endswith(".md"))
      then "repository/documentation-license"
    else "repository/" + $p[0]
    end;
  [inputs | split("\t") | {
    kind: .[0], git_status: .[1],
    old_upstream: (if (.[2] | length) > 0 then .[2] else null end),
    upstream: .[3], rust: .[4], current_port_status: .[5],
    test: (.[6] == "true"), required_action: .[7], source_kind: .[8],
    module: module_id(.[3])
  }] as $changes |
  {
    schema: "ctox.cliproxyapi.upstream-delta.v2",
    repository: $repository,
    base_commit: $base,
    candidate_commit: $candidate,
    clean: ($changes | length == 0),
    summary: {
      changed_files: ($changes | length),
      changed_go_files: ($changes | map(select(.source_kind == "go_production" or .source_kind == "go_test")) | length),
      production: ($changes | map(select(.source_kind == "go_production")) | length),
      tests: ($changes | map(select(.source_kind == "go_test")) | length),
      dependency_manifests: ($changes | map(select(.source_kind == "dependency_manifest")) | length),
      build_release: ($changes | map(select(.source_kind == "build_release")) | length),
      documentation_license: ($changes | map(select(.source_kind == "documentation_license")) | length),
      runtime_assets: ($changes | map(select(.source_kind == "runtime_asset")) | length),
      other: ($changes | map(select(.source_kind == "other")) | length),
      added: ($changes | map(select(.kind == "added")) | length),
      modified: ($changes | map(select(.kind == "modified")) | length),
      deleted: ($changes | map(select(.kind == "deleted")) | length),
      renamed: ($changes | map(select(.kind == "renamed")) | length),
      impacted_modules: ($changes | map(.module) | unique)
    },
    changes: $changes
  }' < "$rows" > "$output"

raw_count=$(wc -l < "$raw" | tr -d ' ')
row_count=$(wc -l < "$rows" | tr -d ' ')
json_count=$(jq -r '.summary.changed_files' "$output")
if [ "$raw_count" -ne "$row_count" ] || [ "$row_count" -ne "$json_count" ]; then
    echo "upstream delta inventory mismatch: raw=$raw_count rows=$row_count json=$json_count" >&2
    exit 1
fi

echo "$output"
