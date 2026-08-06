#!/bin/bash
set -eu

crate_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
map_file=${1:-"$crate_dir/port-map.json"}
output=${2:-"$crate_dir/mirror-closure.json"}

jq '
  def wave:
    if (.upstream | startswith("internal/translator/")) then "01-translator-matrix"
    elif (.upstream | startswith("internal/runtime/executor/")) then "02-provider-runtime"
    elif (.upstream | test("^(internal|sdk)/api/")) then "03-api-control-plane"
    elif (.upstream | test("^(internal|sdk)/auth/")) then "04-auth-subscriptions"
    elif (.upstream | test("^sdk/(cliproxy|access|translator|proxyutil|pluginhost|pluginstore|logging|config)/")) then "05-sdk-runtime"
    elif (.upstream | test("^(internal/(pluginhost|pluginstore|watcher|wsrelay|client|home|store|redisqueue)/)")) then "06-lifecycle-platform"
    elif (.upstream | test("^(examples|cmd|internal/(cmd|tui))/")) then "08-examples-cli-tui"
    else "07-support-core"
    end;
  [.files[] | select(.status == "scaffold" and .test == false) | . + {wave: wave}] as $open |
  {
    schema: "ctox.cliproxyapi.mirror-closure.v1",
    upstream_commit: .upstream_commit,
    policy: "A file leaves this ledger only after a ported, partial, adapted_to_ctox or replaced_by_ctox status is evidenced and tracking gates pass.",
    production_total: .summary.production_go_files,
    production_classified: (.summary.production_go_files - .summary.production_scaffold_files),
    production_open: ($open | length),
    upstream_test_open: .summary.test_scaffold_files,
    waves: [
      $open | sort_by(.wave, .upstream) | group_by(.wave)[] |
      {id: .[0].wave, open: length, status: "pending", files: map(.upstream)}
    ]
  }
' "$map_file" > "$output"
