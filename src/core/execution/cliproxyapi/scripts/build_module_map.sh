#!/bin/bash
set -eu

crate_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
port_map=${1:-"$crate_dir/port-map.json"}
output=${2:-"$crate_dir/module-map.json"}

jq '
  def module_id:
    (.upstream | split("/")) as $p |
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
    elif $p[0] == "examples" then $p[0:2] | join("/")
    elif $p[0] == "cmd" then $p[0:2] | join("/")
    else $p[0]
    end;
  def open: .status == "scaffold" or .status == "partial" or .status == "missing";
  . as $port |
  ($port.files | group_by(module_id) | map(
    . as $files |
    ($files[0] | module_id) as $id |
    {
      id: $id,
      production_total: ($files | map(select(.test == false)) | length),
      production_open: ($files | map(select(.test == false and open)) | length),
      production_scaffolds: ($files | map(select(.test == false and .status == "scaffold")) | length),
      production_partial: ($files | map(select(.test == false and .status == "partial")) | length),
      test_total: ($files | map(select(.test == true)) | length),
      test_open: ($files | map(select(.test == true and open)) | length),
      files: ($files | map(.upstream)),
      status: (
        if ($files | map(select(open)) | length) == 0 then "closed_live"
        elif ($files | map(select(open)) | length) == ($files | length) then "queued"
        else "in_progress"
        end
      )
    }
  )) as $modules |
  {
    schema: "ctox.cliproxyapi.module-map.v1",
    upstream_commit: $port.upstream_commit,
    policy: "A module is live-closed only when no production or test mirror is scaffold, partial, or missing. Common-gate verification is tracked separately.",
    summary: {
      modules: ($modules | length),
      closed_live: ($modules | map(select(.status == "closed_live")) | length),
      in_progress: ($modules | map(select(.status == "in_progress")) | length),
      queued: ($modules | map(select(.status == "queued")) | length)
    },
    modules: ($modules | sort_by([-.production_open, -.test_open, .id]))
  }
' "$port_map" > "$output"
