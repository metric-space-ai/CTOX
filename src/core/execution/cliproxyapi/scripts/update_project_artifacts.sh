#!/bin/bash
set -eu

crate_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
repo_dir=$(CDPATH= cd -- "$crate_dir/../../../.." && pwd)
upstream_dir=${1:-"$repo_dir/runtime/cliproxyapi-upstream"}

"$crate_dir/scripts/build_port_map.sh" "$upstream_dir" "$crate_dir/port-map.json"
"$crate_dir/scripts/build_module_map.sh" "$crate_dir/port-map.json" "$crate_dir/module-map.json"
"$crate_dir/scripts/build_mirror_closure.sh" "$crate_dir/port-map.json" "$crate_dir/mirror-closure.json"
"$crate_dir/scripts/check_tracking.sh" "$upstream_dir"
"$crate_dir/scripts/build_dashboard.sh" "$repo_dir/runtime/cliproxyapi-porting-dashboard.html"
