#!/bin/bash
set -euo pipefail
stage=/home/ctox/.cache/ctox/file-preservation-fbeca32b7
exec >"$stage/native-shell-root-browser.log" 2>&1
cd "$stage/source"
export CTOX_SMOKE_ROOT="$stage/browser-timing-root-final-20260906"
export CTOX_SMOKE_KEEP_ARTIFACTS=1
export CTOX_BIN=/home/ctox/.cache/ctox/file-preservation-fbeca32b7/release-target/release/ctox
export PLAYWRIGHT_MODULE_PATH="$stage/source/src/apps/business-os/node_modules/playwright"
export PLAYWRIGHT_CHROMIUM_EXECUTABLE="$stage/playwright-browsers/chromium_headless_shell-1223/chrome-headless-shell-linux64/chrome-headless-shell"
export SMOKE_MODE=command-roundtrip-timing-browser-to-rust
export SMOKE_PAGE_PATH=/index.html
export BUSINESS_PORT=28877
export SIGNALING_PORT=28876
export SMOKE_PROCESS_LIFECYCLE_PATH="$stage/browser-timing-root-final-lifecycle.json"
test ! -e "$CTOX_SMOKE_ROOT"
test -x "$CTOX_BIN"
sha256sum --check <<'BINARY'
580887fe5133089d77dc6dd1a44c0e865e539c9a1d22285738ce6ee8d602790c  /home/ctox/.cache/ctox/file-preservation-fbeca32b7/release-target/release/ctox
BINARY
mkdir -p "$CTOX_SMOKE_ROOT/runtime"
ln -s "$stage/source/Cargo.toml" "$CTOX_SMOKE_ROOT/Cargo.toml"
ln -s "$stage/source/contracts" "$CTOX_SMOKE_ROOT/contracts"
mkdir -p "$CTOX_SMOKE_ROOT/src"
ln -s "$stage/source/src/core" "$CTOX_SMOKE_ROOT/src/core"
export CTOX_ROOT="$CTOX_SMOKE_ROOT"
test -f "$CTOX_ROOT/src/core/main.rs"
test -f "$CTOX_ROOT/contracts/history/creation-ledger.md"
"$CTOX_BIN" business-os shell-update stage --version 0.1.46-beta.13
"$CTOX_BIN" business-os shell-update activate
python3 - <<'PY'
import os,sqlite3,json,pathlib
r=pathlib.Path(os.environ['CTOX_ROOT']);p=r/'runtime/business-os.sqlite3'
c=sqlite3.connect('file:'+str(p)+'?mode=ro',uri=True)
d=json.loads(c.execute('select state_json from business_os_shell_update_state where singleton=1').fetchone()[0])
assert d['currentSlot']=='0.1.46-beta.13',d
assert (r/'runtime/business-os-shell/slots/0.1.46-beta.13/.ctox-shell-release.v2.json').is_file()
print('verified_fixture_root='+str(r)+' active_slot='+d['currentSlot'])
PY
python3 - <<'PY'
import socket
for port in (28877,28876):
    s=socket.socket();s.bind(('127.0.0.1',port));s.close()
PY
exec node src/core/rxdb/tools/browser_rust_smoke.js
