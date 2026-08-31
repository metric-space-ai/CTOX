// SPDX-License-Identifier: MIT OR AGPL-3.0-only
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const root = new URL('../', import.meta.url);
const html = await readFile(new URL('index.html', root), 'utf8');
const css = await readFile(new URL('app.css', root), 'utf8');
const start = html.indexOf('<div class="topbar-status-bar">');
const end = html.indexOf('<div class="module-nav">', start);
assert.ok(start >= 0 && end > start, 'topbar status region must exist');
const region = html.slice(start, end);

assert.match(region, /data-shell-version-label(?:\s[^>]*)?>v—</);
assert.equal((region.match(/data-shell-release-status/g) || []).length, 1);
assert.doesNotMatch(region, /business-os-shell-v|RxDB\/WebRTC wird|source_commit|commit/i);
assert.ok(
  region.indexOf('data-ctox-shell-warning') > region.indexOf('data-shell-release-panel'),
  'runtime diagnostics must live inside the version status panel',
);
assert.ok(
  region.indexOf('data-recovery-warning') > region.indexOf('data-shell-release-panel'),
  'recovery diagnostics must live inside the version status panel',
);
assert.match(
  css,
  /\.topbar-status-bar \.brand-status\s*\{[^}]*display:\s*none;/s,
  'internal status prose must never consume header width',
);

console.log('ok - shell header is short-version-only with one status action');
