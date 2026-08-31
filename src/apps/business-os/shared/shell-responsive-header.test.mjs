import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import test from 'node:test';

const businessOsRoot = join(import.meta.dirname, '..');
const css = readFileSync(join(businessOsRoot, 'app.css'), 'utf8');
const index = readFileSync(join(businessOsRoot, 'index.html'), 'utf8');
const app = readFileSync(join(businessOsRoot, 'app.js'), 'utf8');

function mediaRules(source) {
  const rules = [];
  let cursor = 0;
  while ((cursor = source.indexOf('@media', cursor)) >= 0) {
    const open = source.indexOf('{', cursor);
    if (open < 0) break;
    let depth = 1;
    let end = open + 1;
    while (end < source.length && depth > 0) {
      if (source[end] === '{') depth += 1;
      if (source[end] === '}') depth -= 1;
      end += 1;
    }
    rules.push({ condition: source.slice(cursor + 6, open).trim(), body: source.slice(open + 1, end - 1) });
    cursor = end;
  }
  return rules;
}

const compactCss = mediaRules(css)
  .filter(({ condition }) => /max-width:\s*(?:600|720|900)px/.test(condition))
  .map(({ body }) => body)
  .join('\n');
const mobileCss = mediaRules(css)
  .filter(({ condition }) => /max-width:\s*767px/.test(condition))
  .map(({ body }) => body)
  .join('\n');

test('compact shell keeps start, app navigation, and account actions in one header row', () => {
  assert.match(compactCss, /grid-template-columns:\s*auto minmax\(0, 1fr\) auto/);
  assert.match(compactCss, /grid-template-rows:\s*40px/);
  assert.match(compactCss, /\.module-nav\s*\{[^}]*display:\s*grid[^}]*grid-column:\s*2[^}]*grid-row:\s*1/);
  assert.doesNotMatch(compactCss, /\.module-nav\s*\{[^}]*display:\s*none/);
  assert.doesNotMatch(compactCss, /grid-template-rows:\s*52px\s+42px/);
  assert.doesNotMatch(css, /grid-template-rows:\s*52px\s+42px/);
});

test('phone shell keeps open apps in the same row instead of growing a second header row', () => {
  assert.match(mobileCss, /--ctox-shell-topbar-h:\s*56px/);
  assert.match(mobileCss, /grid-template-rows:\s*48px/);
  assert.match(mobileCss, /\.module-nav\s*\{[^}]*grid-column:\s*2[^}]*grid-row:\s*1/);
  assert.match(mobileCss, /\.module-tabs\s*\{[^}]*flex-wrap:\s*nowrap[^}]*overflow-x:\s*auto/);
  assert.doesNotMatch(mobileCss, /109px|grid-template-rows:\s*52px\s+48px/);
});

test('one topbar token positions shell chrome at every responsive width', () => {
  assert.equal(css.match(/--shell-topbar-height\s*:/g)?.length, 1);
  assert.match(css, /\.app-shell\s*\{[^}]*grid-template-rows:\s*var\(--shell-topbar-height\)/);
  assert.match(css, /\.shell-start-menu-panel\s*\{[^}]*top:\s*var\(--shell-topbar-height\)/);
  assert.match(compactCss, /inset:\s*var\(--shell-topbar-height\) 0 0 !important/);
  assert.doesNotMatch(css, /\.app-shell\s*\{[^}]*grid-template-rows:\s*(?:48|50|52|54)px/);
});

test('compact warnings remain actionable through persistent shell notifications', () => {
  assert.match(compactCss, /\.topbar-status-bar\s*\{[^}]*display:\s*none/);
  assert.match(app, /function syncCompactShellWarningNotification\(/);
  assert.match(app, /time:\s*0/);
  assert.match(app, /action:\s*\{[\s\S]*callback:\s*visibleWarning\.action/);
  assert.match(app, /openSettingsDrawer\(\{ initialTab:\s*'runtime' \}\)/);
  assert.match(app, /exportBrowserRecoveryFromWarning\(\)/);
});

test('macOS tabs preserve the reachable leading edge and compact CTOX drops island geometry', () => {
  assert.match(css, /:root\[data-shell-style="macos"\] \.module-tabs\s*\{[^}]*justify-content:\s*safe center/);
  assert.match(compactCss, /:root\[data-shell-style="ctox"\] \.topbar\s*\{[^}]*margin:\s*0[^}]*border-radius:\s*0/);
});

test('responsive shell CSS is cache-bound to the current application build', () => {
  const appBuild = app
    .match(/const APP_BUILD = '([^']+)'/)?.[1];
  const cssBuild = index.match(/app\.css\?v=([^"']+)/)?.[1];
  assert.ok(appBuild);
  assert.equal(cssBuild, appBuild);
});
