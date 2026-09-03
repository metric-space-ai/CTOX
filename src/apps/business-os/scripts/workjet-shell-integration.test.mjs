import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const shellDir = new URL('../', import.meta.url);

const read = (relativePath) => readFile(new URL(relativePath, shellDir), 'utf8');

test('public shell loads the versioned Workjet contract before shell overrides', async () => {
  const html = await read('index.html');
  const contractIndex = html.indexOf('ui-contract/v1/workjet-ui-contract.css');
  const appIndex = html.indexOf('app.css');
  assert.ok(contractIndex >= 0, 'index.html must load the Workjet contract');
  assert.ok(contractIndex < appIndex, 'contract must load before app.css');
  assert.match(html, /5173a1155a9a5f1f28ed43afcb004693dd95c073cabfae8157cd01c7e8830419/);
});

test('shell maps public targets to categories and applies them to active chrome', async () => {
  const app = await read('app.js');
  const desktop = await read('modules/desktop/index.js');
  const desktopCss = await read('modules/desktop/index.css');

  assert.match(app, /workjet-theme\.js\?v=20260903-entertainment-import-v336/);
  assert.match(app, /workjetCategoryForModule\(mod\)/);
  assert.match(app, /state\.schemaRegistrations\.delete\(activeModuleId\)/);
  assert.match(app, /applyWorkjetCategory\(win\.element, entry\.category \|\| 'imported'\)/);
  assert.match(app, /applyWorkjetCategory\(win\.element, descriptor\.category\)/);
  assert.match(app, /applyWorkjetCategory\(button, target\.category/);
  assert.match(desktop, /workjetCategoryForModule/);
  assert.match(desktop, /applyWorkjetCategory\(refs\.root/);
  assert.match(desktop, /categoryForLauncherEntry/);
  assert.match(desktopCss, /desktop-icon\[data-workjet-category\].*selected/);
  assert.match(desktopCss, /desktop-module\[data-workjet-category\] \.widget-status/);
});

test('shell primitives stay neutral in light/dark and category color is state-scoped', async () => {
  const css = await read('app.css');
  assert.match(css, /--bg:\s*var\(--ctox-host-bg, #fcfcfc\)/);
  assert.match(css, /--bg:\s*var\(--ctox-host-bg, #0a0a0a\)/);
  assert.match(css, /--font-sans:\s*-apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif/);
  assert.match(css, /\.topbar[\s\S]*background:\s*var\(--workjet-surface-chrome, var\(--surface\)\) !important/);
  assert.match(css, /\.module-tab\[aria-current="page"\][\s\S]*var\(--shell-category-accent\)/);
  assert.match(css, /\.shell-window\.is-focused[\s\S]*var\(--shell-category-accent\)/);
  assert.match(css, /\.start-menu-item:focus-within[\s\S]*var\(--shell-category-soft\)/);
  assert.doesNotMatch(css, /#72b8aa/i, 'legacy dark-green shell accent must not return');
});

test('chat dock and windows use contract category metadata instead of module palettes', async () => {
  const chat = await read('shared/business-chat.js');
  assert.match(chat, /workjet-theme\.js\?v=20260903-entertainment-import-v336/);
  assert.match(chat, /data-workjet-category="\$\{escapeAttr\(category\)\}"/);
  assert.match(chat, /--shell-category-accent/);
  assert.match(chat, /--accent: var\(--shell-category-accent, var\(--workjet-accent, #1b4ed8\)\)/);
  assert.match(chat, /\.ctox-chat-dock[\s\S]*background: var\(--surface-2\)/);
  assert.match(chat, /\.ctox-chat-window[\s\S]*background: var\(--surface\)/);
  assert.doesNotMatch(chat, /\.ctox-chat-(?:chip|window)\[data-chat-module=/,
    'chat accent must not be selected by module id');
});

test('mobile Home ignores stale focused windows and opens apps only from real icon activation', async () => {
  const mobileHost = await read('mobile-host.js');

  assert.match(
    mobileHost,
    /activeAppId !== 'desktop' && focusedAppId && focusedAppId !== activeAppId/,
    'a stale focused window must never replace the canonical mobile Home desk',
  );
  assert.match(mobileHost, /\.desktop-icon\[data-target\]/);
  assert.match(mobileHost, /document\.addEventListener\('click',[\s\S]*applyActiveAppMetadata\(appId\)/);
  assert.match(mobileHost, /document\.addEventListener\('keydown',[\s\S]*applyActiveAppMetadata\(appId\)/);
  assert.doesNotMatch(mobileHost, /code: 'native-home-route'/);
});
