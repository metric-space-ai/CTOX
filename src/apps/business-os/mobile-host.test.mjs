import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { test } from 'node:test';

const root = new URL('./', import.meta.url);
const read = (name) => readFile(new URL(name, root), 'utf8');

test('mobile host is additive, keeps the canonical home, and removes duplicate chrome', async () => {
  const [index, css, script] = await Promise.all([
    read('index.html'),
    read('mobile-host.css'),
    read('mobile-host.js'),
  ]);
  assert.match(index, /mobile-host\.css/);
  assert.match(index, /mobile-host\.js/);
  assert.match(css, /data-workjet-mobile-host="true"/);
  for (const selector of [
    '.topbar',
    '.shell-window-header',
    '.shell-window-resize',
    '.shell-window-switcher',
    '[data-chat-dock]',
    '[data-taskbar]',
  ]) {
    assert.ok(css.includes(selector), `missing mobile chrome guard for ${selector}`);
  }
  assert.match(script, /workjet\.business-os-shell\.v1/);
  assert.match(script, /appId === 'desktop'/);
  assert.match(script, /\.shell-window\.is-focused/);
  assert.match(script, /applyActiveAppMetadata\(focusedAppId\)/);
  assert.doesNotMatch(script, /native-home-route/);
  assert.match(css, /\.desktop-module\s*\{[\s\S]*min-height:\s*100%/);
  assert.doesNotMatch(script, /capabilityToken|roomPassword|signalingUrls|businessRecords/);
});

test('signed mobile catalog follows the canonical system-app order', async () => {
  const [catalogRaw, systemRaw] = await Promise.all([
    read('mobile-apps.json'),
    read('system-apps.json'),
  ]);
  const catalog = JSON.parse(catalogRaw);
  const system = JSON.parse(systemRaw);
  assert.equal(catalog.type, 'workjet.business-os-mobile-apps.v1');
  assert.deepEqual(
    catalog.apps.slice(0, system.apps.length - 1).map((app) => app.id),
    system.apps.filter((id) => id !== 'desktop'),
  );
  assert.ok(catalog.apps.every((app) => !('iconSvg' in app) && !('iconUrl' in app)));
  assert.ok(catalog.apps.every((app) => app.id !== 'desktop'));
  assert.equal(catalog.apps.length, 35);
  assert.equal(new Set(catalog.apps.map((app) => app.iconAssetId)).size, 35);
  assert.ok(catalog.apps.every((app) => app.iconFamilyVersion === 1 && app.iconRequired === true));
});
