import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import test from 'node:test';

const businessOsRoot = join(import.meta.dirname, '..');
const appSource = readFileSync(join(businessOsRoot, 'app.js'), 'utf8');
const indexSource = readFileSync(join(businessOsRoot, 'index.html'), 'utf8');
const syncSource = readFileSync(join(businessOsRoot, 'shared', 'sync.js'), 'utf8');
const registry = JSON.parse(readFileSync(join(businessOsRoot, 'modules', 'registry.json'), 'utf8'));
const browserManifest = JSON.parse(
  readFileSync(join(businessOsRoot, 'modules', 'browser', 'module.json'), 'utf8'),
);

test('Browser package catalog exposes the current manifest version', () => {
  const browserRegistryEntry = registry.modules.find((module) => module.id === 'browser');
  assert.ok(browserRegistryEntry, 'browser registry entry must exist');
  assert.equal(browserRegistryEntry.version, browserManifest.version);
  assert.match(appSource, new RegExp(`"version": "${browserManifest.version.replaceAll('.', '\\.') }"`));
});

test('workspace restore starts saved windows concurrently', () => {
  const restoreStart = appSource.indexOf('async function restoreWorkspaceSession');
  const restoreEnd = appSource.indexOf('\nfunction focusExplicitDesktopAppRoute', restoreStart);
  const restoreSource = appSource.slice(restoreStart, restoreEnd);
  assert.match(restoreSource, /await Promise\.allSettled\(\(snapshot\?\.windows \|\| \[\]\)\.map\(async \(entry\) => \{/);
  assert.doesNotMatch(restoreSource, /for \(const entry of snapshot\?\.windows \|\| \[\]\)/);
});

test('shell, index and multi-tab epoch use one release cache key', () => {
  const appBuild = appSource.match(/const APP_BUILD = '([^']+)'/)?.[1];
  const indexBuild = indexSource.match(/app\.js\?v=([^"']+)/)?.[1];
  const syncEpoch = syncSource.match(/const MULTI_TAB_COORDINATOR_EPOCH = '([^']+)'/)?.[1];
  assert.ok(appBuild);
  assert.equal(indexBuild, appBuild);
  assert.equal(syncEpoch, appBuild);
});
