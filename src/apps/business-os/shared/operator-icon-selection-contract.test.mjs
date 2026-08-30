import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import { OPERATOR_ICON_SELECTION } from './operator-icon-selection.js';

const sharedRoot = dirname(fileURLToPath(import.meta.url));
const businessOsRoot = resolve(sharedRoot, '..');
const manifestPath = resolve(
  sharedRoot,
  'assets/workjet-icons/operator-selection-v1/manifest.json',
);
const registryPath = resolve(businessOsRoot, 'modules/registry.json');

test('all 34 operator-selected raster icons are hash-bound and registered', async () => {
  const [manifest, registry] = await Promise.all([
    readFile(manifestPath, 'utf8').then(JSON.parse),
    readFile(registryPath, 'utf8').then(JSON.parse),
  ]);

  assert.equal(manifest.schema, 'workjet.operator-icon-selection.v1');
  assert.equal(manifest.assetKind, 'raster-reference');
  assert.equal(manifest.count, 34);
  assert.equal(manifest.icons.length, 34);
  assert.equal(new Set(manifest.icons.map(({ appId }) => appId)).size, 34);
  assert.deepEqual(Object.keys(OPERATOR_ICON_SELECTION).sort(), manifest.icons.map(({ appId }) => appId).sort());

  for (const icon of manifest.icons) {
    assert.match(icon.candidateId, /^candidate-(?:0[1-9]|1[0-6])$/);
    assert.match(icon.sha256, /^[a-f0-9]{64}$/);
    assert.equal(icon.mediaType, 'image/jpeg');

    const moduleDefinition = registry.modules.find(({ id }) => id === icon.appId);
    assert.ok(moduleDefinition, `missing registry module ${icon.appId}`);
    assert.equal(moduleDefinition.layout.icon_asset, icon.renderAsset);
    assert.equal(moduleDefinition.layout.icon_asset_sha256, icon.renderSha256);
    assert.equal(moduleDefinition.layout.icon_selection_sha256, icon.sha256);
    assert.equal(moduleDefinition.layout.icon_selection_candidate, icon.candidateId);
    assert.equal(moduleDefinition.layout.icon_asset_kind, 'raster-reference');

    const shellIcon = OPERATOR_ICON_SELECTION[icon.appId];
    assert.equal(shellIcon.asset, icon.renderAsset);
    assert.equal(shellIcon.sha256, icon.renderSha256);
    assert.equal(shellIcon.candidateId, icon.candidateId);

    const originalBytes = await readFile(resolve(businessOsRoot, icon.asset));
    const actualOriginalSha256 = createHash('sha256').update(originalBytes).digest('hex');
    assert.equal(actualOriginalSha256, icon.sha256, `selection hash mismatch for ${icon.appId}`);

    const renderBytes = await readFile(resolve(businessOsRoot, icon.renderAsset));
    const actualRenderSha256 = createHash('sha256').update(renderBytes).digest('hex');
    assert.equal(actualRenderSha256, icon.renderSha256, `render hash mismatch for ${icon.appId}`);
  }
});
