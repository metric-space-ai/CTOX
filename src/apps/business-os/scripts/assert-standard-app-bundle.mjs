import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '..');
const bundle = JSON.parse(readFileSync(resolve(root, 'standard-app-bundle.json'), 'utf8'));
const system = JSON.parse(readFileSync(resolve(root, 'system-apps.json'), 'utf8'));
const registry = JSON.parse(readFileSync(resolve(root, 'modules/registry.json'), 'utf8'));

assert.equal(bundle.schema, 'ctox.business-os.standard-app-bundle.v1');
assert.ok(Array.isArray(bundle.selected_apps) && bundle.selected_apps.length > 0);
assert.equal(new Set(bundle.selected_apps).size, bundle.selected_apps.length, 'standard app ids must be unique');
assert.deepEqual(system.apps, ['desktop', ...bundle.selected_apps], 'system app installation order must follow the selected bundle exactly');

const modules = new Map(registry.modules.map((module) => [module.id, module]));
for (const id of bundle.selected_apps) {
  const module = modules.get(id);
  assert.ok(module, `selected standard app must exist in registry: ${id}`);
  assert.equal(module.install_scope, 'core', `${id} registry install_scope`);
  assert.equal(module.default_installed, true, `${id} registry default_installed`);
  assert.equal(module.source, 'core', `${id} registry source`);
  assert.equal(module.core, true, `${id} registry core flag`);
  assert.equal(module.deletable, false, `${id} registry deletable flag`);
  assert.equal(module.store?.installable, false, `${id} registry store installable flag`);
  assert.equal(module.store?.distribution, 'system-module', `${id} registry distribution`);

  const manifest = JSON.parse(readFileSync(resolve(root, 'modules', id, 'module.json'), 'utf8'));
  for (const field of ['install_scope', 'default_installed', 'source', 'core', 'deletable']) {
    assert.deepEqual(manifest[field], module[field], `${id} ${field} must match source manifest and registry`);
  }
  for (const field of ['repository', 'source_path', 'installable', 'editable_after_install', 'distribution']) {
    assert.deepEqual(
      manifest.store?.[field],
      module.store?.[field],
      `${id} store.${field} must match source manifest and registry`,
    );
  }
}

for (const module of registry.modules) {
  if (module.id === 'desktop') continue;
  assert.equal(
    module.install_scope === 'core',
    bundle.selected_apps.includes(module.id),
    `${module.id} core membership must be owned by standard-app-bundle.json`,
  );
}

console.log(`standard_app_bundle_ok=1 selected=${bundle.selected_apps.length}`);
