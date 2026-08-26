import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

import { loadBusinessOsAppInventory } from '../scripts/business-os-app-inventory.mjs';
import {
  isShellSurfaceModule,
  launchesInWindow,
  resolvePresentation,
  SHELL_SURFACE_MODULE_ID,
  usesLegacyWorkspace,
} from './presentation.js';
import {
  SHELL_WINDOW_CHROME_VERSION,
  SHELL_WINDOW_CONTROL_ACTIONS,
} from './window-manager.js';

const here = dirname(fileURLToPath(import.meta.url));
const businessOsRoot = resolve(here, '..');
const appSource = readFileSync(resolve(businessOsRoot, 'app.js'), 'utf8');
const appCss = readFileSync(resolve(businessOsRoot, 'app.css'), 'utf8');
const windowManagerSource = readFileSync(resolve(here, 'window-manager.js'), 'utf8');
const registry = JSON.parse(readFileSync(resolve(businessOsRoot, 'modules/registry.json'), 'utf8'));
const systemApps = JSON.parse(readFileSync(resolve(businessOsRoot, 'system-apps.json'), 'utf8'));

test('every registry app is classified as a shared-window app or the one shell surface', () => {
  const inventory = loadBusinessOsAppInventory();
  const modules = Array.isArray(registry?.modules) ? registry.modules : [];
  const shellSurfaces = modules.filter(isShellSurfaceModule);

  // The inventory helper verifies registry/source parity and exact counts; the
  // loop below deliberately touches every registry entry so a new app cannot
  // silently skip the windowing contract.
  assert.equal(modules.length, inventory.sourceApps.length);
  assert.deepEqual(shellSurfaces.map((mod) => mod.id), [SHELL_SURFACE_MODULE_ID]);
  assert.deepEqual(new Set(systemApps.apps), new Set(inventory.coreApps.map((app) => app.id)));

  for (const moduleDef of modules) {
    assert.ok(moduleDef?.id, 'every registry entry needs an id');
    if (isShellSurfaceModule(moduleDef)) {
      assert.equal(moduleDef.layout?.shell, 'full-workspace');
      assert.equal(launchesInWindow(moduleDef), false);
      assert.equal(usesLegacyWorkspace(moduleDef), true);
      continue;
    }

    assert.equal(moduleDef.launch_kind, 'desktop-app', `${moduleDef.id} must declare the app launch kind`);
    assert.equal(moduleDef.layout?.shell, 'windowed', `${moduleDef.id} must use the windowed shell contract`);
    assert.ok(moduleDef.presentation && typeof moduleDef.presentation === 'object');
    assert.ok(['window', 'maximized', 'focus'].includes(moduleDef.presentation.default_mode));
    for (const mode of ['window', 'maximized', 'focus']) {
      assert.ok(moduleDef.presentation.supported_modes.includes(mode), `${moduleDef.id} missing ${mode} support`);
    }
    assert.equal(launchesInWindow(moduleDef), true, `${moduleDef.id} must launch in the shared window shell`);
    assert.equal(usesLegacyWorkspace(moduleDef), false, `${moduleDef.id} must not use the workspace bypass`);
  }
});

test('legacy, runtime, and imported app records cannot opt back into full-workspace mounting', () => {
  const records = [
    { id: 'legacy-imported', layout: { shell: 'full-workspace' } },
    { id: 'runtime-missing-presentation' },
    { id: 'runtime-windowed', launch_kind: 'desktop-app' },
    { id: 'imported-desktop-window', layout: { shell: 'desktop-window' } },
  ];

  for (const moduleDef of records) {
    assert.equal(launchesInWindow(moduleDef), true, `${moduleDef.id} must be window-hosted`);
    assert.equal(resolvePresentation(moduleDef).defaultMode, 'window');
    assert.equal(usesLegacyWorkspace(moduleDef), false);
  }
});

test('the shared shell window always exposes one draggable chrome and all three controls', () => {
  assert.deepEqual([...SHELL_WINDOW_CONTROL_ACTIONS].sort(), ['close', 'maximize', 'minimize']);
  assert.equal(SHELL_WINDOW_CHROME_VERSION, 'shared-v1');
  assert.match(windowManagerSource, /winEl\.dataset\.shellWindow = 'true'/);
  assert.match(windowManagerSource, /winEl\.dataset\.shellWindowChrome = SHELL_WINDOW_CHROME_VERSION/);
  assert.match(windowManagerSource, /data-window-header data-window-drag-region/);
  assert.match(windowManagerSource, /data-window-controls data-window-control-strip/);
  assert.match(windowManagerSource, /btn\.dataset\.windowControl = kind/);
  assert.match(windowManagerSource, /querySelector\('\[data-window-drag-region\]'\)/);
  assert.match(windowManagerSource, /windows: SHELL_WINDOW_CONTROL_ACTIONS/);
  assert.match(windowManagerSource, /macos: \['close', 'minimize', 'maximize'\]/);
  assert.match(windowManagerSource, /assertShellWindowChrome\(winEl\)/);
  assert.match(windowManagerSource, /assertShellWindowChrome\(win\.element\)/);
  assert.match(windowManagerSource, /btn\.type = 'button'/);
  assert.match(windowManagerSource, /btn\.setAttribute\('aria-label'/);
  for (const action of SHELL_WINDOW_CONTROL_ACTIONS) {
    assert.match(windowManagerSource, new RegExp(`['"]${action}['"]`));
  }
  assert.match(windowManagerSource, /if \(action === 'close'\) destroy\(win\.id\)/);
  assert.match(windowManagerSource, /else if \(action === 'minimize'\) minimize\(win\.id\)/);
  assert.match(windowManagerSource, /else if \(action === 'maximize'\) toggleMaximize\(win\.id\)/);
  assert.match(windowManagerSource, /setChromeLayout[\s\S]{0,500}renderControls\(/);
});

test('mobile sheets keep one close action while desktop retains the complete window contract', () => {
  assert.match(appCss, /@media \(max-width: 767px\)[\s\S]*\.shell-window-control--minimize,[\s\S]*\.shell-window-control--maximize[\s\S]*display: none/);
  assert.doesNotMatch(appCss, /\.shell-window-control--close[^{}]*\{[^}]*display:\s*none/);
});

test('public shell titles are localized before registry implementation names are exposed', () => {
  assert.match(appSource, /shellText\('moduleTitles'\)\?\.\[mod\.id\]/);
  assert.match(appSource, /documents:\s*'Dokumente'/);
  assert.match(appSource, /documents:\s*'Documents'/);
  assert.match(appSource, /spreadsheets:\s*'Tabellen'/);
  assert.match(appSource, /spreadsheets:\s*'Spreadsheets'/);
});

test('all app launch routes converge on the shared window manager', () => {
  assert.match(appSource, /async function openDesktopApp\(appId, options = \{\}\)/);
  assert.match(appSource, /async function openWindowedModule\(mod, options = \{\}\)/);
  assert.match(appSource, /state\.windowManager\.create\(\{/);
  assert.match(appSource, /ownerId: `desktop-app:\$\{entry\.id\}`/);
  assert.match(appSource, /ownerId: `desktop-app:\$\{mod\.id\}`/);
  for (const staticAppId of ['explorer', 'code-editor', 'file-viewer']) {
    assert.match(appSource, new RegExp(`id: '${staticAppId}'`));
  }
  assert.match(appSource, /if \(moduleLaunchesAsDesktopApp\(mod\)\) \{/);
  assert.doesNotMatch(appSource, /moduleLaunchesAsDesktopApp\(mod\) && !options\.asModule/);
  assert.match(appSource, /Every Business OS app is hosted by the shared window manager/);
});
