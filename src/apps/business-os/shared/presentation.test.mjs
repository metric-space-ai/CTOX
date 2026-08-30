import test from 'node:test';
import assert from 'node:assert/strict';

import {
  isShellSurfaceModule,
  launchesInWindow,
  resolvePresentation,
  usesLegacyWorkspace,
} from './presentation.js';

test('presentation contract resolves explicit window configuration', () => {
  const presentation = resolvePresentation({
    presentation: {
      default_mode: 'maximized',
      supported_modes: ['window', 'focus'],
      initial_size: { width: 1280, height: 800 },
      minimum_size: { width: 720, height: 520 },
      multi_instance: true,
    },
  });

  assert.equal(presentation.defaultMode, 'maximized');
  assert.deepEqual(presentation.supportedModes, ['maximized', 'window', 'focus']);
  assert.deepEqual(presentation.initialSize, { width: 1280, height: 800 });
  assert.deepEqual(presentation.minimumSize, { width: 720, height: 520 });
  assert.equal(presentation.multiInstance, true);
});

test('presentation contract retains bounded legacy behavior', () => {
  const windowed = { layout: { shell: 'desktop-window' } };
  const workspace = { id: 'desktop', layout: { shell: 'full-workspace' } };
  const projectedDesktop = { id: 'desktop' };
  const importedLegacyWorkspace = { id: 'imported-legacy', layout: { shell: 'full-workspace' } };
  const ordinaryPaneModule = {};

  assert.equal(launchesInWindow(windowed), true);
  assert.equal(isShellSurfaceModule(workspace), true);
  assert.equal(usesLegacyWorkspace(projectedDesktop), true);
  assert.equal(isShellSurfaceModule(importedLegacyWorkspace), false);
  assert.equal(usesLegacyWorkspace(workspace), true);
  assert.equal(launchesInWindow(importedLegacyWorkspace), true);
  assert.equal(usesLegacyWorkspace(importedLegacyWorkspace), false);
  assert.equal(usesLegacyWorkspace(ordinaryPaneModule), false);
});

test('modules without a presentation contract still launch in the shared window shell', () => {
  const runtimeModule = { id: 'runtime-imported-app', title: 'Imported app' };
  const presentation = resolvePresentation(runtimeModule);

  assert.equal(presentation.defaultMode, 'window');
  assert.ok(presentation.supportedModes.includes('window'));
  assert.equal(launchesInWindow(runtimeModule), true);
});
