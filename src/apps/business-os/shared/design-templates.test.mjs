import assert from 'node:assert/strict';
import test from 'node:test';

import {
  currentShellDesignValue,
  normalizeDesignTemplates,
  resolveShellDesign,
  shellDesignOptions,
} from './design-templates.js';

const templates = [
  {
    id: 'hypo-rem',
    title: 'Hypo REM',
    description: 'REM product shell',
    base_style: 'windows',
    stylesheet_href: 'design-templates/hypo-rem/theme.css',
  },
];

test('custom templates extend the built-in CTOX, Windows and macOS designs', () => {
  assert.deepEqual(
    shellDesignOptions(templates).map(({ value }) => value),
    ['ctox', 'windows', 'macos', 'custom:hypo-rem'],
  );
  assert.deepEqual(resolveShellDesign('custom:hypo-rem', templates), {
    value: 'custom:hypo-rem',
    id: 'hypo-rem',
    label: 'Hypo REM',
    description: 'REM product shell',
    shellStyle: 'windows',
    stylesheet: 'design-templates/hypo-rem/theme.css',
    templateId: 'hypo-rem',
  });
});

test('invalid local template metadata cannot inject values into the shell', () => {
  assert.deepEqual(normalizeDesignTemplates([
    ...templates,
    { id: '../escape', title: 'Escape', stylesheet_href: '../escape.css' },
    { id: 'remote', title: 'Remote', stylesheet_href: 'https://example.com/theme.css' },
  ]), normalizeDesignTemplates(templates));
});

test('a missing custom selection falls back to CTOX', () => {
  assert.equal(resolveShellDesign('custom:missing', templates).value, 'ctox');
  assert.equal(
    currentShellDesignValue({ dataset: { shellStyle: 'macos', designTemplate: '' } }, templates),
    'macos',
  );
  assert.equal(
    currentShellDesignValue({ dataset: { shellStyle: 'windows', designTemplate: 'hypo-rem' } }, templates),
    'custom:hypo-rem',
  );
});
