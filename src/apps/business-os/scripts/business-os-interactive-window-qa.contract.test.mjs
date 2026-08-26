#!/usr/bin/env node
import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const qaSource = readFileSync(
  fileURLToPath(new URL('./business-os-interactive-window-qa.mjs', import.meta.url)),
  'utf8',
);
const shellSource = readFileSync(
  fileURLToPath(new URL('../app.js', import.meta.url)),
  'utf8',
);

test('interactive QA rejects app recovery surfaces instead of counting their controls as content', () => {
  assert.match(qaSource, /moduleRoot\?\.dataset\.moduleLoadFailed === 'true'/);
  assert.match(qaSource, /content\.querySelector\('\.shell-app-recovery'\)/);
  assert.match(qaSource, /mounted its recovery surface/);
});

test('interactive QA collection grants are confined to the explicit all-source smoke fixture', () => {
  assert.match(qaSource, /!params\.has\('rxdbSmoke'\) \|\| params\.get\('qaCatalog'\) !== 'all-source'/);
  assert.match(qaSource, /subject_type: 'user'/);
  assert.match(qaSource, /permission,\s+scope_type: 'collection'/);
  assert.match(qaSource, /grant_id: `qa\.\$\{permission\}\.\$\{collection\}`/);
});

test('module mount diagnostics remain smoke-only and production recovery copy stays sanitized', () => {
  assert.match(shellSource, /new URLSearchParams\(window\.location\.search\)\.has\('rxdbSmoke'\)/);
  assert.match(shellSource, /state\.qaModuleMountFailures\[mod\.id\]/);
  assert.match(shellSource, /renderWindowAppRecovery\(content/);
});
