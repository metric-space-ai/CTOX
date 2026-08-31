#!/usr/bin/env node
import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const source = readFileSync(fileURLToPath(new URL('./smoke-app-module.mjs', import.meta.url)), 'utf8');

test('generic smoke treats a visible mounted non-CRUD app as valid', () => {
  assert.match(source, /No primary create action is required for a generic mount smoke/);
  assert.doesNotMatch(source, /result\.failures\.push\('no visible primary create action found under module root'\)/);
});

test('generic smoke fails same-origin HTTP request errors', () => {
  assert.match(source, /same-origin request failed/);
  assert.match(source, /url\.origin === pageOrigin/);
});
