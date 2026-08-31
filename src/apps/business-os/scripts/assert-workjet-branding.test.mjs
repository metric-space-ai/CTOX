#!/usr/bin/env node
// SPDX-License-Identifier: MIT OR AGPL-3.0-only

import assert from 'node:assert/strict';
import test from 'node:test';

import {
  FORBIDDEN_VISIBLE_TERMS,
  findForbiddenVisibleTerms,
  runBrandingGuard,
} from './assert-workjet-branding.mjs';

test('the release guard covers retired visible labels and product identities', () => {
  for (const term of [
    'Shard-Ansicht',
    'Shard view',
    'Listen-Ansicht',
    'CTOX App Store',
    'Kandidaten-Shards',
  ]) {
    assert.ok(FORBIDDEN_VISIBLE_TERMS.includes(term), `missing forbidden visible term: ${term}`);
  }
});

test('legacy labels are found in visible HTML copy and product strings', () => {
  const htmlFindings = findForbiddenVisibleTerms(
    '<button aria-label="Shard-Ansicht" title="Listen-Ansicht">Shard-Ansicht</button>',
    'src/apps/business-os/modules/example/index.html',
  );
  assert.deepEqual(htmlFindings.map(({ term }) => term), ['Shard-Ansicht', 'Listen-Ansicht', 'Shard-Ansicht']);

  const jsFindings = findForbiddenVisibleTerms(
    "const copy = { cardsView: 'Card view', legacy: 'Shard view' };",
    'src/apps/business-os/modules/example/index.js',
  );
  assert.deepEqual(jsFindings.map(({ term }) => term), ['Shard view']);
});

test('canonical labels and implementation-only CSS/comments remain allowed', () => {
  assert.deepEqual(
    findForbiddenVisibleTerms(
      '<button aria-label="Kachelansicht" title="Listenansicht">Card view</button>',
      'src/apps/business-os/modules/example/index.html',
    ),
    [],
  );
  assert.deepEqual(
    findForbiddenVisibleTerms(
      '/* Shard-Ansicht is an internal note */\n.shard-grid { --shard-count: 3; }',
      'src/apps/business-os/modules/example/index.css',
    ),
    [],
  );
  assert.deepEqual(
    findForbiddenVisibleTerms(
      "// 'Shard view' is an implementation comment\nconst mode = 'cards';",
      'src/apps/business-os/modules/example/index.js',
    ),
    [],
  );
});

test('all current release-owned Business OS surfaces pass the guard', async () => {
  const result = await runBrandingGuard();
  assert.ok(result.filesAudited > 0);
  assert.ok(result.moduleFiles.some((relativePath) => relativePath.endsWith('/modules/consent/index.html')));
  assert.ok(!result.moduleFiles.some((relativePath) => relativePath.includes('/modules/desktop/')));
});
