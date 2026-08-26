// SPDX-License-Identifier: MIT OR AGPL-3.0-only
import assert from 'node:assert/strict';
import test from 'node:test';

import { customerReleaseViolations } from './assert-customer-app-isolation.mjs';

test('customer runtime paths and private global manifests are rejected', () => {
  const manifests = new Map([
    ['src/apps/business-os/modules/tickets/module.json', JSON.stringify({ id: 'tickets' })],
    ['src/apps/business-os/modules/private/module.json', JSON.stringify({ id: 'private', audience: 'customer' })],
  ]);
  const violations = customerReleaseViolations([
    'src/apps/business-os/installed-modules/rem-private/index.js',
    'runtime/thesen-work/thesen-outbound/index.js',
    ...manifests.keys(),
  ], (path) => manifests.get(path));
  assert.equal(violations.length, 3);
  assert.match(violations.join('\n'), /installed-modules/);
  assert.match(violations.join('\n'), /runtime\/thesen-work/);
  assert.match(violations.join('\n'), /private\/module\.json/);
});

test('public system manifests remain allowed', () => {
  const violations = customerReleaseViolations([
    'src/apps/business-os/modules/tickets/module.json',
  ], () => JSON.stringify({ id: 'tickets', audience: 'public' }));
  assert.deepEqual(violations, []);
});

test('unknown and malformed distribution markers fail closed', () => {
  const paths = [
    'src/apps/business-os/modules/new-private/module.json',
    'src/apps/business-os/modules/malformed/module.json',
    'src/apps/business-os/modules/customer-id/module.json',
  ];
  const values = new Map([
    [paths[0], JSON.stringify({ id: 'new-private', distribution: 'customer-beta' })],
    [paths[1], JSON.stringify({ id: 'malformed', audience: ['public'] })],
    [paths[2], JSON.stringify({ id: 'customer-id', customerId: 42 })],
  ]);
  const violations = customerReleaseViolations(paths, (path) => values.get(path));
  assert.equal(violations.length, 3);
});
