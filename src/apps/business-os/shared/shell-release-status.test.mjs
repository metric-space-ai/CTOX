// SPDX-License-Identifier: MIT OR AGPL-3.0-only
import assert from 'node:assert/strict';
import test from 'node:test';

import { normalizeShellUpdateStatus, normalizeShellVersion, shellChannel } from './shell-release-status.js';

test('shell identity stays short and channel-aware', () => {
  assert.equal(normalizeShellVersion('v1.2.3'), '1.2.3');
  assert.equal(normalizeShellVersion('business-os-shell-v1.2.3'), '');
  assert.equal(shellChannel('1.2.3'), 'stable');
  assert.equal(shellChannel('1.2.3-rc.1'), 'beta');
  assert.equal(shellChannel('1.2.3-nightly.9'), 'nightly');
});

test('all public update states are accepted and unknown values fail closed', () => {
  for (const state of ['current', 'checking', 'available', 'download', 'verify', 'ready', 'restart', 'failed', 'incompatible', 'blocked', 'rollback', 'recovery']) {
    assert.equal(normalizeShellUpdateStatus(state), state);
  }
  assert.equal(normalizeShellUpdateStatus('surprise'), 'failed');
});
