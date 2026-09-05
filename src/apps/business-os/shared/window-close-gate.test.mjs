import test from 'node:test';
import assert from 'node:assert/strict';
import { createWindowCloseGate } from './window-close-gate.js';

test('close waits for native save and coalesces repeated clicks', async () => {
  const gate = createWindowCloseGate();
  let resolveSave;
  let saves = 0;
  let closed = 0;
  gate.add(() => { saves++; return new Promise(resolve => { resolveSave = resolve; }); });
  const first = gate.request(() => closed++);
  const repeated = gate.request(() => closed++);
  assert.equal(first, repeated);
  await Promise.resolve();
  assert.equal(saves, 1);
  assert.equal(closed, 0);
  resolveSave(true);
  assert.equal(await first, true);
  assert.equal(closed, 1);
});

test('failed native save keeps window alive and permits a later retry', async () => {
  let failures = 0;
  let closed = 0;
  let fail = true;
  const gate = createWindowCloseGate(() => failures++);
  gate.add(async () => { if (fail) throw new Error('save conflict'); });
  assert.equal(await gate.request(() => closed++), false);
  assert.equal(closed, 0);
  assert.equal(failures, 1);
  fail = false;
  assert.equal(await gate.request(() => closed++), true);
  assert.equal(closed, 1);
});

test('remaining edits veto closing without running subsequent guards', async () => {
  const gate = createWindowCloseGate();
  const remove = gate.add(() => false);
  let second = 0;
  let closed = 0;
  gate.add(() => { second++; });
  assert.equal(await gate.request(() => closed++), false);
  assert.equal(second, 0);
  assert.equal(closed, 0);
  remove();
  assert.equal(await gate.request(() => closed++), true);
  assert.equal(second, 1);
  assert.equal(closed, 1);
});
