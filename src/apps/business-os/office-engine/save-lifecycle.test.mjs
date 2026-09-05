import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createOfficeSaveTracker } from './src/runtime/ctox-fork-core.mjs';

test('save acknowledges only the serialized revision, preserving later typing', () => {
  const state = createOfficeSaveTracker();
  state.edit();
  const saved = state.snapshot();
  state.edit();
  assert.equal(state.acknowledge(saved), false);
  assert.equal(state.dirty, true);
  assert.equal(state.acknowledge(state.snapshot()), true);
  assert.equal(state.dirty, false);
});

test('failed save remains dirty until a successful acknowledgement', () => {
  const state = createOfficeSaveTracker();
  state.edit();
  const saved = state.snapshot();
  state.fail();
  assert.equal(state.dirty, true);
  state.acknowledge(saved);
  assert.equal(state.dirty, false);
});

test('late acknowledgement cannot clear edits in another opened document', () => {
  const state = createOfficeSaveTracker();
  state.edit();
  const previous = state.snapshot();
  state.reset();
  state.edit();
  assert.equal(state.acknowledge(previous), false);
  assert.equal(state.dirty, true);
});
