import test from 'node:test';
import assert from 'node:assert/strict';
import { createCoalescedRefresh } from './src/coalesced-refresh.mjs';

function clock() {
  const timers = new Map();
  let next = 0;
  return {
    timers,
    setTimer(fn, ms) { assert.equal(ms, 80); const id = ++next; timers.set(id, fn); return id; },
    clearTimer(id) { timers.delete(id); },
    tick() { const [id, fn] = timers.entries().next().value; timers.delete(id); return fn(); },
  };
}

test('Office refresh batches bursts and never overlaps reads', async () => {
  const time = clock();
  const batches = [];
  let release;
  const held = new Promise(resolve => { release = resolve; });
  let active = 0;
  const refresh = createCoalescedRefresh({ ...time, async refresh(changed) {
    assert.equal(++active, 1);
    batches.push([...changed].sort());
    if (batches.length === 1) await held;
    active -= 1;
  } });
  for (let i = 0; i < 100; i++) refresh.notify('documents');
  assert.equal(time.timers.size, 1);
  const first = time.tick();
  for (let i = 0; i < 100; i++) {
    refresh.notify('document_versions');
    refresh.notify('document_blob_chunks');
  }
  assert.equal(time.timers.size, 0);
  release();
  await first;
  assert.equal(time.timers.size, 1);
  await time.tick();
  assert.deepEqual(batches, [['documents'], ['document_blob_chunks', 'document_versions']]);
  assert.equal(time.timers.size, 0);
  refresh.dispose();
});

test('Office refresh disposal cancels queued work and invalidates an in-flight render', async () => {
  const time = clock();
  let release;
  const held = new Promise(resolve => { release = resolve; });
  let rendered = false;
  const refresh = createCoalescedRefresh({ ...time, async refresh(_changed, isActive) {
    await held;
    if (isActive()) rendered = true;
  } });
  refresh.notify('spreadsheets');
  const first = time.tick();
  refresh.notify('spreadsheet_versions');
  refresh.dispose();
  release();
  await first;
  refresh.notify('spreadsheets');
  assert.equal(rendered, false);
  assert.equal(time.timers.size, 0);
});

test('Office refresh continues after a failed read without losing queued changes', async () => {
  const time = clock();
  const errors = [];
  let calls = 0;
  let refresh;
  refresh = createCoalescedRefresh({ ...time, onError: error => errors.push(error.message),
    async refresh() {
      calls += 1;
      if (calls === 1) { refresh.notify('documents'); throw new Error('query cancelled'); }
    },
  });
  refresh.notify('documents');
  await time.tick();
  await time.tick();
  assert.equal(calls, 2);
  assert.deepEqual(errors, ['query cancelled']);
  assert.equal(time.timers.size, 0);
});
