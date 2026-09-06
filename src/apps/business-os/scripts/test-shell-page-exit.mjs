import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';
import { test } from 'node:test';

const source = await readFile(new URL('../app.js', import.meta.url), 'utf8');
const cleanup = source.match(/  window\.addEventListener\('(beforeunload|pagehide)',[^\n]*\n(?:[^\n]*\n)*?    if \(state\.ctoxHealthTimer\)[\s\S]*?\n  \}\);/);
assert.ok(cleanup, 'the shell must register its page-exit cleanup');

function fixture() {
  const window = new EventTarget();
  const cleared = [];
  let closed = false;
  let closes = 0;
  window.clearInterval = id => cleared.push(id);
  const state = {
    ctoxHealthTimer: 17,
    db: {
      close() { closed = true; closes += 1; },
      transaction() {
        if (closed) throw new DOMException('The database connection is closing.', 'InvalidStateError');
        return 'usable';
      },
    },
  };
  vm.runInNewContext(cleanup[0], { window, state });
  return { window, state, cleared, closes: () => closes };
}

test('cancelled navigation leaves the shared database and health timer usable', () => {
  const f = fixture();
  f.window.addEventListener('beforeunload', event => event.preventDefault());
  const event = new Event('beforeunload', { cancelable: true });
  assert.equal(f.window.dispatchEvent(event), false);
  assert.equal(f.state.db.transaction(), 'usable');
  assert.equal(f.closes(), 0);
  assert.deepEqual(f.cleared, []);
});

test('a page retained in the back-forward cache keeps its live database', () => {
  const f = fixture();
  const event = new Event('pagehide');
  Object.defineProperty(event, 'persisted', { value: true });
  f.window.dispatchEvent(event);
  assert.equal(f.state.db.transaction(), 'usable');
  assert.equal(f.closes(), 0);
  assert.deepEqual(f.cleared, []);
});

test('a real non-cached departure closes the shared database and health timer', () => {
  const f = fixture();
  const event = new Event('pagehide');
  Object.defineProperty(event, 'persisted', { value: false });
  f.window.dispatchEvent(event);
  assert.equal(f.closes(), 1);
  assert.deepEqual(f.cleared, [17]);
});
