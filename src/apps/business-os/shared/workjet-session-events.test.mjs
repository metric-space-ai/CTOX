import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const appSource = readFileSync(new URL('../app.js', import.meta.url), 'utf8');
const mobileSource = readFileSync(new URL('../mobile-host.js', import.meta.url), 'utf8');
const eventsStart = appSource.indexOf('const WORKJET_SESSION_EVENT_BUFFER_LIMIT');
const eventsEnd = appSource.indexOf('function installAdvancedStatusInterface', eventsStart);
const eventsSource = appSource.slice(eventsStart, eventsEnd);

const eventShape = {
  type: 'workjet.session.transfer',
  transferId: 'transfer-1',
  sessionId: 'session-1',
  state: 'pause_requested',
  fenceEpoch: 1,
  sourceComputerId: 'computer-source',
  targetComputerId: 'computer-target',
  deadlineAtMs: 1_700_000_060_000,
  updatedAtMs: 1_700_000_000_000,
};

function transferRecord(overrides = {}) {
  return {
    id: 'transfer-1',
    session_id: 'session-1',
    state: 'pause_requested',
    fence_epoch: 1,
    source_computer_id: 'computer-source',
    target_computer_id: 'computer-target',
    deadline_at_ms: 1_700_000_060_000,
    updated_at_ms: 1_700_000_000_000,
    is_deleted: false,
    ...overrides,
  };
}

function createCollection(initial = []) {
  const docs = [...initial];
  const listeners = new Set();
  return {
    docs,
    $: {
      subscribe(listener) {
        listeners.add(listener);
        return { unsubscribe: () => listeners.delete(listener) };
      },
    },
    find() {
      return { async exec() { return [...docs]; } };
    },
    emit(record) {
      const index = docs.findIndex((doc) => doc.id === record.id);
      if (index >= 0) docs[index] = record;
      else docs.push(record);
      for (const listener of listeners) listener({ documentData: record });
    },
    get listenerCount() { return listeners.size; },
  };
}

function boundedText(value, field, maxLength) {
  if (typeof value !== 'string') throw new TypeError(`Invalid ${field}`);
  const normalized = value.trim();
  const hasControlCharacter = Array.from(normalized).some((character) => {
    const code = character.charCodeAt(0);
    return code < 32 || code === 127;
  });
  if (!normalized || [...normalized].length > maxLength || hasControlCharacter) {
    throw new TypeError(`Invalid ${field}`);
  }
  return normalized;
}

function createHarness({ initial = [], ready = Promise.resolve() } = {}) {
  const collection = createCollection(initial);
  const startedCollections = [];
  const syncWaits = [];
  const context = {
    console,
    state: {
      db: { collection: (name) => name === 'workjet_session_transfers' ? collection : null },
      sync: {
        async startCollection(name) {
          startedCollections.push(name);
          return { name };
        },
      },
    },
    waitForDataPlaneReady: () => ready,
    waitForSyncBridgeReady: async (bridge) => { syncWaits.push(bridge.name); },
    boundedWorkjetSessionText: boundedText,
  };
  vm.runInNewContext(
    `${eventsSource}\nglobalThis.__createWorkjetSessionEvents = createWorkjetSessionEvents;`,
    context,
  );
  return {
    context,
    collection,
    startedCollections,
    syncWaits,
    api: context.__createWorkjetSessionEvents(),
  };
}

const settle = () => new Promise((resolve) => setImmediate(resolve));

test('Workjet session transfer events install a bounded host API and mobile output', () => {
  assert.ok(eventsStart >= 0 && eventsEnd > eventsStart, 'event implementation exists');
  assert.match(appSource, /globalThis\.workjetSessionEvents = createWorkjetSessionEvents\(\)/);
  assert.match(eventsSource, /startCollection\?\.\('workjet_session_transfers'\)/);
  assert.match(eventsSource, /await waitForDataPlaneReady\(\)/);
  assert.match(eventsSource, /collection\.\$\.subscribe/);
  assert.match(eventsSource, /postSessionTransferEvent/);
  assert.match(eventsSource, /WORKJET_SESSION_EVENT_BUFFER_LIMIT = 64/);
  assert.doesNotMatch(eventsSource, /fetch\s*\(|XMLHttpRequest|\/api\/|https?:\/\//);

  assert.match(mobileSource, /post\(\{ type: 'session\.transfer\.event', event: bounded \}\)/);
  assert.match(mobileSource, /boundedSessionTransferEvent/);
  assert.match(mobileSource, /transferId: text\(event\.transferId, 160\)/);
  assert.match(mobileSource, /sourceComputerId: text\(event\.sourceComputerId, 256\)/);
});

test('subscription waits for the data plane, filters computers, and deduplicates state updates', async () => {
  let releaseReady;
  const ready = new Promise((resolve) => { releaseReady = resolve; });
  const harness = createHarness({ ready });
  const posted = [];
  harness.context.workjetHostBridge = {
    postSessionTransferEvent(event) { posted.push(JSON.parse(JSON.stringify(event))); },
  };

  assert.deepEqual(JSON.parse(JSON.stringify(harness.api.register({
    computerIds: ['computer-source', 'computer-source'],
  }))), { registered: 1 });
  await settle();
  assert.deepEqual(harness.startedCollections, []);
  assert.equal(harness.collection.listenerCount, 0);

  releaseReady();
  await settle();
  await settle();
  assert.deepEqual(harness.startedCollections, ['workjet_session_transfers']);
  assert.deepEqual(harness.syncWaits, ['workjet_session_transfers']);
  assert.equal(harness.collection.listenerCount, 1);

  const pauseRequested = transferRecord();
  harness.collection.emit(pauseRequested);
  assert.deepEqual(posted, [eventShape]);

  const packing = transferRecord({ state: 'packing', updated_at_ms: eventShape.updatedAtMs + 1 });
  harness.collection.emit(packing);
  harness.collection.emit(packing);
  harness.collection.emit(transferRecord({
    id: 'transfer-foreign',
    source_computer_id: 'computer-foreign-a',
    target_computer_id: 'computer-foreign-b',
    updated_at_ms: eventShape.updatedAtMs + 2,
  }));
  assert.equal(posted.length, 2);
  assert.equal(posted[1].state, 'packing');

  const snapshot = JSON.parse(JSON.stringify(await harness.api.snapshot()));
  assert.deepEqual(snapshot, [{ ...eventShape, state: 'packing', updatedAtMs: eventShape.updatedAtMs + 1 }]);
});

test('events buffer to 64 entries, replay on registration, and stop after unregister', async () => {
  const harness = createHarness();
  harness.api.register({ computerIds: ['computer-source'] });
  await settle();
  await settle();

  for (let index = 1; index <= 65; index += 1) {
    harness.collection.emit(transferRecord({ state: 'packing', updated_at_ms: index }));
  }

  const posted = [];
  harness.context.workjetHostBridge = {
    postSessionTransferEvent(event) { posted.push(JSON.parse(JSON.stringify(event))); },
  };
  harness.api.register({ computerIds: ['computer-source'] });
  assert.equal(posted.length, 64);
  assert.equal(posted[0].updatedAtMs, 2);
  assert.equal(posted.at(-1).updatedAtMs, 65);

  harness.collection.emit(transferRecord({ state: 'packing', updated_at_ms: 65 }));
  assert.equal(posted.length, 64);
  harness.api.unregister();
  assert.equal(harness.collection.listenerCount, 0);
  harness.collection.emit(transferRecord({ state: 'packed', updated_at_ms: 66 }));
  assert.equal(posted.length, 64);
});

test('snapshot excludes terminal transfers and rejects unbounded registration fields', async () => {
  const harness = createHarness({ initial: [
    transferRecord({ id: 'active', state: 'applying', updated_at_ms: 10 }),
    transferRecord({ id: 'done', state: 'completed', updated_at_ms: 11 }),
    transferRecord({ id: 'other', source_computer_id: 'other-a', target_computer_id: 'other-b' }),
  ] });
  harness.api.register({ computerIds: ['computer-target'] });
  const snapshot = JSON.parse(JSON.stringify(await harness.api.snapshot()));
  assert.equal(snapshot.length, 1);
  assert.equal(snapshot[0].transferId, 'active');
  assert.equal(snapshot[0].state, 'applying');

  const isTypeError = (error) => error?.name === 'TypeError';
  assert.throws(() => harness.api.register({ computerIds: ['computer-source'], extra: true }), isTypeError);
  assert.throws(() => harness.api.register({ computerIds: ['x'.repeat(257)] }), isTypeError);
  assert.throws(() => harness.api.register({ computerIds: 'computer-source' }), isTypeError);
});
