import assert from 'node:assert/strict';
import test from 'node:test';

import { __ctoxSyncTestHooks } from './sync.js';

const {
  applyRoomRepairCycleOutcome,
  boundedCollectionStartQueueStep,
  repairRestartBatch,
} = __ctoxSyncTestHooks;

function roomCircuit(overrides = {}) {
  return {
    state: 'closed',
    consecutiveFailures: 0,
    openUntilMs: 0,
    nextProbeAtMs: 0,
    permanent: false,
    lastError: null,
    lastFailureKey: '',
    lastFailureAtMs: 0,
    updatedAtMs: 0,
    ...overrides,
  };
}

test('a failed collection retry does not stop healthy collections in the restart batch', async () => {
  const states = new Map([
    ['documents', { connectionStatus: 'connected', open: true }],
    ['document_versions', { connectionStatus: 'connected', open: true }],
    ['ctox_queue_tasks', { connectionStatus: 'reconnecting', open: false }],
  ]);
  const stopped = [];
  const initial = [...states].map(([collection, state]) => ({
    collection,
    bridge: { collection, state },
  }));

  const result = await repairRestartBatch(initial, {
    async waitForStable({ collection, bridge, error }) {
      if (error) throw error;
      if (!bridge?.state?.open) throw new Error(`${collection} did not open`);
      return bridge;
    },
    async stopFailed({ collection }) {
      stopped.push(collection);
      states.get(collection).connectionStatus = 'reconnecting';
    },
    async restartFailed(failed) {
      return failed.map(({ collection }) => {
        const state = states.get(collection);
        state.open = true;
        state.connectionStatus = 'connected';
        return { collection, bridge: { collection, state } };
      });
    },
  });

  assert.deepEqual(stopped, ['ctox_queue_tasks']);
  assert.deepEqual(
    result.stable.map(({ collection }) => collection).sort(),
    ['ctox_queue_tasks', 'document_versions', 'documents'],
  );
  assert.equal(result.failed.length, 0);
  assert.equal(states.get('documents').connectionStatus, 'connected');
  assert.equal(states.get('document_versions').connectionStatus, 'connected');
});

test('one room repair cycle counts at most one failure for many collection errors', async () => {
  const circuit = roomCircuit();
  const collectionErrors = await Promise.all(
    Array.from({ length: 32 }, async (_, index) => new Error(`collection-${index} failed`)),
  );

  const delayMs = applyRoomRepairCycleOutcome(
    circuit,
    { errors: collectionErrors },
    { now: 10_000, random: () => 0 },
  );

  assert.equal(circuit.consecutiveFailures, 1);
  assert.equal(circuit.state, 'closed');
  assert.equal(delayMs, 1_000);
});

test('a stable room repair cycle closes a half-open circuit and clears failures', () => {
  const circuit = roomCircuit({
    state: 'half_open',
    consecutiveFailures: 4,
    openUntilMs: 9_000,
    nextProbeAtMs: 9_000,
    lastError: { message: 'previous cycle failed' },
  });

  const delayMs = applyRoomRepairCycleOutcome(
    circuit,
    { stableCount: 3, errors: [new Error('one collection is still slow')] },
    { now: 12_000, random: () => 0 },
  );

  assert.equal(delayMs, 0);
  assert.equal(circuit.state, 'closed');
  assert.equal(circuit.consecutiveFailures, 0);
  assert.equal(circuit.openUntilMs, 0);
  assert.equal(circuit.nextProbeAtMs, 0);
  assert.equal(circuit.lastError, null);
});

test('a never-settling bridge promise cannot block the next collection start', async () => {
  const neverSettles = new Promise(() => {});
  let startQueue = Promise.resolve();
  const firstBridge = startQueue.then(() => neverSettles);
  startQueue = boundedCollectionStartQueueStep(firstBridge, 20, 0);

  let followingStarted = false;
  const followingBridge = startQueue.then(() => {
    followingStarted = true;
    return { collection: 'documents' };
  });

  const result = await Promise.race([
    followingBridge,
    new Promise((_, reject) => setTimeout(() => reject(new Error('following collection stayed blocked')), 250)),
  ]);
  assert.equal(followingStarted, true);
  assert.deepEqual(result, { collection: 'documents' });
});
