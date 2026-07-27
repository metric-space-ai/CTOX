// REGRESSION (SYNC-A A0.6): targeted writes, field-merge persistence, and
// demand-cache safety setup must fail closed and surface diagnostics.

import { replicationWebRtcTestInternals } from '../src/replication-webrtc.mjs';
import { QueryMetaStorage } from '../src/query-meta-storage.mjs';

const ReplicationState = replicationWebRtcTestInternals.getReplicationStateClass();

const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};

function mockCollection(name, { conflictStrategy = 'lww' } = {}) {
  let demandLoader = null;
  return {
    name,
    schema: {
      version: 0,
      primaryPath: 'id',
      hash: async () => `hash-${name}`,
    },
    observe() { return { unsubscribe() {} }; },
    setDemandLoader(loader) { demandLoader = loader; },
    get demandLoader() { return demandLoader; },
    storageCollection: {
      databaseName: `db-${name}`,
      conflictStrategy,
      replicationCheckpointStatus: async () => ({ epoch: 'e1', state: 'ready' }),
      getChangedDocumentsSince: async () => ({ documents: [], checkpoint: null }),
      getStoredRecord: async () => null,
      bulkWrite: async () => ({}),
    },
  };
}

async function makeState(name, options = {}) {
  const state = new ReplicationState({
    collection: mockCollection(name, options),
    topic: `room-a06-${name}`,
    pull: { batchSize: 5 },
    push: { batchSize: 5 },
    retryTime: 60,
    ctox: {},
  });
  state.initialReplication?.catch?.(() => {});
  state.shared = {
    peer: { request: async () => [] },
    demandTransport: {},
    getTransportStatus: () => ({}),
    unregister() {},
  };
  return state;
}

// 1. A resolved terminal ctoxError is not an empty conflict array. The direct
// write path must reject immediately and must not retry/report success.
{
  const state = await makeState('business_commands');
  let requests = 0;
  state.shared.peer.request = async () => {
    requests += 1;
    return {
      type: 'ctoxError',
      scope: 'replication',
      code: 'RC_WEBRTC_SCHEMA',
      status: 422,
      direction: 'push',
      collection: 'business_commands',
      message: 'document failed schema validation',
    };
  };

  let caught = null;
  try {
    await state.writeDocumentsToPeer('peer-1', [{ id: 'cmd-1', status: 'pending_sync' }]);
  } catch (error) {
    caught = error;
  }

  assert(caught, 'terminal direct-push rejection must throw');
  assert(caught.code === 'ctox_replication_push_rejected', `unexpected rejection code: ${caught?.code}`);
  assert(caught.terminal === true, 'terminal direct-push error must be marked terminal');
  assert(requests === 1, `terminal direct-push rejection must not be retried (got ${requests})`);
  await state.cancel();
}

// 2. A failed local write of the merged row must be relayed and stop before a
// second masterWrite can push the unpersisted merge.
{
  const state = await makeState('business_records', { conflictStrategy: 'field-merge' });
  const base = { id: 'record-1', title: 'base', status: 'draft' };
  const local = { id: 'record-1', title: 'local edit', status: 'draft' };
  const master = { id: 'record-1', title: 'base', status: 'published' };
  let requests = 0;
  const remoteRows = [];
  state.shared.peer.request = async (_peerId, _method, [rows]) => {
    requests += 1;
    remoteRows.push(rows);
    return [master];
  };
  state.collection.storageCollection.getStoredRecord = async () => ({ base });
  state.collection.storageCollection.bulkWrite = async () => {
    throw new Error('local merge write failed');
  };
  const observed = [];
  state.error$.subscribe((error) => { if (error) observed.push(error); });

  let caught = null;
  try {
    await state.writeDocumentsToPeer('peer-1', [local]);
  } catch (error) {
    caught = error;
  }

  assert(caught?.code === 'ctox_field_merge_persistence_failed', 'merge-write failure must reject with a diagnostic error');
  assert(caught?.operation === 'write-merged-document', `unexpected merge operation: ${caught?.operation}`);
  assert(
    observed.some((error) => error?.code === 'ctox_field_merge_persistence_failed' && error?.documentId === 'record-1'),
    'merge-write failure must be observable on error$',
  );
  assert(requests === 1, `unpersisted merged row must not be remotely retried (got ${requests} masterWrite calls)`);
  assert(remoteRows.length === 1 && remoteRows[0][0]?.newDocumentState === local, 'only the original row may reach masterWrite');
  await state.cancel();
}

// 3. Demand loading must fail closed when its cache budget cannot be persisted:
// no loader attachment, no active marker, and both error$/status diagnostics.
{
  const originalSetBudgetBytes = QueryMetaStorage.prototype.setBudgetBytes;
  QueryMetaStorage.prototype.setBudgetBytes = async function setBudgetBytesFailure() {
    throw new Error('budget backend unavailable');
  };

  const state = await makeState('business_records');
  const observed = [];
  state.error$.subscribe((error) => { if (error) observed.push(error); });
  try {
    const loader = await state.enableDemandLoading({ indexedDbAvailable: false });
    assert(loader === null, 'failed demand-cache safety setup must not return a loader');
    assert(state.demandLoaderActive === false, 'failed budget setup must not mark demand loading initialized');
    assert(state.demandStatus.queryDemandLoadingActive === false, 'failed budget setup must keep demand loading inactive');
    assert(state.collection.demandLoader === null, 'failed budget setup must not attach a collection loader');
    assert(
      observed.some((error) => error?.code === 'ctox_demand_cache_safety_failed' && error?.operation === 'setBudgetBytes'),
      'setBudgetBytes failure must be observable on error$',
    );
    const status = state.getTransportStatus();
    assert(status.demandLoading?.queryDemandLoadingActive === false, 'transport status must report inactive demand loading');
    assert(status.demandLoading?.code === 'ctox_demand_cache_safety_failed', 'transport status must expose the safety failure code');
  } finally {
    QueryMetaStorage.prototype.setBudgetBytes = originalSetBudgetBytes;
    await state.cancel();
  }
}

console.log('write-documents-error-handling-smoke: ok');
