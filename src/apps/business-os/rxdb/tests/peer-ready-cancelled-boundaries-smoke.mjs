import assert from 'node:assert/strict';
import { replicationWebRtcTestInternals } from '../src/replication-webrtc.mjs';

const runPeerReady = replicationWebRtcTestInternals.getReplicationStateClass().prototype.runPeerReady;
for (const boundary of ['checkpoint', 'permission', 'loader']) {
  let release;
  let entered;
  const pending = new Promise((resolve) => { release = resolve; });
  const started = new Promise((resolve) => { entered = resolve; });
  const pause = async (name, value) => {
    if (boundary === name) { entered(); await pending; }
    return value;
  };
  let active = false;
  let peers = new Map();
  let pulls = 0;
  let callbacks = 0;
  const state = {
    cancelled: false,
    ctox: { onPeerCapabilityNegotiated() { callbacks += 1; } },
    demandStatus: {},
    collection: { storageCollection: { replicationCheckpointStatus: () => pause('checkpoint', null) } },
    resolveReadPermissionDigest: () => pause('permission', null),
    retainedCheckpoints: null,
    peerStates$: { getValue: () => peers, next(value) { peers = value; } },
    active$: { next(value) { active = value; } },
    transportStatus$: { getValue: () => ({}), next() {} },
    decorateTransportStatus: (value) => value,
    enableDemandLoading: () => pause('loader', null),
    error$: { next(error) { throw error; } },
    pullFromRemotePeers: async () => { pulls += 1; },
    pushToRemotePeers: async () => {},
    resolveInitialReplication() {},
    rejectInitialReplication(error) { throw error; },
  };
  const run = runPeerReady.call(state, 'native-test', {}, boundary === 'loader');
  await started;
  state.cancelled = true;
  active = false; // cancel() emits active=false before asynchronous cleanup.
  peers = new Map();
  release();
  await run;
  assert.equal(active, false, `${boundary}: cancelled state was reactivated`);
  assert.equal(peers.size, 0, `${boundary}: cancelled state regained a peer`);
  assert.equal(pulls, 0, `${boundary}: cancelled state started replication`);
  assert.equal(callbacks, 0, `${boundary}: cancelled state announced readiness`);
}
const replicationPrototype = replicationWebRtcTestInternals.getReplicationStateClass().prototype;
for (const [direction, boundary] of [['pull', 'read'], ['push', 'read'], ['push', 'dirty'], ['push', 'write']]) {
  let release;
  let entered;
  const pending = new Promise((resolve) => { release = resolve; });
  const started = new Promise((resolve) => { entered = resolve; });
  let writes = 0;
  let checkpoints = 0;
  const pause = async (name, value) => {
    if (boundary === name) { entered(); await pending; }
    return value;
  };
  const batch = { documents: [{ id: 'cancelled-document' }], checkpoint: { sequence: 1 } };
  const state = {
    cancelled: false,
    pull: {}, push: {},
    pullCheckpointsByPeer: new Map(), pushCheckpointsByPeer: new Map(),
    peer: { request() { writes += 1; return pause('write', []); } },
    demandSidecar: { markDirty: () => pause('dirty', undefined) },
    collection: { name: 'cancel-test', schema: { primaryPath: 'id' }, storageCollection: {
      getChangedDocumentsSince: () => pause('read', batch),
      async bulkWrite() { writes += 1; },
    } },
    requestMasterChangesSince: async () => ({ peerId: 'native-test', result: await pause('read', batch) }),
    changedDocumentReadOptionsForPeer: () => ({}),
    recordLocalPushChangedSinceRead() {},
    replicationOriginForPeer: () => 'native-test',
    invalidateDemandCacheForRemoteWrite: async () => {},
    persistCheckpointsForPeer: async () => { checkpoints += 1; },
    requestTimeoutMsFor: () => 1000,
  };
  const run = replicationPrototype[direction === 'pull' ? 'pullFromPeer' : 'pushToPeer'].call(state, 'native-test');
  await started;
  state.cancelled = true;
  state.peer = null;
  release();
  await run;
  assert.equal(writes, boundary === 'write' ? 1 : 0, `${direction}/${boundary}: cancelled transfer issued a new write`);
  assert.equal(checkpoints, 0, `${direction}/${boundary}: cancelled transfer advanced its checkpoint`);
}
console.log('peer-ready and transfer cancelled boundaries smoke passed');
