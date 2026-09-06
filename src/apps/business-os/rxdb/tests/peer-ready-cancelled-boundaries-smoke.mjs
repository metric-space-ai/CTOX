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
console.log('peer-ready cancelled boundaries smoke passed');
