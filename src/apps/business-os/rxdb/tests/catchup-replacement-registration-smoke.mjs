// Replacing a registration while its previous catch-up awaits negotiation
// must start the replacement, not leave an open room with no collection peer.
import assert from 'node:assert/strict';
import { replicationWebRtcTestInternals } from '../src/replication-webrtc.mjs';

const SharedRoomPeer = replicationWebRtcTestInternals.getSharedRoomPeerClass();
const shared = new SharedRoomPeer({
  key: 'catchup-replacement-test', signalingUrl: 'wss://signaling.invalid/',
  room: 'catchup-replacement-room', iceServers: [], expectedNativePeerId: 'native-test',
});
shared.collectionCatchUpQueueSliceMs = 60_000;
shared.isPeerOpen = () => true;
shared.collectCollectionSchemas = async () => ({ widgets: { schemaVersion: 1, schemaHash: 'widgets-hash' } });
let releaseOld;
let startedOld;
let startedFresh;
const oldPending = new Promise((resolve) => { releaseOld = resolve; });
const oldStarted = new Promise((resolve) => { startedOld = resolve; });
const freshStarted = new Promise((resolve) => { startedFresh = resolve; });
let calls = 0;
let oldEffects = 0;
let oldCancellations = 0;
shared.ensureNegotiatedPeer = async () => {
  if (++calls === 1) { startedOld(); await oldPending; }
  return { peerId: 'native-test', remoteProtocol: { collectionSchemas: {} }, queryFetchCapable: false };
};
const oldRegistration = {
  collection: 'widgets',
  state: { peerStates$: { getValue: () => new Map() }, onPeerReady() { oldEffects += 1; } },
};
oldRegistration.state.cancel = async () => {
  oldCancellations += 1;
  shared.unregister('widgets', oldRegistration.state);
};
const freshRegistration = {
  collection: 'widgets',
  state: { peerStates$: { getValue: () => new Map() }, onPeerReady() { startedFresh(); } },
};
shared.register('widgets', oldRegistration);
const oldRun = shared.collectionCatchUps.get('widgets');
await oldStarted;
shared.register('widgets', freshRegistration);
const freshRun = shared.collectionCatchUps.get('widgets');
let deadline;
try {
  assert.notEqual(freshRun, oldRun, 'replacement inherited the obsolete catch-up and will never acquire a collection peer');
  assert.equal(oldCancellations, 1, 'obsolete timers/loaders were not retired');
  assert.equal(shared.collections.get('widgets'), freshRegistration, 'obsolete cleanup removed the new registration');
  shared.register('widgets', freshRegistration);
  assert.equal(shared.refCount, 1, 'identical registration leaked a room reference');
  await Promise.race([
    freshStarted,
    new Promise((_, reject) => { deadline = setTimeout(() => reject(new Error('replacement stayed blocked behind obsolete negotiation')), 1000); }),
  ]);
  releaseOld();
  await oldRun;
  await freshRun;
  assert.equal(oldEffects, 0, 'obsolete registration acquired peer authority');
  assert.equal(shared.refCount, 1, 'replacing one collection leaked a room reference');
  assert.equal(shared.unregister('widgets', oldRegistration.state), false, 'late obsolete cleanup retained room authority');
  assert.equal(shared.collections.get('widgets'), freshRegistration);
} finally {
  clearTimeout(deadline);
  releaseOld();
  shared.unregister('widgets');
}
console.log('catch-up replacement registration smoke passed');
