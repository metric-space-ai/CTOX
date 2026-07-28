// REGRESSION: unregistering a collection must invalidate an in-flight shared
// catch-up. Otherwise a later registration inherits the stale Promise and the
// old generation can activate/persist after its collection has been stopped.

import { replicationWebRtcTestInternals } from '../src/replication-webrtc.mjs';

const SharedRoomPeer = replicationWebRtcTestInternals.getSharedRoomPeerClass();
const shared = new SharedRoomPeer({
  key: 'catchup-unregister-invalidation-test',
  signalingUrl: 'wss://signaling.invalid/',
  room: 'catchup-unregister-invalidation-room',
  iceServers: [],
  expectedNativePeerId: 'native-test',
});

// Make queue release depend on invalidation, not the ordinary pacing timeout.
shared.collectionCatchUpQueueSliceMs = 60_000;
shared.peerOpenQueue = Promise.resolve();
shared.isPeerOpen = () => true;
shared.collectCollectionSchemas = async () => ({
  widgets: { schemaVersion: 1, schemaHash: 'widgets-hash' },
});

const oldNegotiation = deferred();
const oldNegotiationStarted = deferred();
const freshReadyStarted = deferred();
const releaseFreshReady = deferred();
let negotiationCalls = 0;
let oldEffects = 0;
let freshEffects = 0;
let staleErrors = 0;

shared.ensureNegotiatedPeer = async () => {
  negotiationCalls += 1;
  if (negotiationCalls === 1) {
    oldNegotiationStarted.resolve(true);
    await oldNegotiation.promise;
  }
  return {
    peerId: 'native-test',
    remoteProtocol: { collectionSchemas: {} },
    queryFetchCapable: false,
  };
};

const oldRegistration = {
  collection: 'widgets',
  state: {
    peerStates$: { getValue: () => new Map() },
    onPeerReady() { oldEffects += 1; },
    emitError() { staleErrors += 1; },
  },
};
shared.register('widgets', oldRegistration);
const oldRun = shared.collectionCatchUps.get('widgets');
assert(oldRun, 'initial catch-up was not scheduled');
await oldNegotiationStarted.promise;

shared.unregister('widgets');
assert(!shared.collectionCatchUps.has('widgets'), 'unregister did not remove the collectionCatchUps entry');
assert(oldRun.cancelled === true, 'unregister did not observably mark the in-flight catch-up stale');
assert(oldRun.isCurrent() === false, 'invalidated catch-up still reports itself as current');

const freshRegistration = {
  collection: 'widgets',
  state: {
    peerStates$: { getValue: () => new Map() },
    async onPeerReady() {
      freshReadyStarted.resolve(true);
      await releaseFreshReady.promise;
      freshEffects += 1;
    },
    emitError(error) { throw error; },
  },
};
shared.register('widgets', freshRegistration);
const freshRun = shared.collectionCatchUps.get('widgets');
assert(freshRun, 're-register did not schedule a fresh catch-up');
assert(freshRun !== oldRun, 're-register reused the invalidated catch-up Promise');
assert(freshRun.generation !== oldRun.generation, 're-register reused the invalidated catch-up generation');
await withTimeout(freshReadyStarted.promise, 500, 'fresh catch-up did not start after invalidation');
assert(negotiationCalls === 2, 'expected a fresh negotiation call, got ' + negotiationCalls);

// Let the old asynchronous boundary finish while the fresh generation is still
// in flight. The old generation must neither activate its state nor delete the
// fresh generation's deduplication entry from its finally handler.
oldNegotiation.resolve(true);
await oldRun;
assert(oldEffects === 0, 'stale catch-up invoked onPeerReady after unregister');
assert(staleErrors === 0, 'stale catch-up emitted an error after unregister');
assert(
  shared.collectionCatchUps.get('widgets') === freshRun,
  'stale catch-up completion removed the fresh generation map entry',
);

releaseFreshReady.resolve(true);
await freshRun;
assert(freshEffects === 1, 'fresh catch-up effect count was ' + freshEffects + ', expected 1');
assert(!shared.collectionCatchUps.has('widgets'), 'completed fresh catch-up left a stale map entry');

console.log('catch-up unregister invalidation smoke passed');

function deferred() {
  let resolve;
  const promise = new Promise((promiseResolve) => { resolve = promiseResolve; });
  return { promise, resolve };
}

async function withTimeout(promise, timeoutMs, message) {
  let timer;
  try {
    return await Promise.race([
      promise,
      new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error(message)), timeoutMs);
      }),
    ]);
  } finally {
    clearTimeout(timer);
  }
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}
