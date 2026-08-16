// REGRESSION: runtime-installed Business OS apps register module collections
// after the shell-critical shared WebRTC room is already authenticated and
// open. The native handshake already advertises its complete collection schema
// map, so late registration must catch up through that cached handshake without
// renegotiating the room or recycling its multiplexed DataChannel.

import { replicationWebRtcTestInternals } from '../src/replication-webrtc.mjs';

const SharedRoomPeer = replicationWebRtcTestInternals.getSharedRoomPeerClass();
const collectionName = 'runtime_app_items';
const schemaHash = 'runtime-app-items-schema-hash';

const shared = new SharedRoomPeer({
  key: 'late-collection-test',
  signalingUrl: 'wss://signaling.invalid/?token=t&token_iat=1&token_exp=2',
  room: 'room-late-collection',
  iceServers: [],
  expectedNativePeerId: 'native-1',
});

const cachedHandshake = {
  peerId: 'native-1',
  remoteProtocol: {
    marker: 'cached-live-handshake',
    collectionSchemas: {
      [collectionName]: { schemaVersion: 1, schemaHash },
    },
  },
  queryFetchCapable: true,
};
shared.negotiated = cachedHandshake;
shared.peerOpenQueue = Promise.resolve();
shared.isPeerOpen = () => true;
shared.openSharedPeerIds = () => ['native-1'];

let renegotiations = 0;
shared.negotiatePeer = async () => {
  renegotiations += 1;
  throw new Error('late collection registration must not renegotiate a live handshake');
};

let peerReadyCalls = 0;
let peerReadyProtocol = null;
const registration = {
  collection: collectionName,
  state: {
    schemaHashValue: schemaHash,
    collection: {
      schema: {
        version: 1,
        async hash() { return schemaHash; },
      },
    },
    peerStates$: { getValue: () => new Map() },
    async onPeerReady(peerId, remoteProtocol, queryFetchCapable) {
      assert(peerId === 'native-1', 'late collection catch-up used the wrong peer');
      assert(queryFetchCapable === true, 'late collection lost cached capabilities');
      peerReadyCalls += 1;
      peerReadyProtocol = remoteProtocol;
    },
  },
};

shared.register(collectionName, registration);

const catchUp = shared.collectionCatchUps.get(collectionName);
if (!catchUp) throw new Error('late collection catch-up was not scheduled');
await catchUp;

assert(renegotiations === 0, `late collection must not renegotiate, got ${renegotiations}`);
assert(shared.negotiated === cachedHandshake, 'late collection replaced the cached live handshake');
assert(peerReadyCalls === 1, `late collection catch-up ran ${peerReadyCalls} times`);
assert(peerReadyProtocol?.marker === 'cached-live-handshake', 'late collection did not use cached protocol');
assert(peerReadyProtocol?.collection?.name === collectionName, 'late collection protocol was not collection-scoped');

console.log('ctox-rxdb late-collection renegotiation smoke OK');

function assert(condition, message) {
  if (!condition) throw new Error(message);
}
