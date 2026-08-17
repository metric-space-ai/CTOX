// REGRESSION: runtime-installed Business OS apps register module collections
// after the shell-critical shared WebRTC room is already open. The native
// handshake already advertises its complete schema map, so a late browser
// registration must reuse it. Renegotiating once per collection can recycle
// the single multiplexed DataChannel while foreground Browser traffic is live.

import { replicationWebRtcTestInternals } from '../src/replication-webrtc.mjs';

const SharedRoomPeer = replicationWebRtcTestInternals.getSharedRoomPeerClass();

const shared = new SharedRoomPeer({
  key: 'late-collection-test',
  signalingUrl: 'wss://signaling.invalid/?token=t&token_iat=1&token_exp=2',
  room: 'room-late-collection',
  iceServers: [],
  expectedNativePeerId: 'native-1',
});

shared.negotiated = {
  peerId: 'native-1',
  remoteProtocol: { marker: 'old-handshake' },
  queryFetchCapable: false,
};
shared.peerOpenQueue = Promise.resolve();
shared.isPeerOpen = () => true;
shared.openSharedPeerIds = () => ['native-1'];

let renegotiations = 0;
shared.negotiatePeer = async (peerId) => {
  renegotiations += 1;
  const negotiated = {
    peerId,
    remoteProtocol: { marker: 'unexpected-renegotiation' },
    queryFetchCapable: true,
  };
  shared.negotiated = negotiated;
  return negotiated;
};

shared.catchUpRegisteredCollection = async () => {
  await shared.ensureNegotiatedPeer();
};

shared.register('runtime_app_items', {
  collection: 'runtime_app_items',
  state: {},
});

const catchUp = shared.collectionCatchUps.get('runtime_app_items');
if (!catchUp) throw new Error('late collection catch-up was not scheduled');
await catchUp;

assert(renegotiations === 0, `late collection must reuse the live handshake, got ${renegotiations} renegotiations`);
assert(
  shared.negotiated?.remoteProtocol?.marker === 'old-handshake',
  'late collection replaced the authenticated room handshake',
);

console.log('ctox-rxdb late-collection handshake reuse smoke OK');

function assert(condition, message) {
  if (!condition) throw new Error(message);
}
