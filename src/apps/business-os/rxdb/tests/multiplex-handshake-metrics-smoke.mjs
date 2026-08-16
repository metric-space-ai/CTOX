// The Business OS room is one physical WebRTC peer shared by every collection.
// Keep the instrumentation that makes accidental per-collection channels and
// handshakes visible to the boot acceptance harness.

import { replicationWebRtcTestInternals } from '../src/replication-webrtc.mjs';

const SharedRoomPeer = replicationWebRtcTestInternals.getSharedRoomPeerClass();
const shared = new SharedRoomPeer({
  key: 'multiplex-handshake-metrics',
  signalingUrl: 'wss://signaling.invalid',
  room: 'one-room',
  iceServers: [],
  expectedNativePeerId: 'native-1',
});

// Keep catch-up tasks queued: this test is about room registration, not I/O.
shared.peerOpenQueue = new Promise(() => {});
for (let index = 0; index < 25; index += 1) {
  const name = `collection_${index}`;
  shared.register(name, { collection: { name }, state: {} });
}
shared.peer = {
  connections: new Map([
    ['native-1', { channel: { readyState: 'open' }, peer: { connectionState: 'connected' } }],
  ]),
  getTransportStatus: () => ({ transport: 'webrtc' }),
};

const status = shared.getTransportStatus();
const metrics = status.multiplexHandshake;
assert(metrics.registeredCollections === 25, 'all collections are registered on the shared room');
assert(metrics.collectionRegistrations === 25, 'unique registrations are counted');
assert(metrics.openDataChannels === 1, 'one native peer means one open data channel');
assert(metrics.peakOpenDataChannels === 1, 'peak data-channel count is retained');
assert(metrics.protocolNegotiations === 0, 'collection registration itself performs no protocol roundtrip');

console.log('ctox-rxdb multiplex handshake metrics smoke OK', metrics);

function assert(condition, message) {
  if (!condition) throw new Error(message);
}
