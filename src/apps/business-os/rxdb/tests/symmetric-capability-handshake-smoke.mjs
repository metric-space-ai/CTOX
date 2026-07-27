import { replicationWebRtcTestInternals } from '../src/replication-webrtc.mjs';
import { CtoxWebRtcNativePeer } from '../src/webrtc-native.mjs';

const SharedRoomPeer = replicationWebRtcTestInternals.getSharedRoomPeerClass();
const shared = new SharedRoomPeer({
  key: 'symmetric-capability-test',
  signalingUrl: 'wss://signaling.invalid',
  room: 'room-symmetric-capability',
  iceServers: [],
  expectedNativePeerId: 'native-1',
});

let observedTimeoutMs = 0;
shared.peer = {
  async waitForRequest(peerId, method, timeoutMs) {
    assert(peerId === 'native-1', 'remote master readiness uses the negotiated peer');
    assert(method === 'token', 'remote master readiness waits for native token request');
    observedTimeoutMs = timeoutMs;
  },
};
await shared.awaitRemoteMasterReady('native-1');
assert(
  observedTimeoutMs >= 10_000,
  `symmetric capability handshake must tolerate a busy native peer, got ${observedTimeoutMs}ms`,
);

const handshakeError = new Error('native symmetric handshake missing');
shared.peer = {
  async waitForRequest() {
    throw handshakeError;
  },
};
await assertRejects(
  shared.awaitRemoteMasterReady('native-1'),
  handshakeError,
  'missing native authorization handshake must fail closed',
);

const peer = new CtoxWebRtcNativePeer({
  signalingUrl: 'wss://signaling.invalid',
  room: 'room-observed-request-reset',
});
peer.observedRequests.set('native-1|token', Date.now());
assert(peer.hasObservedRequest('native-1', 'token'), 'precondition: token request is observed');
peer.removeConnection('native-1', 'reconnect', null, { reconnect: false });
assert(
  !peer.hasObservedRequest('native-1', 'token'),
  'a reconnect must not reuse an earlier connection token observation',
);

console.log('ctox-rxdb symmetric capability handshake smoke OK');

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

async function assertRejects(promise, expected, message) {
  try {
    await promise;
  } catch (error) {
    assert(error === expected, `${message}: rejected with an unexpected error`);
    return;
  }
  throw new Error(`${message}: promise resolved`);
}
