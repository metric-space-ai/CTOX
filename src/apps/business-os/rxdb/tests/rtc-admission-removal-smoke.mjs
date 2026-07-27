// Regression guard: creating a new room connection must never evict or close
// an existing RTCPeerConnection. The retired global pool used to preempt the
// oldest optional room when the 65th connection arrived.

globalThis.window = {};
globalThis.document = {};

const createdPeerConnections = [];
globalThis.RTCPeerConnection = class FakeRTCPeerConnection {
  constructor() {
    this.connectionState = 'new';
    this.iceConnectionState = 'new';
    this.iceGatheringState = 'new';
    this.signalingState = 'stable';
    this.localDescription = null;
    this.remoteDescription = null;
    this.closeCalls = 0;
    createdPeerConnections.push(this);
  }

  createDataChannel() {
    return {
      readyState: 'connecting',
      send() {},
      close() {},
      addEventListener() {},
      removeEventListener() {},
    };
  }

  async createOffer() {
    return { type: 'offer', sdp: 'fake-offer' };
  }

  async setLocalDescription(description) {
    this.localDescription = description;
    this.signalingState = 'have-local-offer';
  }

  close() {
    this.closeCalls += 1;
    this.connectionState = 'closed';
    this.signalingState = 'closed';
  }
};

const { createCtoxWebRtcNativePeer } = await import('../dist/ctox-rxdb-js.mjs');
const joined = JSON.stringify({
  type: 'joined',
  peers: [{ peerId: 'ctox-core-admission-test', role: 'ctox_instance' }],
});

const peers = [];
const LEGACY_POOL_CAP_PLUS_ONE = 65;
for (let index = 0; index < LEGACY_POOL_CAP_PLUS_ONE; index += 1) {
  const peer = createCtoxWebRtcNativePeer({
    signalingUrl: 'wss://signaling.invalid',
    room: `ctox-business-os:instance:secret:admission-${index}`,
    clientId: `browser-admission-${index}`,
  });
  peers.push(peer);
  peer.handleSignalingMessage(joined);

  assert(
    peer.connections.has('ctox-core-admission-test'),
    `connection ${index + 1} must be granted synchronously`,
  );
  assert(
    createdPeerConnections.slice(0, -1).every((connection) => connection.closeCalls === 0),
    `connection ${index + 1} must not close any existing RTCPeerConnection`,
  );
}

assertEqual(
  createdPeerConnections.length,
  LEGACY_POOL_CAP_PLUS_ONE,
  'every room must receive its own RTCPeerConnection without a global cap',
);
assert(
  createdPeerConnections.every((connection) => connection.closeCalls === 0),
  'no existing RTCPeerConnection may be evicted when a new room arrives',
);

for (const peer of peers) peer.close();
console.log('ctox-rxdb-js RTC admission removal smoke OK');

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function assertEqual(actual, expected, message) {
  if (actual !== expected) {
    throw new Error(`${message}: expected ${expected}, got ${actual}`);
  }
}
