// Pins the Phase-3 multiplex admission contract + the shell-critical set.
//
// HISTORY: before Phase 3 every collection opened its own RTCPeerConnection
// and a per-collection admission gate held optional collections back until
// the shell-critical DataChannels were open. Phase 3 multiplexes EVERY
// collection of a sync room over ONE RTCPeerConnection, so the gate is
// intentionally retired (rtcPeerConnectionPriority always returns 0). This
// test used to pin the retired gate and was red for weeks — which trained
// agents to ignore failing tests. It now pins the CURRENT contract:
//
//   1. SHELL_CRITICAL_COLLECTIONS is the single source of truth app.js
//      derives from; its membership changing silently is a drift bug.
//   2. No admission machinery at all (SYNC-A-B4): every room's connection is
//      created immediately — no pool, no queue, no cap, and a new incoming
//      connection NEVER closes an existing one.
//
// If you (the agent reading this) change either contract, change it in
// src/webrtc-native.mjs FIRST, on purpose, and update this pin in the same
// commit — never by deleting assertions to make the suite pass.

globalThis.window = {};
globalThis.document = {};
let fakeConnectionsCreated = 0;
let fakeConnectionsClosed = 0;
globalThis.RTCPeerConnection = class FakeRTCPeerConnection {
  constructor() {
    fakeConnectionsCreated += 1;
    this.connectionState = 'new';
    this.iceConnectionState = 'new';
    this.iceGatheringState = 'new';
    this.signalingState = 'stable';
    this.localDescription = null;
    this.remoteDescription = null;
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
    if (this.connectionState !== 'closed') fakeConnectionsClosed += 1;
    this.connectionState = 'closed';
    this.signalingState = 'closed';
  }
};

const { createCtoxWebRtcNativePeer, SHELL_CRITICAL_COLLECTIONS } = await import('../dist/ctox-rxdb-js.mjs');

// --- 1. shell-critical set drift guard ---------------------------------------
const EXPECTED_SHELL_CRITICAL = [
  'ctox_runtime_settings',
  'business_module_catalog',
  'business_commands',
  'ctox_queue_tasks',
  'browser_sessions',
  'browser_tabs',
  'browser_frames',
  'browser_input_events',
];
const actualCritical = [...SHELL_CRITICAL_COLLECTIONS].sort();
const expectedCritical = [...EXPECTED_SHELL_CRITICAL].sort();
assertEqual(
  JSON.stringify(actualCritical),
  JSON.stringify(expectedCritical),
  'SHELL_CRITICAL_COLLECTIONS membership changed — update this pin (and app.js consumers) deliberately',
);

// --- 2. no admission machinery: immediate creation, never eviction -----------
// 70 rooms deliberately exceeds the retired 64-connection pool cap: with the
// admission machinery gone, every room connects immediately and no incoming
// connection may close an existing one (SYNC-A-B4 eviction regression).
const joined = JSON.stringify({
  type: 'joined',
  peers: [{ peerId: 'ctox-core-test', role: 'ctox_instance' }],
});

const ROOM_COUNT = 70;
const peers = Array.from({ length: ROOM_COUNT }, (_, index) => createPeer(`room-${index}`, `browser-${index}`));
for (const peer of peers) peer.handleSignalingMessage(joined);
await delay(10);

assertEqual(
  fakeConnectionsCreated >= ROOM_COUNT,
  true,
  `all ${ROOM_COUNT} rooms must create their RTCPeerConnection immediately (created: ${fakeConnectionsCreated})`,
);
assertEqual(fakeConnectionsClosed, 0, 'no incoming connection may evict/close an existing RTCPeerConnection');
assertEqual(
  peers[0].getTransportStatus().rtcConnectionPool,
  undefined,
  'transport status must no longer report admission-pool counters',
);

for (const peer of peers) peer.close();
assertEqual(
  fakeConnectionsClosed >= ROOM_COUNT,
  true,
  'explicit peer.close() must still close the underlying RTCPeerConnection',
);
console.log('ctox-rxdb-js rtc critical pool smoke OK (post-admission contract)');

function createPeer(room, clientId) {
  return createCtoxWebRtcNativePeer({
    signalingUrl: 'wss://signaling.invalid',
    room: `ctox-business-os:instance:secret:${room}`,
    clientId,
  });
}

function assertEqual(actual, expected, message) {
  if (actual !== expected) {
    throw new Error(`${message}: expected ${expected}, got ${actual}`);
  }
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
