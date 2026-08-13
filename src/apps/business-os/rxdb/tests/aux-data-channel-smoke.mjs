// Regression guard for the auxiliary DataChannel extension point.
//
// Before this existed, attachChannel() wrote EVERY incoming channel into the
// single `connection.channel` slot that 25 replication call sites read. A
// consumer opening a second channel therefore did not get an extra channel —
// it silently destroyed replication. This guard pins the three properties the
// consumer depends on:
//
//   1. an auxiliary channel never touches connection.channel,
//   2. a registration survives a reconnect (the channel is re-opened and the
//      'aux-channel' event fires again — a silently dead channel is the exact
//      defect shape this codebase paid for four times in one night),
//   3. closeAuxChannel() actually unregisters, so a reconnect does NOT
//      resurrect a channel the consumer gave up.

globalThis.window = {};
globalThis.document = {};

const createdChannels = [];

class FakeDataChannel {
  constructor(label) {
    this.label = label;
    this.readyState = 'connecting';
    this.closeCalls = 0;
    createdChannels.push(this);
  }
  send() {}
  close() {
    this.closeCalls += 1;
    this.readyState = 'closed';
    this.onclose?.();
  }
  addEventListener() {}
  removeEventListener() {}
}

globalThis.RTCPeerConnection = class FakeRTCPeerConnection {
  constructor() {
    this.connectionState = 'new';
    this.iceConnectionState = 'new';
    this.iceGatheringState = 'new';
    this.signalingState = 'stable';
    this.localDescription = null;
    this.remoteDescription = null;
  }
  createDataChannel(label) {
    return new FakeDataChannel(label);
  }
  async createOffer() {
    return { type: 'offer', sdp: 'fake-offer' };
  }
  async setLocalDescription(description) {
    this.localDescription = description;
    this.signalingState = 'have-local-offer';
  }
  close() {
    this.connectionState = 'closed';
    this.signalingState = 'closed';
  }
};

const { createCtoxWebRtcNativePeer } = await import('../dist/ctox-rxdb-js.mjs');

const REMOTE = 'ctox-core-aux-test';
const joined = JSON.stringify({
  type: 'joined',
  peers: [{ peerId: REMOTE, role: 'ctox_instance' }],
});

const peer = createCtoxWebRtcNativePeer({
  signalingUrl: 'wss://signaling.invalid',
  room: 'ctox-business-os:instance:secret:aux',
  clientId: 'browser-aux',
});

const auxEvents = [];
// The emitter dispatches CustomEvents — the payload lives under .detail.
peer.on('aux-channel', (event) => auxEvents.push(event.detail));

peer.handleSignalingMessage(joined);
const connection = peer.connections.get(REMOTE);
assert(connection, 'the peer must hold a connection after joining');

const replicationChannel = connection.channel;
assert(replicationChannel, 'the replication channel must be created by the initiator');
assertEqual(replicationChannel.label, 'ctox-rxdb', 'the replication channel keeps its label');

// 1. An INCOMING auxiliary channel must not take the replication slot.
peer.attachChannel(connection, new FakeDataChannel('ctox-frames-inbound'));
assertEqual(
  connection.channel,
  replicationChannel,
  'an incoming auxiliary channel must never replace the replication channel',
);
assertEqual(auxEvents.length, 1, 'an incoming auxiliary channel must be announced');
assertEqual(auxEvents[0].label, 'ctox-frames-inbound', 'the event carries the channel label');
assertEqual(auxEvents[0].peerId, REMOTE, 'the event carries the peer id');

// An OUTGOING auxiliary channel must not either.
const opened = peer.openAuxChannel(REMOTE, 'ctox-frames', { ordered: false });
assert(opened, 'openAuxChannel must return the live channel for a connected peer');
assertEqual(opened.label, 'ctox-frames', 'the requested label is used verbatim');
assertEqual(
  connection.channel,
  replicationChannel,
  'an outgoing auxiliary channel must never replace the replication channel',
);
assertEqual(auxEvents.length, 2, 'an outgoing auxiliary channel is announced too');

// The replication label is reserved — taking it would be the original defect.
let rejected = false;
try {
  peer.openAuxChannel(REMOTE, 'ctox-rxdb');
} catch {
  rejected = true;
}
assert(rejected, 'the replication label must be rejected as an auxiliary label');

// 2. A reconnect must re-establish the registered auxiliary channel.
peer.removeConnection(REMOTE, 'test-reconnect', null, { reconnect: false });
assert(opened.closeCalls > 0, 'tearing down a connection must close its auxiliary channels');

peer.handleSignalingMessage(joined);
const reconnected = peer.connections.get(REMOTE);
assert(reconnected && reconnected !== connection, 'the reconnect must build a fresh connection');
assertEqual(
  reconnected.channel.label,
  'ctox-rxdb',
  'the reconnect rebuilds the replication channel',
);

const reopened = reconnected.auxChannels.get('ctox-frames');
assert(reopened, 'a registered auxiliary channel must be re-opened after a reconnect');
assert(reopened !== opened, 'the re-opened channel is a new channel, not the dead one');
assertEqual(auxEvents.length, 3, "the 'aux-channel' event must fire again after a reconnect");
assertEqual(auxEvents[2].label, 'ctox-frames', 'the reconnect event carries the same label');

// The unregistered INBOUND channel must not be resurrected — only registrations are standing.
assert(
  !reconnected.auxChannels.has('ctox-frames-inbound'),
  'an inbound-only channel is not a registration and must not be re-opened',
);

// 3. closeAuxChannel unregisters for good.
peer.closeAuxChannel('ctox-frames');
assert(reopened.closeCalls > 0, 'closeAuxChannel must close the live channel');
peer.removeConnection(REMOTE, 'test-reconnect-2', null, { reconnect: false });
peer.handleSignalingMessage(joined);
const afterClose = peer.connections.get(REMOTE);
assert(
  !afterClose.auxChannels.has('ctox-frames'),
  'a closed registration must not be resurrected by a later reconnect',
);

peer.close();
console.log('ctox-rxdb-js auxiliary data channel smoke OK');

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function assertEqual(actual, expected, message) {
  if (actual !== expected) {
    throw new Error(`${message}: expected ${String(expected)}, got ${String(actual)}`);
  }
}
