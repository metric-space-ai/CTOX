// SYNC-A A0.5 + A0.11: native WebRTC inbound transport hygiene.
// Proves per-connection frame serialization, queue recovery after one frame
// fails, aggregate incoming-transfer admission budgets, and stale signaling
// control-plane error cleanup after a successful join.

import {
  CtoxWebRtcNativePeer,
  webrtcNativeTestInternals,
} from '../src/webrtc-native.mjs';
import { MAX_TRANSFER_BYTES } from '../src/frame-contract.generated.mjs';

const {
  MAX_INCOMING_FRAME_TRANSFERS,
  MAX_INCOMING_FRAME_BUFFERED_BYTES,
} = webrtcNativeTestInternals;

function newPeer() {
  return new CtoxWebRtcNativePeer({
    signalingUrl: 'ws://localhost:0/ignored',
    room: 'ctox-business-os:test:native-frame-ordering',
  });
}

function attachFakeChannel(peer, peerId) {
  const connection = {
    remotePeerId: peerId,
    channel: null,
    lastError: null,
    inboundFrameChain: Promise.resolve(),
    inboundFrameGeneration: 0,
  };
  const channel = {
    readyState: 'open',
    sent: [],
    send(value) { this.sent.push(value); },
    close() { this.readyState = 'closed'; },
  };
  peer.connections.set(peerId, connection);
  peer.attachChannel(connection, channel);
  return { connection, channel };
}

function receive(channel, payload) {
  channel.onmessage({ data: JSON.stringify(payload) });
}

// Frame N+1 must not enter its async handler before frame N has completed.
{
  const peer = newPeer();
  const { connection, channel } = attachFakeChannel(peer, 'native-order');
  const completed = [];
  peer.handleDataChannelFrame = async (_peerId, payload) => {
    if (payload.frame === 1) await delay(30);
    completed.push(payload.frame);
  };

  receive(channel, { frame: 1 });
  receive(channel, { frame: 2 });
  await connection.inboundFrameChain;

  assertDeepEqual(completed, [1, 2], 'slow frame 1 completes before frame 2');
  peer.removeConnection('native-order', 'peer-close');
  assert(connection.inboundFrameChain === null, 'connection teardown clears the inbound frame chain');
}

// A rejected handler is observable but leaves the queue fulfilled so frame 3
// still runs after frame 2 fails.
{
  const peer = newPeer();
  const errors = [];
  peer.on('error', (event) => errors.push(event.detail));
  const { connection, channel } = attachFakeChannel(peer, 'native-error');
  const completed = [];
  peer.handleDataChannelFrame = async (_peerId, payload) => {
    if (payload.frame === 2) throw new Error('intentional frame failure');
    completed.push(payload.frame);
  };

  receive(channel, { frame: 2 });
  receive(channel, { frame: 3 });
  await connection.inboundFrameChain;

  assertDeepEqual(completed, [3], 'frame 3 runs after frame 2 rejects');
  assert(
    errors.some((error) => error.code === 'ctox_webrtc_inbound_frame_failed'
      && error.message === 'intentional frame failure'),
    'handler failure is emitted as an observable inbound-frame error',
  );
  peer.close();
}

// Aggregate admission reserves declared transfer bytes at `start`: normal
// transfers remain accepted, while both byte and transfer-count excesses are
// rejected with diagnostics instead of being silently dropped.
{
  const peer = newPeer();
  const errors = [];
  peer.on('error', (event) => errors.push(event.detail));

  const start = (transferId, totalBytes) => peer.handleTransportFrame('native-budget', {
    ctoxFrame: 'ctox-rxdb-frame-v1',
    kind: 'start',
    transferId,
    totalFrames: 1,
    totalBytes,
  });

  const fullTransfers = MAX_INCOMING_FRAME_BUFFERED_BYTES / MAX_TRANSFER_BYTES;
  assert(Number.isInteger(fullTransfers) && fullTransfers > 0, 'byte budget is transfer-size aligned');
  for (let index = 0; index < fullTransfers; index += 1) {
    await start(`within-bytes-${index}`, MAX_TRANSFER_BYTES);
  }
  assert(
    peer.incomingFrames.size === fullTransfers,
    'transfers at the aggregate byte limit are admitted unchanged',
  );
  await start('over-byte-budget', 1);
  assert(!peer.incomingFrames.has('over-byte-budget'), 'start over aggregate byte budget is rejected');
  assert(
    errors.some((error) => error.code === 'ctox_webrtc_incoming_transfer_budget_exceeded'
      && error.transferId === 'over-byte-budget'
      && error.maxBufferedBytes === MAX_INCOMING_FRAME_BUFFERED_BYTES),
    'byte-budget rejection includes observable diagnostics',
  );

  peer.incomingFrames.clear();
  errors.length = 0;
  for (let index = 0; index < MAX_INCOMING_FRAME_TRANSFERS; index += 1) {
    await start(`within-count-${index}`, 1);
  }
  assert(
    peer.incomingFrames.size === MAX_INCOMING_FRAME_TRANSFERS,
    'transfers at the aggregate count limit are admitted unchanged',
  );
  await start('over-count-budget', 1);
  assert(!peer.incomingFrames.has('over-count-budget'), 'start over aggregate transfer count is rejected');
  assert(
    errors.some((error) => error.code === 'ctox_webrtc_incoming_transfer_budget_exceeded'
      && error.transferId === 'over-count-budget'
      && error.maxTransfers === MAX_INCOMING_FRAME_TRANSFERS),
    'transfer-count rejection includes observable diagnostics',
  );
  peer.close();
}

// A later `joined` proves successful admission and must clear an old
// control-plane rejection before the socket's generic error callback runs.
{
  const originalWebSocket = globalThis.WebSocket;
  class FakeWebSocket {
    static OPEN = 1;
    constructor(url) {
      this.url = url;
      this.readyState = FakeWebSocket.OPEN;
    }
    send() {}
    close() { this.readyState = 3; }
  }
  globalThis.WebSocket = FakeWebSocket;
  try {
    const peer = newPeer();
    const errors = [];
    peer.on('error', (event) => errors.push(event.detail));
    peer.connect();
    peer.handleSignalingMessage(JSON.stringify({
      type: 'ctoxError',
      scope: 'control-plane',
      code: 'temporary_unavailable',
      reason: 'old admission failure',
    }));
    assert(peer.lastControlPlaneError?.code === 'temporary_unavailable', 'control-plane error is retained before recovery');

    peer.handleSignalingMessage(JSON.stringify({ type: 'joined', peers: [] }));
    assert(peer.lastControlPlaneError === null, 'joined clears the stale control-plane error');
    peer.socket.onerror();
    assert(
      errors.at(-1)?.code === 'ctox_signaling_socket_error',
      'post-join socket error is generic instead of reusing the old control-plane classification',
    );
    peer.close();
  } finally {
    globalThis.WebSocket = originalWebSocket;
  }
}

console.log('ctox-rxdb-js native frame ordering smoke OK');

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function assertDeepEqual(actual, expected, message) {
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    throw new Error(`${message}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
  }
}
