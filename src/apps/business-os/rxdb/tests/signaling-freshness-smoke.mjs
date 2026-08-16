// REGRESSION: signaling-connection freshness + transient identity rules.
//
// 1. Token re-stamp: token_iat/token_exp are re-derived on EVERY connect
//    attempt, preserving the TTL length.
// 2. Identity: deterministic options.clientId never changes. A bounded
//    init.yourPeerId is socket-scoped and drives signaling-only identity while
//    live, with prompt status updates on assignment and clear.
// 3. Backoff: reconnect delay resets on the joined acceptance broadcast, not
//    merely on socket-open.

import { createCtoxWebRtcNativePeer } from '../src/webrtc-native.mjs';

const STALE_IAT = 1_000_000;
const TTL_SECONDS = 24 * 60 * 60;
const NativeWebSocket = globalThis.WebSocket;

class FakeWebSocket {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;

  constructor(url) {
    this.url = String(url);
    this.readyState = FakeWebSocket.OPEN;
    this.sent = [];
  }

  send(payload) {
    this.sent.push(String(payload));
  }

  close() {
    if (this.readyState === FakeWebSocket.CLOSED) return;
    this.readyState = FakeWebSocket.CLOSED;
    this.onclose?.({});
  }
}

globalThis.WebSocket = FakeWebSocket;

const peer = createCtoxWebRtcNativePeer({
  signalingUrl: `wss://signaling.invalid/?room=x&token=secret-token&token_iat=${STALE_IAT}&token_exp=${STALE_IAT + TTL_SECONDS}`,
  room: 'ctox-business-os:test:abcdef',
  clientId: 'browser-test-client',
  role: 'browser',
});
const statuses = [];
peer.on('transport-status', (event) => statuses.push(event.detail || event));

try {
  assert(peer.getTransportStatus().localSignalingPeerId === null, 'peer id is absent before init');
  assert(peer.shouldInitiate('browser-z') === true, 'deterministic client id orders peers before init');

  // --- 1. token re-stamp on connect ---------------------------------------
  peer.connect();
  const url = new URL(peer.socket.url);
  const iat = Number(url.searchParams.get('token_iat'));
  const exp = Number(url.searchParams.get('token_exp'));
  const now = Math.floor(Date.now() / 1000);
  assert(Math.abs(iat - now) <= 60, `token_iat re-stamped to now (got ${iat}, now ${now})`);
  assert(exp - iat === TTL_SECONDS, `TTL length preserved (got ${exp - iat})`);
  assert(url.searchParams.get('token') === 'secret-token', 'token value itself unchanged');
  assert(url.searchParams.get('peerId') === 'browser-test-client', 'signaling URL keeps deterministic client id');

  // --- 2. transient server-assigned signaling identity --------------------
  peer.handleSignalingMessage(JSON.stringify({ type: 'ctoxPresence', peerId: 'remote-native-peer' }));
  assert(peer.options.clientId === 'browser-test-client', 'presence peerId must not rename this client');
  assert(peer.localSignalingPeerId === null, 'presence peerId must not assign local signaling identity');

  let emissionCount = statuses.length;
  peer.handleSignalingMessage(JSON.stringify({ type: 'init', yourPeerId: 'zz-server-assigned-id' }));
  assert(peer.options.clientId === 'browser-test-client', 'init must preserve deterministic options.clientId');
  assert(peer.localSignalingPeerId === 'zz-server-assigned-id', 'valid init assigns transient peer id');
  assert(peer.getTransportStatus().localSignalingPeerId === 'zz-server-assigned-id', 'ordinary transport status exposes assigned peer id');
  assert(statuses.length === emissionCount + 1, 'assignment emits a prompt transport status');
  assert(statuses.at(-1).localSignalingPeerId === 'zz-server-assigned-id', 'assignment status carries assigned peer id');
  assert(peer.shouldInitiate('browser-z') === false, 'assigned peer id drives initiator ordering');
  assert(peer.shouldConnectToRemotePeer('zz-server-assigned-id') === false, 'assigned peer id drives self comparison');
  assert(peer.sendSignal('remote-peer', { type: 'candidate' }) === true, 'signal sends over open socket');
  assert(lastSignal(peer).senderPeerId === 'zz-server-assigned-id', 'assigned peer id drives signaling sender');

  emissionCount = statuses.length;
  peer.handleSignalingMessage(JSON.stringify({ type: 'init', yourPeerId: 'zz-server-assigned-id' }));
  assert(peer.localSignalingPeerId === 'zz-server-assigned-id', 'duplicate init keeps assigned peer id');
  assert(statuses.length === emissionCount, 'duplicate init does not emit a redundant status');

  emissionCount = statuses.length;
  peer.handleSignalingMessage(JSON.stringify({ type: 'init', yourPeerId: 'aa-updated-id' }));
  assert(peer.localSignalingPeerId === 'aa-updated-id', 'updated init replaces transient peer id');
  assert(statuses.length === emissionCount + 1, 'updated init emits a prompt status');
  assert(peer.shouldInitiate('browser-z') === true, 'updated peer id immediately changes initiator ordering');
  peer.sendSignal('remote-peer', { type: 'candidate' });
  assert(lastSignal(peer).senderPeerId === 'aa-updated-id', 'updated peer id immediately changes signaling sender');

  emissionCount = statuses.length;
  peer.handleSignalingMessage(JSON.stringify({ type: 'init', yourPeerId: 'bad\npeer-id' }));
  assert(peer.localSignalingPeerId === null, 'control characters clear transient peer id');
  assert(statuses.length === emissionCount + 1, 'invalid init clear emits a prompt status');
  assert(peer.shouldInitiate('browser-z') === true, 'invalid init restores deterministic ordering fallback');
  peer.sendSignal('remote-peer', { type: 'candidate' });
  assert(lastSignal(peer).senderPeerId === 'browser-test-client', 'invalid init restores deterministic sender fallback');

  peer.handleSignalingMessage(JSON.stringify({ type: 'init', yourPeerId: 'valid-before-oversize' }));
  emissionCount = statuses.length;
  peer.handleSignalingMessage(JSON.stringify({ type: 'init', yourPeerId: 'x'.repeat(257) }));
  assert(peer.localSignalingPeerId === null, 'oversized init clears transient peer id');
  assert(statuses.length === emissionCount + 1, 'oversized init clear emits a prompt status');

  // --- 3. joined acceptance resets backoff --------------------------------
  peer.signalingReconnectDelayMs = 30_000;
  peer.handleSignalingMessage(JSON.stringify({ type: 'joined', otherPeerIds: [] }));
  assert(
    peer.signalingReconnectDelayMs < 30_000,
    `joined broadcast resets the reconnect backoff (still ${peer.signalingReconnectDelayMs})`,
  );

  // --- 4. socket and peer close clear the transient identity ---------------
  peer.handleSignalingMessage(JSON.stringify({ type: 'init', yourPeerId: 'socket-live-id' }));
  emissionCount = statuses.length;
  peer.socket.close();
  assert(peer.localSignalingPeerId === null, 'WebSocket close clears transient peer id');
  assert(statuses.length === emissionCount + 1, 'WebSocket close emits a prompt clear status');
  assert(statuses.at(-1).localSignalingPeerId === null, 'WebSocket close status exposes absence');

  peer.handleSignalingMessage(JSON.stringify({ type: 'init', yourPeerId: 'peer-close-id' }));
  emissionCount = statuses.length;
  peer.close();
  assert(peer.localSignalingPeerId === null, 'native peer close clears transient peer id');
  assert(statuses.length === emissionCount + 1, 'native peer close emits a prompt clear status');
  assert(statuses.at(-1).localSignalingPeerId === null, 'native peer close status exposes absence');
  assert(peer.options.clientId === 'browser-test-client', 'deterministic client id remains unchanged for the full lifecycle');
} finally {
  peer.close();
  globalThis.WebSocket = NativeWebSocket;
}

console.log('ctox-rxdb signaling freshness smoke OK');
process.exit(0);

function lastSignal(value) {
  return JSON.parse(value.socket.sent.at(-1));
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}
