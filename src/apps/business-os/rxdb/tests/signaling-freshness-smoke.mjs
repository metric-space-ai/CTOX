// REGRESSION: signaling-connection freshness + identity rules.
//
// 1. Token re-stamp: token_iat/token_exp are re-derived on EVERY connect
//    attempt, preserving the TTL length. They used to be baked into the URL
//    once at page load — a tab older than the TTL (24h) then reconnect-looped
//    forever against "control plane token expired" rejections.
// 2. Identity: init.yourPeerId is tracked as a bounded ephemeral signaling
//    identity, separately from the deterministic client id. Unrelated peerId
//    fields cannot overwrite it, and close transitions clear it promptly.
// 3. Backoff: the reconnect backoff resets on the `joined` broadcast (proof
//    the server ACCEPTED the join), not on socket-open — an open-then-rejected
//    socket must keep backing off instead of hammering at 1s.

import { createCtoxWebRtcNativePeer } from '../src/webrtc-native.mjs';

const STALE_IAT = 1_000_000; // ancient: must be re-stamped on connect
const TTL_SECONDS = 24 * 60 * 60;

const peer = createCtoxWebRtcNativePeer({
  signalingUrl: `wss://signaling.invalid/?room=x&token=secret-token&token_iat=${STALE_IAT}&token_exp=${STALE_IAT + TTL_SECONDS}`,
  room: 'ctox-business-os:test:abcdef',
  clientId: 'browser-test-client',
  role: 'browser',
});
const transportStatuses = [];
peer.on('transport-status', (event) => transportStatuses.push(event.detail || event));

try {
  // --- 1. token re-stamp on connect ---------------------------------------
  peer.connect();
  const url = new URL(peer.socket.url);
  const iat = Number(url.searchParams.get('token_iat'));
  const exp = Number(url.searchParams.get('token_exp'));
  const now = Math.floor(Date.now() / 1000);
  assert(Math.abs(iat - now) <= 60, `token_iat re-stamped to now (got ${iat}, now ${now})`);
  assert(exp - iat === TTL_SECONDS, `TTL length preserved (got ${exp - iat})`);
  assert(url.searchParams.get('token') === 'secret-token', 'token value itself unchanged');

  // --- 2. server signaling id is separate, bounded, and live --------------
  assert(
    peer.getTransportStatus().localSignalingPeerId == null,
    'transport status is empty before init.yourPeerId',
  );
  peer.handleSignalingMessage(JSON.stringify({ type: 'ctoxPresence', peerId: 'remote-native-peer' }));
  assert(peer.options.clientId === 'browser-test-client', 'message.peerId must NOT rename this client');
  assert(
    peer.getTransportStatus().localSignalingPeerId == null,
    'unrelated peerId must not assign the local signaling id',
  );
  peer.handleSignalingMessage(JSON.stringify({ type: 'init', yourPeerId: 'server-assigned-id' }));
  assert(peer.options.clientId === 'browser-test-client', 'deterministic client id remains unchanged after init');
  assert(
    peer.getTransportStatus().localSignalingPeerId === 'server-assigned-id',
    'init.yourPeerId is visible in ordinary transport status',
  );
  assert(
    transportStatuses.at(-1)?.localSignalingPeerId === 'server-assigned-id',
    'init.yourPeerId assignment emits promptly to transport subscribers',
  );
  peer.handleSignalingMessage(JSON.stringify({ type: 'joined', peerId: 'unrelated-peer', otherPeerIds: [] }));
  assert(
    peer.getTransportStatus().localSignalingPeerId === 'server-assigned-id',
    'joined.peerId cannot change the assigned local signaling id',
  );
  peer.handleSignalingMessage(JSON.stringify({ type: 'init', yourPeerId: 'x'.repeat(257) }));
  assert(
    peer.getTransportStatus().localSignalingPeerId === 'server-assigned-id',
    'unbounded init.yourPeerId is rejected rather than truncated',
  );

  // --- 3. authenticated signaling ICE arrives before joined ---------------
  const managedIceServers = [
    { urls: ['stun:stun.cloudflare.com:3478'] },
    {
      urls: ['turn:turn.cloudflare.com:3478?transport=udp'],
      username: `${Math.floor(Date.now() / 1000) + 3600}:test`,
      credential: 'ephemeral-turn-secret',
    },
  ];
  peer.handleSignalingMessage(JSON.stringify({
    type: 'ctoxIceServers',
    iceServers: managedIceServers,
  }));
  assert(peer.options.iceServers.length === 2, 'signaling ICE config is installed before joined');
  assert(
    peer.getTransportStatus().turnCredentialExpiresAtMs > Date.now(),
    'signaling TURN expiry is exposed without exposing credentials',
  );
  peer.handleSignalingMessage(JSON.stringify({
    type: 'ctoxIceServers',
    iceServers: [{ urls: 'https://not-an-ice-server.invalid' }],
  }));
  assert(peer.options.iceServers.length === 2, 'invalid signaling ICE cannot replace valid config');

  // --- 4. backoff resets on joined, not on open ----------------------------
  peer.signalingReconnectDelayMs = 30_000; // pretend we backed off heavily
  peer.handleSignalingMessage(JSON.stringify({ type: 'joined', otherPeerIds: [] }));
  assert(
    peer.signalingReconnectDelayMs < 30_000,
    `joined broadcast resets the reconnect backoff (still ${peer.signalingReconnectDelayMs})`,
  );

  // --- 5. socket and peer close clear the ephemeral status -----------------
  peer.socket.onclose();
  assert(peer.getTransportStatus().localSignalingPeerId == null, 'signaling socket close clears the local signaling id');
  assert(
    transportStatuses.at(-1)?.localSignalingPeerId == null,
    'signaling socket close emits the clear promptly',
  );
  peer.handleSignalingMessage(JSON.stringify({ type: 'init', yourPeerId: 'peer-close-id' }));
  assert(peer.getTransportStatus().localSignalingPeerId === 'peer-close-id', 'peer-close fixture is assigned');
  peer.close();
  assert(peer.getTransportStatus().localSignalingPeerId == null, 'peer close clears the local signaling id');
  assert(transportStatuses.at(-1)?.localSignalingPeerId == null, 'peer close emits the clear promptly');
} finally {
  peer.close();
}

console.log('ctox-rxdb signaling freshness smoke OK');
process.exit(0);

function assert(condition, message) {
  if (!condition) throw new Error(message);
}
