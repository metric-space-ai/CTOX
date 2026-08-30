import { CtoxWebRtcNativePeer } from '../src/webrtc-native.mjs';

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

const expectedNativePeerId = 'ctox-core-stable-instance';
const peer = new CtoxWebRtcNativePeer({
  signalingUrl: 'wss://signaling.invalid',
  room: 'ctox-business-os:instance:room',
  expectedNativePeerId,
});
peer.setLocalSignalingPeerId('browser-self');

peer.rememberPeerMetadata('attacker-peer', {
  role: 'ctox_instance',
  client: 'attacker-controlled-native-label',
});
assert(
  peer.shouldConnectToRemotePeer('attacker-peer') === false,
  'a self-declared ctox_instance must not replace the configured native peer',
);

let connectionAttempts = 0;
peer.ensureConnection = () => {
  connectionAttempts += 1;
};
peer.handleSignalingMessage(JSON.stringify({
  type: 'joined',
  peers: [{
    peerId: 'attacker-peer',
    role: 'ctox_instance',
    client: 'attacker-controlled-native-label',
  }],
}));
assert(connectionAttempts === 0, 'joined presence must fail closed while the expected native peer is absent');

peer.rememberPeerMetadata('server-assigned-native-peer', {
  role: 'ctox_instance',
  client: expectedNativePeerId,
});
assert(
  peer.shouldConnectToRemotePeer('server-assigned-native-peer') === true,
  'the stable configured native identity must remain connectable across process sessions',
);

let authError = null;
peer.on('error', (event) => {
  authError = event.detail || event;
});
peer.handleSignalingMessage(JSON.stringify({
  type: 'ctoxError',
  scope: 'control-plane',
  code: 'role_credential_invalid',
  reason: 'role-bound signaling credential invalid',
}));
assert(authError?.retryable === false, 'role-credential rejection must be terminal');
assert(peer.closed === true, 'terminal role-credential rejection must stop reconnect attempts');

console.log('Native peer identity fail-closed smoke OK');
