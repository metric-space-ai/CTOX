// REGRESSION: native durable-device revocation is returned as the wire error
// string `peer_revoked`. Browser request correlation must preserve it as a
// structured Error code so shell diagnostics and policy do not flatten it.

import { createCtoxWebRtcNativePeer } from '../src/webrtc-native.mjs';

const peer = createCtoxWebRtcNativePeer({
  signalingUrl: 'wss://signaling.invalid',
  room: 'ctox-business-os:revocation-test',
  clientId: 'stable-browser-device',
  localDevicePeerId: 'stable-browser-device',
});

let rejected;
const pending = new Promise((resolve, reject) => {
  peer.pending.set('revoked-request', {
    resolve,
    reject,
    timer: setTimeout(() => reject(new Error('response did not settle')), 1000),
    method: 'ctoxProtocol',
    peerId: 'native-peer',
  });
}).catch((error) => {
  rejected = error;
});

await peer.handleDataChannelFrame('native-peer', {
  id: 'revoked-request',
  result: null,
  error: 'peer_revoked',
  collection: 'business_module_catalog',
});
await pending;

assert(rejected instanceof Error, 'peer_revoked rejects with an Error instance');
assert(rejected.name === 'CtoxWebRTCResponseError', 'response rejection has a stable structured error name');
assert(rejected.code === 'peer_revoked', 'wire error string is preserved as the structured error code');
assert(rejected.retryable === false, 'durable revocation is terminal');
assert(rejected.method === 'ctoxProtocol', 'request method context is preserved');
assert(rejected.peerId === 'native-peer', 'remote peer context is preserved');
assert(rejected.collection === 'business_module_catalog', 'collection context is preserved');
assert(peer.pending.size === 0, 'settled request is removed from pending correlation');

peer.close();
console.log('ctox-rxdb peer revocation smoke OK');

function assert(condition, message) {
  if (!condition) throw new Error(message);
}
