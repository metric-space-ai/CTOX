import { buildProtocolPayload } from '../src/schema.mjs';
import { resolveDeviceProof } from '../src/replication-webrtc.mjs';
import { getBusinessOsDeviceProof } from '../../shared/sync.js';

const nonce = 'nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn';
const receivedNonces = [];
Object.defineProperty(globalThis, 'ctoxWorkjetDeviceProofProvider', {
  value: async (value) => {
    receivedNonces.push(value);
    return {
      publicJwk: {
        kty: 'EC',
        crv: 'P-256',
        x: 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx',
        y: 'yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy',
        privateKey: 'must-not-cross-the-bridge',
      },
      signature: 'ssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss',
      privateKey: 'must-not-cross-the-bridge',
    };
  },
  writable: false,
  configurable: false,
  enumerable: false,
});
const proof = await resolveDeviceProof({
  deviceProofProvider: getBusinessOsDeviceProof,
}, nonce);

if (receivedNonces[0] !== nonce) {
  throw new Error('device proof provider must receive exactly the native nonce');
}
const descriptor = Object.getOwnPropertyDescriptor(globalThis, 'ctoxWorkjetDeviceProofProvider');
if (descriptor?.writable !== false || descriptor.configurable !== false || descriptor.enumerable !== false) {
  throw new Error('native device proof provider property must be immutable and non-enumerable');
}
const reconnectNonce = 'mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm';
await resolveDeviceProof({ deviceProofProvider: getBusinessOsDeviceProof }, reconnectNonce);
if (receivedNonces.length !== 2 || receivedNonces[1] !== reconnectNonce) {
  throw new Error('each native handshake/reconnect challenge must invoke the provider again');
}
const payload = buildProtocolPayload({
  peerSessionId: 'browser:test-room',
  capabilityToken: 'bound-capability',
  deviceProof: proof,
});
if (
  payload.peerSession?.deviceProof?.nonce !== nonce
  || payload.peerSession.deviceProof.publicJwk?.kty !== 'EC'
  || payload.peerSession.deviceProof.signature !== proof.signature
) {
  throw new Error('valid device proof was not carried in peerSession');
}
if (
  'privateKey' in payload.peerSession.deviceProof
  || 'privateKey' in payload.peerSession.deviceProof.publicJwk
) {
  throw new Error('private key material crossed the browser protocol boundary');
}
if (await resolveDeviceProof({}, nonce) !== null) {
  throw new Error('missing native proof provider must not synthesize a software key');
}
if (await resolveDeviceProof({
  async deviceProofProvider() {
    throw new Error('native signing unavailable');
  },
}, nonce) !== null) {
  throw new Error('rejected native proof promises must omit proof and fail closed natively');
}
const malformed = buildProtocolPayload({
  peerSessionId: 'browser:test-room',
  deviceProof: { ...proof, signature: 'not-base64url-p1363' },
});
if (malformed.peerSession?.deviceProof) {
  throw new Error('malformed device proof must be omitted so bound tokens fail closed natively');
}

console.log('device proof callback boundary smoke: ok');
