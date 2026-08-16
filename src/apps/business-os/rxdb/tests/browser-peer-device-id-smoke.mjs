// REGRESSION: browser signaling identity is a device/partition identity, not
// a page-session identity. Workjet recreates a guest WebContents when an
// instance is deactivated and reactivated. A random identity per module load
// let that new page bypass a native peer revocation issued for the same
// paired device.

import { replicationWebRtcTestInternals } from '../src/replication-webrtc.mjs';

const {
  browserInitiatorPeerIdForDevice,
  resolveBrowserPeerDeviceId,
  browserPeerDeviceIdStorageKey,
} = replicationWebRtcTestInternals;

const storage = memoryStorage();
const firstDeviceId = resolveBrowserPeerDeviceId(storage, () => '0123456789abcdef');
assert(firstDeviceId === '0123456789abcdef', 'first page creates a bounded device id');
assert(
  storage.getItem(browserPeerDeviceIdStorageKey) === firstDeviceId,
  'first page persists its device id in the browser partition',
);

const recreatedPageDeviceId = resolveBrowserPeerDeviceId(storage, () => 'fedcba9876543210');
assert(recreatedPageDeviceId === firstDeviceId, 'a recreated page reuses the partition device id');

const firstPeerId = browserInitiatorPeerIdForDevice(
  'ctox-business-os:instance-a',
  'ctox-business-os://shell',
  firstDeviceId,
);
const recreatedPeerId = browserInitiatorPeerIdForDevice(
  'ctox-business-os:instance-a',
  'ctox-business-os://shell',
  recreatedPageDeviceId,
);
assert(recreatedPeerId === firstPeerId, 'recreated page keeps the same signaling peer id');

storage.removeItem(browserPeerDeviceIdStorageKey);
const replacementDeviceId = resolveBrowserPeerDeviceId(storage, () => 'fedcba9876543210');
assert(replacementDeviceId !== firstDeviceId, 'clearing the partition rotates the device id');
assert(
  browserInitiatorPeerIdForDevice('ctox-business-os:instance-a', 'ctox-business-os://shell', replacementDeviceId)
    !== firstPeerId,
  'clearing the partition rotates the signaling peer id',
);

storage.setItem(browserPeerDeviceIdStorageKey, 'not-a-valid-device-id');
assert(
  resolveBrowserPeerDeviceId(storage, () => '1111111111111111') === '1111111111111111',
  'invalid persisted ids are replaced',
);
assert(
  storage.getItem(browserPeerDeviceIdStorageKey) === '1111111111111111',
  'replacement for an invalid id is persisted',
);

const unavailableStorageDeviceId = resolveBrowserPeerDeviceId({
  getItem() { throw new Error('storage unavailable'); },
  setItem() { throw new Error('storage unavailable'); },
}, () => '2222222222222222');
assert(unavailableStorageDeviceId === '2222222222222222', 'storage failure falls back without aborting sync');

assert(
  browserInitiatorPeerIdForDevice('ctox-business-os:instance-b', 'ctox-business-os://shell', firstDeviceId)
    !== firstPeerId,
  'room topic remains part of peer identity scope',
);
assert(
  browserInitiatorPeerIdForDevice('ctox-business-os:instance-a', 'https://other.invalid', firstDeviceId)
    !== firstPeerId,
  'origin remains part of peer identity scope',
);

// The shared room computes the durable id once, then uses that exact value for
// signaling clientId, transport status, and every returned ctoxProtocol payload.
storage.setItem(browserPeerDeviceIdStorageKey, firstDeviceId);
globalThis.localStorage = storage;
globalThis.location = { origin: 'ctox-business-os://shell' };
const SharedRoomPeer = replicationWebRtcTestInternals.getSharedRoomPeerClass();
const shared = new SharedRoomPeer({
  key: 'device-id-test',
  signalingUrl: 'wss://signaling.invalid',
  room: 'ctox-business-os:instance-a',
  iceServers: [],
});
shared.collections.set('documents', {
  collection: 'documents',
  state: {
    async buildProtocolPayload() {
      return { peerSession: { role: 'browser', sessionId: 'legacy-page-session' } };
    },
  },
});
const stablePayload = await shared.buildProtocolPayload('documents');
const transportPeer = shared.ensurePeer();
assert(shared.localDevicePeerId === firstPeerId, 'shared room stores the scoped durable device id once');
assert(transportPeer.options.clientId === firstPeerId, 'durable device id is the connection clientId');
assert(transportPeer.options.localDevicePeerId === firstPeerId, 'transport retains the durable device id explicitly');
assert(
  stablePayload.peerSession.sessionId === firstPeerId,
  'ctoxProtocol peerSession.sessionId is the same durable device id',
);
assert(
  transportPeer.getTransportStatus().localDevicePeerId === firstPeerId,
  'transport status exposes the durable device id',
);

console.log('ctox-rxdb browser peer device identity smoke OK');

function memoryStorage() {
  const values = new Map();
  return {
    getItem(key) { return values.has(key) ? values.get(key) : null; },
    setItem(key, value) { values.set(String(key), String(value)); },
    removeItem(key) { values.delete(String(key)); },
  };
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}
