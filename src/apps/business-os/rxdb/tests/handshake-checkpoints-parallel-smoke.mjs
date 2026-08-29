// REGRESSION: the browser used to read every IndexedDB checkpoint before
// answering the native peer's symmetric ctoxProtocol request. A production
// room with 192 collections exceeded the native 60 s handshake, so the native
// peer never advanced to its token request and Browser/managed tenant reconnected
// forever. The browser is always the fork for a native CTOX peer: it must send
// every schema hash, but the native side never consumes browser checkpoints.

import { replicationWebRtcTestInternals } from '../src/replication-webrtc.mjs';

const SharedRoomPeer = replicationWebRtcTestInternals.getSharedRoomPeerClass();
const names = Array.from({ length: 192 }, (_, index) => `collection_${String(index).padStart(3, '0')}`);
const shared = new SharedRoomPeer({
  key: 'handshake-checkpoints-test',
  signalingUrl: 'wss://signaling.invalid',
  room: 'ctox-business-os:test',
  iceServers: [],
  expectedNativePeerId: 'native-1',
});

const checkpointStarts = [];
let representativePayloadBuilds = 0;

for (const name of names) {
  shared.collections.set(name, {
    state: {
      schemaHashValue: `hash-${name}`,
      collection: {
        name,
        schema: {
          version: 1,
          hash: async () => `hash-${name}`,
        },
        storageCollection: {
          async replicationCheckpointStatus(hash) {
            checkpointStarts.push({ name, hash });
            return new Promise(() => {});
          },
        },
      },
      async buildProtocolPayload() {
        representativePayloadBuilds += 1;
        return {
          protocol: 'ctox-rxdb-protocol-v1',
          capabilities: [],
          collection: { name, schemaVersion: 1, schemaHash: `hash-${name}` },
          checkpoint: { state: 'advertised', epoch: `representative-${name}` },
        };
      },
    },
  });
}

const firstBuild = shared.buildProtocolPayload();
const symmetricBuild = shared.buildProtocolPayload(names[1]);
const [payload, symmetricPayload] = await Promise.race([
  Promise.all([firstBuild, symmetricBuild]),
  delay(1_000).then(() => {
    throw new Error('schema-only symmetric handshake did not resolve within 1 second');
  }),
]);

assert(payload !== symmetricPayload, 'each protocol request keeps its collection-specific envelope');
assert(payload.collection?.name === names[0], 'representative request keeps the representative collection');
assert(symmetricPayload.collection?.name === names[1], 'symmetric request keeps the requested collection envelope');
assert(representativePayloadBuilds === 2, 'each collection-specific envelope is built once');
assert(checkpointStarts.length === 0, 'room-level browser response does not read unused IndexedDB checkpoints');
for (const name of names) {
  assert(payload.collectionSchemas?.[name]?.schemaHash === `hash-${name}`, `${name}: schema hash included`);
}
assert(payload.collectionCheckpoints == null, 'browser room response omits unused checkpoint map');

console.log('ctox-rxdb schema-only symmetric handshake smoke OK');

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}
