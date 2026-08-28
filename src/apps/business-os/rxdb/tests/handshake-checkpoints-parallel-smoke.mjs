// REGRESSION: multiplexed protocol handshakes used to read collection
// checkpoints sequentially. With many Business OS collections, reconnect paid
// one IndexedDB/storage await per collection before the native peer saw the
// protocol payload. The shared room peer now collects per-collection protocol
// maps with bounded parallelism.

import { replicationWebRtcTestInternals } from '../src/replication-webrtc.mjs';

const SharedRoomPeer = replicationWebRtcTestInternals.getSharedRoomPeerClass();
const names = ['business_commands', 'business_records', 'desktop_files'];
const shared = new SharedRoomPeer({
  key: 'handshake-checkpoints-test',
  signalingUrl: 'wss://signaling.invalid',
  room: 'ctox-business-os:test',
  iceServers: [],
  expectedNativePeerId: 'native-1',
});

let releaseAllStarted;
const allStarted = new Promise((resolve) => { releaseAllStarted = resolve; });
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
            if (checkpointStarts.length === names.length) releaseAllStarted();
            await allStarted;
            return {
              state: 'advertised',
              epoch: `epoch-${name}`,
              hash,
            };
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
const symmetricBuild = shared.buildProtocolPayload('business_records');
const [payload, symmetricPayload] = await Promise.race([
  Promise.all([firstBuild, symmetricBuild]),
  delay(500).then(() => {
    throw new Error(`checkpoint collection did not run concurrently; started ${checkpointStarts.map((entry) => entry.name).join(',')}`);
  }),
]);

assert(payload !== symmetricPayload, 'each protocol request keeps its collection-specific envelope');
assert(payload.collection?.name === 'business_commands', 'representative request keeps the representative collection');
assert(symmetricPayload.collection?.name === 'business_records', 'symmetric request keeps the requested collection');
assert(representativePayloadBuilds === 2, 'each collection-specific envelope is built once');
assert(checkpointStarts.length === names.length, 'all checkpoint reads started before any completed');
for (const name of names) {
  assert(payload.collectionSchemas?.[name]?.schemaHash === `hash-${name}`, `${name}: schema hash included`);
  assert(payload.collectionCheckpoints?.[name]?.epoch === `epoch-${name}`, `${name}: checkpoint included`);
  assert(payload.collectionCheckpoints?.[name]?.collection === name, `${name}: checkpoint collection normalized`);
}

console.log('ctox-rxdb handshake checkpoint parallel smoke OK');

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}
