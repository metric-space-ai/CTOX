// REGRESSION: the shell starts its representative collection before runtime
// app schemas have all registered. If the room grows from one collection to
// many while the representative payload awaits IndexedDB, the multiplex map
// build must be acquired after that growth instead of dereferencing null.

import { replicationWebRtcTestInternals } from '../src/replication-webrtc.mjs';

const SharedRoomPeer = replicationWebRtcTestInternals.getSharedRoomPeerClass();
const shared = new SharedRoomPeer({
  key: 'handshake-registration-race-test',
  signalingUrl: 'wss://signaling.invalid',
  room: 'ctox-business-os:test',
  iceServers: [],
  expectedNativePeerId: 'native-1',
});

let releaseRepresentative;
const representativeMayFinish = new Promise((resolve) => { releaseRepresentative = resolve; });

shared.collections.set('business_commands', registration('business_commands', representativeMayFinish));
const payloadBuild = shared.buildProtocolPayload();

// Grow the room while buildProtocolPayloadUncached() is awaiting the
// representative collection payload.
await Promise.resolve();
shared.collections.set('outbound_lead_generation_adapters', registration('outbound_lead_generation_adapters'));
releaseRepresentative();

const payload = await payloadBuild;
assert(payload.collectionSchemas?.business_commands?.schemaHash === 'hash-business_commands', 'representative schema is present');
assert(payload.collectionSchemas?.outbound_lead_generation_adapters?.schemaHash === 'hash-outbound_lead_generation_adapters', 'late schema is present');
assert(payload.collectionCheckpoints == null, 'browser symmetric response omits unused checkpoint map');

console.log('ctox-rxdb handshake collection-registration race smoke OK');

function registration(name, gate = Promise.resolve()) {
  const collection = {
    name,
    schema: { version: 1, hash: async () => `hash-${name}` },
    storageCollection: {
      async replicationCheckpointStatus(hash) {
        return { state: 'advertised', collection: name, hash };
      },
    },
  };
  return {
    state: {
      schemaHashValue: `hash-${name}`,
      collection,
      async buildProtocolPayload() {
        await gate;
        return {
          protocol: 'ctox-rxdb-protocol-v1',
          capabilities: [],
          collection: { name, schemaVersion: 1, schemaHash: `hash-${name}` },
          checkpoint: { state: 'advertised', collection: name },
        };
      },
    },
  };
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}
