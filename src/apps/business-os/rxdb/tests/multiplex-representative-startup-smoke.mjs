// REGRESSION: the browser may start its room handshake after registering only
// one collection while the native peer already advertises the full multiplexed
// room. Different representative envelopes must not be compared as if this
// were a single-collection connection.

import { replicationWebRtcTestInternals } from '../src/replication-webrtc.mjs';

const { protocolHandshakeIsMultiplexed } = replicationWebRtcTestInternals;

const browser = {
  collection: { name: 'outbound_lead_generation_leads', schemaVersion: 0, schemaHash: 'lead-hash' },
};
const native = {
  collection: { name: 'desktop_files', schemaVersion: 0, schemaHash: 'file-hash' },
  collectionSchemas: {
    desktop_files: { schemaVersion: 0, schemaHash: 'file-hash' },
    outbound_lead_generation_leads: { schemaVersion: 0, schemaHash: 'lead-hash' },
  },
};

assert(
  protocolHandshakeIsMultiplexed(1, browser, native),
  'remote room schema map must make the asymmetric startup handshake multiplexed',
);
assert(
  !protocolHandshakeIsMultiplexed(1, browser, {
    collection: { name: 'desktop_files', schemaVersion: 0, schemaHash: 'file-hash' },
  }),
  'two genuinely single-collection peers must still validate representative identity',
);

console.log('ctox-rxdb multiplex representative startup smoke OK');

function assert(condition, message) {
  if (!condition) throw new Error(message);
}
