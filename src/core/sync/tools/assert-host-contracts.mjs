// Check the generated Workjet consumer using the canonical native setup shapes.
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
const root = process.argv[2];
if (!root) throw new Error('Expected Workjet checkout path');
const schemaPath = resolve(root, 'packages/contracts/src/ctoxSync.schema.generated.ts');
const require = createRequire(schemaPath);
const Schema = await import(pathToFileURL(require.resolve('effect/Schema')).href);
const generated = await import(pathToFileURL(schemaPath).href);
const decode = (name) => Schema.decodeUnknownSync(generated[name], { onExcessProperty: 'error' });
const host = decode('SyncHostConfigurationSchema');
const transport = decode('SyncHostTransportSchema');
const peer = { identity: `ed25519:${'01'.repeat(32)}`, executor: true, dataReplica: true };
const input = {
  version: 1, scopeId: 'network', local: { type: 'voter', nodeId: 1 },
  voters: { '1': peer, '2': peer, '3': peer },
  timing: { heartbeatMs: 250, electionMinMs: 1500, electionMaxMs: 3000 },
};
// Shape decoding does not certify membership, distinct pins or authorization;
// native configuration validation enforces those semantic constraints.
assert.deepEqual(host(input), input);
const worker = { ...input, local: { type: 'worker', member: { nodeId: 4, identity: peer.identity, dataReplica: true, revoked: false } } };
assert.deepEqual(host(worker), worker);
for (const bad of [
  { ...input, local: { type: 'voter', nodeId: Number.MAX_SAFE_INTEGER + 1 } },
  { ...input, timing: { ...input.timing, heartbeatMs: -1 } },
  { ...input, ipcEndpoint: '/untrusted/socket' },
  { ...input, local: { ...input.local, role: 'admin' } },
]) assert.throws(() => host(bad));
const settings = { signalingUrls: ['wss://signal.example?role=workjet_executor'], iceServers: [] };
assert.deepEqual(transport(settings), settings);
assert.throws(() => transport({ signalingUrls: settings.signalingUrls }));
assert.throws(() => transport({ ...settings, iceServers: [{ urls: ['turn:example'], username: '' }] }));
console.log('CHECK generated host setup contracts: voter, worker, transport, exact fields and safe integers');
