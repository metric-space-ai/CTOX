import test from 'node:test';
import assert from 'node:assert/strict';
import { createBusinessOsOfficeBridge } from './src/business-os-bridge.mjs';

for (const kind of ['document', 'spreadsheet']) {
  const names = kind === 'document'
    ? ['document_blob_chunks', 'document_versions', 'documents']
    : ['spreadsheet_blob_chunks', 'spreadsheet_versions', 'spreadsheets'];
  test(`${kind}: native prepare waits for all source pushes in a follower tab`, async () => {
    const remote = new Set();
    const released = [];
    let acknowledge;
    const held = new Promise(resolve => { acknowledge = resolve; });
    let started;
    const pushing = new Promise(resolve => { started = resolve; });
    let dispatched = false;
    const bridge = createBusinessOsOfficeBridge({
      sync: {
        async leaseCollection(name) {
          return { bridge: { mode: 'follower' }, async release() { released.push(name); } };
        },
        async startCollection(name, options) {
          assert.deepEqual(options, { pin: false, forceDirect: true });
          return { state: {
            async waitForOpenPeerId(timeoutMs) { assert.equal(timeoutMs, 60000); return 'native'; },
            async pushToPeer(peer) {
              assert.equal(peer, 'native');
              if (name === names[0]) { started(); await held; }
              remote.add(name);
            },
            async awaitInSync() { assert.fail('must not download full collections'); },
            async pushToRemotePeers() { assert.fail('must not use unconfirmed background sweep'); },
          } };
        },
      },
      commandBus: { async dispatch() {
        assert.deepEqual([...remote], names);
        dispatched = true;
        return { status: 'completed', result: { ok: true } };
      } },
    }, kind);
    const pending = bridge.prepare({ recordId: 'new', versionId: 'v1' });
    await pushing;
    assert.equal(dispatched, false);
    acknowledge();
    await pending;
    assert.equal(dispatched, true);
    assert.deepEqual(released, [names[1], names[2], names[0]]);
  });

  test(`${kind}: a failed source push prevents native prepare and releases leases`, async () => {
    let dispatched = false;
    const released = [];
    const bridge = createBusinessOsOfficeBridge({
      sync: { async leaseCollection(name) {
        return { bridge: { state: {
          async waitForOpenPeerId() { return 'native'; },
          async pushToPeer() { throw new Error('source transfer rejected'); },
        } }, async release() { released.push(name); } };
      } },
      commandBus: { async dispatch() { dispatched = true; } },
    }, kind);
    await assert.rejects(bridge.prepare({ recordId: 'new', versionId: 'v1' }), /source transfer rejected/);
    assert.equal(dispatched, false);
    assert.deepEqual(released, [names[0]]);
  });
}
