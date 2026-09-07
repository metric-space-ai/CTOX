import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { test } from 'node:test';
import { createBusinessOsOfficeBridge } from './src/business-os-bridge.mjs';

function deferred() {
  let resolve;
  const promise = new Promise((done) => { resolve = done; });
  return { promise, resolve };
}

function fixture(kind, transport, startCollection) {
  const events = [];
  const staged = [];
  const commands = [];
  const chunks = `${kind}_blob_chunks`;
  const lease = { bridge: transport, async release() { events.push('release'); } };
  const ctx = {
    permissions: { canWriteCollection: () => true },
    db: { collection(name) {
      assert.equal(name, chunks);
      return { async bulkUpsert(rows) { staged.push(...rows); events.push('stage'); } };
    } },
    sync: {
      async leaseCollection(name) { assert.equal(name, chunks); return lease; },
      async startCollection(name, options) {
        assert.equal(name, chunks);
        assert.deepEqual(options, { pin: false, forceDirect: true });
        events.push('direct');
        return startCollection?.();
      },
    },
    commandBus: { async dispatch(command) {
      events.push('dispatch'); commands.push(command);
      return { status: 'completed', result: { ok: true, version_id: 'v2' } };
    } },
  };
  const bytes = new TextEncoder().encode(kind === 'document' ? 'DOCY;v10;payload' : 'XLSY;v10;payload');
  const commit = () => createBusinessOsOfficeBridge(ctx, kind).commit({
    recordId: 'owned_record', baseVersionId: 'v1', bytes,
    editorProtocol: `office-${kind}`, editorProtocolVersion: 10,
    implementedFeatures: [`${kind}.edit-save`], reason: 'test',
  });
  return { commit, events, staged, commands, bytes };
}

for (const kind of ['document', 'spreadsheet']) {
  test(`${kind} commit waits for acknowledged peer upload before native dispatch`, async () => {
    const entered = deferred();
    const acknowledged = deferred();
    let f;
    const state = {
      async waitForOpenPeerId(timeout) { assert.equal(timeout, 60000); f.events.push('peer'); return 'native'; },
      async pushToPeer(peer) {
        assert.equal(peer, 'native'); f.events.push('upload'); entered.resolve();
        await acknowledged.promise; f.events.push('ack');
      },
      async pushToRemotePeers() { f.events.push('background'); },
      async awaitInSync() { assert.fail('commit must not download the entire collection'); },
    };
    f = fixture(kind, { mode: 'leader', state });
    const pending = f.commit();
    try {
      const first = await Promise.race([
        entered.promise.then(() => 'upload'), pending.then(() => 'completed'),
      ]);
      assert.equal(first, 'upload', 'native dispatch must not race ahead of chunk acknowledgement');
      assert.equal(f.commands.length, 0);
      acknowledged.resolve();
      assert.equal((await pending).version_id, 'v2');
      assert.deepEqual(f.events, ['stage', 'peer', 'upload', 'ack', 'dispatch', 'release']);
      assert.equal(f.commands[0].payload.base_version_id, 'v1');
      assert.equal(f.commands[0].payload.editor_sha256, createHash('sha256').update(f.bytes).digest('hex'));
      assert.equal(f.commands[0].payload.editor_blob_id, f.staged[0].blob_id);
    } finally { acknowledged.resolve(); await pending.catch(() => null); }
  });

  test(`${kind} commit propagates failed upload without dispatching a native command`, async () => {
    const failure = Object.assign(new Error('native peer disconnected'), { code: 'sync_unavailable' });
    const f = fixture(kind, { mode: 'leader', state: {
      async waitForOpenPeerId() { return 'native'; },
      async pushToPeer() { throw failure; },
      async pushToRemotePeers() { /* background sweep may defer failed peers */ },
    } });
    await assert.rejects(f.commit(), (error) => error === failure);
    assert.deepEqual(f.events, ['stage', 'release']);
    assert.equal(f.commands.length, 0);
  });

  for (const mode of ['pending', 'follower', 'missing']) {
    test(`${kind} commit obtains a direct upload bridge from a ${mode} lease`, async () => {
      let f;
      const initial = mode === 'missing' ? null : { mode, state: null };
      f = fixture(kind, initial, () => ({ mode: 'leader', state: {
        async waitForOpenPeerId() { return 'native'; },
        async pushToPeer() { f.events.push('ack'); },
      } }));
      await f.commit();
      assert.deepEqual(f.events, ['stage', 'direct', 'ack', 'dispatch', 'release']);
      assert.equal(f.commands.length, 1);
    });
  }

  test(`${kind} commit fails closed when no push bridge becomes available`, async () => {
    const f = fixture(kind, { mode: 'pending', state: null }, () => null);
    await assert.rejects(f.commit(), /sync push unavailable/);
    assert.equal(f.commands.length, 0);
    assert.deepEqual(f.events, ['stage', 'direct', 'release']);
  });
}
