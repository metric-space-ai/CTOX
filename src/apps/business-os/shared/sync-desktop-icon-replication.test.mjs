import test from 'node:test';
import assert from 'node:assert/strict';
import { createSyncRuntime, __ctoxSyncTestHooks } from './sync.js';

const { collectionForReplication, projectDesktopIconForReplication } = __ctoxSyncTestHooks;
const unsafeIcon = () => ({
  id: `desk_icon_${'x'.repeat(180)}`,
  target_type: 'application-with-an-unbounded-type-name',
  target_module: `module-${'m'.repeat(180)}`,
  target_record_id: `record-${'r'.repeat(400)}`,
  label: `label-${'l'.repeat(400)}`,
  glyph: `data:image/svg+xml,${'<svg>'.repeat(20_000)}`,
  x: Infinity,
  y: -900_000,
  pinned: 1,
  hidden: 0,
  sort_index: 900_000,
  updated_at_ms: 42,
  browser_only_manifest: { payload: 'z'.repeat(96 * 1024) },
  _deleted: false,
  _rev: `7-${'a'.repeat(400)}`,
  _meta: {
    lwt: 43,
    ctoxHlc: '43:0:browser',
    ctoxReplicationOrigin: { role: 'ctox_instance', secret: 'must-not-cross-wire' },
    oversized: 'q'.repeat(96 * 1024),
  },
  _attachments: { icon: { type: 'image/svg+xml', data: 'A'.repeat(96 * 1024) } },
});

test('desktop icon replication projects unsafe local cache rows at the wire boundary', async () => {
  const local = unsafeIcon();
  let inboundRows;
  let inboundOptions;
  const storage = {
    async getChangedDocumentsSince() {
      return { documents: [local], checkpoint: { lwt: 43, id: local.id }, scanned: 1 };
    },
    async bulkWrite(rows, options) {
      inboundRows = rows;
      inboundOptions = options;
      return { success: {} };
    },
    identity() { return this; },
  };
  const collection = { name: 'desktop_icons', storageCollection: storage, identity() { return this; } };
  const replicationCollection = collectionForReplication('desktop_icons', collection);
  assert.notStrictEqual(replicationCollection, collection);
  assert.strictEqual(collectionForReplication('desktop_icons', collection), replicationCollection);
  assert.strictEqual(replicationCollection.identity(), collection);
  assert.strictEqual(replicationCollection.storageCollection.identity(), storage);

  const changed = await replicationCollection.storageCollection.getChangedDocumentsSince(null, 10);
  const projected = changed.documents[0];
  assert.deepEqual(projected, projectDesktopIconForReplication(local));
  assert.deepEqual(changed.checkpoint, { lwt: 43, id: local.id });
  assert.equal(projected.glyph, '◻︎');
  assert.equal(projected.id.length, 128);
  assert.equal(projected.target_type.length, 32);
  assert.equal(projected.target_module.length, 128);
  assert.equal(projected.target_record_id.length, 256);
  assert.equal(projected.label.length, 256);
  assert.equal(projected.x, 0);
  assert.equal(projected.y, -100_000);
  assert.equal(projected.sort_index, 100_000);
  assert.equal(projected.updated_at_ms, 42);
  assert.deepEqual(projected._meta, { lwt: 43, ctoxHlc: '43:0:browser' });
  assert.equal(Object.hasOwn(projected, '_attachments'), false);
  assert.equal(Object.hasOwn(projected, 'browser_only_manifest'), false);
  assert.ok(new TextEncoder().encode(JSON.stringify(projected)).byteLength < 64 * 1024);
  assert.ok(local.browser_only_manifest.payload.length > 64 * 1024);
  assert.ok(local._attachments.icon.data.length > 64 * 1024);

  const origin = { role: 'ctox_instance', peerId: 'native' };
  await replicationCollection.storageCollection.bulkWrite([local], { replicationOrigin: origin });
  assert.strictEqual(inboundOptions.replicationOrigin, origin);
  assert.deepEqual(inboundRows, [projected]);
  await replicationCollection.storageCollection.bulkWrite([
    { document: local, previous: { id: local.id } },
  ], { replicationOrigin: origin });
  assert.deepEqual(inboundRows, [{ document: projected, previous: { id: local.id } }]);
});

test('non-desktop collections retain their original collection', () => {
  const collection = { storageCollection: {} };
  assert.strictEqual(collectionForReplication('desktop_layout', collection), collection);
});

test('desktop_icons startup gives WebRTC the projected collection', async () => {
  const previousWindow = globalThis.window;
  globalThis.window = {
    location: { href: 'https://business-os.test/' },
    addEventListener() {},
    removeEventListener() {},
  };
  let replicatedCollection;
  const subscriptions = () => ({ subscribe() { return { unsubscribe() {} }; } });
  const coordinator = {
    onRoleChange() { return () => {}; },
    onDirty() { return () => {}; },
    async start() { return { isLeader: true, role: 'leader' }; },
    isLeader() { return true; },
    snapshot() { return { isLeader: true, role: 'leader' }; },
    stop() {},
  };
  const storage = {
    async getChangedDocumentsSince() { return { documents: [unsafeIcon()], checkpoint: null }; },
    async bulkWrite() { return { success: {} }; },
    async replicationCheckpointStatus() { return { latestLwt: 0, epoch: 'browser:test' }; },
  };
  const rawCollection = {
    name: 'desktop_icons',
    schema: { version: 0, primaryPath: 'id', async hash() { return 'desktop-icons-hash'; } },
    storageCollection: storage,
    observe() { return () => {}; },
  };
  const replicationState = {
    error$: subscriptions(), active$: subscriptions(), canceled$: subscriptions(),
    transportStatus$: subscriptions(),
    peerStates$: { ...subscriptions(), getValue() { return new Map(); } },
    async awaitInitialReplication() { return true; },
    getTransportStatus() { return {}; },
    async cancel() {},
  };
  const db = {
    mode: 'rxdb', name: 'i-009-test', raw: { desktop_icons: rawCollection },
    rxdb: {
      getMultiTabSyncCoordinator() { return coordinator; },
      getConnectionHandlerSimplePeer(options) { return options; },
      async replicateWebRTC(options) { replicatedCollection = options.collection; return replicationState; },
    },
  };
  const runtime = createSyncRuntime({
    db,
    config: { transport: 'webrtc', sync_room: 'ctox-business-os:i-009', signaling_urls: ['wss://signal.test/room'] },
  });
  try {
    const bridge = await runtime.startCollection('desktop_icons', { forceDirect: true, pin: false });
    assert.notStrictEqual(replicatedCollection, rawCollection);
    assert.notStrictEqual(replicatedCollection.storageCollection, storage);
    const result = await replicatedCollection.storageCollection.getChangedDocumentsSince(null, 10);
    assert.equal(Object.hasOwn(result.documents[0], '_attachments'), false);
    assert.equal(Object.hasOwn(result.documents[0], 'browser_only_manifest'), false);
    await bridge.stop();
  } finally {
    await runtime.stop();
    globalThis.window = previousWindow;
  }
});
