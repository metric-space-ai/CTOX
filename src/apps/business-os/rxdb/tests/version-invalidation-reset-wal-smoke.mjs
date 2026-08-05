import http from 'node:http';
import { existsSync, readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const playwrightModule = process.env.PLAYWRIGHT_MODULE_PATH
  ? pathToFileURL(resolve(process.env.PLAYWRIGHT_MODULE_PATH, 'index.mjs')).href
  : '../../node_modules/playwright/index.mjs';
const { chromium } = await import(playwrightModule);
const testDir = dirname(fileURLToPath(import.meta.url));
const bundle = readFileSync(resolve(testDir, '../dist/ctox-rxdb-js.mjs'));
const server = http.createServer((request, response) => {
  if (request.url === '/bundle.mjs') {
    response.writeHead(200, { 'content-type': 'text/javascript' });
    response.end(bundle);
    return;
  }
  response.writeHead(200, { 'content-type': 'text/html' });
  response.end('<!doctype html><title>version invalidation reset wal smoke</title>');
});
await new Promise((resolveReady) => server.listen(0, '127.0.0.1', resolveReady));
const { port } = server.address();
const systemChrome = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const browser = await chromium.launch({
  headless: true,
  ...(existsSync(systemChrome) ? { executablePath: systemChrome } : {}),
});

try {
  const page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${port}/`);
  const result = await page.evaluate(async () => {
    const {
      createRxDatabase,
      openRecoveryJournal,
      removeRxDatabase,
      schemaHash,
    } = await import('/bundle.mjs');
    const suffix = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    const databaseName = `version-invalidation-reset-wal-${suffix}`;
    const recoveryName = `${databaseName}__recovery_v2`;
    const schemaV0 = {
      version: 0,
      type: 'object',
      primaryKey: 'id',
      properties: {
        id: { type: 'string', maxLength: 128 },
        title: { type: 'string' },
      },
      required: ['id'],
    };
    const schemaV1 = {
      version: 1,
      type: 'object',
      primaryKey: 'id',
      properties: {
        id: { type: 'string', maxLength: 128 },
        title: { type: 'string' },
        added: { type: 'string' },
      },
      required: ['id', 'added'],
    };

    // Phase 1: establish a ready v0 marker + a primary master-origin row, then a
    // pending WAL batch. The LocalStorage recovery mirror is intentionally left
    // "export newer than oldest pending" so a resetBusinessDb-style primary wipe
    // would be allowed — but the WAL itself must survive the primary delete.
    let db = await createRxDatabase({ name: databaseName });
    await db.addCollections({ widgets: { schema: schemaV0 } });
    await db.collections.widgets.storageCollection.bulkWrite([
      { id: 'master-cache', title: 'primary only' },
    ], { replicationOrigin: { role: 'ctox_instance', peerId: 'native-smoke' } });
    const markerBeforeReset = await db.storage.collectionSchemaMarker('widgets');
    await db.close();

    const journal = await openRecoveryJournal({
      databaseName,
      instanceId: `${databaseName}-reset`,
    });
    const schemaHashV0 = await schemaHash(schemaV0, 'widgets');
    const batchId = await journal.appendBatch({
      collection: 'widgets',
      schemaHash: schemaHashV0,
      operation: 'upsert',
      rows: [{ id: 'wal-survives-reset', title: 'must still block after primary reset' }],
    });
    const pendingBeforeReset = await journal.pendingSummaryForCollection('widgets');
    const statusBeforeReset = await journal.getStatus();
    journal.close();

    // Mimic resetBusinessDb: delete ONLY the primary IndexedDB. The recovery WAL
    // namespace (${name}__recovery_v2) is intentionally not deleted.
    await removeRxDatabase(databaseName);
    const primaryStillOpen = await new Promise((resolveOpen) => {
      const request = indexedDB.open(databaseName);
      request.onsuccess = () => {
        const opened = request.result;
        // A freshly recreated empty DB has no stores/version from the old handle.
        const names = [...opened.objectStoreNames];
        opened.close();
        resolveOpen({ objectStoreNames: names, version: opened.version });
      };
      request.onerror = () => resolveOpen({ error: String(request.error || 'open failed') });
    });

    const journalAfterReset = await openRecoveryJournal({
      databaseName,
      instanceId: `${databaseName}-reset-after`,
    });
    const pendingAfterReset = await journalAfterReset.pendingSummaryForCollection('widgets');
    const batchAfterReset = (await journalAfterReset.listBatches('pending', 'widgets'))
      .find((entry) => entry.batchId === batchId);
    journalAfterReset.close();

    // Phase 2: reopen primary, redeclare v0 to get a marker again on the fresh
    // primary (missing marker is treated like a mismatch — but first we install
    // v0 cleanly with no pending for the clean marker path), then bump to v1
    // while the WAL still has the pending batch.
    //
    // The live WAL pending batch must block the v0->v1 invalidation even though
    // the primary was wiped and has zero pushable rows.
    db = await createRxDatabase({ name: databaseName });
    // First bring-up after primary wipe with pending WAL: v0 declaration also
    // needs invalidation of sidecars/markers. The pending WAL must block that
    // too — same guard, missing-marker path.
    let firstBringUpCode = '';
    let firstBringUpDetail = null;
    try {
      await db.addCollections({ widgets: { schema: schemaV0 } });
    } catch (error) {
      firstBringUpCode = error?.code || '';
      firstBringUpDetail = {
        pushableRows: error?.pushableRows,
        pendingBatches: error?.pendingBatches,
        pendingWrites: error?.pendingWrites,
      };
    }
    await db.close();

    // Also exercise an explicit v0->v1 path: install v0 by temporarily acknowledging
    // is not possible without clearing WAL. The production contract is that any
    // prepareCollectionSchema while pending>0 is fail-closed. Capture the live
    // journal state after the blocked bring-up.
    const journalFinal = await openRecoveryJournal({
      databaseName,
      instanceId: `${databaseName}-reset-final`,
    });
    const pendingFinal = await journalFinal.pendingSummaryForCollection('widgets');
    const batchFinal = (await journalFinal.listBatches('pending', 'widgets'))
      .find((entry) => entry.batchId === batchId);
    journalFinal.close();

    // Prove a v1 reopen is likewise blocked by the live WAL (fresh primary DB).
    db = await createRxDatabase({ name: databaseName });
    let v1BlockedCode = '';
    let v1BlockedDetail = null;
    let v1BlockedMessage = '';
    try {
      await db.addCollections({ widgets: { schema: schemaV1 } });
    } catch (error) {
      v1BlockedCode = error?.code || '';
      v1BlockedMessage = error?.message || String(error);
      v1BlockedDetail = {
        pushableRows: error?.pushableRows,
        pendingBatches: error?.pendingBatches,
        pendingWrites: error?.pendingWrites,
      };
    }
    const markerAfterBlocked = await db.storage.collectionSchemaMarker('widgets');
    const rowsAfterBlocked = await db.storage.collection('widgets', { schema: schemaV0 })
      .allDocuments({ withDeleted: true });
    await db.close();

    for (const name of [databaseName, recoveryName]) {
      await new Promise((resolveDelete) => {
        const request = indexedDB.deleteDatabase(name);
        request.onsuccess = request.onerror = request.onblocked = () => resolveDelete();
      });
    }

    return {
      markerBeforeReset,
      pendingBeforeReset,
      statusBeforeReset,
      batchId,
      primaryStillOpen,
      pendingAfterReset,
      batchAfterReset,
      firstBringUpCode,
      firstBringUpDetail,
      pendingFinal,
      batchFinal,
      v1BlockedCode,
      v1BlockedDetail,
      v1BlockedMessage,
      markerAfterBlocked,
      rowsAfterBlocked,
    };
  });

  assert(result.markerBeforeReset?.state === 'ready' && result.markerBeforeReset?.declaredVersion === 0, 'setup did not persist a ready v0 marker');
  assert(result.pendingBeforeReset?.pendingBatches === 1, 'pending WAL batch was not installed before the primary reset');
  assert(result.pendingAfterReset?.pendingBatches === 1, 'resetBusinessDb-style primary delete dropped __recovery_v2 pending batches');
  assert(
    result.batchAfterReset?.batchId === result.batchId && result.batchAfterReset?.state === 'pending',
    'resetBusinessDb-style primary delete mutated the surviving WAL batch',
  );
  assert(
    result.firstBringUpCode === 'collection_version_invalidation_blocked',
    `post-reset missing-marker bring-up did not fail closed on live WAL: ${result.firstBringUpCode}`,
  );
  assert(
    result.firstBringUpDetail?.pushableRows === 0 && result.firstBringUpDetail?.pendingBatches === 1,
    `post-reset guard evidence wrong: ${JSON.stringify(result.firstBringUpDetail)}`,
  );
  assert(
    result.v1BlockedCode === 'collection_version_invalidation_blocked',
    `post-reset v1 declaration did not fail closed on live WAL: ${result.v1BlockedCode}`,
  );
  assert(
    result.v1BlockedDetail?.pushableRows === 0 && result.v1BlockedDetail?.pendingBatches === 1,
    `post-reset v1 guard evidence wrong: ${JSON.stringify(result.v1BlockedDetail)}`,
  );
  assert(result.v1BlockedMessage.includes('Nothing was discarded'), 'post-reset blocked message does not state data preservation');
  assert(
    result.pendingFinal?.pendingBatches === 1
      && result.batchFinal?.batchId === result.batchId
      && result.batchFinal?.state === 'pending',
    'blocked post-reset invalidation mutated the live WAL batch',
  );
  assert((result.markerAfterBlocked == null) || result.markerAfterBlocked?.declaredVersion === 0, 'blocked post-reset invalidation wrote a v1 marker');
  assert(result.rowsAfterBlocked.length === 0, 'blocked post-reset path unexpectedly left primary rows');
  console.log('ctox-rxdb version invalidation reset-wal smoke OK assertions=12');
} finally {
  await browser.close();
  await new Promise((resolveClose) => server.close(resolveClose));
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}
