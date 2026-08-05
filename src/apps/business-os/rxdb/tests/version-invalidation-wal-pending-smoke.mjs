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
  response.end('<!doctype html><title>version invalidation wal pending smoke</title>');
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
    const { createRxDatabase, openRecoveryJournal, schemaHash } = await import('/bundle.mjs');
    const suffix = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    const databaseName = `version-invalidation-wal-pending-${suffix}`;
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

    let db = await createRxDatabase({ name: databaseName });
    await db.addCollections({ widgets: { schema: schemaV0 } });
    // Master-origin cache row: pushable=0 so only the WAL pending signal can block.
    await db.collections.widgets.storageCollection.bulkWrite([
      { id: 'master-cache', title: 'synced cache row' },
    ], { replicationOrigin: { role: 'ctox_instance', peerId: 'native-smoke' } });
    const markerV0 = await db.storage.collectionSchemaMarker('widgets');
    const unsyncedBefore = await db.getUnsyncedWriteSummary();
    await db.close();

    const journal = await openRecoveryJournal({
      databaseName,
      instanceId: `${databaseName}-wal-only`,
    });
    const schemaHashV0 = await schemaHash(schemaV0, 'widgets');
    const batchId = await journal.appendBatch({
      collection: 'widgets',
      schemaHash: schemaHashV0,
      operation: 'upsert',
      rows: [{ id: 'wal-only-pending', title: 'must block version change' }],
    });
    const pendingBefore = await journal.pendingSummaryForCollection('widgets');
    const batchBefore = (await journal.listBatches('pending', 'widgets'))
      .find((entry) => entry.batchId === batchId);
    journal.close();

    db = await createRxDatabase({ name: databaseName });
    let blockedCode = '';
    let blockedMessage = '';
    let blockedDetail = null;
    try {
      await db.addCollections({ widgets: { schema: schemaV1 } });
    } catch (error) {
      blockedCode = error?.code || '';
      blockedMessage = error?.message || String(error);
      blockedDetail = {
        pushableRows: error?.pushableRows,
        pendingBatches: error?.pendingBatches,
        pendingWrites: error?.pendingWrites,
      };
    }
    const markerAfter = await db.storage.collectionSchemaMarker('widgets');
    const rowsAfter = await db.storage.collection('widgets', { schema: schemaV0 })
      .allDocuments({ withDeleted: true });
    const unsyncedAfter = await db.getUnsyncedWriteSummary();
    await db.close();

    const journalAfter = await openRecoveryJournal({
      databaseName,
      instanceId: `${databaseName}-wal-only-after`,
    });
    const pendingAfter = await journalAfter.pendingSummaryForCollection('widgets');
    const batchAfter = (await journalAfter.listBatches('pending', 'widgets'))
      .find((entry) => entry.batchId === batchId);
    journalAfter.close();

    for (const name of [databaseName, `${databaseName}__recovery_v2`]) {
      await new Promise((resolveDelete) => {
        const request = indexedDB.deleteDatabase(name);
        request.onsuccess = request.onerror = request.onblocked = () => resolveDelete();
      });
    }

    return {
      markerV0,
      unsyncedBefore,
      pendingBefore,
      batchBefore,
      blockedCode,
      blockedMessage,
      blockedDetail,
      markerAfter,
      rowsAfter,
      unsyncedAfter,
      pendingAfter,
      batchAfter,
      batchId,
    };
  });

  assert(result.markerV0?.state === 'ready' && result.markerV0?.declaredVersion === 0, 'v0 marker was not ready before the WAL-pending version change');
  assert(result.unsyncedBefore?.total === 0, 'setup accidentally left pushable primary rows');
  assert(result.pendingBefore?.pendingBatches === 1 && result.pendingBefore?.pendingWrites === 1, 'pending WAL batch was not installed before the version change');
  assert(result.batchBefore?.state === 'pending' && result.batchBefore?.batchId === result.batchId, 'WAL batch was not pending before the version change');
  assert(result.blockedCode === 'collection_version_invalidation_blocked', `WAL-pending guard did not fail closed: ${result.blockedCode}`);
  assert(
    result.blockedDetail?.pushableRows === 0 && result.blockedDetail?.pendingBatches === 1,
    `fail-closed evidence did not isolate the WAL-pending guard: ${JSON.stringify(result.blockedDetail)}`,
  );
  assert(result.blockedMessage.includes('Nothing was discarded'), 'fail-closed error message does not state the data-preservation outcome');
  assert(result.markerAfter?.declaredVersion === 0 && result.markerAfter?.state === 'ready', 'WAL-pending block modified the persisted v0 marker');
  assert(
    result.rowsAfter.length === 1 && result.rowsAfter[0]?.id === 'master-cache',
    'WAL-pending block discarded or mutated the primary cache row',
  );
  assert((result.unsyncedAfter?.total || 0) === 0, 'WAL-pending block created unexpected pushable rows');
  assert(
    result.pendingAfter?.pendingBatches === 1
      && result.batchAfter?.batchId === result.batchId
      && result.batchAfter?.state === 'pending',
    'WAL-pending block mutated or dropped the recovery WAL batch',
  );
  console.log('ctox-rxdb version invalidation wal-pending smoke OK assertions=11');
} finally {
  await browser.close();
  await new Promise((resolveClose) => server.close(resolveClose));
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}
