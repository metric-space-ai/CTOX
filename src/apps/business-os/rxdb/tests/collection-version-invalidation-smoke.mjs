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
  response.end('<!doctype html><title>collection version invalidation smoke</title>');
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
    const { createRxDatabase, schemaHash } = await import('/bundle.mjs');
    const suffix = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    const cleanName = `collection-version-clean-${suffix}`;
    const blockedName = `collection-version-blocked-${suffix}`;
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

    let cleanDb = await createRxDatabase({ name: cleanName });
    await cleanDb.addCollections({ widgets: { schema: schemaV0 } });
    await cleanDb.collections.widgets.storageCollection.bulkWrite([
      { id: 'old-clean', title: 'v0 cache row' },
    ], { replicationOrigin: { role: 'ctox_instance', peerId: 'native-smoke' } });
    const markerV0 = await cleanDb.storage.collectionSchemaMarker('widgets');
    await cleanDb.close();

    const retainedKey = 'ctox.rxdb.checkpoints.v1.smoke-room.widgets';
    localStorage.setItem(retainedKey, JSON.stringify({
      validityKey: 'old',
      pull: { id: 'old-clean', lwt: 1 },
      push: { id: 'old-clean', lwt: 1 },
      firstPullCompletedAtMs: Date.now(),
    }));
    cleanDb = await createRxDatabase({ name: cleanName });
    let browserMigrationStrategyRan = false;
    await cleanDb.addCollections({
      widgets: {
        schema: schemaV1,
        migrationStrategies: {
          1: () => {
            browserMigrationStrategyRan = true;
            throw new Error('browser migration strategy must not execute');
          },
        },
      },
    });
    const cleanRowsAfter = await cleanDb.collections.widgets.storageCollection.allDocuments({ withDeleted: true });
    const markerV1 = await cleanDb.storage.collectionSchemaMarker('widgets');
    const invalidation = cleanDb.collections.widgets.versionInvalidation;
    const expectedV1Hash = await schemaHash(schemaV1, 'widgets');
    const retainedAfter = localStorage.getItem(retainedKey);
    await cleanDb.close();

    let blockedDb = await createRxDatabase({ name: blockedName });
    await blockedDb.addCollections({ widgets: { schema: schemaV0 } });
    await blockedDb.collections.widgets.storageCollection.bulkWrite([
      { id: 'local-only', title: 'must survive' },
    ], { skipJournal: true });
    await blockedDb.close();

    blockedDb = await createRxDatabase({ name: blockedName });
    let blockedCode = '';
    let blockedMessage = '';
    let blockedDetail = null;
    try {
      await blockedDb.addCollections({ widgets: { schema: schemaV1 } });
    } catch (error) {
      blockedCode = error?.code || '';
      blockedMessage = error?.message || String(error);
      blockedDetail = {
        pushableRows: error?.pushableRows,
        pendingBatches: error?.pendingBatches,
      };
    }
    const blockedMarker = await blockedDb.storage.collectionSchemaMarker('widgets');
    const blockedRows = await blockedDb.storage.collection('widgets', { schema: schemaV0 }).allDocuments({ withDeleted: true });
    const blockedUnsynced = await blockedDb.getUnsyncedWriteSummary();
    await blockedDb.close();

    for (const name of [cleanName, `${cleanName}__recovery_v2`, blockedName, `${blockedName}__recovery_v2`, 'ctox_business_os_v1_5_meta_widgets']) {
      await new Promise((resolveDelete) => {
        const request = indexedDB.deleteDatabase(name);
        request.onsuccess = request.onerror = request.onblocked = () => resolveDelete();
      });
    }
    localStorage.removeItem(retainedKey);

    return {
      markerV0,
      markerV1,
      expectedV1Hash,
      invalidation,
      cleanRowsAfter,
      retainedAfter,
      browserMigrationStrategyRan,
      blockedCode,
      blockedMessage,
      blockedDetail,
      blockedMarker,
      blockedRows,
      blockedUnsynced,
    };
  });

  assert(result.markerV0?.declaredVersion === 0, 'v0 marker was not persisted');
  assert(result.cleanRowsAfter.length === 0, 'clean v0 cache rows were not cleared on v1 bring-up');
  assert(result.invalidation?.invalidated === true && result.invalidation?.clearedRows === 1, 'version invalidation did not report the collection clear');
  assert(result.markerV1?.state === 'ready' && result.markerV1?.declaredVersion === 1, 'v1 marker was not finalized');
  assert(result.markerV1?.effectiveSchemaHash === result.expectedV1Hash, 'marker did not persist the effective schema hash');
  assert(result.retainedAfter === null, 'retained pull/push/readiness checkpoint record survived invalidation');
  assert(result.browserMigrationStrategyRan === false, 'browser runtime executed migrationStrategies');
  assert(result.blockedCode === 'collection_version_invalidation_blocked', `pushable guard did not fail closed: ${result.blockedCode}`);
  assert(result.blockedDetail?.pushableRows === 1 && result.blockedDetail?.pendingBatches === 0, 'fail-closed evidence did not isolate the pushable-row guard');
  assert(result.blockedRows.length === 1 && result.blockedRows[0]?.id === 'local-only', 'fail-closed version change discarded the local row');
  assert(result.blockedMarker?.declaredVersion === 0 && result.blockedMarker?.state === 'ready', 'fail-closed version change modified the persisted v0 marker');
  assert(result.blockedUnsynced?.byCollection?.widgets === 1, 'fail-closed row no longer remains pushable');
  assert(result.blockedMessage.includes('Nothing was discarded'), 'fail-closed error message does not state the data-preservation outcome');
  console.log('ctox-rxdb collection version invalidation smoke OK assertions=13');
} finally {
  await browser.close();
  await new Promise((resolveClose) => server.close(resolveClose));
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}
