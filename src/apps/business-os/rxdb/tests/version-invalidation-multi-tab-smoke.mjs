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
  response.end('<!doctype html><title>version invalidation multi-tab smoke</title>');
});
await new Promise((resolveReady) => server.listen(0, '127.0.0.1', resolveReady));
const { port } = server.address();
const systemChrome = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const browser = await chromium.launch({
  headless: true,
  ...(existsSync(systemChrome) ? { executablePath: systemChrome } : {}),
});

try {
  const context = await browser.newContext();
  const setupPage = await context.newPage();
  const writerPage = await context.newPage();
  const invalidatorPage = await context.newPage();
  const databaseName = `version-invalidation-multi-tab-${Date.now()}-${Math.random().toString(36).slice(2)}`;

  await Promise.all([
    setupPage.goto(`http://127.0.0.1:${port}/`),
    writerPage.goto(`http://127.0.0.1:${port}/`),
    invalidatorPage.goto(`http://127.0.0.1:${port}/`),
  ]);

  await setupPage.evaluate(async ({ databaseName }) => {
    const { createRxDatabase } = await import('/bundle.mjs');
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
    const db = await createRxDatabase({ name: databaseName });
    await db.addCollections({ widgets: { schema: schemaV0 } });
    await db.collections.widgets.storageCollection.bulkWrite([
      { id: 'master-cache', title: 'synced cache row' },
    ], { replicationOrigin: { role: 'ctox_instance', peerId: 'native-smoke' } });
    const marker = await db.storage.collectionSchemaMarker('widgets');
    const unsynced = await db.getUnsyncedWriteSummary();
    await db.close();
    if (marker?.state !== 'ready' || marker?.declaredVersion !== 0) {
      throw new Error('setup did not persist a ready v0 marker');
    }
    if ((unsynced?.total || 0) !== 0) {
      throw new Error('setup accidentally left pushable rows');
    }
  }, { databaseName });

  // Writer tab keeps a live v0 collection handle and races a local write against
  // the version invalidation in the other tab. Web Locks + realm serialization
  // must keep the dirty check and clear atomic: either the write wins and the
  // version change fails closed, or invalidation wins and the write fails closed.
  await writerPage.evaluate(async ({ databaseName }) => {
    const { createRxDatabase } = await import('/bundle.mjs');
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
    globalThis.__writerDb = await createRxDatabase({ name: databaseName });
    await globalThis.__writerDb.addCollections({ widgets: { schema: schemaV0 } });
    globalThis.__writerReady = true;
  }, { databaseName });

  await invalidatorPage.evaluate(async ({ databaseName }) => {
    const { createRxDatabase } = await import('/bundle.mjs');
    globalThis.__invalidatorDb = await createRxDatabase({ name: databaseName });
    globalThis.__invalidatorReady = true;
  }, { databaseName });

  await writerPage.waitForFunction(() => globalThis.__writerReady === true);
  await invalidatorPage.waitForFunction(() => globalThis.__invalidatorReady === true);

  const race = await Promise.all([
    writerPage.evaluate(async () => {
      // Yield once so both sides contend under the collection mutation lock.
      await new Promise((resolve) => setTimeout(resolve, 0));
      let writeErrorCode = '';
      let writeErrorMessage = '';
      try {
        await globalThis.__writerDb.collections.widgets.storageCollection.bulkWrite([
          { id: 'concurrent-local', title: 'must not sneak past invalidation' },
        ]);
        return { ok: true, writeErrorCode, writeErrorMessage };
      } catch (error) {
        writeErrorCode = error?.code || '';
        writeErrorMessage = error?.message || String(error);
        return { ok: false, writeErrorCode, writeErrorMessage };
      }
    }),
    invalidatorPage.evaluate(async () => {
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
      let blockedCode = '';
      let blockedMessage = '';
      let blockedDetail = null;
      let invalidation = null;
      try {
        await globalThis.__invalidatorDb.addCollections({ widgets: { schema: schemaV1 } });
        invalidation = globalThis.__invalidatorDb.collections.widgets.versionInvalidation;
        return {
          ok: true,
          blockedCode,
          blockedMessage,
          blockedDetail,
          invalidation,
        };
      } catch (error) {
        blockedCode = error?.code || '';
        blockedMessage = error?.message || String(error);
        blockedDetail = {
          pushableRows: error?.pushableRows,
          pendingBatches: error?.pendingBatches,
        };
        return {
          ok: false,
          blockedCode,
          blockedMessage,
          blockedDetail,
          invalidation,
        };
      }
    }),
  ]);

  const [writeResult, invalidationResult] = race;

  const after = await invalidatorPage.evaluate(async ({ databaseName, invalidationSucceeded }) => {
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
    // Prefer the invalidator handle (already open). If invalidation failed closed,
    // the collection is not registered there, so open a fresh read path.
    let marker = null;
    let rows = [];
    let unsynced = { total: 0, byCollection: {} };
    if (invalidationSucceeded && globalThis.__invalidatorDb?.collections?.widgets) {
      marker = await globalThis.__invalidatorDb.storage.collectionSchemaMarker('widgets');
      rows = await globalThis.__invalidatorDb.collections.widgets.storageCollection
        .allDocuments({ withDeleted: true });
      unsynced = await globalThis.__invalidatorDb.getUnsyncedWriteSummary();
    } else {
      const db = await (await import('/bundle.mjs')).createRxDatabase({ name: databaseName });
      marker = await db.storage.collectionSchemaMarker('widgets');
      rows = await db.storage.collection('widgets', { schema: schemaV0 })
        .allDocuments({ withDeleted: true });
      unsynced = await db.getUnsyncedWriteSummary();
      await db.close();
    }
    return {
      marker,
      rowIds: rows.map((row) => row.id).sort(),
      unsynced,
      hasConcurrentLocal: rows.some((row) => row.id === 'concurrent-local'),
      schemaProbe: schemaV1.version,
    };
  }, { databaseName, invalidationSucceeded: invalidationResult.ok });

  await writerPage.evaluate(async () => {
    try { await globalThis.__writerDb?.close?.(); } catch {}
  });
  await invalidatorPage.evaluate(async () => {
    try { await globalThis.__invalidatorDb?.close?.(); } catch {}
  });
  await setupPage.evaluate(async ({ databaseName }) => {
    for (const name of [databaseName, `${databaseName}__recovery_v2`]) {
      await new Promise((resolveDelete) => {
        const request = indexedDB.deleteDatabase(name);
        request.onsuccess = request.onerror = request.onblocked = () => resolveDelete();
      });
    }
  }, { databaseName });

  const writeWon = writeResult.ok === true;
  const invalidationWon = invalidationResult.ok === true
    && invalidationResult.invalidation?.invalidated === true;
  const invalidationBlocked = invalidationResult.ok === false
    && invalidationResult.blockedCode === 'collection_version_invalidation_blocked';
  const writeFailedClosed = writeResult.ok === false
    && (
      writeResult.writeErrorCode === 'collection_version_not_ready'
      || writeResult.writeErrorCode === 'collection_version_invalidation_blocked'
      || /not writable until browser cache version invalidation finishes/i.test(writeResult.writeErrorMessage)
      || /Nothing was discarded/i.test(writeResult.writeErrorMessage)
    );

  assert(writeWon || writeFailedClosed, `concurrent write neither succeeded nor failed closed: ${JSON.stringify(writeResult)}`);
  assert(
    (invalidationWon && writeFailedClosed)
      || (invalidationBlocked && writeWon)
      || (invalidationBlocked && writeFailedClosed),
    `TOCTOU race produced an illegal combination: write=${JSON.stringify(writeResult)} invalidation=${JSON.stringify(invalidationResult)} after=${JSON.stringify(after)}`,
  );
  // The TOCTOU core: a write must never land between dirty-check and clear.
  // If invalidation reports success, the concurrent local row must not exist.
  assert(
    !(invalidationWon && after.hasConcurrentLocal),
    'concurrent local write landed between dirty-check and collection clear (TOCTOU)',
  );
  if (invalidationWon) {
    assert(after.marker?.state === 'ready' && after.marker?.declaredVersion === 1, 'successful invalidation did not finalize the v1 marker');
    assert(after.rowIds.length === 0, 'successful invalidation left primary rows behind');
    assert((after.unsynced?.total || 0) === 0, 'successful invalidation left pushable rows behind');
    assert(invalidationResult.invalidation?.clearedRows === 1, 'successful invalidation did not report clearing the master cache row');
  }
  if (invalidationBlocked) {
    assert(invalidationResult.blockedMessage.includes('Nothing was discarded'), 'blocked invalidation message does not state data preservation');
    assert(after.marker?.declaredVersion === 0 && after.marker?.state === 'ready', 'blocked invalidation modified the v0 marker');
    assert(after.rowIds.includes('master-cache'), 'blocked invalidation discarded the master cache row');
    if (writeWon) {
      assert(after.hasConcurrentLocal, 'winning concurrent write is missing after blocked invalidation');
      assert(
        Number(invalidationResult.blockedDetail?.pushableRows || 0) >= 1
          || Number(after.unsynced?.byCollection?.widgets || 0) >= 1,
        'blocked invalidation did not observe the concurrent pushable write',
      );
    }
  }
  console.log('ctox-rxdb version invalidation multi-tab smoke OK assertions=8');
} finally {
  await browser.close();
  await new Promise((resolveClose) => server.close(resolveClose));
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}
