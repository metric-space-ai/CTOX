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
  response.end('<!doctype html><title>version invalidation sidecar window smoke</title>');
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
      createIndexedDbMetaBackend,
      QueryMetaStorage,
      SIDECAR_DATABASE_NAME,
    } = await import('/bundle.mjs');
    const suffix = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    const databaseName = `version-invalidation-sidecar-${suffix}`;
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
    const widgetsSidecarName = `${SIDECAR_DATABASE_NAME}_widgets`;
    const ticketsSidecarName = `${SIDECAR_DATABASE_NAME}_tickets`;

    // Establish ready v0 markers first. Missing markers invalidate sidecars too,
    // so demand windows must be seeded only after markers exist.
    let db = await createRxDatabase({ name: databaseName });
    await db.addCollections({
      widgets: { schema: schemaV0 },
      tickets: { schema: schemaV0 },
    });
    await db.collections.widgets.storageCollection.bulkWrite([
      { id: 'widget-cache', title: 'v0 cache' },
    ], { replicationOrigin: { role: 'ctox_instance', peerId: 'native-smoke' } });
    await db.collections.tickets.storageCollection.bulkWrite([
      { id: 'ticket-cache', title: 'other collection cache' },
    ], { replicationOrigin: { role: 'ctox_instance', peerId: 'native-smoke' } });
    await db.close();

    const widgetsBackend = createIndexedDbMetaBackend({ databaseName: widgetsSidecarName });
    const ticketsBackend = createIndexedDbMetaBackend({ databaseName: ticketsSidecarName });
    const widgetsSidecar = new QueryMetaStorage(widgetsBackend, { databaseName: widgetsSidecarName });
    const ticketsSidecar = new QueryMetaStorage(ticketsBackend, { databaseName: ticketsSidecarName });

    await widgetsSidecar.upsertQueryWindow({
      collection: 'widgets',
      queryFingerprint: 'widgets-open',
      offset: 0,
      limit: 50,
      documentIds: ['widget-1', 'widget-2'],
      complete: true,
      authoritativeRevision: 'widgets-rev-1',
    });
    await ticketsSidecar.upsertQueryWindow({
      collection: 'tickets',
      queryFingerprint: 'tickets-open',
      offset: 0,
      limit: 50,
      documentIds: ['ticket-1'],
      complete: true,
      authoritativeRevision: 'tickets-rev-1',
    });
    const widgetsWindowBefore = await widgetsSidecar.getQueryWindow(['widgets', 'widgets-open', 0, 50]);
    const ticketsWindowBefore = await ticketsSidecar.getQueryWindow(['tickets', 'tickets-open', 0, 50]);
    await widgetsSidecar.close?.();
    await ticketsSidecar.close?.();
    await widgetsBackend.close?.();
    await ticketsBackend.close?.();

    db = await createRxDatabase({ name: databaseName });
    await db.addCollections({
      widgets: { schema: schemaV1 },
      tickets: { schema: schemaV0 },
    });
    const invalidation = db.collections.widgets.versionInvalidation;
    const ticketsInvalidation = db.collections.tickets.versionInvalidation;
    const widgetsRows = await db.collections.widgets.storageCollection.allDocuments({ withDeleted: true });
    const ticketsRows = await db.collections.tickets.storageCollection.allDocuments({ withDeleted: true });
    await db.close();

    const widgetsBackendAfter = createIndexedDbMetaBackend({ databaseName: widgetsSidecarName });
    const ticketsBackendAfter = createIndexedDbMetaBackend({ databaseName: ticketsSidecarName });
    const widgetsSidecarAfter = new QueryMetaStorage(widgetsBackendAfter, { databaseName: widgetsSidecarName });
    const ticketsSidecarAfter = new QueryMetaStorage(ticketsBackendAfter, { databaseName: ticketsSidecarName });
    const widgetsWindowAfter = await widgetsSidecarAfter.getQueryWindow(['widgets', 'widgets-open', 0, 50]);
    const ticketsWindowAfter = await ticketsSidecarAfter.getQueryWindow(['tickets', 'tickets-open', 0, 50]);
    const widgetsWindowsScan = await widgetsBackendAfter.scanQueryWindows?.()
      || (await widgetsSidecarAfter.backend.scanQueryWindows());
    await widgetsSidecarAfter.close?.();
    await ticketsSidecarAfter.close?.();
    await widgetsBackendAfter.close?.();
    await ticketsBackendAfter.close?.();

    for (const name of [
      databaseName,
      `${databaseName}__recovery_v2`,
      widgetsSidecarName,
      ticketsSidecarName,
    ]) {
      await new Promise((resolveDelete) => {
        const request = indexedDB.deleteDatabase(name);
        request.onsuccess = request.onerror = request.onblocked = () => resolveDelete();
      });
    }

    return {
      widgetsWindowBefore,
      ticketsWindowBefore,
      invalidation,
      ticketsInvalidation,
      widgetsRows,
      ticketsRows,
      widgetsWindowAfter,
      ticketsWindowAfter,
      widgetsWindowsScan,
    };
  });

  assert(result.widgetsWindowBefore?.complete === true, 'widgets sidecar window was not seeded complete');
  assert(result.ticketsWindowBefore?.complete === true, 'tickets sidecar window was not seeded complete');
  assert(result.invalidation?.invalidated === true && result.invalidation?.clearedRows === 1, 'widgets version invalidation did not clear the primary collection');
  assert(result.ticketsInvalidation?.invalidated === false, 'tickets collection was unexpectedly invalidated');
  assert(result.widgetsRows.length === 0, 'widgets primary rows survived invalidation');
  assert(result.ticketsRows.length === 1 && result.ticketsRows[0]?.id === 'ticket-cache', 'other collection primary rows were touched');
  assert(result.widgetsWindowAfter === null, 'widgets demand/query sidecar window survived invalidation');
  assert(
    Array.isArray(result.widgetsWindowsScan) && result.widgetsWindowsScan.length === 0,
    'widgets sidecar still contains query windows after invalidation',
  );
  assert(
    result.ticketsWindowAfter?.complete === true
      && result.ticketsWindowAfter?.documentIds?.join(',') === 'ticket-1',
    'other collection sidecar window was cleared or mutated',
  );
  console.log('ctox-rxdb version invalidation sidecar-window smoke OK assertions=9');
} finally {
  await browser.close();
  await new Promise((resolveClose) => server.close(resolveClose));
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}
