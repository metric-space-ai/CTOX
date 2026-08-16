import http from 'node:http';
import { existsSync, readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const testDir = dirname(fileURLToPath(import.meta.url));
const bundle = readFileSync(resolve(testDir, '../dist/ctox-rxdb-js.mjs'));
const server = http.createServer((request, response) => {
  if (request.url === '/bundle.mjs') {
    response.writeHead(200, { 'content-type': 'text/javascript' });
    response.end(bundle);
    return;
  }
  response.writeHead(200, { 'content-type': 'text/html' });
  response.end('<!doctype html><title>demand cache migration smoke</title>');
});
await new Promise((resolveReady) => server.listen(0, '127.0.0.1', resolveReady));
const { port } = server.address();
const playwrightModule = process.env.PLAYWRIGHT_MODULE_PATH
  ? pathToFileURL(resolve(process.env.PLAYWRIGHT_MODULE_PATH, 'index.mjs')).href
  : '../../node_modules/playwright/index.mjs';
const { chromium } = await import(playwrightModule);
const systemChrome = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const browser = await chromium.launch({
  headless: true,
  ...(existsSync(systemChrome) ? { executablePath: systemChrome } : {}),
});

try {
  const page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${port}/`);
  const result = await page.evaluate(async () => {
    const { openCtoxIndexedDbStorage } = await import('/bundle.mjs');
    const databaseName = `demand-cache-${Date.now()}-${Math.random().toString(36).slice(2)}`;
    const storage = await openCtoxIndexedDbStorage({ databaseName });
    const schema = {
      version: 0,
      type: 'object',
      primaryKey: 'id',
      properties: { id: { type: 'string', maxLength: 128 }, title: { type: 'string' } },
      required: ['id'],
    };
    const cached = storage.collection('sellify_people', { schema });
    const local = storage.collection('sellify_companies', { schema });
    const retained = storage.collection('business_commands', { schema });
    await cached._bulkUpsertOnce([
      { id: 'person-1', title: 'cached one' },
      { id: 'person-2', title: 'cached two' },
    ], { replicationOrigin: { role: 'native', peerId: 'native-test' } });
    await retained._bulkUpsertOnce([
      { id: 'command-1', title: 'must remain' },
    ], { replicationOrigin: { role: 'native', peerId: 'native-test' } });
    await local.bulkUpsert([{ id: 'company-local', title: 'unsynced write' }]);

    let localWriteCode = '';
    try {
      await storage.clearCachedCollections(['sellify_companies']);
    } catch (error) {
      localWriteCode = error?.code || '';
    }
    const localStillThere = await local.findOne('company-local');
    const cleared = await storage.clearCachedCollections(['sellify_people']);
    const cachedAfter = await Promise.all([
      cached.findOne('person-1'),
      cached.findOne('person-2'),
    ]);
    const retainedAfter = await retained.findOne('command-1');

    cached.close();
    local.close();
    retained.close();
    storage.close();
    indexedDB.deleteDatabase(databaseName);
    indexedDB.deleteDatabase(`${databaseName}__recovery_v2`);
    return {
      localWriteCode,
      localStillThere: Boolean(localStillThere),
      cleared,
      cachedAfter: cachedAfter.filter(Boolean).length,
      retainedAfter: Number(Boolean(retainedAfter)),
    };
  });
  assert(result.localWriteCode === 'CTOX_DEMAND_CACHE_UNSYNCED_WRITES', 'unsynced local rows must fail closed');
  assert(result.localStillThere, 'a collection with local writes must remain intact');
  assert(result.cleared.total === 2, 'native cache rows must be counted before deletion');
  assert(result.cachedAfter === 0, 'targeted native cache rows must be deleted');
  assert(result.retainedAfter === 1, 'non-target collections must remain intact');
  console.log('ctox-rxdb demand cache migration browser smoke OK', result);
} finally {
  await browser.close();
  await new Promise((resolveClose) => server.close(resolveClose));
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}
