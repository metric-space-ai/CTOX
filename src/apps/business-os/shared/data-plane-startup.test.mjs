import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const appSource = readFileSync(new URL('../app.js', import.meta.url), 'utf8');

test('shell reopens IndexedDB only after a settled connection-closing error', () => {
  assert.match(appSource, /openBusinessDbAndRegisterCoreCollections\(dbName\)/);
  assert.match(appSource, /const maxAttempts = 3/);
  assert.match(
    appSource,
    /const retryable = isIndexedDbConnectionClosingError\(error\) && attempt < maxAttempts/,
  );
  assert.match(appSource, /await state\.db\?\.close\?\.\(\)/);
  assert.match(appSource, /state\.db = null/);
});

test('slow core registration is awaited without racing or closing its live IndexedDB', () => {
  const openStart = appSource.indexOf('async function openBusinessDbAndRegisterCoreCollections');
  const openEnd = appSource.indexOf('function isIndexedDbConnectionClosingError', openStart);
  const openBlock = appSource.slice(openStart, openEnd);
  const registerStart = appSource.indexOf('async function registerCoreCollections');
  const registerEnd = appSource.indexOf('async function primeWindowGeometryCache', registerStart);
  const registerBlock = appSource.slice(registerStart, registerEnd);

  assert.match(openBlock, /await registerCoreCollections\(\)/);
  assert.doesNotMatch(openBlock, /CtoxCoreCollectionRegistrationTimeout|timeoutMs/);
  assert.match(registerBlock, /await state\.db\.addCollections\(consolidated\)/);
  assert.doesNotMatch(registerBlock, /Promise\.race|setTimeout|timeoutMs/);
});

test('generic IndexedDB timeouts are not treated as schema corruption', () => {
  const classifierStart = appSource.indexOf('function isRxDbSchemaDriftError');
  const classifierEnd = appSource.indexOf('function hasLiveModulePreloadDataPlane', classifierStart);
  const classifierBlock = appSource.slice(classifierStart, classifierEnd);

  assert.match(classifierBlock, /RxDB Error-Code: DB6/);
  assert.match(classifierBlock, /previousSchemaHash/);
  assert.doesNotMatch(classifierBlock, /timed out|IndexedDB lock|open blocked/);
});
