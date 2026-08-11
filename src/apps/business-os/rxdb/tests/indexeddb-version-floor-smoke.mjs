// Once a browser has opened an IndexedDB version, later releases must never
// request a lower version. Version 4 shipped before the collection schema
// marker feature was temporarily removed, so the storage schema must retain
// that version floor and its object store even when the store is currently
// unused.

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const testDir = dirname(fileURLToPath(import.meta.url));
const source = readFileSync(resolve(testDir, '../src/storage-indexeddb.mjs'), 'utf8');

const versionMatch = source.match(/const DB_VERSION = (\d+);/);
assert.ok(versionMatch, 'storage source must declare an IndexedDB version');
assert.ok(
  Number(versionMatch[1]) >= 4,
  `IndexedDB version must never fall below the shipped v4 floor (got ${versionMatch[1]})`,
);
assert.match(source, /const COLLECTION_SCHEMA_MARKER_STORE = 'collectionSchemaMarkers';/);
assert.match(
  source,
  /createObjectStore\(COLLECTION_SCHEMA_MARKER_STORE, \{ keyPath: 'collection' \}\)/,
  'v4 profiles must retain the shipped collectionSchemaMarkers object store',
);

console.log('indexeddb version floor smoke OK');
