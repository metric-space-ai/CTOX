import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';

// Replace only the network import boundary; exercise the shipped loader's
// promise lifetime, concurrency and retry behavior without a browser network.
const source = readFileSync(new URL('../../shared/rxdb-runtime.js', import.meta.url), 'utf8');
let attempts = 0;
let rejectImport;
const runtime = { createRxDatabase() {} };
const urls = [];
const load = runInNewContext(
  `${source.replace(/export /g, '').replace('import(RXDB_BUNDLE_URL)', 'importRuntime(RXDB_BUNDLE_URL)')}\nloadRxdbRuntime;`,
  {
    importRuntime(url) {
      urls.push(url);
      attempts += 1;
      return attempts === 1
        ? new Promise((_, reject) => { rejectImport = reject; })
        : Promise.resolve(runtime);
    },
  },
);
const first = load();
assert.equal(load(), first, 'concurrent consumers share the pending import');
rejectImport(new Error('transient IndexedDB/module import failure'));
await assert.rejects(first, /transient/);
const retry = load();
assert.notEqual(retry, first, 'a rejected import is not memoized forever');
assert.equal(await retry, runtime);
assert.equal(load(), retry, 'successful imports stay memoized');
assert.equal(attempts, 2);
assert.equal(new Set(urls).size, 1, 'retry must not create a second bundle URL');
console.log('canonical RxDB loader retry OK');
