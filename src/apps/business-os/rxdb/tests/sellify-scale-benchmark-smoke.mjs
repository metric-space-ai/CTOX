import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const tool = resolve(here, '../../../../core/rxdb/tools/sellify_scale_benchmark.mjs');
const run = spawnSync(process.execPath, [
  tool,
  '--runs', '2',
  '--scale-divisor', '10000',
  '--output', '-',
], {
  encoding: 'utf8',
  maxBuffer: 16 * 1024 * 1024,
});
assert.equal(run.status, 0, run.stderr || run.stdout);
const result = JSON.parse(run.stdout);
assert.equal(result.schema, 'ctox.business_os_sellify_scale.v1');
assert.equal(result.synthetic, true);
assert.equal(result.runs, 2);
assert.deepEqual(Object.keys(result.populations), [
  'sellify_activities',
  'sellify_campaigns',
  'sellify_people',
  'sellify_companies',
  'business_commands',
  'desktop_file_chunks',
]);
assert.ok(result.measurements.every((measurement) => measurement.queryRpcEquivalent === 4));
assert.ok(result.measurements.every((measurement) => measurement.materializedDocuments <= 800));
assert.ok(result.summary.maxMaterializedDocuments <= 800);
assert.ok(result.summary.maxQueryRpcEquivalent <= 5);
console.log('sellify scale benchmark smoke passed');
