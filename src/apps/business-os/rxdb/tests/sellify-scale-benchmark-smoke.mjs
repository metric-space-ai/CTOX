import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
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

const browserSmoke = readFileSync(
  resolve(here, '../../../../core/rxdb/tools/browser_rust_smoke.js'),
  'utf8',
);
const scaleSetupStart = browserSmoke.indexOf('let sellifyScaleSeed = null;');
const nativePeerStart = browserSmoke.indexOf('let ctox = startCtoxServer();', scaleSetupStart);
const scaleSetupBlock = browserSmoke.slice(scaleSetupStart, nativePeerStart);
assert.ok(scaleSetupStart >= 0 && nativePeerStart > scaleSetupStart);
assert.match(scaleSetupBlock, /prepareBusinessOsSellifyScaleModuleFixture\(\);/);
assert.match(scaleSetupBlock, /sellifyScaleSeed = await seedBusinessOsSellifyScaleNativeSetup\(\);/);
const fixtureStart = browserSmoke.indexOf('function prepareBusinessOsSellifyScaleModuleFixture()');
const seedStart = browserSmoke.indexOf('async function seedBusinessOsSellifyScaleNativeSetup()', fixtureStart);
const fixtureBlock = browserSmoke.slice(fixtureStart, seedStart);
assert.ok(fixtureStart >= 0 && seedStart > fixtureStart);
assert.match(fixtureBlock, /'runtime',\s*'business-os',\s*'local-modules',\s*'sellify-scale-smoke'/);
assert.match(fixtureBlock, /syncProfile: 'demand-only'/);
assert.match(fixtureBlock, /schema_format: 'ctox-business-os-module-collections-v1'/);
const seedEnd = browserSmoke.indexOf('function computeBusinessOsReleaseModuleBundle', seedStart);
const seedBlock = browserSmoke.slice(seedStart, seedEnd);
assert.match(seedBlock, /sqlite\(Object\.keys\(tables\)\.map/);
console.log('sellify scale benchmark smoke passed');
