#!/usr/bin/env node

import { spawnSync } from 'node:child_process';
import { mkdtempSync, mkdirSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, resolve } from 'node:path';
import { performance } from 'node:perf_hooks';

const POPULATIONS = Object.freeze({
  sellify_activities: 139_804,
  sellify_campaigns: 86_551,
  sellify_people: 60_640,
  sellify_companies: 17_520,
  business_commands: 5_964,
  desktop_file_chunks: 3_690,
});
const LARGE_COLLECTIONS = Object.freeze([
  'sellify_activities',
  'sellify_campaigns',
  'sellify_people',
  'sellify_companies',
]);

function parseArguments(argv) {
  const options = {
    runs: 30,
    output: '-',
    database: '',
    keepDatabase: false,
    scaleDivisor: 1,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    const next = () => {
      const value = argv[index + 1];
      if (!value) throw new Error(`${argument} requires a value`);
      index += 1;
      return value;
    };
    if (argument === '--runs') options.runs = Number(next());
    else if (argument === '--output') options.output = next();
    else if (argument === '--database') options.database = resolve(next());
    else if (argument === '--keep-database') options.keepDatabase = true;
    else if (argument === '--scale-divisor') options.scaleDivisor = Number(next());
    else throw new Error(`unknown argument: ${argument}`);
  }
  if (!Number.isInteger(options.runs) || options.runs < 1 || options.runs > 1_000) {
    throw new Error('--runs must be an integer between 1 and 1000');
  }
  if (!Number.isInteger(options.scaleDivisor) || options.scaleDivisor < 1) {
    throw new Error('--scale-divisor must be a positive integer');
  }
  return options;
}

function sqlite(database, sql) {
  const result = spawnSync('sqlite3', [database], {
    input: sql,
    encoding: 'utf8',
    maxBuffer: 64 * 1024 * 1024,
  });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    throw new Error(`sqlite3 failed (${result.status}): ${String(result.stderr || '').trim()}`);
  }
  return String(result.stdout || '').trim();
}

function sqlString(value) {
  return String(value).replaceAll("'", "''");
}

function tableName(collection) {
  return `ctox_business_os__${collection}__v0`;
}

function syntheticDocument(collection, index, population, baseTime) {
  const padded = String(index).padStart(7, '0');
  const id = `${collection}_${padded}`;
  const updatedAtMs = baseTime - (population - index);
  const common = {
    id,
    status: index % 3 === 0 ? 'active' : index % 3 === 1 ? 'paused' : 'completed',
    sort_key: population - index,
    updated_at_ms: updatedAtMs,
    _deleted: false,
    _rev: '1-sellify-scale',
    _meta: { lwt: updatedAtMs },
    _attachments: {},
  };
  if (collection === 'sellify_activities') {
    return {
      ...common,
      activity_type: ['email', 'call', 'meeting'][index % 3],
      campaign_id: `sellify_campaigns_${String(index % 86_551).padStart(7, '0')}`,
      person_id: `sellify_people_${String(index % 60_640).padStart(7, '0')}`,
      summary: `Synthetic activity ${index}`,
    };
  }
  if (collection === 'sellify_campaigns') {
    return {
      ...common,
      name: `Synthetic campaign ${index}`,
      owner_id: `owner-${index % 32}`,
      market: ['de', 'at', 'ch'][index % 3],
    };
  }
  if (collection === 'sellify_people') {
    return {
      ...common,
      full_name: `Synthetic Person ${index}`,
      company_id: `sellify_companies_${String(index % 17_520).padStart(7, '0')}`,
      email: `person-${index}@example.invalid`,
    };
  }
  if (collection === 'sellify_companies') {
    return {
      ...common,
      name: `Synthetic Company ${index}`,
      domain: `company-${index}.example.invalid`,
      country: ['DE', 'AT', 'CH'][index % 3],
    };
  }
  if (collection === 'business_commands') {
    return {
      ...common,
      command_id: id,
      module: 'sellify-scale',
      command_type: 'ctox.scale.fixture',
      record_id: `scale-record-${index}`,
      payload: { synthetic: true, index },
    };
  }
  return {
    ...common,
    file_id: `synthetic-file-${Math.floor(index / 8)}`,
    chunk_index: index % 8,
    data: `synthetic-chunk-${padded}`,
  };
}

function createTables(database) {
  const statements = Object.keys(POPULATIONS).map((collection) => `
    CREATE TABLE IF NOT EXISTS "${tableName(collection)}" (
      id TEXT PRIMARY KEY,
      revision TEXT NOT NULL,
      deleted INTEGER NOT NULL DEFAULT 0,
      lastWriteTime REAL NOT NULL,
      data TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS "idx_${collection}_lwt"
      ON "${tableName(collection)}" (lastWriteTime DESC);
  `);
  sqlite(database, `PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; ${statements.join('\n')}`);
}

function seedCollection(database, collection, population, baseTime) {
  const batchSize = 500;
  for (let start = 1; start <= population; start += batchSize) {
    const end = Math.min(population, start + batchSize - 1);
    const rows = [];
    for (let index = start; index <= end; index += 1) {
      const document = syntheticDocument(collection, index, population, baseTime);
      rows.push(`('${sqlString(document.id)}','1-sellify-scale',0,${document.updated_at_ms},'${sqlString(JSON.stringify(document))}')`);
    }
    sqlite(database, `
      BEGIN IMMEDIATE;
      INSERT INTO "${tableName(collection)}" (id, revision, deleted, lastWriteTime, data)
      VALUES ${rows.join(',')};
      COMMIT;
    `);
  }
}

function percentile(values, quantile) {
  const sorted = [...values].sort((left, right) => left - right);
  const index = Math.max(0, Math.ceil(sorted.length * quantile) - 1);
  return sorted[index] ?? 0;
}

function round(value) {
  return Math.round(value * 1000) / 1000;
}

function queryWindow(database, collection) {
  const startedAt = performance.now();
  const output = sqlite(database, `
    SELECT json_extract(data, '$.id')
    FROM "${tableName(collection)}"
    WHERE deleted = 0 AND json_extract(data, '$.status') = 'active'
    ORDER BY lastWriteTime DESC
    LIMIT 200;
  `);
  return {
    elapsedMs: round(performance.now() - startedAt),
    documentCount: output ? output.split('\n').length : 0,
  };
}

function main() {
  const options = parseArguments(process.argv.slice(2));
  const temporaryRoot = options.database ? '' : mkdtempSync(`${tmpdir()}/ctox-sellify-scale-`);
  const database = options.database || resolve(temporaryRoot, 'sellify-scale.sqlite3');
  mkdirSync(dirname(database), { recursive: true });
  const populations = Object.fromEntries(
    Object.entries(POPULATIONS).map(([collection, count]) => [
      collection,
      Math.ceil(count / options.scaleDivisor),
    ]),
  );
  const seedStartedAt = performance.now();
  createTables(database);
  const baseTime = 1_800_000_000_000;
  for (const [collection, population] of Object.entries(populations)) {
    seedCollection(database, collection, population, baseTime);
  }
  sqlite(database, 'PRAGMA wal_checkpoint(TRUNCATE);');
  const seedMs = round(performance.now() - seedStartedAt);
  const runs = [];
  for (let run = 1; run <= options.runs; run += 1) {
    const windows = Object.fromEntries(
      LARGE_COLLECTIONS.map((collection) => [collection, queryWindow(database, collection)]),
    );
    const elapsedMs = round(
      Object.values(windows).reduce((total, window) => total + window.elapsedMs, 0),
    );
    runs.push({
      run,
      elapsedMs,
      materializedDocuments: Object.values(windows)
        .reduce((total, window) => total + window.documentCount, 0),
      queryRpcEquivalent: LARGE_COLLECTIONS.length,
      windows,
    });
  }
  const elapsedValues = runs.map((run) => run.elapsedMs);
  const result = {
    schema: 'ctox.business_os_sellify_scale.v1',
    generatedAt: new Date().toISOString(),
    synthetic: true,
    populations,
    sellifyServerDocuments: LARGE_COLLECTIONS.reduce(
      (total, collection) => total + populations[collection],
      0,
    ),
    totalDocuments: Object.values(populations).reduce((total, count) => total + count, 0),
    runs: options.runs,
    seedMs,
    databaseBytes: statSync(database).size,
    queryWindowLimit: 200,
    summary: {
      p50Ms: round(percentile(elapsedValues, 0.5)),
      p95Ms: round(percentile(elapsedValues, 0.95)),
      maxMs: round(Math.max(...elapsedValues)),
      maxMaterializedDocuments: Math.max(...runs.map((run) => run.materializedDocuments)),
      maxQueryRpcEquivalent: Math.max(...runs.map((run) => run.queryRpcEquivalent)),
    },
    measurements: runs,
  };
  const serialized = `${JSON.stringify(result, null, 2)}\n`;
  if (options.output === '-') process.stdout.write(serialized);
  else {
    const output = resolve(options.output);
    mkdirSync(dirname(output), { recursive: true });
    writeFileSync(output, serialized);
  }
  if (!options.database && !options.keepDatabase) rmSync(temporaryRoot, { recursive: true });
}

main();
