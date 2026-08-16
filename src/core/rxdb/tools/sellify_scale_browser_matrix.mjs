#!/usr/bin/env node

import { spawnSync } from 'node:child_process';
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  rmSync,
  statSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const toolRoot = dirname(fileURLToPath(import.meta.url));
const repositoryRoot = resolve(toolRoot, '../../../..');
const smokeRunner = resolve(toolRoot, 'browser_rust_smoke.js');

function parseArguments(argv) {
  const options = {
    runs: 30,
    output: '-',
    runtimeRoot: '',
    profileRoot: '',
    keepArtifacts: false,
    timeoutMs: 300_000,
    businessPort: 8877,
    signalingPort: 18876,
    selfTest: false,
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
    else if (argument === '--runtime-root') options.runtimeRoot = resolve(next());
    else if (argument === '--profile-root') options.profileRoot = resolve(next());
    else if (argument === '--timeout-ms') options.timeoutMs = Number(next());
    else if (argument === '--business-port') options.businessPort = Number(next());
    else if (argument === '--signaling-port') options.signalingPort = Number(next());
    else if (argument === '--keep-artifacts') options.keepArtifacts = true;
    else if (argument === '--self-test') options.selfTest = true;
    else throw new Error(`unknown argument: ${argument}`);
  }
  if (!Number.isInteger(options.runs) || options.runs < 1 || options.runs > 100) {
    throw new Error('--runs must be an integer between 1 and 100');
  }
  if (!Number.isInteger(options.timeoutMs) || options.timeoutMs < 1_000 || options.timeoutMs > 900_000) {
    throw new Error('--timeout-ms must be an integer between 1000 and 900000');
  }
  for (const [name, value] of [
    ['--business-port', options.businessPort],
    ['--signaling-port', options.signalingPort],
  ]) {
    if (!Number.isInteger(value) || value < 1 || value > 65535) {
      throw new Error(`${name} must be an integer between 1 and 65535`);
    }
  }
  if (options.businessPort === options.signalingPort) {
    throw new Error('--business-port and --signaling-port must differ');
  }
  return options;
}

function parseEvidence(output) {
  const evidence = {};
  for (const line of String(output || '').split(/\r?\n/)) {
    const match = /^([a-z0-9_]+)=(.*)$/.exec(line.trim());
    if (!match) continue;
    const [, key, value] = match;
    evidence[key] = value;
  }
  return evidence;
}

function numberEvidence(evidence, key) {
  const value = Number(evidence[key]);
  if (!Number.isFinite(value)) throw new Error(`missing numeric smoke evidence: ${key}`);
  return value;
}

function percentile(values, quantile) {
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.max(0, Math.ceil(sorted.length * quantile) - 1)] ?? 0;
}

function summarize(measurements) {
  const usable = measurements.map((measurement) => measurement.usableMs);
  return {
    runs: measurements.length,
    p50UsableMs: percentile(usable, 0.5),
    p95UsableMs: percentile(usable, 0.95),
    maxUsableMs: Math.max(...usable),
    maxQueryRpcs: Math.max(...measurements.map((measurement) => measurement.queryRpcs)),
    maxMaterializedDocuments: Math.max(
      ...measurements.map((measurement) => measurement.materializedDocuments),
    ),
    maxIndexedDbUsageBytes: Math.max(
      ...measurements.map((measurement) => measurement.indexedDbUsageAfter),
    ),
  };
}

function directoryBytes(target) {
  if (!existsSync(target)) return 0;
  const entry = statSync(target);
  if (entry.isFile()) return entry.size;
  if (!entry.isDirectory()) return 0;
  return readdirSync(target, { withFileTypes: true }).reduce((total, child) => (
    total + directoryBytes(join(target, child.name))
  ), 0);
}

function sha256File(target) {
  if (!target || !existsSync(target)) return null;
  const result = spawnSync('shasum', ['-a', '256', target], { encoding: 'utf8' });
  if (result.status !== 0) return null;
  return String(result.stdout || '').trim().split(/\s+/)[0] || null;
}

function gitRevision() {
  const result = spawnSync('git', ['rev-parse', 'HEAD'], {
    cwd: repositoryRoot,
    encoding: 'utf8',
  });
  return result.status === 0 ? String(result.stdout || '').trim() : null;
}

function validateMeasurement(measurement, { provisionOnly = false } = {}) {
  const failures = [];
  if (measurement.serverDocuments !== 304_515) failures.push('serverDocuments');
  if (measurement.queryRpcs > 5) failures.push('queryRpcs');
  if (measurement.materializedDocuments > 1_000) failures.push('materializedDocuments');
  if (measurement.visibleRows < 1 || measurement.visibleRows > 200) failures.push('visibleRows');
  if (!measurement.interactionReady) failures.push('interactionReady');
  if (measurement.fullPullBeforeRender) failures.push('fullPullBeforeRender');
  if (!measurement.windowedCoverage) failures.push('windowedCoverage');
  if (!measurement.budgetPassed) failures.push('budgetPassed');
  if (!provisionOnly && measurement.usableMs <= 0) failures.push('usableMs');
  if (failures.length) {
    throw new Error(`Sellify browser measurement failed: ${failures.join(', ')}`);
  }
}

function measurementFromOutput(output, phase, run) {
  const evidence = parseEvidence(output);
  const measurement = {
    phase,
    run,
    serverDocuments: numberEvidence(evidence, 'business_os_sellify_scale_server_documents'),
    totalDocuments: numberEvidence(evidence, 'business_os_sellify_scale_total_documents'),
    queryRpcs: numberEvidence(evidence, 'business_os_sellify_scale_query_rpcs'),
    queryResponses: numberEvidence(evidence, 'business_os_sellify_scale_query_responses'),
    materializedDocuments: numberEvidence(evidence, 'business_os_sellify_scale_materialized_documents'),
    visibleRows: numberEvidence(evidence, 'business_os_sellify_scale_visible_rows'),
    firstVisibleDataMs: numberEvidence(evidence, 'business_os_sellify_scale_first_visible_data_ms'),
    usableMs: numberEvidence(evidence, 'business_os_sellify_scale_first_render_ms'),
    queryWindowRenderMs: numberEvidence(evidence, 'business_os_sellify_scale_query_window_render_ms'),
    cachedDocumentsBeforeQuery: numberEvidence(
      evidence,
      'business_os_sellify_scale_cached_documents_before_query',
    ),
    indexedDbUsageBefore: numberEvidence(evidence, 'business_os_sellify_scale_indexeddb_usage_before'),
    indexedDbUsageAfter: numberEvidence(evidence, 'business_os_sellify_scale_indexeddb_usage_after'),
    nativeDatabaseBytes: numberEvidence(evidence, 'business_os_sellify_scale_native_database_bytes'),
    interactionReady: evidence.business_os_sellify_scale_interaction_ready === '1',
    fullPullBeforeRender: evidence.business_os_sellify_scale_full_pull_before_render === '1',
    windowedCoverage: evidence.business_os_sellify_scale_windowed_coverage === '1',
    budgetPassed: evidence.business_os_sellify_scale_budget_passed === '1',
    latencyTargetPassed: evidence.business_os_sellify_scale_latency_target_passed === '1',
    fixtureReused: evidence.business_os_sellify_scale_fixture_reused === '1',
    collectionReadiness: JSON.parse(
      evidence.business_os_sellify_scale_collection_readiness || '{}',
    ),
    startupMarks: JSON.parse(evidence.business_os_sellify_scale_startup_marks || '{}'),
    outerPhaseTimings: JSON.parse(evidence.outer_phase_timings || '{}'),
  };
  return measurement;
}

function runSelfTest() {
  const fixture = [
    'business_os_sellify_scale_server_documents=304515',
    'business_os_sellify_scale_total_documents=314169',
    'business_os_sellify_scale_query_rpcs=4',
    'business_os_sellify_scale_query_responses=4',
    'business_os_sellify_scale_materialized_documents=800',
    'business_os_sellify_scale_visible_rows=200',
    'business_os_sellify_scale_first_visible_data_ms=500',
    'business_os_sellify_scale_first_render_ms=600',
    'business_os_sellify_scale_query_window_render_ms=100',
    'business_os_sellify_scale_cached_documents_before_query=0',
    'business_os_sellify_scale_indexeddb_usage_before=10',
    'business_os_sellify_scale_indexeddb_usage_after=20',
    'business_os_sellify_scale_native_database_bytes=30',
    'business_os_sellify_scale_interaction_ready=1',
    'business_os_sellify_scale_full_pull_before_render=0',
    'business_os_sellify_scale_windowed_coverage=1',
    'business_os_sellify_scale_budget_passed=1',
    'business_os_sellify_scale_latency_target_passed=1',
    'business_os_sellify_scale_fixture_reused=1',
    'business_os_sellify_scale_collection_readiness={}',
    'business_os_sellify_scale_startup_marks={"usableAtMs":600}',
    'outer_phase_timings={"shellReadyWaitMs":400}',
  ].join('\n');
  const measurement = measurementFromOutput(fixture, 'self-test', 1);
  validateMeasurement(measurement);
  const summary = summarize([measurement, { ...measurement, usableMs: 900 }]);
  if (summary.p95UsableMs !== 900 || summary.maxQueryRpcs !== 4) {
    throw new Error('Sellify scale browser matrix percentile self-test failed');
  }
  process.stdout.write('sellify_scale_browser_matrix_self_test=ok\n');
}

function main() {
  const options = parseArguments(process.argv.slice(2));
  if (options.selfTest) return runSelfTest();

  const temporaryRoot = options.runtimeRoot
    ? ''
    : mkdtempSync(join(tmpdir(), 'ctox-sellify-browser-matrix-'));
  const runtimeRoot = options.runtimeRoot || join(temporaryRoot, 'runtime-root');
  const profileRoot = options.profileRoot || join(temporaryRoot, 'profiles');
  const warmProfile = join(profileRoot, 'warm');
  mkdirSync(runtimeRoot, { recursive: true });
  mkdirSync(profileRoot, { recursive: true });

  const ctoxBinary = process.env.CTOX_BIN
    || resolve(repositoryRoot, 'runtime/build/core-rxdb-integration-target/debug/ctox');
  const baseEnvironment = {
    ...process.env,
    CTOX_BIN: ctoxBinary,
    CTOX_SMOKE_ROOT: runtimeRoot,
    SMOKE_MODE: 'business-os-sellify-scale-ui',
    SMOKE_PAGE_PATH: '/index.html',
    SMOKE_DB_ID: 'sellify_scale_browser_matrix',
    SMOKE_BROWSER_PROFILE_DIR: '',
    SMOKE_SELLIFY_SCALE_PROVISION_ONLY: '0',
    BUSINESS_PORT: String(options.businessPort),
    SIGNALING_PORT: String(options.signalingPort),
  };

  const execute = (phase, run, extraEnvironment = {}) => {
    process.stderr.write(`[sellify-scale] ${phase} ${run}\n`);
    const result = spawnSync(process.execPath, [smokeRunner], {
      cwd: repositoryRoot,
      env: { ...baseEnvironment, ...extraEnvironment },
      encoding: 'utf8',
      maxBuffer: 64 * 1024 * 1024,
      timeout: options.timeoutMs,
    });
    if (result.error || result.status !== 0) {
      const tail = `${result.stdout || ''}\n${result.stderr || ''}`.trim().split(/\r?\n/).slice(-80).join('\n');
      throw new Error(`${phase} run ${run} failed (${result.status ?? 'spawn'}): ${result.error?.message || ''}\n${tail}`);
    }
    const measurement = measurementFromOutput(result.stdout, phase, run);
    validateMeasurement(measurement, { provisionOnly: phase === 'provision' });
    return measurement;
  };

  const startedAt = new Date().toISOString();
  execute('provision', 0, { SMOKE_SELLIFY_SCALE_PROVISION_ONLY: '1' });
  const cold = [];
  for (let run = 1; run <= options.runs; run += 1) cold.push(execute('cold', run));
  execute('warmup', 0, { SMOKE_BROWSER_PROFILE_DIR: warmProfile });
  const warm = [];
  for (let run = 1; run <= options.runs; run += 1) {
    warm.push(execute('warm', run, { SMOKE_BROWSER_PROFILE_DIR: warmProfile }));
  }

  const coldSummary = summarize(cold);
  const warmSummary = summarize(warm);
  const acceptance = {
    coldP95UnderFiveSeconds: coldSummary.p95UsableMs < 5_000,
    warmP95UnderOneSecond: warmSummary.p95UsableMs < 1_000,
    queryRpcBudget: Math.max(coldSummary.maxQueryRpcs, warmSummary.maxQueryRpcs) <= 5,
    materializationBudget: Math.max(
      coldSummary.maxMaterializedDocuments,
      warmSummary.maxMaterializedDocuments,
    ) <= 1_000,
  };
  const artifact = {
    schema: 'ctox.business_os_sellify_scale_browser_matrix.v1',
    generatedAt: new Date().toISOString(),
    startedAt,
    synthetic: true,
    gitRevision: gitRevision(),
    source: {
      smokeRunner,
      smokeRunnerSha256: sha256File(smokeRunner),
      ctoxBinary,
      ctoxBinarySha256: sha256File(ctoxBinary),
    },
    configuration: {
      runsPerState: options.runs,
      businessPort: options.businessPort,
      signalingPort: options.signalingPort,
      runtimeRoot,
      warmProfile,
    },
    profileDiskBytes: directoryBytes(warmProfile),
    summary: { cold: coldSummary, warm: warmSummary },
    acceptance,
    ok: Object.values(acceptance).every(Boolean),
    measurements: { cold, warm },
  };
  const serialized = `${JSON.stringify(artifact, null, 2)}\n`;
  if (options.output === '-') process.stdout.write(serialized);
  else {
    const output = resolve(options.output);
    mkdirSync(dirname(output), { recursive: true });
    writeFileSync(output, serialized);
  }

  if (!options.keepArtifacts && !options.runtimeRoot && temporaryRoot) {
    rmSync(temporaryRoot, { recursive: true, force: true });
  }
  if (!artifact.ok) process.exitCode = 1;
}

main();
