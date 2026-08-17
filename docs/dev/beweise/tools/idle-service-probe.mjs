#!/usr/bin/env node

import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { writeFileSync } from "node:fs";

function parseArgs(argv) {
  const values = new Map();
  for (let index = 0; index < argv.length; index += 2) {
    const key = argv[index];
    const value = argv[index + 1];
    if (!key?.startsWith("--") || value === undefined) {
      throw new Error(`invalid argument near ${key ?? "<end>"}`);
    }
    values.set(key.slice(2), value);
  }
  return values;
}

function requireArg(args, name) {
  const value = args.get(name);
  if (!value) {
    throw new Error(`missing --${name}`);
  }
  return value;
}

function positiveNumber(args, name, fallback) {
  const raw = args.get(name);
  const value = raw === undefined ? fallback : Number(raw);
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`--${name} must be a positive number`);
  }
  return value;
}

function run(command, commandArgs) {
  return execFileSync(command, commandArgs, {
    encoding: "utf8",
    maxBuffer: 16 * 1024 * 1024,
  }).trim();
}

function percentile(values, fraction) {
  if (values.length === 0) return null;
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.ceil(fraction * sorted.length) - 1];
}

function parseCpuTime(raw) {
  const [dayPart, clockPart] = raw.includes("-") ? raw.split("-", 2) : ["0", raw];
  const parts = clockPart.split(":").map(Number);
  let hours = 0;
  let minutes = 0;
  let seconds = 0;
  if (parts.length === 3) [hours, minutes, seconds] = parts;
  else if (parts.length === 2) [minutes, seconds] = parts;
  else [seconds] = parts;
  return Number(dayPart) * 86400 + hours * 3600 + minutes * 60 + seconds;
}

function processSample(pid) {
  const raw = run("ps", ["-p", String(pid), "-o", "%cpu=,rss=,state=,time="]);
  const match = raw.match(/^\s*([0-9.]+)\s+(\d+)\s+(\S+)\s+(\S+)\s*$/);
  if (!match) throw new Error(`cannot parse ps sample: ${raw}`);
  return {
    captured_at: new Date().toISOString(),
    cpu_percent: Number(match[1]),
    rss_kib: Number(match[2]),
    state: match[3],
    cpu_time_seconds: parseCpuTime(match[4]),
  };
}

function jsonCommand(binary, commandArgs) {
  return JSON.parse(run(binary, commandArgs));
}

function sqliteScalar(database, sql) {
  return Number(run("sqlite3", ["-readonly", database, sql]) || "0");
}

function sha256(value) {
  return createHash("sha256").update(value).digest("hex");
}

function databaseSnapshot(stateRoot) {
  const coreDatabase = `${stateRoot}/ctox.sqlite3`;
  const businessDatabase = `${stateRoot}/business-os.sqlite3`;
  const rxdbDatabase = `${stateRoot}/business-os-rxdb.sqlite3`;
  const commandTable = "ctox_business_os__business_commands__v1";
  const retryCandidate = `(
    json_extract(data, '$.status') IN ('pending_sync', 'waiting_dependencies')
    OR (
      (
        json_extract(data, '$.status') = 'accepted'
        OR (
          json_extract(data, '$.status') = 'failed'
          AND COALESCE(json_extract(data, '$.terminal_status'), 'none') = 'none'
        )
      )
      AND json_extract(data, '$.command_type') IN (
        'external_sql.sync.refresh',
        'external_sql.write',
        'outbound.research_source.generate_adapter',
        'outbound.research_source.test',
        'outbound.research_source.auth_assist',
        'web_stack.person_research'
      )
    )
  )`;
  const revisionRows = run("sqlite3", [
    "-readonly",
    rxdbDatabase,
    `SELECT id || char(9) || COALESCE(revision, '') || char(9) || printf('%.3f', lastWriteTime)
     FROM ${commandTable} WHERE deleted = 0 ORDER BY id;`,
  ]);
  return {
    captured_at: new Date().toISOString(),
    retry_candidate_count: sqliteScalar(
      rxdbDatabase,
      `SELECT COUNT(*) FROM ${commandTable} WHERE deleted = 0 AND ${retryCandidate};`,
    ),
    command_document_count: sqliteScalar(
      rxdbDatabase,
      `SELECT COUNT(*) FROM ${commandTable} WHERE deleted = 0;`,
    ),
    command_revision_sha256: sha256(revisionRows),
    command_changed_table_revision: sqliteScalar(
      rxdbDatabase,
      `SELECT COALESCE((SELECT changed_at FROM __rxdb_changed_tables
        WHERE table_name = '${commandTable}'), 0);`,
    ),
    core_open_intake_failures: sqliteScalar(
      coreDatabase,
      "SELECT COUNT(*) FROM business_command_intake_failures WHERE resolved_at_ms IS NULL;",
    ),
    compatibility_open_intake_failures: sqliteScalar(
      businessDatabase,
      "SELECT COUNT(*) FROM business_command_intake_failures WHERE resolved_at_ms IS NULL;",
    ),
    pending_command_outbox: sqliteScalar(
      coreDatabase,
      "SELECT COUNT(*) FROM business_command_outbox WHERE delivered_at_ms IS NULL;",
    ),
  };
}

function runtimeSnapshot(binary) {
  const service = jsonCommand(binary, ["status"]);
  const rxdb = jsonCommand(binary, ["business-os", "rxdb", "status", "--json"]);
  return {
    captured_at: new Date().toISOString(),
    service: {
      running: service.running,
      pid: service.pid,
      busy: service.busy,
      pending_count: service.pending_count,
      worker_active_count: service.worker_active_count,
      current_goal: service.current_goal,
      last_error: service.last_error,
    },
    rxdb: {
      running: rxdb.running,
      replicationUp: rxdb.replicationUp,
      heartbeat_fresh: rxdb.heartbeat?.fresh ?? null,
      lifecycle: rxdb.lifecycle ?? null,
      business_commands_loop: rxdb.performance?.loops?.business_commands ?? null,
      browser_runtime_loop: rxdb.performance?.loops?.browser_runtime ?? null,
      command_plane: rxdb.command_plane ?? null,
    },
  };
}

const args = parseArgs(process.argv.slice(2));
const binary = requireArg(args, "binary");
const output = requireArg(args, "output");
const stateRoot = requireArg(args, "state-root");
const pid = Number(requireArg(args, "pid"));
const durationSeconds = positiveNumber(args, "duration-seconds", 3600);
const intervalSeconds = positiveNumber(args, "interval-seconds", 5);
const startedAt = new Date();
const runtimeBefore = runtimeSnapshot(binary);
if (!runtimeBefore.service.running || runtimeBefore.service.pid !== pid) {
  throw new Error(`service PID mismatch: expected ${pid}, got ${runtimeBefore.service.pid}`);
}
if (runtimeBefore.service.busy || runtimeBefore.service.pending_count !== 0 || runtimeBefore.service.worker_active_count !== 0) {
  throw new Error("service is not idle at probe start");
}
const databaseBefore = databaseSnapshot(stateRoot);
const samples = [];
const deadline = Date.now() + durationSeconds * 1000;
let nextProgressAt = Date.now() + 60_000;
while (Date.now() < deadline) {
  samples.push(processSample(pid));
  if (Date.now() >= nextProgressAt) {
    const remaining = Math.max(0, Math.ceil((deadline - Date.now()) / 1000));
    process.stderr.write(`idle probe: ${samples.length} samples, ${remaining}s remaining\n`);
    nextProgressAt += 60_000;
  }
  const sleepMilliseconds = Math.min(intervalSeconds * 1000, Math.max(0, deadline - Date.now()));
  if (sleepMilliseconds > 0) {
    Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, sleepMilliseconds);
  }
}
samples.push(processSample(pid));
const databaseAfter = databaseSnapshot(stateRoot);
const runtimeAfter = runtimeSnapshot(binary);
const finishedAt = new Date();
const cpuValues = samples.map((sample) => sample.cpu_percent);
const cpuTimeDelta = samples.at(-1).cpu_time_seconds - samples[0].cpu_time_seconds;
const elapsedSeconds = (finishedAt.getTime() - startedAt.getTime()) / 1000;
// The database and runtime snapshots can block behind SQLite work for much
// longer than the requested sampling window. CPU time is measured from the
// first through the last process sample, so its denominator must use exactly
// those same endpoints instead of diluting the result with snapshot time.
const sampleElapsedSeconds = Math.max(
  0.001,
  (new Date(samples.at(-1).captured_at).getTime()
    - new Date(samples[0].captured_at).getTime()) / 1000,
);
const result = {
  schema: "ctox.idle_service_probe.v1",
  started_at: startedAt.toISOString(),
  finished_at: finishedAt.toISOString(),
  requested_duration_seconds: durationSeconds,
  measured_elapsed_seconds: elapsedSeconds,
  interval_seconds: intervalSeconds,
  pid,
  runtime_before: runtimeBefore,
  runtime_after: runtimeAfter,
  database_before: databaseBefore,
  database_after: databaseAfter,
  summary: {
    sample_count: samples.length,
    sampled_cpu_average_percent: cpuValues.reduce((sum, value) => sum + value, 0) / cpuValues.length,
    sampled_cpu_p50_percent: percentile(cpuValues, 0.5),
    sampled_cpu_p95_percent: percentile(cpuValues, 0.95),
    sampled_cpu_max_percent: Math.max(...cpuValues),
    sample_elapsed_seconds: sampleElapsedSeconds,
    process_cpu_time_delta_seconds: cpuTimeDelta,
    process_cpu_duration_percent: (cpuTimeDelta / sampleElapsedSeconds) * 100,
    retry_candidates_stable_zero:
      databaseBefore.retry_candidate_count === 0 && databaseAfter.retry_candidate_count === 0,
    command_revisions_unchanged:
      databaseBefore.command_revision_sha256 === databaseAfter.command_revision_sha256 &&
      databaseBefore.command_changed_table_revision === databaseAfter.command_changed_table_revision,
    intake_failures_stable_zero:
      databaseBefore.core_open_intake_failures === 0 &&
      databaseAfter.core_open_intake_failures === 0 &&
      databaseBefore.compatibility_open_intake_failures === 0 &&
      databaseAfter.compatibility_open_intake_failures === 0,
    idle_ticks_advanced:
      (runtimeAfter.rxdb.business_commands_loop?.idle_ticks ?? 0) >
      (runtimeBefore.rxdb.business_commands_loop?.idle_ticks ?? 0),
  },
  samples,
};
writeFileSync(output, `${JSON.stringify(result, null, 2)}\n`);
process.stdout.write(`${JSON.stringify(result.summary, null, 2)}\n`);
