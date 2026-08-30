// AP3: health follows observed command/collection progress, not self-reported
// connection flags. Reproduces the customer stall: peerConnected/replicationUp
// and a fresh heartbeat stay green while a pushed command never comes back.

import {
  DATA_PLANE_NO_PROGRESS_CODE,
  DATA_PLANE_STALL_MS,
  createDataPlaneProgressMonitor,
  createV1_5StatusState,
  evaluateDataPlaneProgress,
  expectDataPlaneProgress,
  noteDataPlaneProgress,
  resetDefaultDataPlaneProgressMonitorForTests,
  tryDataPlaneStallRepair,
} from '../src/v1_5_status.mjs';
import { buildBusinessOsAdvancedStatus as buildStatusFromBridge } from '../src/advanced-status-bridge.mjs';
import { createCommandBus } from '../../shared/command-bus.js';

globalThis.CTOX_BUSINESS_OS_SESSION = {
  capability_token: 'data-plane-progress-capability',
  capability_expires_at_ms: Date.now() + 60 * 60 * 1000,
};

const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};

function greenConnectionFlags() {
  const status = createV1_5StatusState();
  status.peerConnected = true;
  status.replicationUp = true;
  status.heartbeatFresh = true;
  status.queryFetchErrorCount = 0;
  status.fileStreamErrors = 0;
  return status;
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitFor(predicate, timeoutMs = 2_000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (predicate()) return true;
    await delay(10);
  }
  return predicate();
}

// ---------------------------------------------------------------------------
// 1. Pure progress observer: green flags, no movement → typed stall < 10s
// ---------------------------------------------------------------------------
{
  let now = 1_000;
  const monitor = createDataPlaneProgressMonitor({
    now: () => now,
    stallMs: DATA_PLANE_STALL_MS,
  });
  resetDefaultDataPlaneProgressMonitorForTests({ now: () => now, stallMs: DATA_PLANE_STALL_MS });
  Object.assign(globalThis.__ctoxDataPlaneProgressMonitor, {
    now: monitor.now,
    stallMs: monitor.stallMs,
    watches: monitor.watches,
    lastProgressByCollection: monitor.lastProgressByCollection,
    repairAttempts: 0,
    stallReports: 0,
    lastEvaluation: monitor.lastEvaluation,
  });

  expectDataPlaneProgress(monitor, {
    collection: 'business_commands',
    token: 'pending_sync|',
    atMs: now,
  });

  const healthy = buildStatusFromBridge({
    v15Status: greenConnectionFlags(),
    dataPlaneMonitor: monitor,
  });
  assert(healthy.ok === true, 'no stall before the progress budget');
  assert(healthy.dataPlane.ok === true, 'data-plane snapshot starts healthy');

  now += 7_999;
  const stillGreen = buildStatusFromBridge({
    v15Status: greenConnectionFlags(),
    dataPlaneMonitor: monitor,
  });
  assert(stillGreen.ok === true, '7.999s without progress must not alarm');

  now += 1;
  const stalled = buildStatusFromBridge({
    v15Status: greenConnectionFlags(),
    dataPlaneMonitor: monitor,
  });
  assert(stalled.ok === false, '8s without progress must flip ok=false');
  assert(stalled.code === DATA_PLANE_NO_PROGRESS_CODE, `typed code, got ${stalled.code}`);
  assert(stalled.dataPlane.collection === 'business_commands', 'stall names the collection');
  assert(stalled.dataPlane.stalledMs >= DATA_PLANE_STALL_MS, 'stall duration reported');
  assert(stalled.dataPlane.stallReports === 1, 'exactly one stall report');
  assert(DATA_PLANE_STALL_MS <= 10_000, 'user-visible budget is 10s');

  const firstRepair = await tryDataPlaneStallRepair(monitor, 'business_commands', async () => {
    monitor.repairProbe = (monitor.repairProbe || 0) + 1;
  });
  const secondRepair = await tryDataPlaneStallRepair(monitor, 'business_commands', async () => {
    monitor.repairProbe = (monitor.repairProbe || 0) + 1;
  });
  assert(firstRepair.attempted === true, 'first stall repair runs');
  assert(secondRepair.attempted === false, 'second stall repair is locked');
  assert(monitor.repairAttempts === 1, `exactly one repair, got ${monitor.repairAttempts}`);
  assert(monitor.repairProbe === 1, 'repair function invoked once');

  now += 4_000;
  noteDataPlaneProgress(monitor, {
    collection: 'business_commands',
    token: 'accepted|native_observed',
    atMs: now,
  });
  const recovered = buildStatusFromBridge({
    v15Status: greenConnectionFlags(),
    dataPlaneMonitor: monitor,
  });
  assert(recovered.ok === true, 'observed progress restores ok=true');
  assert(recovered.code == null || recovered.code !== DATA_PLANE_NO_PROGRESS_CODE, 'stall code cleared');
  assert(monitor.repairAttempts === 0, 'repair counter resets after progress');
  assert(monitor.stallReports === 0, 'stall reports reset after progress');
}

// ---------------------------------------------------------------------------
// 2. Customer injection: command pushed, pull permanently suppressed
// ---------------------------------------------------------------------------
{
  let now = 50_000;
  const monitor = createDataPlaneProgressMonitor({ now: () => now, stallMs: DATA_PLANE_STALL_MS });
  resetDefaultDataPlaneProgressMonitorForTests({ now: () => now, stallMs: DATA_PLANE_STALL_MS });
  const shared = globalThis.__ctoxDataPlaneProgressMonitor;
  shared.now = () => now;
  shared.stallMs = DATA_PLANE_STALL_MS;
  shared.watches = monitor.watches;
  shared.lastProgressByCollection = monitor.lastProgressByCollection;
  shared.repairAttempts = 0;
  shared.stallReports = 0;

  let stored = null;
  const listeners = new Set();
  const collection = {
    async insert(doc) {
      stored = { ...doc };
    },
    findOne(id) {
      return {
        $: {
          subscribe(listener) {
            listeners.add(listener);
            if (stored?.id === id) listener({ toJSON: () => ({ ...stored }) });
            return { unsubscribe: () => listeners.delete(listener) };
          },
        },
        async exec() {
          return stored?.id === id ? { toJSON: () => ({ ...stored }) } : null;
        },
      };
    },
  };
  let pullCount = 0;
  const syncState = {
    demandStatus: { peerConnected: true, replicationUp: true },
    getTransportStatus() {
      return {
        demandLoading: { peerConnected: true },
        peerConnected: true,
        replicationUp: true,
        heartbeatAgeMs: 2500,
      };
    },
    async pushDocumentsToRemotePeers() {
      return true;
    },
    async pullFromRemotePeers() {
      pullCount += 1;
      // Reply permanently suppressed: the local pending_sync row never moves.
    },
  };
  const bus = createCommandBus({
    db: { raw: { business_commands: collection, ctox_queue_tasks: collection } },
    sync: {
      dataPlaneMonitor: shared,
      async startCollection() {
        return { state: syncState };
      },
      recordCommandMetric() {},
    },
  });

  const waiting = bus.dispatch({
    id: 'cmd-customer-stall',
    command_type: 'business_os.chat.task',
    module: 'customers',
    wait_timeout_ms: 12_000,
  });

  await waitFor(() => stored?.id === 'cmd-customer-stall');
  assert(stored?.id === 'cmd-customer-stall', 'command was pushed locally');
  assert(stored.status === 'pending_sync', 'command still waiting for native ack');

  const before = buildStatusFromBridge({
    v15Status: greenConnectionFlags(),
    dataPlaneMonitor: shared,
  });
  assert(before.ok === true, 'still green immediately after push');

  now += DATA_PLANE_STALL_MS;
  await delay(350);

  const after = buildStatusFromBridge({
    v15Status: greenConnectionFlags(),
    dataPlaneMonitor: shared,
  });
  assert(after.ok === false, 'customer stall must surface within 10s');
  assert(after.code === DATA_PLANE_NO_PROGRESS_CODE, `typed stall code, got ${after.code}`);
  assert(after.dataPlane.collection === 'business_commands', 'affected collection named');
  assert(shared.repairAttempts === 1, `exactly one automatic repair, got ${shared.repairAttempts}`);

  await delay(400);
  assert(shared.repairAttempts === 1, 'no second automatic repair');

  stored = {
    ...stored,
    status: 'accepted',
    replication_phase: 'native_observed',
    execution_task_id: 'queue:customers::1',
    updated_at_ms: now + 1,
  };
  now += 1;
  listeners.forEach((listener) => listener({ toJSON: () => ({ ...stored }) }));
  const receipt = await waiting;
  assert(receipt.status === 'accepted', 'command proceeds once progress returns');

  const recovered = buildStatusFromBridge({
    v15Status: greenConnectionFlags(),
    dataPlaneMonitor: shared,
  });
  assert(recovered.ok === true, 'status recovers after observed progress');
  assert(shared.repairAttempts === 0, 'counters reset after recovery');
  assert(pullCount >= 1, 'one repair/rebind pull was attempted');
}

// ---------------------------------------------------------------------------
// 3. Normal flowing commands: zero stall reports
// ---------------------------------------------------------------------------
{
  let now = 90_000;
  const monitor = createDataPlaneProgressMonitor({ now: () => now, stallMs: DATA_PLANE_STALL_MS });
  resetDefaultDataPlaneProgressMonitorForTests({ now: () => now, stallMs: DATA_PLANE_STALL_MS });
  const shared = globalThis.__ctoxDataPlaneProgressMonitor;
  shared.now = () => now;
  shared.stallMs = DATA_PLANE_STALL_MS;
  shared.watches = monitor.watches;
  shared.lastProgressByCollection = monitor.lastProgressByCollection;
  shared.repairAttempts = 0;
  shared.stallReports = 0;

  let stored = null;
  const listeners = new Set();
  const collection = {
    async insert(doc) {
      stored = { ...doc };
    },
    findOne(query) {
      const id = typeof query === 'string' ? query : query?.selector?.id;
      return {
        $: {
          subscribe(listener) {
            listeners.add(listener);
            if (stored?.id === id) listener({ toJSON: () => ({ ...stored }) });
            return { unsubscribe: () => listeners.delete(listener) };
          },
        },
        async exec() {
          // business_commands is demand-only. An authoritative revision read
          // is the normal healthy path now; model the native exact-id response
          // instead of relying on the removed collection-wide pull fallback.
          if (query?.requireRevision && stored?.id === id) {
            stored = {
              ...stored,
              status: 'accepted',
              replication_phase: 'native_observed',
              execution_task_id: 'queue:ok::1',
              updated_at_ms: now + 5,
            };
            listeners.forEach((listener) => listener({ toJSON: () => ({ ...stored }) }));
          }
          return stored?.id === id ? { toJSON: () => ({ ...stored }) } : null;
        },
      };
    },
  };
  const bus = createCommandBus({
    db: { raw: { business_commands: collection, ctox_queue_tasks: collection } },
    sync: {
      dataPlaneMonitor: shared,
      async startCollection() {
        return {
          state: {
            demandStatus: { peerConnected: true },
            async pushDocumentsToRemotePeers() {
              stored = {
                ...stored,
                status: 'accepted',
                replication_phase: 'native_observed',
                execution_task_id: 'queue:ok::1',
                updated_at_ms: now + 5,
              };
              listeners.forEach((listener) => listener({ toJSON: () => ({ ...stored }) }));
              return true;
            },
            async pullFromRemotePeers() {
              if (!stored) return;
              stored = {
                ...stored,
                status: 'accepted',
                replication_phase: 'native_observed',
                execution_task_id: 'queue:ok::1',
                updated_at_ms: now + 5,
              };
              listeners.forEach((listener) => listener({ toJSON: () => ({ ...stored }) }));
            },
          },
        };
      },
    },
  });

  const receipt = await bus.dispatch({
    id: 'cmd-healthy-flow',
    command_type: 'business_os.chat.task',
    wait_timeout_ms: 2500,
  });
  assert(receipt.status === 'accepted', 'healthy command is accepted');
  now += DATA_PLANE_STALL_MS + 1_000;
  const env = buildStatusFromBridge({
    v15Status: greenConnectionFlags(),
    dataPlaneMonitor: shared,
  });
  assert(env.ok === true, 'healthy flow stays ok');
  assert(shared.stallReports === 0, `healthy flow must emit zero stall reports, got ${shared.stallReports}`);
  assert(shared.repairAttempts === 0, 'healthy flow must not repair');
}

// Keep the unused import from triggering "no coverage" if a later test wants the
// snapshot helper from the same module surface.
assert(typeof evaluateDataPlaneProgress === 'function', 'evaluate helper exported');

console.log('ctox-rxdb data-plane progress health smoke OK', {
  stallMs: DATA_PLANE_STALL_MS,
  code: DATA_PLANE_NO_PROGRESS_CODE,
});
