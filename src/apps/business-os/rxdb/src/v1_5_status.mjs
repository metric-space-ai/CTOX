// V1.5 status surface. Filled progressively by later waves.
// Values reflect the V1-only baseline until Wave 1 lights up the capability
// and Wave 3 wires demand-loading into the query path. Do NOT widen the
// schema without bumping the field list documented in docs/rxdb_on-demand-load.md.

import { CTOX_QUERY_FETCH_CAPABILITY, CTOX_QUERY_RPC } from './protocol-contract.generated.mjs';

export const V1_5_QUERY_FETCH_CAPABILITY = CTOX_QUERY_FETCH_CAPABILITY;
export const V1_5_QUERY_RPC = CTOX_QUERY_RPC;

export const V1_5_STATUS_FIELDS = Object.freeze([
  'rxdbRuntime',
  'rxdbProtocolVersion',
  'transport',
  'peerConnected',
  'peerCapabilityQueryFetchV1',
  'queryDemandLoadingEnabled',
  'queryDemandLoadingActive',
  'queryFetchInFlight',
  'pendingQueryFetchCollectors',
  'queuedQueryFetchRequests',
  'maxPendingQueryFetchCollectors',
  'queryFetchSuccessCount',
  'queryFetchErrorCount',
  'queryFetchDedupHitCount',
  'indexedDbWorkingSetBytes',
  'indexedDbEvictionCount',
  'pinnedDocCount',
  'pinnedBytes',
  'lastQueryFetchMs',
  'lastTransportBackpressureMs',
  'lastReloadHydrationMs',
  'activeFileStreams',
  'pendingFileFetchCollectors',
  'maxPendingFileFetchCollectors',
  'fileBytesReceived',
  'fileStreamErrors',
  'fileStreamDedupHits',
  'lastFileFetchMs',
  'localPushChangedSinceCalls',
  'localPushChangedSinceScannedRows',
  'localPushChangedSinceScanLimitHits',
  'localPushChangedSinceMaxScannedRows',
  'clockSkewDetected',
  'nativeClockOffsetMs',
  'nativeClockObservedAtMs',
  'code',
]);


export const DATA_PLANE_NO_PROGRESS_CODE = 'data_plane_no_progress';
// 8s is above a couple of 1.5s command-bus rebinds (slow device / one missed
// pull) and strictly under the 10s user-visible stall budget.
export const DATA_PLANE_STALL_MS = 8000;

function progressToken(value) {
  if (value == null) return '';
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
    return String(value);
  }
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function healthyEvaluation() {
  return {
    ok: true,
    code: null,
    collection: null,
    stalledMs: 0,
  };
}

const DEFAULT_MONITOR_KEY = '__ctoxDataPlaneProgressMonitor';

export function createDataPlaneProgressMonitor(options = {}) {
  const stallMs = Number(options.stallMs);
  return {
    now: typeof options.now === 'function' ? options.now : () => Date.now(),
    stallMs: Number.isFinite(stallMs) && stallMs > 0 ? stallMs : DATA_PLANE_STALL_MS,
    watches: new Map(),
    lastProgressByCollection: new Map(),
    repairAttempts: 0,
    stallReports: 0,
    lastEvaluation: healthyEvaluation(),
  };
}

export function getDefaultDataPlaneProgressMonitor() {
  const root = globalThis;
  if (!root[DEFAULT_MONITOR_KEY]) {
    root[DEFAULT_MONITOR_KEY] = createDataPlaneProgressMonitor();
  }
  return root[DEFAULT_MONITOR_KEY];
}

export function resetDefaultDataPlaneProgressMonitorForTests(options = {}) {
  const monitor = createDataPlaneProgressMonitor(options);
  globalThis[DEFAULT_MONITOR_KEY] = monitor;
  return monitor;
}

export function expectDataPlaneProgress(monitor, observation = {}) {
  const current = monitor || getDefaultDataPlaneProgressMonitor();
  const collection = String(observation.collection || '').trim() || 'unknown';
  const now = Number.isFinite(Number(observation.atMs)) ? Number(observation.atMs) : current.now();
  const baselineToken = progressToken(observation.token);
  const existing = current.watches.get(collection);
  if (existing) {
    existing.waiters += 1;
    return current;
  }
  current.watches.set(collection, {
    baselineToken,
    sinceMs: now,
    waiters: 1,
    repairAttempted: false,
    stallReported: false,
  });
  current.lastProgressByCollection.set(collection, { token: baselineToken, atMs: now });
  return current;
}

export function releaseDataPlaneProgressExpectation(monitor, collection) {
  const current = monitor || getDefaultDataPlaneProgressMonitor();
  const name = String(collection || '').trim() || 'unknown';
  const watch = current.watches.get(name);
  if (!watch) return current;
  watch.waiters = Math.max(0, Number(watch.waiters || 1) - 1);
  if (watch.waiters > 0) return current;
  const last = current.lastProgressByCollection.get(name);
  const token = last?.token ?? watch.baselineToken;
  const stalled = token === watch.baselineToken
    && (current.now() - Number(watch.sinceMs || current.now())) >= current.stallMs;
  if (stalled) {
    // Keep the stall visible until observed progress returns.
    return current;
  }
  current.watches.delete(name);
  if (current.lastEvaluation.collection === name || current.watches.size === 0) {
    current.lastEvaluation = healthyEvaluation();
    current.repairAttempts = 0;
    current.stallReports = 0;
  }
  return current;
}

export function noteDataPlaneProgress(monitor, observation = {}) {
  const current = monitor || getDefaultDataPlaneProgressMonitor();
  const collection = String(observation.collection || '').trim() || 'unknown';
  const token = progressToken(observation.token);
  const atMs = Number(observation.atMs);
  const now = Number.isFinite(atMs) ? atMs : current.now();
  current.lastProgressByCollection.set(collection, { token, atMs: now });
  const watch = current.watches.get(collection);
  if (watch && token !== watch.baselineToken) {
    watch.baselineToken = token;
    watch.sinceMs = now;
    watch.repairAttempted = false;
    watch.stallReported = false;
    current.repairAttempts = 0;
    current.stallReports = 0;
    if (current.lastEvaluation.collection === collection) {
      current.lastEvaluation = healthyEvaluation();
    }
  }
  return current;
}

export function evaluateDataPlaneProgress(monitor, _connectionFlags = {}) {
  const current = monitor || getDefaultDataPlaneProgressMonitor();
  const now = current.now();
  // peerConnected / replicationUp / heartbeat freshness are self-reports.
  // They must not keep health green while an expected collection is frozen.
  for (const [collection, watch] of current.watches) {
    const last = current.lastProgressByCollection.get(collection);
    const token = last?.token ?? watch.baselineToken;
    if (token !== watch.baselineToken) continue;
    const stalledMs = Math.max(0, now - Number(watch.sinceMs || now));
    if (stalledMs < current.stallMs) continue;
    if (!watch.stallReported) {
      watch.stallReported = true;
      current.stallReports += 1;
    }
    const evaluation = {
      ok: false,
      code: DATA_PLANE_NO_PROGRESS_CODE,
      collection,
      stalledMs,
    };
    current.lastEvaluation = evaluation;
    return evaluation;
  }
  const evaluation = healthyEvaluation();
  current.lastEvaluation = evaluation;
  return evaluation;
}

export async function tryDataPlaneStallRepair(monitor, collection, repairFn) {
  const current = monitor || getDefaultDataPlaneProgressMonitor();
  const name = String(collection || '').trim() || 'unknown';
  const watch = current.watches.get(name);
  if (!watch || watch.repairAttempted) {
    return { attempted: false, locked: Boolean(watch?.repairAttempted) };
  }
  watch.repairAttempted = true;
  current.repairAttempts += 1;
  try {
    await repairFn?.();
    return { attempted: true, locked: true };
  } catch {
    return { attempted: true, locked: true };
  }
}

export function applyDataPlaneHealthToStatus(state, evaluation) {
  if (!state) return state;
  if (evaluation?.ok === false && evaluation.code) {
    state.code = evaluation.code;
  } else if (state.code === DATA_PLANE_NO_PROGRESS_CODE) {
    state.code = null;
  }
  return state;
}

export function snapshotDataPlaneHealth(monitor) {
  const current = monitor || getDefaultDataPlaneProgressMonitor();
  const evaluation = current.lastEvaluation || healthyEvaluation();
  return {
    ok: evaluation.ok !== false,
    code: evaluation.code || null,
    collection: evaluation.collection || null,
    stalledMs: Number(evaluation.stalledMs || 0),
    repairAttempts: Number(current.repairAttempts || 0),
    stallReports: Number(current.stallReports || 0),
  };
}

export function createV1_5StatusState() {
  return {
    rxdbRuntime: 'ctox-rxdb-js',
    rxdbProtocolVersion: '1',
    transport: 'webrtc',
    peerConnected: false,
    peerCapabilityQueryFetchV1: false,
    queryDemandLoadingEnabled: false,
    queryDemandLoadingActive: false,
    queryFetchInFlight: 0,
    pendingQueryFetchCollectors: 0,
    queuedQueryFetchRequests: 0,
    maxPendingQueryFetchCollectors: 0,
    queryFetchSuccessCount: 0,
    queryFetchErrorCount: 0,
    queryFetchDedupHitCount: 0,
    indexedDbWorkingSetBytes: 0,
    indexedDbEvictionCount: 0,
    pinnedDocCount: 0,
    pinnedBytes: 0,
    lastQueryFetchMs: null,
    lastTransportBackpressureMs: null,
    lastReloadHydrationMs: null,
    activeFileStreams: 0,
    pendingFileFetchCollectors: 0,
    maxPendingFileFetchCollectors: 0,
    fileBytesReceived: 0,
    fileStreamErrors: 0,
    fileStreamDedupHits: 0,
    lastFileFetchMs: null,
    localPushChangedSinceCalls: 0,
    localPushChangedSinceScannedRows: 0,
    localPushChangedSinceScanLimitHits: 0,
    localPushChangedSinceMaxScannedRows: 0,
    clockSkewDetected: false,
    nativeClockOffsetMs: 0,
    nativeClockObservedAtMs: null,
    code: null,
  };
}

export function projectStatusFromSidecar(state, sidecarStats, registry = null) {
  const next = { ...state };
  if (sidecarStats) {
    next.indexedDbWorkingSetBytes = sidecarStats.estimatedBytes || 0;
  }
  if (registry?.pinnedDocCount !== undefined) next.pinnedDocCount = registry.pinnedDocCount;
  if (registry?.pinnedBytes !== undefined) next.pinnedBytes = registry.pinnedBytes;
  return next;
}

export function snapshotV1_5Status(state) {
  const snapshot = {};
  for (const field of V1_5_STATUS_FIELDS) {
    snapshot[field] = state?.[field] ?? null;
  }
  return snapshot;
}
