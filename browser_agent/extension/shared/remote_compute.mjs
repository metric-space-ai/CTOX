import { getSupportedLocalQwenModels } from "./local_qwen_runtime.mjs";

function asText(value) {
  return String(value == null ? "" : value).trim();
}

export const REMOTE_COMPUTE_ROLLOUT_ENABLED = false;

const SUPPORTED_REMOTE_COMPUTE_MODELS = Object.freeze(
  getSupportedLocalQwenModels().map((entry) => entry.vanillaModelName),
);

export const DEFAULT_REMOTE_COMPUTE_MODEL =
  SUPPORTED_REMOTE_COMPUTE_MODELS[0] || "unsloth/Qwen3.5-0.8B (Vanilla)";
export const REMOTE_COMPUTE_TOPIC = "sinepanel-compute.v1";
export const REMOTE_COMPUTE_MAX_CONCURRENT_JOBS = 1;
export const REMOTE_COMPUTE_SETTINGS_DEFAULTS = Object.freeze({
  computeOfferEnabled: false,
  computeModelName: DEFAULT_REMOTE_COMPUTE_MODEL,
  remoteExecutionEnabled: false,
});

export function listRemoteComputeModelNames() {
  if (!REMOTE_COMPUTE_ROLLOUT_ENABLED) return [];
  return SUPPORTED_REMOTE_COMPUTE_MODELS.slice();
}

export function normalizeRemoteComputeSettings(value = {}) {
  const raw = value && typeof value === "object" ? value : {};
  const requestedModelName = asText(raw.computeModelName) || DEFAULT_REMOTE_COMPUTE_MODEL;
  const computeModelName = SUPPORTED_REMOTE_COMPUTE_MODELS.includes(requestedModelName)
    ? requestedModelName
    : DEFAULT_REMOTE_COMPUTE_MODEL;
  if (!REMOTE_COMPUTE_ROLLOUT_ENABLED) {
    return {
      computeOfferEnabled: false,
      computeModelName,
      remoteExecutionEnabled: false,
    };
  }
  return {
    computeOfferEnabled: raw.computeOfferEnabled === true,
    computeModelName,
    remoteExecutionEnabled: raw.remoteExecutionEnabled === true,
  };
}

export function buildComputeWorkerId(deviceId = "") {
  const normalizedDeviceId = asText(deviceId);
  return normalizedDeviceId ? `compute-worker:${normalizedDeviceId}` : "";
}

export function buildComputeJobId() {
  if (typeof crypto?.randomUUID === "function") {
    return `compute-job:${crypto.randomUUID()}`;
  }
  return `compute-job:${Date.now().toString(16)}-${Math.random().toString(16).slice(2)}`;
}

export function buildComputeRequestId() {
  if (typeof crypto?.randomUUID === "function") {
    return `compute-request:${crypto.randomUUID()}`;
  }
  return `compute-request:${Date.now().toString(16)}-${Math.random().toString(16).slice(2)}`;
}

export function clampComputeProgress(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 0;
  if (numeric <= 0) return 0;
  if (numeric >= 1) return 1;
  return numeric;
}

export function summarizeComputeWorker(record = null) {
  if (!record || typeof record !== "object") return null;
  return {
    id: asText(record.id),
    deviceId: asText(record.deviceId),
    name: asText(record.name),
    modelName: asText(record.modelName),
    available: record.available === true,
    busy: record.busy === true,
    activeJobId: asText(record.activeJobId),
    lastSeen: asText(record.lastSeen || record.last_seen),
    maxConcurrentJobs: Math.max(1, Number(record.maxConcurrentJobs || record.max_concurrent_jobs || 1) || 1),
  };
}

export function summarizeComputeJob(record = null) {
  if (!record || typeof record !== "object") return null;
  return {
    id: asText(record.id),
    kind: asText(record.kind),
    status: asText(record.status),
    modelName: asText(record.modelName),
    requesterDeviceId: asText(record.requesterDeviceId),
    workerDeviceId: asText(record.workerDeviceId),
    error: asText(record.error),
    progress: clampComputeProgress(record.progress),
    createdAt: asText(record.createdAt || record.created_at),
    startedAt: asText(record.startedAt || record.started_at),
    completedAt: asText(record.completedAt || record.completed_at),
    updatedAt: asText(record.updatedAt || record.updated_at),
  };
}
