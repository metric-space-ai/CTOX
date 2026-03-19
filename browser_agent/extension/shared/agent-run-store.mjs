import "./craft-sync.js";

export const AGENT_RUN_STATE_ARTIFACT_KIND = "agent_run_state";
const AGENT_RUN_STORAGE_TIMEOUT_MS = 1800;

function asText(value) {
  return String(value == null ? "" : value).trim();
}

function cloneJson(value, fallback = null) {
  try {
    if (typeof globalThis.structuredClone === "function") {
      return globalThis.structuredClone(value);
    }
    return JSON.parse(JSON.stringify(value));
  } catch (_error) {
    return fallback;
  }
}

function asPlainObject(value) {
  return value && typeof value === "object" && !Array.isArray(value) ? value : {};
}

function getCraftSync() {
  return globalThis.SinepanelCraftSync || null;
}

async function withTimeout(promiseLike, timeoutMs, timeoutMessage) {
  let timeoutId = 0;
  const pending = Promise.resolve(promiseLike);
  pending.catch(() => {});
  try {
    return await Promise.race([
      pending,
      new Promise((_, reject) => {
        timeoutId = globalThis.setTimeout(() => {
          reject(new Error(timeoutMessage));
        }, Math.max(0, Number(timeoutMs) || 0));
      }),
    ]);
  } finally {
    if (timeoutId) {
      globalThis.clearTimeout(timeoutId);
    }
  }
}

async function callCraftSyncMethod(methodName, args = [], fallbackValue = null, timeoutMessage = "") {
  const craftSync = getCraftSync();
  if (!craftSync || typeof craftSync?.[methodName] !== "function") {
    return fallbackValue;
  }
  try {
    return await withTimeout(
      craftSync[methodName](...args),
      AGENT_RUN_STORAGE_TIMEOUT_MS,
      timeoutMessage || `Agent run store ${methodName} timed out.`,
    );
  } catch (error) {
    console.warn(`[agent-run-store] craftSync.${methodName} failed`, error);
    return fallbackValue;
  }
}

export function getAgentRunStateArtifactId(craftId = "") {
  const normalizedCraftId = asText(craftId);
  return normalizedCraftId ? `agent-run-state:${normalizedCraftId}` : "";
}

function normalizeAgentQuestions(questions) {
  if (!Array.isArray(questions)) return [];
  return questions
    .map((entry) => {
      const value = asPlainObject(entry);
      const id = asText(value.id);
      const question = asText(value.question);
      if (!id && !question) return null;
      return {
        id: id || `question-${Math.random().toString(16).slice(2, 10)}`,
        question,
        reason: asText(value.reason),
        answer: asText(value.answer),
      };
    })
    .filter(Boolean);
}

function mergeSnapshotQuestions(previousQuestions, nextQuestions) {
  const previousById = new Map(
    normalizeAgentQuestions(previousQuestions).map((entry) => [entry.id, entry]),
  );
  return normalizeAgentQuestions(nextQuestions).map((entry) => {
    const previous = previousById.get(entry.id);
    return {
      ...entry,
      answer: asText(entry.answer) || asText(previous?.answer),
    };
  });
}

function mergeRunSnapshot(previousSnapshot, nextSnapshot) {
  const previous = asPlainObject(previousSnapshot);
  const next = asPlainObject(nextSnapshot);
  if (!Object.keys(previous).length) return cloneJson(next, {});
  if (!Object.keys(next).length) return cloneJson(previous, {});

  const merged = {
    ...cloneJson(previous, {}),
    ...cloneJson(next, {}),
  };

  if (Array.isArray(previous.questions) || Array.isArray(next.questions)) {
    merged.questions = mergeSnapshotQuestions(previous.questions, next.questions);
  }
  if (!Object.prototype.hasOwnProperty.call(next, "activityRecorded")) {
    merged.activityRecorded = previous.activityRecorded === true;
  }
  if (!Object.prototype.hasOwnProperty.call(next, "useSurfaceApplied")) {
    merged.useSurfaceApplied = previous.useSurfaceApplied === true;
  }

  return merged;
}

function normalizeAgentRunStatePayload(payload, craftId = "") {
  const raw = asPlainObject(payload);
  const snapshot = asPlainObject(raw.snapshot);
  const runtime = asPlainObject(raw.runtime);
  const normalizedCraftId =
    asText(raw.craftId) ||
    asText(craftId) ||
    asText(snapshot.craftId) ||
    asText(runtime.craftId);

  return {
    version: 1,
    craftId: normalizedCraftId,
    snapshot: Object.keys(snapshot).length ? cloneJson(snapshot, {}) : null,
    runtime: Object.keys(runtime).length ? cloneJson(runtime, {}) : null,
    updatedAt: asText(raw.updatedAt),
  };
}

async function listLocalRunArtifacts() {
  const synced = await callCraftSyncMethod(
    "listLocalArtifacts",
    [{ kind: AGENT_RUN_STATE_ARTIFACT_KIND }],
    [],
    "Listing agent run artifacts timed out.",
  );
  return Array.isArray(synced) ? synced : [];
}

async function readLocalRunArtifact(artifactId) {
  const synced = await callCraftSyncMethod(
    "readLocalArtifact",
    [artifactId, null],
    null,
    `Reading agent run artifact ${artifactId} timed out.`,
  );
  return synced || null;
}

async function putLocalRunArtifact(record) {
  const synced = await callCraftSyncMethod(
    "putLocalArtifact",
    [record],
    null,
    `Writing agent run artifact ${record?.id || ""} timed out.`,
  );
  if (synced) {
    return synced;
  }
  throw new Error("No RxDB-backed local artifact store is available for agent runs.");
}

async function deleteLocalRunArtifact(artifactId) {
  await callCraftSyncMethod(
    "deleteLocalArtifact",
    [artifactId],
    null,
    `Deleting agent run artifact ${artifactId} timed out.`,
  );
}

export async function readAgentRunState(craftId = "", fallback = null) {
  const normalizedCraftId = asText(craftId);
  if (!normalizedCraftId) return cloneJson(fallback, fallback);
  const artifactId = getAgentRunStateArtifactId(normalizedCraftId);
  const record = await readLocalRunArtifact(artifactId);
  if (!record?.payload || typeof record.payload !== "object") {
    return cloneJson(fallback, fallback);
  }
  const normalized = normalizeAgentRunStatePayload(record.payload, normalizedCraftId);
  return {
    craftId: normalized.craftId,
    snapshot: normalized.snapshot,
    runtime: normalized.runtime,
    updatedAt: normalized.updatedAt || asText(record?.updatedAt),
    meta: cloneJson(record?.meta, {}),
  };
}

export async function listAgentRunStates() {
  const records = await listLocalRunArtifacts();
  return records
    .map((record) => {
      const payload = normalizeAgentRunStatePayload(record?.payload, record?.craftId);
      if (!payload.craftId) return null;
      return {
        craftId: payload.craftId,
        snapshot: payload.snapshot,
        runtime: payload.runtime,
        updatedAt: payload.updatedAt || asText(record?.updatedAt),
        meta: cloneJson(record?.meta, {}),
      };
    })
    .filter(Boolean)
    .sort((left, right) => Date.parse(String(right.updatedAt || "")) - Date.parse(String(left.updatedAt || "")));
}

export async function upsertAgentRunState({
  craftId = "",
  snapshot = undefined,
  runtime = undefined,
  meta = {},
} = {}) {
  const normalizedCraftId =
    asText(craftId) ||
    asText(snapshot?.craftId) ||
    asText(runtime?.craftId);
  if (!normalizedCraftId) {
    throw new Error("craftId is required to persist an agent run.");
  }

  const existing = await readAgentRunState(normalizedCraftId, null);
  const nextSnapshot =
    snapshot === undefined
      ? cloneJson(existing?.snapshot, null)
      : mergeRunSnapshot(existing?.snapshot, snapshot);
  const nextRuntime =
    runtime === undefined
      ? cloneJson(existing?.runtime, null)
      : cloneJson(runtime, null);
  const updatedAt = new Date().toISOString();

  await putLocalRunArtifact({
    id: getAgentRunStateArtifactId(normalizedCraftId),
    craftId: normalizedCraftId,
    kind: AGENT_RUN_STATE_ARTIFACT_KIND,
    payload: {
      version: 1,
      craftId: normalizedCraftId,
      snapshot: nextSnapshot,
      runtime: nextRuntime,
      updatedAt,
    },
    meta: {
      ...(existing?.meta && typeof existing.meta === "object" ? existing.meta : {}),
      ...(meta && typeof meta === "object" ? cloneJson(meta, {}) : {}),
      updatedAt: Date.now(),
    },
  });

  return await readAgentRunState(normalizedCraftId, null);
}

export async function deleteAgentRunState(craftId = "") {
  const artifactId = getAgentRunStateArtifactId(craftId);
  if (!artifactId) return;
  await deleteLocalRunArtifact(artifactId);
}
