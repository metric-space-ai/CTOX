import { rxdbCore, rxdbWebRTC } from "./craft-sync-rxdb-shim.js"
import {
  BROWSER_CAPABILITY_BUNDLE_ARTIFACT_KIND,
  POLICY_BUNDLE_ARTIFACT_KIND,
  TOOL_SCRIPTS_ARTIFACT_KIND,
  TRAINING_DATA_ARTIFACT_KIND,
  WEIGHTS_ARTIFACT_KIND,
  buildCapabilityBundle,
  extractBundleArtifactRecords,
  getBrowserCapabilityBundleArtifactId,
  getPolicyBundleArtifactId,
  getToolScriptsArtifactId,
  getTrainingDataArtifactId,
  getWeightsArtifactId,
} from "./capability-bundle.mjs"
import {
  REMOTE_COMPUTE_ROLLOUT_ENABLED,
  buildComputeWorkerId,
  normalizeRemoteComputeSettings,
  summarizeComputeJob,
  summarizeComputeWorker,
} from "./remote_compute.mjs"

;(function registerCraftSync(globalScope) {
  const SINGLETON_KEY = Symbol.for("sinepanel.craft-sync.singleton.v1")
  const EXISTING_SINGLETON = globalScope[SINGLETON_KEY] || null
  if (EXISTING_SINGLETON?.api) {
    globalScope.SinepanelCraftSync = EXISTING_SINGLETON.api
    return
  }
  const singleton = EXISTING_SINGLETON && typeof EXISTING_SINGLETON === "object"
    ? EXISTING_SINGLETON
    : { api: null }
  globalScope[SINGLETON_KEY] = singleton
  const sharedDbState = singleton.dbState && typeof singleton.dbState === "object"
    ? singleton.dbState
    : (singleton.dbState = {
        dbPromise: null,
        dbInstance: null,
      })

  const SETTINGS_KEY = "sinepanel.craft-sync.settings.v1"
  const DEVICE_ID_KEY = "sinepanel.craft-sync.device-id.v1"
  const LOCAL_SETTINGS_READY_KEY = "sinepanel.craft-sync.local-settings-ready.v1"
  const LOCAL_CRAFTS_READY_KEY = "sinepanel.craft-sync.local-crafts-ready.v1"
  const LOCAL_ARTIFACT_CHUNK_MIGRATION_KEY = "sinepanel.craft-sync.local-artifact-chunks.v1"
  const LOCAL_SETTINGS_CHANNEL_NAME = "sinepanel.craft-sync.local-settings.v1"
  const DATABASE_NAME = "sinepanel.craft-share.rxdb.v1"
  const REPLICATION_ROOM = "sinepanel-craft-share-v1"
  const DEFAULT_SIGNALING_URL = "wss://api.metricspace.org/signal"
  const ASSET_INLINE_THRESHOLD = 12_000
  const ASSET_CHUNK_CHARS = 6_000
  const CHUNKED_JSON_FORMAT = "json_text_v1"
  const PRESENCE_HEARTBEAT_MS = 8_000
  const PRESENCE_MAX_AGE_MS = 30_000
  const MANUAL_SYNC_DURATION_MS = 12_000

  let settingsPromise = null
  let deviceIdPromise = null
  let remoteCraftRefreshPromise = null
  let localCraftRefreshPromise = null
  let computeWorkerRefreshPromise = null
  let computeJobRefreshPromise = null
  let localArtifactChunkMigrationPromise = null
  let presenceRefreshPromise = null
  let replicationTeardown = null
  let presenceTeardown = null
  let restartTimer = 0
  let manualStopTimer = 0
  let transportPeerCount = 0
  let localSettingsSnapshot = null
  const localSettingsChannelId = createRuntimeInstanceId()
  const localSettingsChannel = createLocalSettingsChannel()

  const subscribers = new Set()
  const querySubscriptions = []

  const syncState = {
    running: false,
    requestedMode: "off",
    connection: "stopped",
    remotePeerCount: 0,
    transportPeerCount: 0,
    remoteCraftCount: 0,
    remoteComputeWorkerCount: 0,
    computeJobCount: 0,
    publishedCount: 0,
    lastError: "",
    lastSyncedAt: "",
    pageName: "",
  }

  function cloneJson(value, fallback = null) {
    try {
      if (typeof globalScope.structuredClone === "function") {
        return globalScope.structuredClone(value)
      }
      return JSON.parse(JSON.stringify(value))
    } catch (_error) {
      return fallback
    }
  }

  function asText(value) {
    return String(value == null ? "" : value).trim()
  }

  function asStoredChunkText(value) {
    return String(value == null ? "" : value)
  }

  function serializeJson(value, fallback = "null") {
    try {
      return JSON.stringify(value == null ? null : value)
    } catch (_error) {
      return fallback
    }
  }

  function parseJson(text, fallback = null) {
    try {
      return JSON.parse(String(text || "null"))
    } catch (_error) {
      return cloneJson(fallback, fallback)
    }
  }

  function generateRandomToken() {
    const bytes = new Uint8Array(8)
    if (globalScope.crypto?.getRandomValues) {
      globalScope.crypto.getRandomValues(bytes)
    } else {
      for (let index = 0; index < bytes.length; index += 1) {
        bytes[index] = Math.floor(Math.random() * 256)
      }
    }
    const text = Array.from(bytes, (value) => value.toString(16).padStart(2, "0")).join("")
    return `room-${text.slice(0, 6)}-${text.slice(6, 12)}`
  }

  function createRuntimeInstanceId() {
    if (typeof globalScope.crypto?.randomUUID === "function") {
      try {
        return globalScope.crypto.randomUUID()
      } catch (_error) {}
    }
    return `page-${Math.random().toString(16).slice(2)}-${Date.now().toString(16)}`
  }

  function createLocalSettingsChannel() {
    if (typeof globalScope.BroadcastChannel !== "function") return null
    try {
      const channel = new globalScope.BroadcastChannel(LOCAL_SETTINGS_CHANNEL_NAME)
      channel.onmessage = handleLocalSettingsBroadcastMessage
      return channel
    } catch (_error) {
      return null
    }
  }

  function broadcastLocalSettingsMutation(type, key, value = null, updatedAt = 0) {
    if (!localSettingsChannel) return
    try {
      localSettingsChannel.postMessage({
        scope: "local-settings",
        source: localSettingsChannelId,
        type: asText(type),
        key: asText(key),
        value: cloneJson(value, null),
        updatedAt: Math.max(0, Number(updatedAt) || 0),
      })
    } catch (_error) {}
  }

  function handleLocalSettingsBroadcastMessage(event) {
    const payload = event?.data && typeof event.data === "object" ? event.data : null
    if (!payload || asText(payload.scope) !== "local-settings") return
    if (asText(payload.source) === localSettingsChannelId) return

    const type = asText(payload.type)
    const key = asText(payload.key)
    if (!key || (type !== "kv-upsert" && type !== "kv-delete")) return

    if (type === "kv-upsert") {
      rememberLocalSettingSnapshotEntry(
        key,
        serializeLocalSettingSnapshotValue(cloneJson(payload.value, null), Number(payload.updatedAt) || 0),
      )
    } else {
      forgetLocalSettingSnapshotEntry(key)
    }

    if (key === SETTINGS_KEY) {
      settingsPromise = null
    }

    notifySubscribers({ type, key })
  }

  function toEpochMs(value) {
    const number = Number(value)
    if (Number.isFinite(number) && number >= 0) return Math.round(number)
    const parsed = Date.parse(String(value || ""))
    return Number.isFinite(parsed) && parsed >= 0 ? Math.round(parsed) : 0
  }

  function isoFromEpochMs(value) {
    const epochMs = toEpochMs(value)
    return epochMs > 0 ? new Date(epochMs).toISOString() : ""
  }

  function normalizeSettings(raw) {
    const remoteComputeSettings = normalizeRemoteComputeSettings(raw)
    const mode = asText(raw?.mode).toLowerCase()
    const rawToken = asText(raw?.token)
    const token = rawToken || generateRandomToken()
    const tokenAutoGenerated =
      !rawToken || (raw?.tokenAutoGenerated === true && Boolean(rawToken))
    return {
      displayName: asText(raw?.displayName),
      signalingUrls: asText(raw?.signalingUrls) || DEFAULT_SIGNALING_URL,
      token,
      tokenAutoGenerated,
      mode: mode === "continuous" || mode === "manual" ? mode : "off",
      computeOfferEnabled: remoteComputeSettings.computeOfferEnabled,
      computeModelName: remoteComputeSettings.computeModelName,
      remoteExecutionEnabled: remoteComputeSettings.remoteExecutionEnabled,
    }
  }

  function getPublicState() {
    return {
      settings: cloneJson(settingsPromise?.value || null, null),
      sync: cloneJson(syncState, {}),
    }
  }

  function notifySubscribers(detail = {}) {
    const payload = {
      settings: cloneJson(settingsPromise?.value || null, null),
      sync: cloneJson(syncState, {}),
      detail: cloneJson(detail, {}),
    }
    for (const listener of [...subscribers]) {
      try {
        listener(payload)
      } catch (error) {
        console.error("[craft-sync] subscriber failed", error)
      }
    }
  }

  function subscribe(listener) {
    if (typeof listener !== "function") return () => {}
    subscribers.add(listener)
    try {
      listener({
        settings: cloneJson(settingsPromise?.value || null, null),
        sync: cloneJson(syncState, {}),
        detail: {},
      })
    } catch (error) {
      console.error("[craft-sync] subscriber failed", error)
    }
    return () => {
      subscribers.delete(listener)
    }
  }

  function reportBackgroundFailure(type, error, fallbackMessage) {
    console.warn(`[craft-sync] ${type} failed`, error)
    syncState.lastError = String(error?.message || error || fallbackMessage || `${type} failed.`)
    notifySubscribers({ type })
  }

  function runInBackground(type, task, fallbackMessage) {
    try {
      const pending = Promise.resolve(task())
      pending.catch((error) => {
        reportBackgroundFailure(type, error, fallbackMessage)
      })
    } catch (error) {
      reportBackgroundFailure(type, error, fallbackMessage)
    }
  }

  function warmRemoteState(db = null) {
    runInBackground(
      "remote-peers-refresh-failed",
      () => refreshRemotePeerCount(db),
      "Refreshing remote peer presence failed.",
    )
    runInBackground(
      "remote-crafts-refresh-failed",
      () => refreshRemoteCrafts(db),
      "Refreshing remote crafts failed.",
    )
    if (REMOTE_COMPUTE_ROLLOUT_ENABLED) {
      runInBackground(
        "compute-workers-refresh-failed",
        () => refreshComputeWorkers(db),
        "Refreshing remote compute workers failed.",
      )
      runInBackground(
        "compute-jobs-refresh-failed",
        () => refreshComputeJobs(db),
        "Refreshing remote compute jobs failed.",
      )
    }
  }

  function warmDatabaseCaches(db) {
    runInBackground(
      "local-crafts-refresh-failed",
      () => refreshLocalCrafts(db),
      "Refreshing local crafts failed.",
    )
    runInBackground(
      "local-artifact-chunk-migration-failed",
      () => ensureLocalArtifactChunksMigrated(db),
      "Migrating local artifact chunks failed.",
    )
    warmRemoteState(db)
  }

  async function getValue(key, fallback = null) {
    const normalizedKey = asText(key)
    if (!normalizedKey) return cloneJson(fallback, fallback)
    const db = await getDatabase()
    return await readLocalSettingValueFromDb(db, normalizedKey, fallback)
  }

  async function setValue(key, value) {
    const normalizedKey = asText(key)
    if (!normalizedKey) return cloneJson(value, value)
    const db = await getDatabase()
    const nextValue = cloneJson(value, null)
    const updatedAt = Date.now()
    await writeLocalSettingValueToDb(db, normalizedKey, nextValue, updatedAt)
    rememberLocalSettingSnapshotEntry(normalizedKey, serializeLocalSettingSnapshotValue(nextValue, updatedAt))
    broadcastLocalSettingsMutation("kv-upsert", normalizedKey, nextValue, updatedAt)
    notifySubscribers({ type: "kv-upsert", key: normalizedKey })
    return cloneJson(nextValue, nextValue)
  }

  async function deleteValue(key) {
    const normalizedKey = asText(key)
    if (!normalizedKey) return
    const db = await getDatabase()
    const doc = await db.local_settings.findOne(normalizedKey).exec()
    try {
      await doc?.remove?.()
    } catch (_error) {}
    forgetLocalSettingSnapshotEntry(normalizedKey)
    broadcastLocalSettingsMutation("kv-delete", normalizedKey)
    notifySubscribers({ type: "kv-delete", key: normalizedKey })
  }

  async function readSettings() {
    if (settingsPromise) return settingsPromise

    settingsPromise = getValue(SETTINGS_KEY, null)
      .then(async (stored) => {
        const settings = normalizeSettings(stored)
        const needsPersist =
          !stored ||
          asText(stored?.displayName) !== settings.displayName ||
          asText(stored?.signalingUrls) !== settings.signalingUrls ||
          asText(stored?.token) !== settings.token ||
          stored?.tokenAutoGenerated !== settings.tokenAutoGenerated ||
          asText(stored?.mode).toLowerCase() !== settings.mode ||
          stored?.computeOfferEnabled !== settings.computeOfferEnabled ||
          asText(stored?.computeModelName) !== settings.computeModelName ||
          stored?.remoteExecutionEnabled !== settings.remoteExecutionEnabled
        if (needsPersist) {
          await setValue(SETTINGS_KEY, settings)
        }
        return settings
      })
      .then((settings) => {
        settingsPromise.value = settings
        return settings
      })
      .catch((error) => {
        settingsPromise = null
        throw error
      })

    return settingsPromise
  }

  async function refreshSettings() {
    settingsPromise = null
    return await readSettings()
  }

  async function updateSettings(patch = {}) {
    const current = await readSettings()
    const next = normalizeSettings({
      ...current,
      ...(patch && typeof patch === "object" ? patch : {}),
    })

    await setValue(SETTINGS_KEY, next)
    settingsPromise = Promise.resolve(next)
    settingsPromise.value = next

    if (next.mode === "continuous") {
      await startContinuous({ pageName: syncState.pageName || "Options" })
    } else if (next.mode === "off") {
      await stopSync()
    }

    notifySubscribers({ type: "settings-updated" })
    return cloneJson(next, {})
  }

  async function getDeviceId() {
    if (deviceIdPromise) return deviceIdPromise

    deviceIdPromise = (async () => {
      let deviceId = asText(await getValue(DEVICE_ID_KEY, ""))
      if (!deviceId) {
        deviceId = crypto.randomUUID()
        await setValue(DEVICE_ID_KEY, deviceId)
      }
      return deviceId
    })().catch((error) => {
      deviceIdPromise = null
      throw error
    })

    return deviceIdPromise
  }

  function stripRxMeta(doc) {
    if (!doc || typeof doc !== "object") return doc
    if (Array.isArray(doc)) return doc.map((item) => stripRxMeta(item))

    const out = {}
    for (const [key, value] of Object.entries(doc)) {
      if (key === "_rev" || key === "_meta" || key === "_deleted" || key === "_attachments") continue
      out[key] = stripRxMeta(value)
    }
    return out
  }

  const conflictHandler = {
    isEqual(left, right) {
      try {
        return JSON.stringify(stripRxMeta(left)) === JSON.stringify(stripRxMeta(right))
      } catch (_error) {
        return false
      }
    },
    resolve(input) {
      const candidate = input?.newDocumentState || null
      const master = input?.realMasterState || null
      if (!master) return candidate
      const candidateTs = Math.max(
        toEpochMs(candidate?.updated_at),
        toEpochMs(candidate?.updatedAt),
        toEpochMs(candidate?.last_seen),
      )
      const masterTs = Math.max(
        toEpochMs(master?.updated_at),
        toEpochMs(master?.updatedAt),
        toEpochMs(master?.last_seen),
      )
      if (candidateTs !== masterTs) {
        return candidateTs >= masterTs ? candidate : master
      }
      return String(candidate?.id || "") <= String(master?.id || "") ? candidate : master
    },
  }

  const presenceSchema = {
    title: "presence",
    version: 0,
    primaryKey: "id",
    type: "object",
    properties: {
      id: { type: "string", maxLength: 256 },
      deviceId: { type: "string", maxLength: 128 },
      name: { type: "string", maxLength: 200 },
      user_agent: { type: "string" },
      last_seen: { type: "number", minimum: 0, maximum: 1000000000000000, multipleOf: 1 },
    },
    required: ["id", "deviceId", "last_seen"],
    indexes: ["deviceId", "last_seen"],
  }

  const sharedCraftsSchema = {
    title: "shared_crafts",
    version: 0,
    primaryKey: "id",
    type: "object",
    properties: {
      id: { type: "string", maxLength: 256 },
      ownerDeviceId: { type: "string", maxLength: 128 },
      ownerName: { type: "string", maxLength: 200 },
      sourceCraftId: { type: "string", maxLength: 160 },
      name: { type: "string", maxLength: 400 },
      summary: { type: "string" },
      stage: { type: "string", maxLength: 120 },
      targetSlot: { type: "string", maxLength: 120 },
      useStatus: { type: "string", maxLength: 120 },
      starterMode: { type: "string", maxLength: 120 },
      payload: {
        type: "object",
        additionalProperties: true,
      },
      published_at: { type: "number", minimum: 0, maximum: 1000000000000000, multipleOf: 1 },
      updated_at: { type: "number", minimum: 0, maximum: 1000000000000000, multipleOf: 1 },
    },
    required: ["id", "ownerDeviceId", "sourceCraftId", "name", "payload", "published_at", "updated_at"],
    indexes: ["ownerDeviceId", "sourceCraftId", "updated_at"],
  }

  const sharedAssetChunksSchema = {
    title: "shared_asset_chunks",
    version: 0,
    primaryKey: "id",
    type: "object",
    properties: {
      id: { type: "string", maxLength: 320 },
      assetId: { type: "string", maxLength: 256 },
      shareId: { type: "string", maxLength: 256 },
      ownerDeviceId: { type: "string", maxLength: 128 },
      idx: { type: "number", minimum: 0, maximum: 1000000000, multipleOf: 1 },
      total: { type: "number", minimum: 1, maximum: 1000000000, multipleOf: 1 },
      mime: { type: "string", maxLength: 160 },
      data: { type: "string" },
      updated_at: { type: "number", minimum: 0, maximum: 1000000000000000, multipleOf: 1 },
    },
    required: ["id", "assetId", "shareId", "ownerDeviceId", "idx", "total", "mime", "data", "updated_at"],
    indexes: ["shareId", "assetId", "updated_at"],
  }

  const computeWorkersSchema = {
    title: "compute_workers",
    version: 0,
    primaryKey: "id",
    type: "object",
    properties: {
      id: { type: "string", maxLength: 256 },
      deviceId: { type: "string", maxLength: 128 },
      name: { type: "string", maxLength: 200 },
      modelName: { type: "string", maxLength: 200 },
      available: { type: "boolean" },
      busy: { type: "boolean" },
      activeJobId: { type: "string", maxLength: 256 },
      max_concurrent_jobs: { type: "number", minimum: 1, maximum: 16, multipleOf: 1 },
      last_seen: { type: "number", minimum: 0, maximum: 1000000000000000, multipleOf: 1 },
      updated_at: { type: "number", minimum: 0, maximum: 1000000000000000, multipleOf: 1 },
    },
    required: ["id", "deviceId", "modelName", "available", "busy", "max_concurrent_jobs", "last_seen", "updated_at"],
    indexes: ["deviceId", "modelName", "available", "updated_at"],
  }

  const computeJobsSchema = {
    title: "compute_jobs",
    version: 1,
    primaryKey: "id",
    type: "object",
    properties: {
      id: { type: "string", maxLength: 256 },
      kind: { type: "string", maxLength: 80 },
      status: { type: "string", maxLength: 80 },
      modelName: { type: "string", maxLength: 200 },
      requesterDeviceId: { type: "string", maxLength: 128 },
      requesterName: { type: "string", maxLength: 200 },
      workerDeviceId: { type: "string", maxLength: 128 },
      workerName: { type: "string", maxLength: 200 },
      error: { type: "string" },
      progress: { type: "number", minimum: 0, maximum: 1 },
      summary: {
        type: "object",
        additionalProperties: true,
      },
      created_at: { type: "number", minimum: 0, maximum: 1000000000000000, multipleOf: 1 },
      started_at: { type: "number", minimum: 0, maximum: 1000000000000000, multipleOf: 1 },
      completed_at: { type: "number", minimum: 0, maximum: 1000000000000000, multipleOf: 1 },
      updated_at: { type: "number", minimum: 0, maximum: 1000000000000000, multipleOf: 1 },
    },
    required: ["id", "kind", "status", "modelName", "requesterDeviceId", "workerDeviceId", "created_at", "updated_at"],
    indexes: ["status", "workerDeviceId", "requesterDeviceId", "updated_at"],
  }

  const computeJobsMigrationStrategies = {
    1: (docData) => {
      if (!docData || typeof docData !== "object") return docData
      return {
        ...docData,
        workerDeviceId: asText(docData.workerDeviceId),
      }
    },
  }

  const localCraftsSchema = {
    title: "local_crafts",
    version: 1,
    primaryKey: "id",
    type: "object",
    properties: {
      id: { type: "string", maxLength: 256 },
      sort_index: { type: "number", minimum: 0, maximum: 1000000000, multipleOf: 1 },
      artifact_mode: {
        type: "string",
        enum: ["artifact_only"],
        maxLength: 32,
      },
      payload: {
        type: "object",
        additionalProperties: true,
      },
      updated_at: { type: "number", minimum: 0, maximum: 1000000000000000, multipleOf: 1 },
    },
    required: ["id", "sort_index", "artifact_mode", "payload", "updated_at"],
    indexes: ["sort_index", "updated_at"],
  }

  const localCraftArtifactsSchema = {
    title: "local_craft_artifacts",
    version: 0,
    primaryKey: "id",
    type: "object",
    properties: {
      id: { type: "string", maxLength: 320 },
      craftId: { type: "string", maxLength: 256 },
      kind: { type: "string", maxLength: 160 },
      payload: {
        type: "object",
        additionalProperties: true,
      },
      meta: {
        type: "object",
        additionalProperties: true,
      },
      created_at: { type: "number", minimum: 0, maximum: 1000000000000000, multipleOf: 1 },
      updated_at: { type: "number", minimum: 0, maximum: 1000000000000000, multipleOf: 1 },
    },
    required: ["id", "craftId", "kind", "payload", "created_at", "updated_at"],
    indexes: ["craftId", "kind", "updated_at"],
  }

  const localArtifactChunksSchema = {
    title: "local_artifact_chunks",
    version: 0,
    primaryKey: "id",
    type: "object",
    properties: {
      id: { type: "string", maxLength: 384 },
      assetId: { type: "string", maxLength: 352 },
      artifactId: { type: "string", maxLength: 320 },
      craftId: { type: "string", maxLength: 256 },
      kind: { type: "string", maxLength: 160 },
      idx: { type: "number", minimum: 0, maximum: 1000000000, multipleOf: 1 },
      total: { type: "number", minimum: 1, maximum: 1000000000, multipleOf: 1 },
      mime: { type: "string", maxLength: 160 },
      data: { type: "string" },
      updated_at: { type: "number", minimum: 0, maximum: 1000000000000000, multipleOf: 1 },
    },
    required: ["id", "assetId", "artifactId", "craftId", "kind", "idx", "total", "mime", "data", "updated_at"],
    indexes: ["artifactId", "craftId", "updated_at"],
  }

  const localSettingsSchema = {
    title: "local_settings",
    version: 0,
    primaryKey: "key",
    type: "object",
    properties: {
      key: { type: "string", maxLength: 256 },
      value_json: { type: "string" },
      updated_at: { type: "number", minimum: 0, maximum: 1000000000000000, multipleOf: 1 },
    },
    required: ["key", "value_json", "updated_at"],
    indexes: ["updated_at"],
  }

  function toLocalSettingDoc(key, value, updatedAt = Date.now()) {
    return {
      key: asText(key),
      value_json: serializeJson(value),
      updated_at: toEpochMs(updatedAt) || Date.now(),
    }
  }

  async function readLocalSettingDoc(db, key) {
    const normalizedKey = asText(key)
    if (!normalizedKey) return null
    const doc = await db.local_settings.findOne(normalizedKey).exec()
    return doc?.toJSON?.() || null
  }

  async function readLocalSettingValueFromDb(db, key, fallback = null) {
    const doc = await readLocalSettingDoc(db, key)
    if (!doc) return cloneJson(fallback, fallback)
    return parseJson(doc.value_json, fallback)
  }

  async function writeLocalSettingValueToDb(db, key, value, updatedAt = Date.now()) {
    const normalizedKey = asText(key)
    if (!normalizedKey) return cloneJson(value, value)
    await db.local_settings.upsert(toLocalSettingDoc(normalizedKey, value, updatedAt))
    return cloneJson(value, value)
  }

  function serializeLocalSettingSnapshotValue(value, updatedAt = 0) {
    return `${Math.max(0, Number(updatedAt) || 0)}:${serializeJson(value)}`
  }

  function buildLocalSettingsSnapshot(docs = []) {
    const snapshot = new Map()
    for (const entry of Array.isArray(docs) ? docs : []) {
      const doc = entry?.toJSON?.() || entry
      const key = asText(doc?.key)
      if (!key) continue
      snapshot.set(
        key,
        serializeLocalSettingSnapshotValue(parseJson(doc?.value_json, null), Number(doc?.updated_at) || 0),
      )
    }
    return snapshot
  }

  function rememberLocalSettingSnapshotEntry(key, signature) {
    if (!(localSettingsSnapshot instanceof Map)) return
    const normalizedKey = asText(key)
    if (!normalizedKey) return
    localSettingsSnapshot.set(normalizedKey, String(signature || ""))
  }

  function forgetLocalSettingSnapshotEntry(key) {
    if (!(localSettingsSnapshot instanceof Map)) return
    const normalizedKey = asText(key)
    if (!normalizedKey) return
    localSettingsSnapshot.delete(normalizedKey)
  }

  function emitLocalSettingsSnapshotDiff(nextSnapshot) {
    if (!(nextSnapshot instanceof Map)) return
    if (!(localSettingsSnapshot instanceof Map)) {
      localSettingsSnapshot = new Map(nextSnapshot)
      return
    }

    for (const [key, signature] of nextSnapshot.entries()) {
      if (localSettingsSnapshot.get(key) !== signature) {
        notifySubscribers({ type: "kv-upsert", key })
      }
    }

    for (const key of localSettingsSnapshot.keys()) {
      if (!nextSnapshot.has(key)) {
        notifySubscribers({ type: "kv-delete", key })
      }
    }

    localSettingsSnapshot = new Map(nextSnapshot)
  }

  function toLocalCraftDoc(craft, index = 0) {
    const artifactOnlyCraft = stripEmbeddedBundleFromLocalCraft(craft)
    return {
      id: asText(craft?.id) || `craft-${index + 1}`,
      sort_index: Math.max(0, Math.round(Number(index) || 0)),
      artifact_mode: "artifact_only",
      payload: cloneJson(artifactOnlyCraft, {}),
      updated_at: Date.now(),
    }
  }

  function isLocalChunkedPayload(value) {
    return Boolean(getLocalChunkedPayloadDescriptor(value))
  }

  function getLocalChunkedPayloadDescriptor(value) {
    if (!value || typeof value !== "object") return null
    const descriptor =
      (value.__localChunkedJson && typeof value.__localChunkedJson === "object"
        ? value.__localChunkedJson
        : null) ||
      (value["|localChunkedJson"] && typeof value["|localChunkedJson"] === "object"
        ? value["|localChunkedJson"]
        : null)
    return descriptor && asText(descriptor.assetId) ? descriptor : null
  }

  function buildLocalChunkedPayload(value, artifactId, craftId, kind, now) {
    const serialized = JSON.stringify(value == null ? {} : value)
    const assetId = `${artifactId}:payload`
    const parts = chunkString(serialized, ASSET_CHUNK_CHARS)
    return {
      replacement: {
        "|localChunkedJson": {
          assetId,
          artifactId,
          total: Math.max(1, parts.length),
          format: CHUNKED_JSON_FORMAT,
        },
      },
      docs: parts.map((data, idx) => ({
        id: `${assetId}:${idx}`,
        assetId,
        artifactId,
        craftId,
        kind,
        idx,
        total: Math.max(1, parts.length),
        mime: "application/json",
        data,
        updated_at: now,
      })),
    }
  }

  async function removeLocalArtifactChunks(db, artifactId) {
    const normalizedArtifactId = asText(artifactId)
    if (!normalizedArtifactId || !db?.local_artifact_chunks) return
    const docs = await db.local_artifact_chunks.find({ selector: { artifactId: normalizedArtifactId } }).exec()
    await Promise.all(
      docs.map((doc) =>
        doc.remove().catch(() => {}),
      ),
    )
  }

  async function loadLocalArtifactChunkMap(db, artifactId) {
    const normalizedArtifactId = asText(artifactId)
    const grouped = new Map()
    if (!normalizedArtifactId || !db?.local_artifact_chunks) return grouped
    const docs = await db.local_artifact_chunks.find({ selector: { artifactId: normalizedArtifactId } }).exec()
    for (const doc of docs
      .map((entry) => entry.toJSON())
      .sort((left, right) => Number(left?.idx || 0) - Number(right?.idx || 0))) {
      const assetId = asText(doc?.assetId)
      if (!assetId) continue
      const current = grouped.get(assetId) || {
        total: Math.max(1, Number(doc?.total || 1)),
        parts: [],
      }
      // Chunk payloads must round-trip byte-for-byte; trimming corrupts embedded JSON strings and scripts.
      current.parts[Number(doc?.idx || 0)] = asStoredChunkText(doc?.data)
      grouped.set(assetId, current)
    }
    return grouped
  }

  function hydrateLocalChunkedPayload(value, chunkMap) {
    if (!isLocalChunkedPayload(value)) {
      return cloneJson(value, {})
    }
    const descriptor = getLocalChunkedPayloadDescriptor(value)
    const assetId = asText(descriptor?.assetId)
    const current = chunkMap.get(assetId)
    if (!current || current.parts.length < current.total) {
      throw new Error(`Local artifact chunk payload is incomplete for ${assetId}.`)
    }
    return JSON.parse(current.parts.join(""))
  }

  async function upsertChunkedLocalArtifact(db, record, existingCreatedAt = null) {
    const normalizedId = asText(record?.id)
    if (!normalizedId) {
      throw new Error("Local artifact id is required.")
    }
    const createdAt = toEpochMs(existingCreatedAt || record?.created_at || record?.createdAt || Date.now()) || Date.now()
    const now = Date.now()
    const craftId = asText(record?.craftId)
    const kind = asText(record?.kind)
    const payload = cloneJson(record?.payload, {})
    const meta = cloneJson(record?.meta, {})
    const chunked = buildLocalChunkedPayload(payload, normalizedId, craftId, kind, now)

    await removeLocalArtifactChunks(db, normalizedId)
    await writeChunkDocs(db.local_artifact_chunks, chunked.docs)

    const doc = {
      id: normalizedId,
      craftId,
      kind,
      payload: chunked.replacement,
      meta,
      created_at: createdAt,
      updated_at: now,
    }
    await db.local_craft_artifacts.upsert(doc)
    return doc
  }

  function buildStableOriginKey(ownerDeviceId, craftLike) {
    const sourceCraftId = asText(craftLike?.sync?.sourceCraftId || craftLike?.id)
    if (!ownerDeviceId || !sourceCraftId) return ""
    return `origin:${ownerDeviceId}:${sourceCraftId}`
  }

  function isLocalOriginPlaceholder(value) {
    const candidate = asText(value)
    return !candidate || candidate.startsWith("local:")
  }

  function canonicalizeLocalCraftForOwner(craft, ownerDeviceId, ownerName) {
    const cleaned = stripEmbeddedBundleFromLocalCraft(craft)
    const sync = cleaned?.sync && typeof cleaned.sync === "object" ? { ...cleaned.sync } : {}
    const nextOriginKey = isLocalOriginPlaceholder(sync.originKey)
      ? buildStableOriginKey(ownerDeviceId, {
          ...cleaned,
          sync,
        })
      : asText(sync.originKey)
    const hasRemoteOrigin =
      Boolean(nextOriginKey) &&
      !nextOriginKey.startsWith(`origin:${ownerDeviceId}:`) &&
      !isLocalOriginPlaceholder(nextOriginKey)
    const forkDepth = Math.max(0, Number(sync.forkDepth || 0))
    const isFork = asText(sync.origin) === "fork" || forkDepth > 0 || asText(sync.parentId) || asText(sync.parentShareId)

    cleaned.sync = {
      origin: isFork ? "fork" : "local",
      readOnly: false,
      sourceCraftId: asText(sync.sourceCraftId || cleaned?.id),
      originKey: nextOriginKey || buildStableOriginKey(ownerDeviceId, cleaned),
      originName: asText(sync.originName || cleaned?.name),
      originOwnerName: hasRemoteOrigin ? asText(sync.originOwnerName) : asText(sync.originOwnerName || ownerName),
      originOwnerDeviceId: hasRemoteOrigin
        ? asText(sync.originOwnerDeviceId)
        : asText(sync.originOwnerDeviceId || ownerDeviceId),
      parentId: asText(sync.parentId),
      parentShareId: asText(sync.parentShareId),
      forkDepth,
      forkedAt: asText(sync.forkedAt),
    }

    return cleaned
  }

  async function ensureLocalSettingsReady(db) {
    const marker = await readLocalSettingValueFromDb(db, LOCAL_SETTINGS_READY_KEY, null)
    if (marker?.done === true) return

    const existingDocs = await db.local_settings.find().exec()
    await writeLocalSettingValueToDb(db, LOCAL_SETTINGS_READY_KEY, {
      done: true,
      source: existingDocs.length ? "rxdb_existing" : "fresh",
      count: existingDocs.length,
      completedAt: new Date().toISOString(),
    })
  }

  async function ensureLocalCraftDataReady(db) {
    const marker = await readLocalSettingValueFromDb(db, LOCAL_CRAFTS_READY_KEY, null)
    if (marker?.done !== true) {
      const localDocs = await db.local_crafts.find().exec()
      const artifactDocs = await db.local_craft_artifacts.find().exec()
      await writeLocalSettingValueToDb(db, LOCAL_CRAFTS_READY_KEY, {
        done: true,
        source: localDocs.length || artifactDocs.length ? "rxdb_existing" : "fresh",
        craftCount: localDocs.length,
        artifactCount: artifactDocs.length,
        completedAt: new Date().toISOString(),
      })
    }
  }

  async function ensureLocalArtifactChunksMigrated(db) {
    if (localArtifactChunkMigrationPromise) return localArtifactChunkMigrationPromise

    localArtifactChunkMigrationPromise = (async () => {
      const marker = await readLocalSettingValueFromDb(db, LOCAL_ARTIFACT_CHUNK_MIGRATION_KEY, null)
      if (marker?.done === true) return

      const docs = await db.local_craft_artifacts.find().exec()
      let migratedCount = 0

      for (const doc of docs) {
        const snapshot = doc.toJSON()
        const artifactId = asText(snapshot?.id)
        if (!artifactId) continue

        const currentDoc = await db.local_craft_artifacts.findOne(artifactId).exec()
        const json = currentDoc?.toJSON?.() || null
        if (!json || isLocalChunkedPayload(json?.payload)) continue

        await upsertChunkedLocalArtifact(
          db,
          {
            id: artifactId,
            craftId: asText(json?.craftId),
            kind: asText(json?.kind),
            payload: cloneJson(json?.payload, {}),
            meta: cloneJson(json?.meta, {}),
            created_at: json?.created_at,
          },
          json?.created_at,
        )
        migratedCount += 1
      }

      await writeLocalSettingValueToDb(db, LOCAL_ARTIFACT_CHUNK_MIGRATION_KEY, {
        done: true,
        migratedCount,
        completedAt: new Date().toISOString(),
      })
    })().finally(() => {
      localArtifactChunkMigrationPromise = null
    })

    return localArtifactChunkMigrationPromise
  }

  async function refreshLocalCrafts(db = null) {
    if (localCraftRefreshPromise) return localCraftRefreshPromise

    localCraftRefreshPromise = (async () => {
      try {
        const resolvedDb = db || await getDatabase()
        const docs = await resolvedDb.local_crafts.find().exec()
        const crafts = docs
          .map((doc) => doc.toJSON())
          .sort((left, right) => Number(left?.sort_index || 0) - Number(right?.sort_index || 0))
          .map((doc) => cloneJson(doc?.payload, {}))
          .filter((craft) => craft && typeof craft === "object")
        refreshLocalCrafts.cache = crafts
        notifySubscribers({ type: "local-crafts" })
      } finally {
        localCraftRefreshPromise = null
      }
    })()

    return localCraftRefreshPromise
  }

  async function readLocalCrafts() {
    if (!refreshLocalCrafts.cache) {
      await refreshLocalCrafts()
    }
    return cloneJson(refreshLocalCrafts.cache || [], [])
  }

  async function writeLocalCrafts(localCrafts) {
    const crafts = Array.isArray(localCrafts) ? localCrafts : []
    const db = await getDatabase()
    const ownerDeviceId = await getDeviceId()
    const settings = await readSettings()
    const ownerName = asText(settings.displayName) || `Peer ${ownerDeviceId.slice(0, 6)}`
    const existingDocs = await db.local_crafts.find().exec()
    const keepIds = new Set()
    const canonicalCrafts = crafts.map((craft) =>
      canonicalizeLocalCraftForOwner(craft, ownerDeviceId, ownerName),
    )

    for (const [index, craft] of canonicalCrafts.entries()) {
      const doc = toLocalCraftDoc(craft, index)
      keepIds.add(doc.id)
      await db.local_crafts.upsert(doc)
    }

    for (const doc of existingDocs) {
      const id = asText(doc.toJSON()?.id)
      if (!id || keepIds.has(id)) continue
      try {
        await doc.remove()
      } catch (_error) {}
    }

    runInBackground(
      "publish-local-crafts-failed",
      () => publishLocalCrafts(canonicalCrafts),
      "Publishing local crafts failed.",
    )
    await refreshLocalCrafts()
    return await readLocalCrafts()
  }

  async function readLocalArtifact(id, fallback = null) {
    const artifactId = asText(id)
    if (!artifactId) return cloneJson(fallback, fallback)
    const db = await getDatabase()
    const doc = await db.local_craft_artifacts.findOne(artifactId).exec()
    if (!doc) return cloneJson(fallback, fallback)
    const json = doc.toJSON()
    const chunkMap = isLocalChunkedPayload(json?.payload)
      ? await loadLocalArtifactChunkMap(db, artifactId)
      : new Map()
    return {
      id: asText(json?.id),
      craftId: asText(json?.craftId),
      kind: asText(json?.kind),
      payload: hydrateLocalChunkedPayload(json?.payload, chunkMap),
      meta: cloneJson(json?.meta, {}),
      createdAt: isoFromEpochMs(json?.created_at),
      updatedAt: isoFromEpochMs(json?.updated_at),
    }
  }

  async function listLocalArtifacts(filters = {}) {
    const craftId = asText(filters?.craftId)
    const kind = asText(filters?.kind)
    const db = await getDatabase()
    const docs = await db.local_craft_artifacts.find().exec()
    const records = docs
      .map((doc) => doc.toJSON())
      .filter((record) => {
        if (craftId && asText(record?.craftId) !== craftId) return false
        if (kind && asText(record?.kind) !== kind) return false
        return true
      })
      .sort((left, right) => Number(right?.updated_at || 0) - Number(left?.updated_at || 0))
    return await Promise.all(records.map(async (record) => {
      const artifactId = asText(record?.id)
      const chunkMap = isLocalChunkedPayload(record?.payload)
        ? await loadLocalArtifactChunkMap(db, artifactId)
        : new Map()
      return {
        id: asText(record?.id),
        craftId: asText(record?.craftId),
        kind: asText(record?.kind),
        payload: hydrateLocalChunkedPayload(record?.payload, chunkMap),
        meta: cloneJson(record?.meta, {}),
        createdAt: isoFromEpochMs(record?.created_at),
        updatedAt: isoFromEpochMs(record?.updated_at),
      }
    }))
  }

  function isBundleArtifactKind(kind) {
    const candidate = asText(kind)
    return (
      candidate === TRAINING_DATA_ARTIFACT_KIND ||
      candidate === TOOL_SCRIPTS_ARTIFACT_KIND ||
      candidate === BROWSER_CAPABILITY_BUNDLE_ARTIFACT_KIND ||
      candidate === WEIGHTS_ARTIFACT_KIND ||
      candidate === POLICY_BUNDLE_ARTIFACT_KIND
    )
  }

  function stripEmbeddedBundleFromLocalCraft(craft) {
    const cleaned = cloneJson(craft, {})
    if (cleaned && typeof cleaned === "object") {
      delete cleaned.bundle
    }
    return cleaned
  }

  function getArtifactIdForKind(craftId, kind) {
    const normalizedCraftId = asText(craftId)
    const normalizedKind = asText(kind)
    if (!normalizedCraftId || !normalizedKind) return ""
    if (normalizedKind === TRAINING_DATA_ARTIFACT_KIND) return getTrainingDataArtifactId(normalizedCraftId)
    if (normalizedKind === TOOL_SCRIPTS_ARTIFACT_KIND) return getToolScriptsArtifactId(normalizedCraftId)
    if (normalizedKind === BROWSER_CAPABILITY_BUNDLE_ARTIFACT_KIND) return getBrowserCapabilityBundleArtifactId(normalizedCraftId)
    if (normalizedKind === WEIGHTS_ARTIFACT_KIND) return getWeightsArtifactId(normalizedCraftId)
    if (normalizedKind === POLICY_BUNDLE_ARTIFACT_KIND) return getPolicyBundleArtifactId(normalizedCraftId)
    return ""
  }

  async function getNewestLocalArtifact(craftId, kind) {
    const records = await listLocalArtifacts({
      craftId: asText(craftId),
      kind: asText(kind),
    })
    return records[0] || null
  }

  async function cloneLocalArtifactsForCraft(sourceCraftId = "", targetCraftId = "") {
    const normalizedSourceCraftId = asText(sourceCraftId)
    const normalizedTargetCraftId = asText(targetCraftId)
    if (!normalizedSourceCraftId || !normalizedTargetCraftId || normalizedSourceCraftId === normalizedTargetCraftId) {
      return []
    }

    const sourceRecords = await Promise.all([
      getNewestLocalArtifact(normalizedSourceCraftId, TRAINING_DATA_ARTIFACT_KIND),
      getNewestLocalArtifact(normalizedSourceCraftId, TOOL_SCRIPTS_ARTIFACT_KIND),
      getNewestLocalArtifact(normalizedSourceCraftId, BROWSER_CAPABILITY_BUNDLE_ARTIFACT_KIND),
      getNewestLocalArtifact(normalizedSourceCraftId, WEIGHTS_ARTIFACT_KIND),
      getNewestLocalArtifact(normalizedSourceCraftId, POLICY_BUNDLE_ARTIFACT_KIND),
    ])

    const clonedRecords = []
    const clonedAt = Date.now()
    for (const sourceRecord of sourceRecords) {
      const kind = asText(sourceRecord?.kind)
      const artifactId = getArtifactIdForKind(normalizedTargetCraftId, kind)
      if (!artifactId || !sourceRecord?.payload || typeof sourceRecord.payload !== "object") continue
      clonedRecords.push(await putLocalArtifact({
        id: artifactId,
        craftId: normalizedTargetCraftId,
        kind,
        payload: cloneJson(sourceRecord.payload, {}),
        meta: {
          ...cloneJson(sourceRecord.meta, {}),
          clonedFromArtifactId: asText(sourceRecord.id),
          clonedFromCraftId: normalizedSourceCraftId,
          clonedAt,
        },
      }))
    }
    return clonedRecords
  }

  async function putLocalArtifact(record) {
    const db = await getDatabase()
    const existing = await readLocalArtifact(record?.id, null)
    const doc = await upsertChunkedLocalArtifact(
      db,
      {
        ...record,
        created_at: existing?.createdAt || record?.createdAt || Date.now(),
      },
      existing?.createdAt || record?.createdAt || Date.now(),
    )
    notifySubscribers({
      type: "local-artifact-upsert",
      artifactId: doc.id,
      craftId: doc.craftId,
      kind: doc.kind,
    })
    await republishSharedCraftArtifactsIfNeeded(doc.craftId, doc.kind)
    return await readLocalArtifact(doc.id, null)
  }

  async function deleteLocalArtifact(id) {
    const artifactId = asText(id)
    if (!artifactId) return
    const db = await getDatabase()
    const existing = await db.local_craft_artifacts.findOne(artifactId).exec()
    const json = existing?.toJSON?.() || {}
    try {
      await existing?.remove?.()
    } catch (_error) {}
    await removeLocalArtifactChunks(db, artifactId)
    notifySubscribers({
      type: "local-artifact-delete",
      artifactId,
      craftId: asText(json?.craftId),
      kind: asText(json?.kind),
    })
    await republishSharedCraftArtifactsIfNeeded(asText(json?.craftId), asText(json?.kind))
  }

  async function republishSharedCraftArtifactsIfNeeded(craftId, kind) {
    const normalizedCraftId = asText(craftId)
    if (!normalizedCraftId || !isBundleArtifactKind(kind)) return
    const localCrafts = await readLocalCrafts()
    const craft = localCrafts.find((entry) => asText(entry?.id) === normalizedCraftId)
    if (!craft || craft?.sharing?.enabled !== true) return
    await publishLocalCrafts(localCrafts)
  }

  async function getDatabase() {
    if (sharedDbState.dbPromise) return sharedDbState.dbPromise

    sharedDbState.dbPromise = (async () => {
      const { createRxDatabase, getRxStorageDexie } = rxdbCore()
      let db = null
      try {
        db = await createRxDatabase({
          name: DATABASE_NAME,
          storage: getRxStorageDexie(),
          multiInstance: true,
        })

        const collections = {
          presence: { schema: presenceSchema, conflictHandler },
          shared_crafts: { schema: sharedCraftsSchema, conflictHandler },
          shared_asset_chunks: { schema: sharedAssetChunksSchema, conflictHandler },
          local_crafts: { schema: localCraftsSchema, conflictHandler },
          local_craft_artifacts: { schema: localCraftArtifactsSchema, conflictHandler },
          local_artifact_chunks: { schema: localArtifactChunksSchema, conflictHandler },
          local_settings: { schema: localSettingsSchema, conflictHandler },
        }
        if (REMOTE_COMPUTE_ROLLOUT_ENABLED) {
          collections.compute_workers = { schema: computeWorkersSchema, conflictHandler }
          collections.compute_jobs = {
            schema: computeJobsSchema,
            conflictHandler,
            migrationStrategies: computeJobsMigrationStrategies,
          }
        }
        await db.addCollections(collections)

        await ensureLocalSettingsReady(db)
        await ensureLocalCraftDataReady(db)
        sharedDbState.dbInstance = db
        installQuerySubscriptions(db)
        warmDatabaseCaches(db)
        return db
      } catch (error) {
        if (db) {
          try {
            await db.close()
          } catch (_closeError) {}
        }
        throw error
      }
    })().catch((error) => {
      sharedDbState.dbPromise = null
      sharedDbState.dbInstance = null
      throw error
    })

    return sharedDbState.dbPromise
  }

  function installQuerySubscriptions(db) {
    while (querySubscriptions.length) {
      try {
        querySubscriptions.pop()?.unsubscribe?.()
      } catch (_error) {}
    }
    localSettingsSnapshot = null

    const sharedCraftSub = db.shared_crafts
      .find()
      .$?.subscribe?.(() => {
        void refreshRemoteCrafts()
      })

    const localCraftSub = db.local_crafts
      .find()
      .$?.subscribe?.(() => {
        void refreshLocalCrafts(db)
      })

    const localArtifactSub = db.local_craft_artifacts
      .find()
      .$?.subscribe?.(() => {
        notifySubscribers({ type: "local-artifacts" })
      })

    const presenceSub = db.presence
      .find()
      .$?.subscribe?.(() => {
        void refreshRemotePeerCount()
      })

    const localSettingsSub = db.local_settings
      .find()
      .$?.subscribe?.((docs) => {
        emitLocalSettingsSnapshotDiff(buildLocalSettingsSnapshot(docs))
      })

    if (sharedCraftSub) querySubscriptions.push(sharedCraftSub)
    if (localCraftSub) querySubscriptions.push(localCraftSub)
    if (localArtifactSub) querySubscriptions.push(localArtifactSub)
    if (presenceSub) querySubscriptions.push(presenceSub)
    if (REMOTE_COMPUTE_ROLLOUT_ENABLED) {
      const computeWorkersSub = db.compute_workers
        .find()
        .$?.subscribe?.(() => {
          void refreshComputeWorkers(db)
        })

      const computeJobsSub = db.compute_jobs
        .find()
        .$?.subscribe?.(() => {
          void refreshComputeJobs(db)
        })

      if (computeWorkersSub) querySubscriptions.push(computeWorkersSub)
      if (computeJobsSub) querySubscriptions.push(computeJobsSub)
    }
    if (localSettingsSub) querySubscriptions.push(localSettingsSub)
  }

  function normalizeSignalingUrl(raw) {
    const value = asText(raw)
    if (!value) throw new Error("Signaling URL is empty.")
    const canonicalValue = /^ss:\/\//i.test(value) ? `wss://${value.slice(5)}` : value
    if (!/^wss:\/\//i.test(canonicalValue)) {
      throw new Error("Signaling URL must start with ss:// or wss://")
    }
    const url = new URL(canonicalValue)
    url.search = ""
    url.hash = ""
    return url.toString()
  }

  function withToken(url, token) {
    const trimmedToken = asText(token)
    if (!trimmedToken) return url
    const parsed = new URL(url)
    if (!parsed.searchParams.has("token")) {
      parsed.searchParams.set("token", trimmedToken)
    }
    return parsed.toString()
  }

  async function waitForDrain(channel, {
    maxBuffered = 2_000_000,
    lowThreshold = 512_000,
    timeoutMs = 15_000,
    isAbort = null,
  } = {}) {
    if (!channel || typeof channel.bufferedAmount !== "number") return
    if (channel.bufferedAmount <= maxBuffered) return

    try {
      channel.bufferedAmountLowThreshold = Math.max(0, Math.min(lowThreshold, maxBuffered))
    } catch (_error) {}

    await new Promise((resolve, reject) => {
      let settled = false
      let abortTimer = 0
      let timeoutId = 0

      const cleanup = () => {
        if (settled) return
        settled = true
        try {
          channel.removeEventListener("bufferedamountlow", onLow)
        } catch (_error) {}
        globalScope.clearInterval(abortTimer)
        globalScope.clearTimeout(timeoutId)
      }

      const onLow = () => {
        if (channel.bufferedAmount <= maxBuffered) {
          cleanup()
          resolve()
        }
      }

      if (typeof isAbort === "function") {
        abortTimer = globalScope.setInterval(() => {
          try {
            if (isAbort()) {
              cleanup()
              reject(new Error("datachannel-abort"))
            }
          } catch (_error) {}
        }, 100)
      }

      timeoutId = globalScope.setTimeout(() => {
        cleanup()
        reject(new Error("datachannel-backpressure-timeout"))
      }, timeoutMs)

      try {
        channel.addEventListener("bufferedamountlow", onLow)
      } catch (_error) {}
      onLow()
    })
  }

  function wrapSendGuard(handler) {
    if (!handler || typeof handler.send !== "function") return handler
    const originalSend = handler.send.bind(handler)
    const encoder = new TextEncoder()

    handler.send = async (peer, message) => {
      if (!peer) throw new Error("peer-missing")
      if (peer.destroyed) throw new Error("peer-destroyed")

      const channel = peer._channel
      const connection = peer._pc
      if (channel?.readyState && channel.readyState !== "open") {
        throw new Error("datachannel-not-open")
      }

      if (typeof message === "string") {
        const bytes = encoder.encode(message).byteLength
        const maxMessageSize = connection?.sctp?.maxMessageSize
        if (typeof maxMessageSize === "number" && maxMessageSize > 0 && bytes > maxMessageSize) {
          throw new Error(`msg-too-large bytes=${bytes} max=${maxMessageSize}`)
        }
      }

      if (channel && typeof channel.bufferedAmount === "number") {
        await waitForDrain(channel, {
          isAbort: () => Boolean(peer.destroyed),
        })
      }

      return await originalSend(peer, message)
    }

    return handler
  }

  async function createConnectionHandlerCreator() {
    const settings = await readSettings()
    const urls = asText(settings.signalingUrls)
      .split(",")
      .map((entry) => asText(entry))
      .filter(Boolean)
      .map((entry) => normalizeSignalingUrl(entry))

    const signalingUrl = withToken(urls[0] || DEFAULT_SIGNALING_URL, settings.token)
    const { getConnectionHandlerSimplePeer } = rxdbWebRTC()

    const iceServers = [
      {
        urls: ["stun:stun.l.google.com:19302", "stun:global.stun.twilio.com:3478"],
      },
    ]

    const NativeWebSocket = globalScope.WebSocket
    class SafeWebSocket {
      constructor(url) {
        this._ws = new NativeWebSocket(url)
        this._ws.onopen = (event) => {
          try {
            this.onopen?.(event)
          } catch (_error) {}
        }
        this._ws.onclose = (event) => {
          try {
            this.onclose?.(event)
          } catch (_error) {}
        }
        this._ws.onerror = (event) => {
          try {
            this.onerror?.(event)
          } catch (_error) {}
        }
        this._ws.onmessage = (event) => {
          try {
            this.onmessage?.(event)
          } catch (_error) {
            try {
              this._ws.close()
            } catch (_closeError) {}
          }
        }
      }

      get readyState() {
        return this._ws.readyState
      }

      send(data) {
        if (this._ws.readyState !== NativeWebSocket.OPEN) return
        this._ws.send(data)
      }

      close() {
        try {
          this._ws.close()
        } catch (_error) {}
      }
    }

    const baseCreator = getConnectionHandlerSimplePeer({
      signalingServerUrl: signalingUrl,
      config: { iceServers },
      webSocketConstructor: SafeWebSocket,
      debug: false,
    })

    return async (...args) => {
      const handler = await baseCreator(...args)
      return wrapSendGuard(handler)
    }
  }

  function setTransportPeerCount(nextCount) {
    transportPeerCount = Math.max(0, Number(nextCount) || 0)
    syncState.transportPeerCount = transportPeerCount
    notifySubscribers({ type: "transport-peers" })
  }

  function scheduleRestart(reason) {
    if (syncState.requestedMode !== "continuous") return
    if (restartTimer) return
    restartTimer = globalScope.setTimeout(async () => {
      restartTimer = 0
      try {
        await stopReplicationOnly()
        await startReplication("continuous")
      } catch (error) {
        syncState.lastError = String(error?.message || error || reason || "Restart failed.")
        notifySubscribers({ type: "restart-failed" })
        scheduleRestart(error)
      }
    }, 1_500)
  }

  async function startReplication(requestedMode) {
    await stopReplicationOnly()
    const db = await getDatabase()
    const { replicateWebRTC } = rxdbWebRTC()
    const handlerCreator = await createConnectionHandlerCreator()

    const collections = [
      { name: "shared_crafts", batchSize: 5 },
      { name: "shared_asset_chunks", batchSize: 1 },
      { name: "presence", batchSize: 10 },
    ]
    if (REMOTE_COMPUTE_ROLLOUT_ENABLED) {
      collections.push(
        { name: "compute_workers", batchSize: 10 },
        { name: "compute_jobs", batchSize: 10 },
      )
    }

    const cleanupFns = []

    const trackPool = (pool) => {
      let collectionConnected = false

      const connectSub = pool?.connect$?.subscribe?.(() => {
        if (!collectionConnected) {
          collectionConnected = true
          setTransportPeerCount(transportPeerCount + 1)
        }
        syncState.connection = "connected"
        syncState.lastError = ""
        notifySubscribers({ type: "webrtc-connect" })
      })

      const disconnectSub = pool?.disconnect$?.subscribe?.(() => {
        if (collectionConnected) {
          collectionConnected = false
          setTransportPeerCount(transportPeerCount - 1)
        }
        if (!transportPeerCount) {
          syncState.connection = syncState.running ? "waiting" : "stopped"
          notifySubscribers({ type: "webrtc-disconnect" })
        }
      })

      const errorSub = pool?.error$?.subscribe?.((error) => {
        syncState.connection = "error"
        syncState.lastError = String(error?.message || error || "WebRTC sync failed.")
        notifySubscribers({ type: "webrtc-error" })
        scheduleRestart(error)
      })

      cleanupFns.push(() => connectSub?.unsubscribe?.())
      cleanupFns.push(() => disconnectSub?.unsubscribe?.())
      cleanupFns.push(() => errorSub?.unsubscribe?.())
      cleanupFns.push(() => pool?.cleanup?.())
      cleanupFns.push(() => pool?.cancel?.())
    }

    for (const { name, batchSize } of collections) {
      const pool = await replicateWebRTC({
        collection: db[name],
        connectionHandlerCreator: handlerCreator,
        topic: `${REPLICATION_ROOM}-${name}`,
        pull: { batchSize },
        push: { batchSize },
      })
      trackPool(pool)
    }

    replicationTeardown = async () => {
      while (cleanupFns.length) {
        try {
          await cleanupFns.pop()()
        } catch (_error) {}
      }
      setTransportPeerCount(0)
    }

    syncState.requestedMode = requestedMode
    syncState.running = true
    syncState.connection = "waiting"
    notifySubscribers({ type: "replication-started" })
  }

  async function stopReplicationOnly() {
    globalScope.clearTimeout(restartTimer)
    restartTimer = 0
    if (!replicationTeardown) {
      setTransportPeerCount(0)
      return
    }
    const teardown = replicationTeardown
    replicationTeardown = null
    await teardown()
  }

  async function startPresence(displayName) {
    await stopPresence()
    const db = await getDatabase()
    const deviceId = await getDeviceId()
    const runtimeId = crypto.randomUUID()
    const presenceId = `${deviceId}:${runtimeId}`
    const resolvedName = asText(displayName) || `Peer ${deviceId.slice(0, 6)}`

    const writePresence = async () => {
      await db.presence.upsert({
        id: presenceId,
        deviceId,
        name: resolvedName,
        user_agent: asText(globalScope.navigator?.userAgent),
        last_seen: Date.now(),
      })
    }

    await writePresence()
    const intervalId = globalScope.setInterval(() => {
      void writePresence()
    }, PRESENCE_HEARTBEAT_MS)

    presenceTeardown = async () => {
      globalScope.clearInterval(intervalId)
      try {
        const doc = await db.presence.findOne(presenceId).exec()
        await doc?.remove?.()
      } catch (_error) {}
      syncState.remotePeerCount = 0
      notifySubscribers({ type: "presence-stopped" })
    }
  }

  async function stopPresence() {
    if (!presenceTeardown) return
    const teardown = presenceTeardown
    presenceTeardown = null
    await teardown()
  }

  async function refreshRemotePeerCount(db = null) {
    if (presenceRefreshPromise) return presenceRefreshPromise
    presenceRefreshPromise = (async () => {
      try {
        const resolvedDb = db || await getDatabase()
        const selfDeviceId = await getDeviceId()
        const cutoff = Date.now() - PRESENCE_MAX_AGE_MS
        const docs = await resolvedDb.presence.find().exec()
        const peers = new Set()

        for (const doc of docs) {
          const json = doc.toJSON()
          const lastSeen = toEpochMs(json?.last_seen)
          const deviceId = asText(json?.deviceId)
          if (!deviceId || deviceId === selfDeviceId) continue
          if (lastSeen < cutoff) continue
          peers.add(deviceId)
        }

        syncState.remotePeerCount = peers.size
        notifySubscribers({ type: "remote-peers" })
      } finally {
        presenceRefreshPromise = null
      }
    })()

    return presenceRefreshPromise
  }

  function chunkString(value, size) {
    const out = []
    for (let index = 0; index < value.length; index += size) {
      out.push(value.slice(index, index + size))
    }
    return out
  }

  function normalizeDataUrlAsset(value) {
    const raw = asText(value)
    if (!raw || !/^data:/i.test(raw)) return null
    const commaIndex = raw.indexOf(",")
    if (commaIndex < 0) return null

    const head = raw.slice(0, commaIndex).trim()
    const body = raw.slice(commaIndex + 1).trim().replace(/\s+/g, "")
    const match = head.match(/^data:([^;]+);base64$/i)
    if (!match || !body) return null
    if (!/^[A-Za-z0-9+/]+={0,2}$/.test(body)) return null

    const mime = asText(match[1]).toLowerCase()
    if (!mime) return null

    return { mime, base64: body }
  }

  function buildChunkedAsset(value, shareId, assetIndex, ownerDeviceId, now) {
    const normalized = normalizeDataUrlAsset(value)
    if (!normalized || normalized.base64.length < ASSET_INLINE_THRESHOLD) {
      return null
    }

    const assetId = `${shareId}:asset:${assetIndex}`
    const parts = chunkString(normalized.base64, ASSET_CHUNK_CHARS)
    return {
      replacement: {
        __sharedAsset: {
          assetId,
          shareId,
          mime: normalized.mime,
          total: parts.length,
        },
      },
      docs: parts.map((data, idx) => ({
        id: `${assetId}:${idx}`,
        assetId,
        shareId,
        ownerDeviceId,
        idx,
        total: parts.length,
        mime: normalized.mime,
        data,
        updated_at: now,
      })),
    }
  }

  function buildSharedChunkedJsonAsset(value, shareId, assetIndex, ownerDeviceId, now) {
    const serialized = JSON.stringify(value == null ? {} : value)
    const assetId = `${shareId}:bundle:${assetIndex}`
    const parts = chunkString(serialized, ASSET_CHUNK_CHARS)
    return {
      replacement: {
        __sharedChunkedJson: {
          assetId,
          shareId,
          total: Math.max(1, parts.length),
          format: CHUNKED_JSON_FORMAT,
        },
      },
      docs: parts.map((data, idx) => ({
        id: `${assetId}:${idx}`,
        assetId,
        shareId,
        ownerDeviceId,
        idx,
        total: Math.max(1, parts.length),
        mime: "application/json",
        data,
        updated_at: now,
      })),
    }
  }

  function chunkSharedBundlePayloads(bundle, { shareId, ownerDeviceId, now }) {
    const nextBundle = cloneJson(bundle, {})
    const chunkDocs = []
    let assetIndex = 0

    for (const sectionName of ["trainingData", "toolScripts", "browserCapabilities", "weights", "policy"]) {
      const section = nextBundle?.[sectionName]
      if (!section || typeof section !== "object" || !section.payload || typeof section.payload !== "object") {
        continue
      }
      const asset = buildSharedChunkedJsonAsset(section.payload, shareId, assetIndex, ownerDeviceId, now)
      assetIndex += 1
      section.payload = asset.replacement
      chunkDocs.push(...asset.docs)
    }

    return {
      bundle: nextBundle,
      chunkDocs,
    }
  }

  function extractSharedAssets(value, { shareId, ownerDeviceId, now }) {
    let assetIndex = 0
    const chunkDocs = []

    const visit = (input) => {
      if (Array.isArray(input)) {
        return input.map((entry) => visit(entry))
      }
      if (!input || typeof input !== "object") {
        if (typeof input === "string") {
          const asset = buildChunkedAsset(input, shareId, assetIndex, ownerDeviceId, now)
          if (asset) {
            assetIndex += 1
            chunkDocs.push(...asset.docs)
            return asset.replacement
          }
        }
        return input
      }

      const out = {}
      for (const [key, entry] of Object.entries(input)) {
        out[key] = visit(entry)
      }
      return out
    }

    return {
      payload: visit(value),
      chunkDocs,
    }
  }

  async function writeChunkDocs(collection, docs) {
    if (!docs.length) return
    if (typeof collection.bulkUpsert === "function") {
      const result = await collection.bulkUpsert(docs)
      if (result?.error?.length) {
        throw new Error(`Chunk bulkUpsert failed (${result.error.length}).`)
      }
      return
    }
    if (typeof collection.bulkInsert === "function") {
      const result = await collection.bulkInsert(docs)
      const errors = Array.isArray(result?.error) ? result.error : []
      const nonConflicts = errors.filter((entry) => !String(entry?.status || entry?.error || "").includes("conflict"))
      if (nonConflicts.length) {
        throw new Error(`Chunk bulkInsert failed (${nonConflicts.length}).`)
      }
      return
    }
    for (const doc of docs) {
      await collection.upsert(doc)
    }
  }

  async function removeShareAssets(db, shareId) {
    const docs = await db.shared_asset_chunks.find({ selector: { shareId } }).exec()
    await Promise.all(
      docs.map((doc) =>
        doc.remove().catch(() => {}),
      ),
    )
  }

  function sanitizeCraftForShare(craft, ownerDeviceId, ownerName) {
    const cleaned = canonicalizeLocalCraftForOwner(craft, ownerDeviceId, ownerName)
    delete cleaned.sharing
    cleaned.sync = {
      origin: asText(cleaned?.sync?.origin),
      sourceCraftId: asText(cleaned?.sync?.sourceCraftId || cleaned?.id),
      originKey: asText(cleaned?.sync?.originKey),
      originName: asText(cleaned?.sync?.originName || cleaned?.name),
      originOwnerName: asText(cleaned?.sync?.originOwnerName),
      originOwnerDeviceId: asText(cleaned?.sync?.originOwnerDeviceId),
      parentId: asText(cleaned?.sync?.parentId),
      parentShareId: asText(cleaned?.sync?.parentShareId),
      forkDepth: Math.max(0, Number(cleaned?.sync?.forkDepth || 0)),
      forkedAt: asText(cleaned?.sync?.forkedAt),
    }
    return cleaned
  }

  function isSharedLocalCraft(craft) {
    return craft && typeof craft === "object" && craft?.sharing?.enabled === true
  }

  function getShareId(ownerDeviceId, craft) {
    return `${ownerDeviceId}:${asText(craft?.id)}`
  }

  async function publishLocalCrafts(localCrafts) {
    const db = await getDatabase()
    const ownerDeviceId = await getDeviceId()
    const settings = await readSettings()
    const ownerName = asText(settings.displayName) || `Peer ${ownerDeviceId.slice(0, 6)}`
    const publishable = Array.isArray(localCrafts)
      ? localCrafts.filter((craft) => isSharedLocalCraft(craft))
      : []

    const existingDocs = await db.shared_crafts.find().exec()
    const ownDocs = existingDocs.filter((doc) => asText(doc.toJSON()?.ownerDeviceId) === ownerDeviceId)
    const keepIds = new Set()

    for (const craft of publishable) {
      const shareId = getShareId(ownerDeviceId, craft)
      keepIds.add(shareId)

      const now = Date.now()
      const [trainingDataRecord, toolScriptsRecord, browserCapabilitiesRecord, weightsRecord, policyRecord] = await Promise.all([
        getNewestLocalArtifact(craft?.id, TRAINING_DATA_ARTIFACT_KIND),
        getNewestLocalArtifact(craft?.id, TOOL_SCRIPTS_ARTIFACT_KIND),
        getNewestLocalArtifact(craft?.id, BROWSER_CAPABILITY_BUNDLE_ARTIFACT_KIND),
        getNewestLocalArtifact(craft?.id, WEIGHTS_ARTIFACT_KIND),
        getNewestLocalArtifact(craft?.id, POLICY_BUNDLE_ARTIFACT_KIND),
      ])
      const bundle = buildCapabilityBundle({
        craft,
        trainingDataRecord,
        toolScriptsRecord,
        browserCapabilitiesRecord,
        weightsRecord,
        policyRecord,
        generatedAt: new Date(now).toISOString(),
        preserveStoredBrowserCapabilities: true,
        toolScriptsOptions: {
          inferPlaceholderScripts: false,
          allowToolFallback: false,
        },
        browserCapabilityOptions: {
          allowToolNameInference: false,
          allowSyntheticCapabilities: false,
          allowFallbackExecuteScript: false,
          allowBundleFallback: false,
        },
      })
      const { bundle: chunkedBundle, chunkDocs: bundleChunkDocs } = chunkSharedBundlePayloads(bundle, {
        shareId,
        ownerDeviceId,
        now,
      })
      const sanitized = sanitizeCraftForShare(
        {
          ...craft,
          bundle: chunkedBundle,
        },
        ownerDeviceId,
        ownerName,
      )
      const { payload, chunkDocs } = extractSharedAssets(sanitized, {
        shareId,
        ownerDeviceId,
        now,
      })

      await removeShareAssets(db, shareId)
      await writeChunkDocs(db.shared_asset_chunks, [...bundleChunkDocs, ...chunkDocs])

      const existingDoc = ownDocs.find((doc) => doc.toJSON()?.id === shareId)
      const publishedAt = toEpochMs(existingDoc?.toJSON?.()?.published_at) || now

      await db.shared_crafts.upsert({
        id: shareId,
        ownerDeviceId,
        ownerName,
        sourceCraftId: asText(craft?.id),
        name: asText(craft?.name) || "Shared craft",
        summary: asText(craft?.summary),
        stage: asText(craft?.stage),
        targetSlot: asText(craft?.targetSlot),
        useStatus: asText(craft?.useStatus),
        starterMode: asText(craft?.starterMode),
        payload,
        published_at: publishedAt,
        updated_at: now,
      })
    }

    for (const doc of ownDocs) {
      const shareId = asText(doc.toJSON()?.id)
      if (!shareId || keepIds.has(shareId)) continue
      await removeShareAssets(db, shareId)
      try {
        await doc.remove()
      } catch (_error) {}
    }

    syncState.publishedCount = publishable.length
    notifySubscribers({ type: "publish-local-crafts" })
  }

  function gatherSharedAssetRefs(value, refs = []) {
    if (Array.isArray(value)) {
      value.forEach((entry) => gatherSharedAssetRefs(entry, refs))
      return refs
    }
    if (!value || typeof value !== "object") return refs

    if (
      value.__sharedAsset &&
      typeof value.__sharedAsset === "object" &&
      asText(value.__sharedAsset.assetId)
    ) {
      refs.push({
        assetId: asText(value.__sharedAsset.assetId),
        shareId: asText(value.__sharedAsset.shareId),
        mime: asText(value.__sharedAsset.mime) || "application/octet-stream",
        total: Math.max(1, Number(value.__sharedAsset.total || 1)),
      })
      return refs
    }

    if (
      value.__sharedChunkedJson &&
      typeof value.__sharedChunkedJson === "object" &&
      asText(value.__sharedChunkedJson.assetId)
    ) {
      refs.push({
        assetId: asText(value.__sharedChunkedJson.assetId),
        shareId: asText(value.__sharedChunkedJson.shareId),
        mime: "application/json",
        total: Math.max(1, Number(value.__sharedChunkedJson.total || 1)),
        format: asText(value.__sharedChunkedJson.format) || CHUNKED_JSON_FORMAT,
      })
      return refs
    }

    Object.values(value).forEach((entry) => gatherSharedAssetRefs(entry, refs))
    return refs
  }

  function hydrateSharedAssets(value, assetsById) {
    if (Array.isArray(value)) {
      return value.map((entry) => hydrateSharedAssets(entry, assetsById))
    }
    if (!value || typeof value !== "object") return value

    if (
      value.__sharedAsset &&
      typeof value.__sharedAsset === "object" &&
      asText(value.__sharedAsset.assetId)
    ) {
      const assetId = asText(value.__sharedAsset.assetId)
      const asset = assetsById.get(assetId)
      if (!asset || asset.parts.length < asset.total) return ""
      return `data:${asset.mime};base64,${asset.parts.join("")}`
    }

    if (
      value.__sharedChunkedJson &&
      typeof value.__sharedChunkedJson === "object" &&
      asText(value.__sharedChunkedJson.assetId)
    ) {
      const assetId = asText(value.__sharedChunkedJson.assetId)
      const asset = assetsById.get(assetId)
      if (!asset || asset.parts.length < asset.total) return {}
      try {
        return JSON.parse(asset.parts.join(""))
      } catch (_error) {
        return {}
      }
    }

    const out = {}
    for (const [key, entry] of Object.entries(value)) {
      out[key] = hydrateSharedAssets(entry, assetsById)
    }
    return out
  }

  async function loadAssetMapForShare(db, shareId, refs) {
    if (!refs.length) return new Map()
    const docs = await db.shared_asset_chunks.find({ selector: { shareId } }).exec()
    const allowedIds = new Set(refs.map((ref) => ref.assetId))
    const grouped = new Map()

    for (const doc of docs
      .map((entry) => entry.toJSON())
      .sort((left, right) => Number(left?.idx || 0) - Number(right?.idx || 0))) {
      const assetId = asText(doc?.assetId)
      if (!allowedIds.has(assetId)) continue
      const current = grouped.get(assetId) || {
        mime: asText(doc?.mime) || "application/octet-stream",
        total: Math.max(1, Number(doc?.total || 1)),
        parts: [],
      }
      // Shared chunk payloads use the same chunked JSON format and must preserve exact chunk boundaries.
      current.parts[Number(doc?.idx || 0)] = asStoredChunkText(doc?.data)
      grouped.set(assetId, current)
    }

    return grouped
  }

  function buildRemoteCraft(sharedDoc) {
    const payload = cloneJson(sharedDoc?.payload, {})
    const payloadSync = payload?.sync && typeof payload.sync === "object" ? payload.sync : {}
    const updatedAt = isoFromEpochMs(sharedDoc?.updated_at)
    const publishedAt = isoFromEpochMs(sharedDoc?.published_at)

    return {
      ...payload,
      id: `shared:${asText(sharedDoc?.id)}`,
      name: asText(payload?.name) || asText(sharedDoc?.name) || "Shared craft",
      summary: asText(payload?.summary) || asText(sharedDoc?.summary),
      createdAt: asText(payload?.createdAt) || publishedAt,
      updatedAt: updatedAt || asText(payload?.updatedAt),
      sync: {
        origin: "remote_share",
        readOnly: true,
        shareId: asText(sharedDoc?.id),
        ownerDeviceId: asText(sharedDoc?.ownerDeviceId),
        ownerName: asText(sharedDoc?.ownerName),
        sourceCraftId: asText(sharedDoc?.sourceCraftId),
        syncedAt: updatedAt,
        publishedAt,
        mode: syncState.running ? "live" : "snapshot",
        originKey:
          asText(payloadSync.originKey) ||
          buildStableOriginKey(
            asText(payloadSync.originOwnerDeviceId || sharedDoc?.ownerDeviceId),
            {
              id: asText(payloadSync.sourceCraftId || sharedDoc?.sourceCraftId),
            },
          ),
        originName: asText(payloadSync.originName) || asText(payload?.name) || asText(sharedDoc?.name),
        originOwnerName: asText(payloadSync.originOwnerName) || asText(sharedDoc?.ownerName),
        originOwnerDeviceId: asText(payloadSync.originOwnerDeviceId || sharedDoc?.ownerDeviceId),
        parentId: asText(payloadSync.parentId),
        parentShareId: asText(payloadSync.parentShareId),
        forkDepth: Math.max(0, Number(payloadSync.forkDepth || 0)),
        forkedAt: asText(payloadSync.forkedAt),
      },
      sharing: {
        enabled: false,
      },
    }
  }

  async function refreshRemoteCrafts(db = null) {
    if (remoteCraftRefreshPromise) return remoteCraftRefreshPromise

    remoteCraftRefreshPromise = (async () => {
      try {
        const resolvedDb = db || await getDatabase()
        const selfDeviceId = await getDeviceId()
        const docs = await resolvedDb.shared_crafts.find().exec()
        const remoteCrafts = []

        for (const doc of docs) {
          const json = doc.toJSON()
          if (asText(json?.ownerDeviceId) === selfDeviceId) continue
          const refs = gatherSharedAssetRefs(json?.payload, [])
          const assetsById = await loadAssetMapForShare(resolvedDb, asText(json?.id), refs)
          const hydratedPayload = hydrateSharedAssets(json?.payload, assetsById)
          remoteCrafts.push(
            buildRemoteCraft({
              ...json,
              payload: hydratedPayload,
            }),
          )
        }

        remoteCrafts.sort((left, right) => {
          const rightTs = toEpochMs(right?.updatedAt || right?.sync?.syncedAt)
          const leftTs = toEpochMs(left?.updatedAt || left?.sync?.syncedAt)
          return rightTs - leftTs
        })

        syncState.remoteCraftCount = remoteCrafts.length
        syncState.lastSyncedAt = new Date().toISOString()
        refreshRemoteCrafts.cache = remoteCrafts
        notifySubscribers({ type: "remote-crafts" })
      } finally {
        remoteCraftRefreshPromise = null
      }
    })()

    return remoteCraftRefreshPromise
  }

  async function readRemoteCrafts() {
    if (!refreshRemoteCrafts.cache) {
      if (!remoteCraftRefreshPromise) {
        warmRemoteState(sharedDbState.dbInstance)
      }
      return []
    }
    return cloneJson(refreshRemoteCrafts.cache || [], [])
  }

  async function refreshComputeWorkers(db = null) {
    if (!REMOTE_COMPUTE_ROLLOUT_ENABLED) {
      syncState.remoteComputeWorkerCount = 0
      refreshComputeWorkers.cache = []
      return []
    }
    if (computeWorkerRefreshPromise) return computeWorkerRefreshPromise

    computeWorkerRefreshPromise = (async () => {
      try {
        const resolvedDb = db || await getDatabase()
        const selfDeviceId = await getDeviceId()
        const docs = await resolvedDb.compute_workers.find().exec()
        const workers = docs
          .map((doc) => summarizeComputeWorker(doc.toJSON()))
          .filter((record) => record && record.deviceId && record.deviceId !== selfDeviceId)
          .sort((left, right) => {
            const availabilityDelta = Number(right?.available === true) - Number(left?.available === true)
            if (availabilityDelta !== 0) return availabilityDelta
            return toEpochMs(right?.lastSeen) - toEpochMs(left?.lastSeen)
          })
        syncState.remoteComputeWorkerCount = workers.length
        refreshComputeWorkers.cache = workers
        notifySubscribers({ type: "compute-workers" })
      } finally {
        computeWorkerRefreshPromise = null
      }
    })()

    return computeWorkerRefreshPromise
  }

  async function listComputeWorkers() {
    if (!REMOTE_COMPUTE_ROLLOUT_ENABLED) return []
    if (!refreshComputeWorkers.cache) {
      if (!computeWorkerRefreshPromise) {
        warmRemoteState(sharedDbState.dbInstance)
      }
      return []
    }
    return cloneJson(refreshComputeWorkers.cache || [], [])
  }

  async function upsertComputeWorker(record = {}) {
    if (!REMOTE_COMPUTE_ROLLOUT_ENABLED) return null
    const normalized = summarizeComputeWorker(record)
    if (!normalized?.deviceId) {
      throw new Error("Compute worker deviceId is required.")
    }
    const workerId = asText(normalized.id) || buildComputeWorkerId(normalized.deviceId)
    const now = Date.now()
    const db = await getDatabase()
    await db.compute_workers.upsert({
      id: workerId,
      deviceId: normalized.deviceId,
      name: normalized.name,
      modelName: normalized.modelName,
      available: normalized.available === true,
      busy: normalized.busy === true,
      activeJobId: normalized.activeJobId,
      max_concurrent_jobs: Math.max(1, Number(normalized.maxConcurrentJobs || 1) || 1),
      last_seen: Math.max(now, toEpochMs(normalized.lastSeen) || 0),
      updated_at: now,
    })
    await refreshComputeWorkers(db)
    return (await listComputeWorkers()).find((entry) => entry.id === workerId) || null
  }

  async function removeComputeWorker(workerId = "") {
    if (!REMOTE_COMPUTE_ROLLOUT_ENABLED) return
    const normalizedWorkerId = asText(workerId)
    if (!normalizedWorkerId) return
    const db = await getDatabase()
    const doc = await db.compute_workers.findOne(normalizedWorkerId).exec()
    try {
      await doc?.remove?.()
    } catch (_error) {}
    await refreshComputeWorkers(db)
  }

  async function refreshComputeJobs(db = null) {
    if (!REMOTE_COMPUTE_ROLLOUT_ENABLED) {
      syncState.computeJobCount = 0
      refreshComputeJobs.cache = []
      return []
    }
    if (computeJobRefreshPromise) return computeJobRefreshPromise

    computeJobRefreshPromise = (async () => {
      try {
        const resolvedDb = db || await getDatabase()
        const docs = await resolvedDb.compute_jobs.find().exec()
        const jobs = docs
          .map((doc) => summarizeComputeJob(doc.toJSON()))
          .filter(Boolean)
          .sort((left, right) => toEpochMs(right?.updatedAt) - toEpochMs(left?.updatedAt))
        syncState.computeJobCount = jobs.length
        refreshComputeJobs.cache = jobs
        notifySubscribers({ type: "compute-jobs" })
      } finally {
        computeJobRefreshPromise = null
      }
    })()

    return computeJobRefreshPromise
  }

  async function listComputeJobs() {
    if (!REMOTE_COMPUTE_ROLLOUT_ENABLED) return []
    if (!refreshComputeJobs.cache) {
      if (!computeJobRefreshPromise) {
        warmRemoteState(sharedDbState.dbInstance)
      }
      return []
    }
    return cloneJson(refreshComputeJobs.cache || [], [])
  }

  async function getComputeJob(jobId = "") {
    if (!REMOTE_COMPUTE_ROLLOUT_ENABLED) return null
    const normalizedJobId = asText(jobId)
    if (!normalizedJobId) return null
    const db = await getDatabase()
    const doc = await db.compute_jobs.findOne(normalizedJobId).exec()
    return summarizeComputeJob(doc?.toJSON?.() || null)
  }

  async function upsertComputeJob(record = {}) {
    if (!REMOTE_COMPUTE_ROLLOUT_ENABLED) return null
    const normalized = summarizeComputeJob(record)
    if (!normalized?.id) {
      throw new Error("Compute job id is required.")
    }
    const db = await getDatabase()
    const now = Date.now()
    await db.compute_jobs.upsert({
      id: normalized.id,
      kind: normalized.kind,
      status: normalized.status || "queued",
      modelName: normalized.modelName,
      requesterDeviceId: normalized.requesterDeviceId,
      requesterName: asText(record?.requesterName),
      workerDeviceId: normalized.workerDeviceId,
      workerName: asText(record?.workerName),
      error: normalized.error,
      progress: Math.max(0, Math.min(1, Number(normalized.progress || 0))),
      summary: cloneJson(record?.summary, {}),
      created_at: toEpochMs(normalized.createdAt) || now,
      started_at: toEpochMs(normalized.startedAt),
      completed_at: toEpochMs(normalized.completedAt),
      updated_at: now,
    })
    await refreshComputeJobs(db)
    return await getComputeJob(normalized.id)
  }

  async function deleteComputeJob(jobId = "") {
    if (!REMOTE_COMPUTE_ROLLOUT_ENABLED) return
    const normalizedJobId = asText(jobId)
    if (!normalizedJobId) return
    const db = await getDatabase()
    const doc = await db.compute_jobs.findOne(normalizedJobId).exec()
    try {
      await doc?.remove?.()
    } catch (_error) {}
    await refreshComputeJobs(db)
  }

  async function materializeBundleArtifactsForCraft(craft, overrideCraftId = "") {
    const targetCraftId = asText(overrideCraftId || craft?.id)
    const bundle = craft?.bundle && typeof craft.bundle === "object" ? craft.bundle : null
    if (!targetCraftId || !bundle) return []

    const records = extractBundleArtifactRecords(bundle, targetCraftId)
    const written = []
    for (const record of records) {
      written.push(await putLocalArtifact(record))
    }
    return written
  }

  async function startContinuous({ pageName = "Options" } = {}) {
    const settings = await readSettings()
    syncState.pageName = pageName
    syncState.requestedMode = "continuous"
    if (settings.mode !== "continuous") {
      await updateSettings({ ...settings, mode: "continuous" })
      return
    }

    await startPresence(settings.displayName)
    await startReplication("continuous")
    warmRemoteState(sharedDbState.dbInstance)
  }

  async function syncNow({ pageName = "Options", durationMs = MANUAL_SYNC_DURATION_MS } = {}) {
    const settings = await readSettings()
    syncState.pageName = pageName
    syncState.requestedMode = "manual"
    syncState.connection = "starting"
    notifySubscribers({ type: "manual-sync-starting" })

    await startPresence(settings.displayName)
    await startReplication("manual")
    warmRemoteState(sharedDbState.dbInstance)

    globalScope.clearTimeout(manualStopTimer)
    manualStopTimer = globalScope.setTimeout(() => {
      void stopSync({ keepRequestedMode: true })
    }, Math.max(2_000, Number(durationMs) || MANUAL_SYNC_DURATION_MS))
  }

  async function stopSync({ keepRequestedMode = false } = {}) {
    globalScope.clearTimeout(manualStopTimer)
    manualStopTimer = 0
    await stopReplicationOnly()
    await stopPresence()
    syncState.running = false
    syncState.connection = "stopped"
    syncState.transportPeerCount = 0
    if (!keepRequestedMode) {
      syncState.requestedMode = "off"
    }
    notifySubscribers({ type: "sync-stopped" })
  }

  async function ensureStartedFromSettings({ pageName = "Options" } = {}) {
    const settings = await readSettings()
    syncState.pageName = pageName
    syncState.requestedMode = settings.mode

    if (settings.mode === "continuous" && !syncState.running) {
      await startContinuous({ pageName })
    } else {
      const db = await getDatabase()
      warmRemoteState(db)
    }

    return {
      settings: cloneJson(settings, {}),
      sync: cloneJson(syncState, {}),
    }
  }

  const api = {
    SETTINGS_KEY,
    DEFAULT_SIGNALING_URL,
    generateRandomToken,
    getValue,
    setValue,
    deleteValue,
    readSettings,
    refreshSettings,
    updateSettings,
    getDeviceId,
    getState: getPublicState,
    subscribe,
    ensureStartedFromSettings,
    startContinuous,
    syncNow,
    stopSync,
    readLocalCrafts,
    writeLocalCrafts,
    readLocalArtifact,
    listLocalArtifacts,
    putLocalArtifact,
    deleteLocalArtifact,
    cloneLocalArtifactsForCraft,
    publishLocalCrafts,
    readRemoteCrafts,
    listComputeWorkers,
    upsertComputeWorker,
    removeComputeWorker,
    listComputeJobs,
    getComputeJob,
    upsertComputeJob,
    deleteComputeJob,
    materializeBundleArtifactsForCraft,
  }
  singleton.api = api
  globalScope.SinepanelCraftSync = api
})(globalThis)
