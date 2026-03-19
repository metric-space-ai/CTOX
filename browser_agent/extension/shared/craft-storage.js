(function registerCraftStorage(globalScope) {
  const CRAFT_SYNC_LOCAL_TIMEOUT_MS = 20000;
  const CRAFT_SYNC_LOCAL_TIMEOUT_RETRY_MS = 45000;
  const CRAFT_SYNC_REMOTE_TIMEOUT_MS = 1800;
  const CRAFT_SYNC_REMOTE_TIMEOUT_RETRY_MS = 6000;
  const DEFAULT_DEBUG_FIXED_TRAINING_CRAFT_ID = "__debug_fixed_training__";

  const ROLLOUT_STAGES = [
    "Task Spec",
    "Golden Pairs",
    "Seed Collection",
    "Canary",
    "Pilot",
    "Training",
    "Evaluation",
  ];

  const DEBUG_FIXED_TRAINING = Object.freeze({
    craftId: DEFAULT_DEBUG_FIXED_TRAINING_CRAFT_ID,
    shardId: "debug-fixed-run",
    modelName: "unsloth/Qwen3.5-0.8B",
  });

  function asText(value) {
    return String(value == null ? "" : value).trim();
  }

  function cloneJson(value, fallback) {
    try {
      return JSON.parse(JSON.stringify(value));
    } catch (_error) {
      return fallback;
    }
  }

  function normalizeNumber(value, fallback = 0) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  }

  function normalizeOptionalNumber(value) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }

  function normalizeInteger(value, fallback = 0) {
    const parsed = Number(value);
    if (!Number.isFinite(parsed)) return fallback;
    return Math.max(0, Math.round(parsed));
  }

  function normalizeTextArray(values) {
    if (!Array.isArray(values)) return [];
    return values
      .map((value) => asText(value))
      .filter(Boolean);
  }

  function normalizeTooling(tooling, tools) {
    const totalFromTools = Array.isArray(tools) ? tools.length : 0;
    if (!tooling || typeof tooling !== "object") {
      return { ready: totalFromTools, total: totalFromTools };
    }

    const total = Math.max(0, normalizeInteger(tooling.total, totalFromTools));
    const ready = Math.min(total, Math.max(0, normalizeInteger(tooling.ready, total)));
    return { ready, total };
  }

  function normalizeNavigatorRecord(record, index = 0) {
    const updatedAt = asText(record?.updatedAt || record?.createdAt);
    return {
      id: asText(record?.id) || `record-${index + 1}`,
      label: asText(record?.label) || "activity",
      status: asText(record?.status) || "recorded",
      source: asText(record?.source),
      input: asText(record?.input),
      output: asText(record?.output),
      meta: asText(record?.meta),
      updatedAt,
    };
  }

  function normalizeTrainingConfig(training) {
    if (!training || typeof training !== "object") return null;
    return cloneJson(training, null);
  }

  function normalizeBundle(bundle) {
    if (!bundle || typeof bundle !== "object") return null;
    return cloneJson(bundle, null);
  }

  function shouldPersistEmbeddedBundle(sync, craft) {
    if (sync?.readOnly === true || sync?.origin === "remote_share") return true;
    return false;
  }

  function normalizeSharing(sharing) {
    if (!sharing || typeof sharing !== "object") {
      return {
        enabled: false,
      };
    }

    return {
      enabled: sharing.enabled === true,
      publishedAt: asText(sharing.publishedAt),
    };
  }

  function normalizeIntegerOrZero(value) {
    const parsed = Number(value);
    if (!Number.isFinite(parsed)) return 0;
    return Math.max(0, Math.round(parsed));
  }

  function normalizeCraftNameSource(value) {
    const source = asText(value).toLowerCase();
    return ["user", "agent", "placeholder"].includes(source) ? source : "";
  }

  function buildLocalOriginKey(craftId) {
    const normalizedId = asText(craftId);
    return normalizedId ? `local:${normalizedId}` : "";
  }

  function normalizeSync(sync, craft) {
    if (!sync || typeof sync !== "object") {
      return {
        origin: "local",
        readOnly: false,
        originKey: buildLocalOriginKey(craft?.id),
        originName: asText(craft?.name),
        originOwnerName: "",
        originOwnerDeviceId: "",
        parentId: "",
        parentShareId: "",
        forkDepth: 0,
      };
    }

    const origin = asText(sync.origin) || "local";
    const shareId = asText(sync.shareId);
    const ownerDeviceId = asText(sync.ownerDeviceId);
    const ownerName = asText(sync.ownerName);
    const fallbackOriginKey =
      asText(sync.originKey) ||
      (shareId ? `share:${shareId}` : buildLocalOriginKey(sync.sourceCraftId || craft?.id));

    return {
      origin,
      readOnly: sync.readOnly === true || origin === "remote_share",
      shareId,
      ownerDeviceId,
      ownerName,
      sourceCraftId: asText(sync.sourceCraftId),
      syncedAt: asText(sync.syncedAt),
      publishedAt: asText(sync.publishedAt),
      forkedAt: asText(sync.forkedAt),
      mode: asText(sync.mode),
      originKey: fallbackOriginKey,
      originName: asText(sync.originName) || asText(craft?.name),
      originOwnerName: asText(sync.originOwnerName) || ownerName,
      originOwnerDeviceId: asText(sync.originOwnerDeviceId) || ownerDeviceId,
      parentId: asText(sync.parentId),
      parentShareId: asText(sync.parentShareId),
      forkDepth: normalizeIntegerOrZero(sync.forkDepth),
    };
  }

  function getCraftSyncApi() {
    return globalScope.SinepanelCraftSync || null;
  }

  async function withTimeout(promiseLike, timeoutMs, timeoutMessage) {
    let timeoutId = 0;
    const pending = Promise.resolve(promiseLike);
    pending.catch(() => {});
    try {
      return await Promise.race([
        pending,
        new Promise((_, reject) => {
          timeoutId = globalScope.setTimeout(() => {
            reject(new Error(timeoutMessage));
          }, Math.max(0, Number(timeoutMs) || 0));
        }),
      ]);
    } finally {
      if (timeoutId) {
        globalScope.clearTimeout(timeoutId);
      }
    }
  }

  function isTimeoutError(error, timeoutMessage) {
    return error instanceof Error && asText(error.message) === asText(timeoutMessage);
  }

  async function withRetryableTimeout(taskFactory, timeoutMs, retryTimeoutMs, timeoutMessage) {
    const pending = Promise.resolve().then(() => taskFactory());
    try {
      return await withTimeout(pending, timeoutMs, timeoutMessage);
    } catch (error) {
      if (!isTimeoutError(error, timeoutMessage) || !(retryTimeoutMs > timeoutMs)) {
        throw error;
      }
      return await withTimeout(pending, retryTimeoutMs, timeoutMessage);
    }
  }

  async function readCraftSyncLocalCraftsWithTimeout(craftSync, fallback = []) {
    if (!craftSync?.readLocalCrafts) return normalizeCrafts(fallback);
    try {
      const crafts = await withRetryableTimeout(
        () => craftSync.readLocalCrafts(),
        CRAFT_SYNC_LOCAL_TIMEOUT_MS,
        CRAFT_SYNC_LOCAL_TIMEOUT_RETRY_MS,
        "Craft sync local read timed out.",
      );
      return normalizeCrafts(crafts);
    } catch (error) {
      console.warn("[craft-storage] failed to read local crafts from craft sync", error);
      return normalizeCrafts(fallback);
    }
  }

  async function writeCraftSyncLocalCraftsWithTimeout(craftSync, crafts) {
    if (!craftSync?.writeLocalCrafts) return null;
    try {
      const written = await withRetryableTimeout(
        () => craftSync.writeLocalCrafts(normalizeCrafts(crafts)),
        CRAFT_SYNC_LOCAL_TIMEOUT_MS,
        CRAFT_SYNC_LOCAL_TIMEOUT_RETRY_MS,
        "Craft sync local write timed out.",
      );
      return normalizeCrafts(written);
    } catch (error) {
      console.warn("[craft-storage] failed to write local crafts into craft sync", error);
      throw error;
    }
  }

  async function readCraftSyncRemoteCraftsWithTimeout(craftSync, fallback = []) {
    if (!craftSync?.readRemoteCrafts) return normalizeCrafts(fallback);
    try {
      const crafts = await withRetryableTimeout(
        () => craftSync.readRemoteCrafts(),
        CRAFT_SYNC_REMOTE_TIMEOUT_MS,
        CRAFT_SYNC_REMOTE_TIMEOUT_RETRY_MS,
        "Craft sync remote read timed out.",
      );
      return normalizeCrafts(crafts);
    } catch (error) {
      console.warn("[craft-storage] failed to read remote crafts from craft sync", error);
      return normalizeCrafts(fallback);
    }
  }

  function normalizeCraft(craft, index = 0) {
    const starterMode = asText(craft?.starterMode) || "vanilla_target";
    const tools = normalizeTextArray(craft?.tools);
    const coverageGaps = normalizeTextArray(craft?.coverageGaps);
    const navigatorRecords = Array.isArray(craft?.navigatorRecords)
      ? craft.navigatorRecords.map((record, recordIndex) => normalizeNavigatorRecord(record, recordIndex))
      : [];
    const createdAt = asText(craft?.createdAt || craft?.updatedAt);
    const updatedAt = asText(craft?.updatedAt || craft?.createdAt);
    const sync = normalizeSync(craft?.sync, craft);

    return {
      id: asText(craft?.id) || `craft-${index + 1}`,
      name: asText(craft?.name) || `Craft ${index + 1}`,
      nameSource: normalizeCraftNameSource(craft?.nameSource),
      summary: asText(craft?.summary),
      accuracy: normalizeOptionalNumber(craft?.accuracy),
      tools,
      tooling: normalizeTooling(craft?.tooling, tools),
      useStatus: asText(craft?.useStatus) || (starterMode === "vanilla_target" ? "vanilla" : "ready"),
      inputMode: asText(craft?.inputMode) || "free_text",
      inputHint: asText(craft?.inputHint),
      inputExamples: normalizeTextArray(craft?.inputExamples).slice(0, 6),
      actionLabel: asText(craft?.actionLabel) || (starterMode === "vanilla_target" ? "Chat" : "Run"),
      tokenSpend: Math.max(0, normalizeInteger(craft?.tokenSpend)),
      costUsd: Math.max(0, normalizeNumber(craft?.costUsd)),
      inputPlaceholder: asText(craft?.inputPlaceholder),
      stage: asText(craft?.stage) || (starterMode === "vanilla_target" ? "Vanilla" : "Task Spec"),
      targetSlot: asText(craft?.targetSlot) || "target",
      accuracyFloor: asText(craft?.accuracyFloor),
      augmentationMode: asText(craft?.augmentationMode),
      seedRows: Math.max(0, normalizeInteger(craft?.seedRows)),
      datasetRows: Math.max(0, normalizeInteger(craft?.datasetRows)),
      openGaps:
        craft?.openGaps == null || craft?.openGaps === ""
          ? null
          : Math.max(0, normalizeInteger(craft?.openGaps)),
      agentPrompt: asText(craft?.agentPrompt),
      metricsReady: craft?.metricsReady === true,
      starterMode,
      starterModelName: asText(craft?.starterModelName),
      coverageGaps,
      navigatorRecords,
      training: normalizeTrainingConfig(craft?.training),
      bundle: shouldPersistEmbeddedBundle(sync, craft) ? normalizeBundle(craft?.bundle) : null,
      sharing: normalizeSharing(craft?.sharing),
      sync,
      createdAt,
      updatedAt,
    };
  }

  function normalizeCrafts(crafts) {
    if (!Array.isArray(crafts)) return [];
    const seen = new Set();
    const out = [];
    for (const [index, craft] of crafts.entries()) {
      const normalized = normalizeCraft(craft, index);
      if (seen.has(normalized.id)) continue;
      seen.add(normalized.id);
      out.push(normalized);
    }
    return out;
  }

  async function readLocalCrafts() {
    const craftSync = getCraftSyncApi();
    if (!craftSync?.readLocalCrafts) {
      return [];
    }
    return await readCraftSyncLocalCraftsWithTimeout(craftSync);
  }

  async function readCrafts() {
    const localCrafts = await readLocalCrafts();
    const craftSync = getCraftSyncApi();
    if (!craftSync?.readRemoteCrafts) {
      return localCrafts;
    }

    const remoteCrafts = await readCraftSyncRemoteCraftsWithTimeout(craftSync);
    return [...localCrafts, ...remoteCrafts];
  }

  function isRemoteSharedCraft(craft) {
    return normalizeSync(craft?.sync, craft).origin === "remote_share";
  }

  function isForkCraft(craft) {
    const sync = normalizeSync(craft?.sync, craft);
    return sync.origin === "fork" || sync.forkDepth > 0 || Boolean(sync.parentId || sync.parentShareId);
  }

  function getCraftOriginKey(craft) {
    const sync = normalizeSync(craft?.sync, craft);
    return asText(sync.originKey) || buildLocalOriginKey(craft?.id);
  }

  function getLineageForkCount(craft, crafts = []) {
    const originKey = getCraftOriginKey(craft);
    if (!originKey) return 0;
    return (Array.isArray(crafts) ? crafts : []).filter((entry) => {
      if (getCraftOriginKey(entry) !== originKey) return false;
      return isForkCraft(entry);
    }).length;
  }

  function isEditableCraft(craft) {
    return !isRemoteSharedCraft(craft);
  }

  function buildForkName(name, existingNames) {
    const base = `${asText(name) || "Shared Craft"} Fork`;
    if (!existingNames.has(base)) return base;

    let counter = 2;
    let candidate = `${base} ${counter}`;
    while (existingNames.has(candidate)) {
      counter += 1;
      candidate = `${base} ${counter}`;
    }
    return candidate;
  }

  function slugify(value) {
    return (
      String(value || "")
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "-")
        .replace(/^-+|-+$/g, "") || `craft-${Date.now()}`
    );
  }

  function createForkFromCraft(craft, existingCrafts = []) {
    const existingIds = new Set(existingCrafts.map((entry) => asText(entry?.id)).filter(Boolean));
    const existingNames = new Set(existingCrafts.map((entry) => asText(entry?.name)).filter(Boolean));
    const forkName = buildForkName(craft?.name, existingNames);

    let forkId = slugify(forkName);
    let counter = 2;
    while (existingIds.has(forkId)) {
      forkId = `${slugify(forkName)}-${counter}`;
      counter += 1;
    }

    const now = new Date().toISOString();
    const fork = normalizeCraft(
      {
        ...cloneJson(craft, {}),
        id: forkId,
        name: forkName,
        createdAt: now,
        updatedAt: now,
        sharing: {
          enabled: false,
        },
        sync: {
          origin: "fork",
          readOnly: false,
          shareId: "",
          ownerDeviceId: "",
          ownerName: "",
          sourceCraftId: asText(craft?.sync?.sourceCraftId || craft?.id),
          originKey: getCraftOriginKey(craft),
          originName: asText(craft?.sync?.originName || craft?.name),
          originOwnerName: asText(craft?.sync?.originOwnerName || craft?.sync?.ownerName),
          originOwnerDeviceId: asText(craft?.sync?.originOwnerDeviceId || craft?.sync?.ownerDeviceId),
          parentId: asText(craft?.id),
          parentShareId: asText(craft?.sync?.shareId),
          forkDepth: normalizeIntegerOrZero(craft?.sync?.forkDepth) + 1,
          forkedAt: now,
        },
      },
      existingCrafts.length,
    );

    return fork;
  }

  async function writeCrafts(crafts) {
    const normalized = normalizeCrafts(crafts);
    const localCrafts = normalized.filter((craft) => !isRemoteSharedCraft(craft));
    const craftSync = getCraftSyncApi();
    if (!craftSync?.writeLocalCrafts) {
      throw new Error("Craft sync local store is unavailable.");
    }
    await writeCraftSyncLocalCraftsWithTimeout(craftSync, localCrafts);
    return await readCrafts();
  }

  globalScope.SinepanelCraftStorage = {
    DEBUG_FIXED_TRAINING,
    ROLLOUT_STAGES,
    normalizeCraft,
    normalizeCrafts,
    readLocalCrafts,
    readCrafts,
    writeCrafts,
    isRemoteSharedCraft,
    isForkCraft,
    isEditableCraft,
    getCraftOriginKey,
    getLineageForkCount,
    createForkFromCraft,
  };
})(globalThis);
