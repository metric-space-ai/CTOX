(function attachSinepanelModelConfig(globalScope) {
  const PROVIDER_STORAGE_KEY = "sinepanel.providers.v1";
  const MODEL_SLOT_STORAGE_KEY = "sinepanel.model-slots.v1";
  const CONFIG_READY_KEY = "sinepanel.config.ready.v1";
  const PROVIDER_CACHE_KEY = "sinepanel.providers.cache.v1";
  const MODEL_SLOT_CACHE_KEY = "sinepanel.model-slots.cache.v1";
  const CONFIG_READY_CACHE_KEY = "sinepanel.config.ready.cache.v1";
  const STORAGE_TIMEOUT_MS = 1200;
  const MODEL_REF_SEPARATOR = ">";
  const LOCAL_QWEN_MODEL_VARIANTS = Object.freeze([
    Object.freeze({
      sizeLabel: "0.8B",
      vanillaModelName: "unsloth/Qwen3.5-0.8B (Vanilla)",
      canonicalModelName: "unsloth/Qwen3.5-0.8B",
      packagedLocalRuntime: true,
    }),
    Object.freeze({
      sizeLabel: "2B",
      vanillaModelName: "unsloth/Qwen3.5-2B (Vanilla)",
      canonicalModelName: "unsloth/Qwen3.5-2B",
      packagedLocalRuntime: false,
    }),
    Object.freeze({
      sizeLabel: "4B",
      vanillaModelName: "unsloth/Qwen3.5-4B (Vanilla)",
      canonicalModelName: "unsloth/Qwen3.5-4B",
      packagedLocalRuntime: false,
    }),
  ]);
  const DEFAULT_LOCAL_QWEN_MODEL = LOCAL_QWEN_MODEL_VARIANTS[0].vanillaModelName;
  const LOCAL_QWEN_MODEL_NAMES = Object.freeze(
    LOCAL_QWEN_MODEL_VARIANTS
      .filter((entry) => entry.packagedLocalRuntime === true)
      .map((entry) => entry.vanillaModelName),
  );
  const DEFAULT_MODEL_SLOTS = Object.freeze({
    agent: {
      providerId: "openai",
      modelName: "gpt-5.4",
      reasoningEffort: "medium",
    },
    batch: {
      providerId: "openai",
      modelName: "gpt-5-mini",
      reasoningEffort: "low",
    },
    vision: {
      providerId: "openai",
      modelName: "gpt-5-mini",
      reasoningEffort: "",
    },
    target: {
      providerId: "local_qwen",
      modelName: DEFAULT_LOCAL_QWEN_MODEL,
      reasoningEffort: "",
    },
  });
  const FIRST_RUN_REQUIRED_SLOT_IDS = Object.freeze(["agent", "batch", "vision"]);

  const MODEL_SLOT_DEFS = Object.freeze([
    {
      id: "agent",
      label: "Agent",
      purpose: "Main planning and crafting model",
      requiredOnFirstStart: true,
    },
    {
      id: "batch",
      label: "Batch",
      purpose: "Cheap model for large prompt/JSON batch runs",
      requiredOnFirstStart: true,
    },
    {
      id: "vision",
      label: "Vision",
      purpose: "Vision model for browser automation and page understanding",
      requiredOnFirstStart: true,
    },
    {
      id: "target",
      label: "Target",
      purpose: "Local runtime role for the tuned capability",
      requiredOnFirstStart: false,
    },
  ]);

  const AGENT_PROVIDER_TYPES = Object.freeze(["openai", "azure_openai"]);
  const AGENT_MODEL_ALLOWLIST = Object.freeze([
    "gpt-5.2",
    "gpt-5.2-pro",
    "gpt-5.4",
    "gpt-5.4-pro",
    "gpt-5-mini",
    "gpt-5",
  ]);

  const REASONING_EFFORT_VALUES = Object.freeze([
    "",
    "minimal",
    "low",
    "medium",
    "high",
  ]);

  const BUILTIN_PROVIDER_CATALOG = Object.freeze({
    local_qwen: {
      type: "local_qwen",
      label: "Local Qwen",
      enabled: true,
      baseUrl: "",
      apiKey: "",
      modelNames: LOCAL_QWEN_MODEL_NAMES,
    },
    openai: {
      type: "openai",
      label: "OpenAI",
      enabled: true,
      baseUrl: "https://api.openai.com/v1",
      apiKey: "",
      modelNames: [
        "gpt-5.2",
        "gpt-5.2-pro",
        "gpt-5.4",
        "gpt-5.4-pro",
        "gpt-5-mini",
        "gpt-5-nano",
        "gpt-5",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4o-mini",
        "computer-use-preview",
      ],
    },
    azure_openai: {
      type: "azure_openai",
      label: "Azure OpenAI",
      enabled: false,
      baseUrl: "",
      apiKey: "",
      modelNames: ["gpt-5.2", "gpt-5.2-pro", "gpt-5-mini", "gpt-5", "gpt-4.1"],
    },
    anthropic: {
      type: "anthropic",
      label: "Anthropic",
      enabled: false,
      baseUrl: "https://api.anthropic.com",
      apiKey: "",
      modelNames: ["claude-3-5-haiku-latest", "claude-sonnet-4-0"],
    },
    openrouter: {
      type: "openrouter",
      label: "OpenRouter",
      enabled: false,
      baseUrl: "https://openrouter.ai/api/v1",
      apiKey: "",
      modelNames: [
        "google/gemini-2.5-flash",
        "deepseek/deepseek-chat-v3.1",
        "openai/gpt-4.1-mini",
      ],
    },
    deepseek: {
      type: "deepseek",
      label: "DeepSeek",
      enabled: true,
      baseUrl: "https://api.deepseek.com/v1",
      apiKey: "",
      modelNames: ["deepseek-chat", "deepseek-reasoner"],
    },
    groq: {
      type: "groq",
      label: "Groq",
      enabled: false,
      baseUrl: "https://api.groq.com/openai/v1",
      apiKey: "",
      modelNames: ["llama-3.3-70b-versatile"],
    },
    cerebras: {
      type: "cerebras",
      label: "Cerebras",
      enabled: false,
      baseUrl: "https://api.cerebras.ai/v1",
      apiKey: "",
      modelNames: ["llama-3.3-70b"],
    },
    ollama: {
      type: "ollama",
      label: "Ollama",
      enabled: false,
      baseUrl: "http://localhost:11434/v1",
      apiKey: "",
      modelNames: [],
    },
  });

  const CUSTOM_PROVIDER_TEMPLATE = Object.freeze({
    type: "custom_openai",
    label: "Custom OpenAI",
    enabled: false,
    baseUrl: "",
    apiKey: "",
    modelNames: [],
  });

  function asText(value) {
    return String(value == null ? "" : value).trim();
  }

  function clone(value) {
    return JSON.parse(JSON.stringify(value));
  }

  function getStorageApi() {
    return globalScope.SinepanelCraftSync || null;
  }

  function getCacheStorage() {
    try {
      return globalScope.localStorage || null;
    } catch (_error) {
      return null;
    }
  }

  function readCachedJson(cacheKey, fallback = null) {
    try {
      const storage = getCacheStorage();
      if (!storage?.getItem) return clone(fallback);
      const raw = storage.getItem(cacheKey);
      if (!raw) return clone(fallback);
      return JSON.parse(raw);
    } catch (_error) {
      return clone(fallback);
    }
  }

  function writeCachedJson(cacheKey, value) {
    try {
      const storage = getCacheStorage();
      if (!storage?.setItem) return;
      storage.setItem(cacheKey, JSON.stringify(value));
    } catch (_error) {}
  }

  async function withStorageTimeout(promiseLike, timeoutMessage) {
    let timeoutId = 0;
    try {
      return await Promise.race([
        Promise.resolve(promiseLike),
        new Promise((_, reject) => {
          timeoutId = globalScope.setTimeout(() => {
            reject(new Error(timeoutMessage));
          }, STORAGE_TIMEOUT_MS);
        }),
      ]);
    } finally {
      if (timeoutId) {
        globalScope.clearTimeout(timeoutId);
      }
    }
  }

  async function readConfigValue(storageKey, cacheKey, fallback = null) {
    const cachedValue = readCachedJson(cacheKey, fallback);
    const storageApi = getStorageApi();
    if (!storageApi?.getValue) {
      return clone(cachedValue);
    }
    try {
      const storedValue = await withStorageTimeout(
        storageApi.getValue(storageKey, cachedValue),
        `Reading ${storageKey} from craft sync timed out.`,
      );
      writeCachedJson(cacheKey, storedValue);
      return clone(storedValue);
    } catch (error) {
      console.warn(`[model-config] falling back to cached ${storageKey}`, error);
      return clone(cachedValue);
    }
  }

  async function writeConfigValue(storageKey, cacheKey, value) {
    writeCachedJson(cacheKey, value);
    const storageApi = getStorageApi();
    if (!storageApi?.setValue) {
      return clone(value);
    }
    try {
      await withStorageTimeout(
        storageApi.setValue(storageKey, value),
        `Writing ${storageKey} into craft sync timed out.`,
      );
    } catch (error) {
      console.warn(`[model-config] failed to persist ${storageKey} in craft sync`, error);
    }
    return clone(value);
  }

  function normalizeModelNames(raw) {
    const values = Array.isArray(raw)
      ? raw
      : String(raw == null ? "" : raw)
          .split(/[\n,]/g)
          .map((item) => item.trim());
    const out = [];
    const seen = new Set();

    for (const value of values) {
      const item = asText(value);
      if (!item || seen.has(item)) continue;
      seen.add(item);
      out.push(item);
    }

    return out;
  }

  function normalizeReasoningEffort(value, fallback = "") {
    const normalized = asText(value).toLowerCase();
    if (REASONING_EFFORT_VALUES.includes(normalized)) return normalized;
    return asText(fallback).toLowerCase();
  }

  function modelNamesToTextarea(modelNames) {
    return normalizeModelNames(modelNames).join("\n");
  }

  function textareaToModelNames(raw) {
    return normalizeModelNames(raw);
  }

  function createProviderRecord(id, partial) {
    const builtin = BUILTIN_PROVIDER_CATALOG[id];
    const template = builtin ? clone(builtin) : clone(CUSTOM_PROVIDER_TEMPLATE);
    const type = asText(partial?.type || template.type || id);
    const name = asText(partial?.name || template.label || id);
    const baseUrl = asText(partial?.baseUrl || template.baseUrl);
    const apiKey = String(partial?.apiKey || template.apiKey || "");
    const modelNames = normalizeModelNames(partial?.modelNames || template.modelNames || []);

    return {
      id,
      type,
      name,
      enabled:
        typeof partial?.enabled === "boolean"
          ? partial.enabled
          : template.enabled !== false,
      baseUrl,
      apiKey,
      modelNames,
      createdAt: Number(partial?.createdAt) || Date.now(),
    };
  }

  function createDefaultProviders() {
    const out = {};
    for (const id of Object.keys(BUILTIN_PROVIDER_CATALOG)) {
      out[id] = createProviderRecord(id, BUILTIN_PROVIDER_CATALOG[id]);
    }
    return out;
  }

  function normalizeProviders(rawProviders) {
    const defaults = createDefaultProviders();
    const merged = {};

    for (const [id, defaultsForId] of Object.entries(defaults)) {
      merged[id] = createProviderRecord(id, {
        ...defaultsForId,
        ...(rawProviders?.[id] || {}),
      });
    }

    for (const [id, provider] of Object.entries(rawProviders || {})) {
      if (merged[id]) continue;
      merged[id] = createProviderRecord(id, provider);
    }

    return merged;
  }

  function supportsVisionModel(providerType, modelName) {
    const type = asText(providerType).toLowerCase();
    const model = asText(modelName).toLowerCase();
    if (!model) return false;

    if (type === "local_qwen") {
      return model.includes("qwen3.5");
    }
    if (model === "computer-use-preview") return true;

    if (["openai", "azure_openai", "openrouter", "custom_openai"].includes(type)) {
      return (
        model.includes("gpt-4o") ||
        model.includes("gpt-4.1") ||
        model.includes("gpt-5") ||
        model.includes("gemini") ||
        model.includes("vision") ||
        model.includes("-vl")
      );
    }

    if (type === "anthropic") {
      return model.includes("claude-3") || model.includes("claude-sonnet-4");
    }

    return model.includes("vision") || model.includes("-vl") || model.includes("llava");
  }

  function collectProviderChoices(providers, slotId) {
    const out = [];

    for (const provider of Object.values(providers || {})) {
      if (slotId === "target") {
        if (provider.type !== "local_qwen") continue;
      } else if (slotId === "agent") {
        if (!AGENT_PROVIDER_TYPES.includes(provider.type) && provider.type !== "local_qwen") continue;
      }

      if (!provider.enabled && slotId !== "target") continue;

      out.push({
        id: provider.id,
        label: provider.name,
      });
    }

    return out;
  }

  function collectModelOptions(providers, slotId) {
    const out = [];
    const seen = new Set();

    for (const provider of Object.values(providers || {})) {
      if (slotId === "target") {
        if (provider.type !== "local_qwen") continue;
      } else if (slotId === "agent") {
        if (!AGENT_PROVIDER_TYPES.includes(provider.type) && provider.type !== "local_qwen") continue;
      }

      if (!provider.enabled && slotId !== "target") continue;

      for (const modelName of normalizeModelNames(provider.modelNames)) {
        if (
          slotId === "agent" &&
          provider.type !== "local_qwen" &&
          !AGENT_MODEL_ALLOWLIST.includes(modelName)
        ) continue;
        if (slotId === "vision" && !supportsVisionModel(provider.type, modelName)) continue;

        const value = buildModelRef(provider.id, modelName);
        if (seen.has(value)) continue;
        seen.add(value);
        out.push({
          value,
          providerId: provider.id,
          providerName: provider.name,
          modelName,
          label: `${provider.name} > ${modelName}`,
        });
      }
    }

    return out;
  }

  function createDefaultModelSlots() {
    return clone(DEFAULT_MODEL_SLOTS);
  }

  function chooseFallbackModel(providers, slotId, providerId) {
    const options = collectModelOptions(providers, slotId).filter(
      (option) => option.providerId === providerId,
    );
    const preferredModelName = asText(DEFAULT_MODEL_SLOTS?.[slotId]?.modelName);
    if (preferredModelName && options.some((option) => option.modelName === preferredModelName)) {
      return preferredModelName;
    }
    if (options[0]) return options[0].modelName;
    if (slotId === "target") return DEFAULT_LOCAL_QWEN_MODEL;
    return "";
  }

  function normalizeModelSlots(rawSlots, providers) {
    const defaults = createDefaultModelSlots();
    const normalized = {};

    for (const def of MODEL_SLOT_DEFS) {
      const candidate = rawSlots?.[def.id] || {};
      const fallback = defaults[def.id];
      const providerChoices = collectProviderChoices(providers, def.id);
      const allowedIds = new Set(providerChoices.map((item) => item.id));

      const providerId = allowedIds.has(candidate.providerId)
        ? candidate.providerId
        : allowedIds.has(fallback.providerId)
          ? fallback.providerId
          : providerChoices[0]?.id || fallback.providerId || "";
      const modelName =
        asText(candidate.modelName) ||
        chooseFallbackModel(providers, def.id, providerId) ||
        fallback.modelName ||
        "";
      const reasoningEffort = normalizeReasoningEffort(
        candidate.reasoningEffort,
        fallback.reasoningEffort,
      );

      normalized[def.id] = {
        providerId,
        modelName,
        reasoningEffort,
      };
    }

    return normalized;
  }

  let configReadyPromise = null;

  async function ensureConfigReady() {
    if (configReadyPromise) return configReadyPromise;

    configReadyPromise = (async () => {
      const marker = await readConfigValue(CONFIG_READY_KEY, CONFIG_READY_CACHE_KEY, null);
      if (marker?.done === true) return;

      const [idbProviders, idbSlots] = await Promise.all([
        readConfigValue(PROVIDER_STORAGE_KEY, PROVIDER_CACHE_KEY, null),
        readConfigValue(MODEL_SLOT_STORAGE_KEY, MODEL_SLOT_CACHE_KEY, null),
      ]);
      const providers = normalizeProviders(idbProviders);
      const slots = normalizeModelSlots(idbSlots, providers);

      await writeConfigValue(PROVIDER_STORAGE_KEY, PROVIDER_CACHE_KEY, providers);
      await writeConfigValue(MODEL_SLOT_STORAGE_KEY, MODEL_SLOT_CACHE_KEY, slots);
      await writeConfigValue(CONFIG_READY_KEY, CONFIG_READY_CACHE_KEY, {
        done: true,
        source: idbProviders !== null || idbSlots !== null ? "existing" : "fresh",
        completedAt: new Date().toISOString(),
      });
    })().catch((error) => {
      configReadyPromise = null;
      throw error;
    });

    return configReadyPromise;
  }

  async function readProviders() {
    await ensureConfigReady();
    return normalizeProviders(await readConfigValue(PROVIDER_STORAGE_KEY, PROVIDER_CACHE_KEY, null));
  }

  async function writeProviders(providers) {
    const normalized = normalizeProviders(providers);
    await ensureConfigReady();
    await writeConfigValue(PROVIDER_STORAGE_KEY, PROVIDER_CACHE_KEY, normalized);
    return normalized;
  }

  async function readModelSlots(providers) {
    const activeProviders = providers || (await readProviders());
    await ensureConfigReady();
    return normalizeModelSlots(
      await readConfigValue(MODEL_SLOT_STORAGE_KEY, MODEL_SLOT_CACHE_KEY, null),
      activeProviders,
    );
  }

  async function writeModelSlots(slots, providers) {
    const activeProviders = providers || (await readProviders());
    const normalized = normalizeModelSlots(slots, activeProviders);
    await ensureConfigReady();
    await writeConfigValue(MODEL_SLOT_STORAGE_KEY, MODEL_SLOT_CACHE_KEY, normalized);
    return normalized;
  }

  function patchProviderRecord(providers, providerId, patch) {
    const current = providers[providerId] || createProviderRecord(providerId, {});
    return {
      ...providers,
      [providerId]: createProviderRecord(providerId, {
        ...current,
        ...patch,
      }),
    };
  }

  function patchModelSlot(slots, slotId, patch, providers) {
    const next = {
      ...slots,
      [slotId]: {
        ...(slots?.[slotId] || {}),
        ...patch,
      },
    };
    return normalizeModelSlots(next, providers);
  }

  function createCustomProvider(providers) {
    let index = 1;
    let id = `custom_openai_${index}`;
    while (providers[id]) {
      index += 1;
      id = `custom_openai_${index}`;
    }
    return createProviderRecord(id, {
      type: "custom_openai",
      name: `Custom ${index}`,
      enabled: true,
    });
  }

  function orderProviderIds(providers) {
    const preferred = [
      "local_qwen",
      "openai",
      "azure_openai",
      "anthropic",
      "openrouter",
      "deepseek",
      "groq",
      "cerebras",
      "ollama",
    ];
    const ids = Object.keys(providers || {});
    const rest = ids.filter((id) => !preferred.includes(id)).sort();
    return [...preferred.filter((id) => ids.includes(id)), ...rest];
  }

  function getSlotDisplayLabel(slotId, slots, providers) {
    const slot = slots?.[slotId];
    const provider = providers?.[slot?.providerId];
    const providerName = provider?.name || slot?.providerId || "";
    const modelName = slot?.modelName || "";

    return {
      title: providerName && modelName ? `${providerName} > ${modelName}` : modelName || providerName || "Unset",
      value: modelName || providerName || "Unset",
    };
  }

  function buildModelRef(providerId, modelName) {
    const provider = asText(providerId);
    const model = asText(modelName);
    return provider && model ? `${provider}${MODEL_REF_SEPARATOR}${model}` : "";
  }

  function parseModelRef(modelRef) {
    const raw = asText(modelRef);
    if (!raw) return { providerId: "", modelName: "" };
    const splitIndex = raw.indexOf(MODEL_REF_SEPARATOR);
    if (splitIndex < 0) return { providerId: raw, modelName: "" };
    return {
      providerId: asText(raw.slice(0, splitIndex)),
      modelName: asText(raw.slice(splitIndex + MODEL_REF_SEPARATOR.length)),
    };
  }

  function getModelRefForSlot(slotId, slots) {
    const slot = slots?.[slotId];
    return buildModelRef(slot?.providerId, slot?.modelName);
  }

  function getReasoningEffortForSlot(slotId, slots) {
    return normalizeReasoningEffort(slots?.[slotId]?.reasoningEffort);
  }

  function getSlotDefinition(slotId) {
    return MODEL_SLOT_DEFS.find((def) => def.id === slotId) || null;
  }

  function validateSlotSelection(slotId, provider, modelName) {
    const normalizedSlotId = asText(slotId);
    const normalizedModelName = asText(modelName);
    const providerType = asText(provider?.type).toLowerCase();

    if (!provider || !providerType) {
      return {
        ok: false,
        code: "provider_missing",
        reason: "Choose a provider first.",
      };
    }

    if (!normalizedModelName) {
      return {
        ok: false,
        code: "model_missing",
        reason: "Add a model name for this role.",
      };
    }

    if (normalizedSlotId === "target" && providerType !== "local_qwen") {
      return {
        ok: false,
        code: "target_must_be_local_qwen",
        reason: "Target must stay on the local Qwen runtime.",
      };
    }

    if (
      normalizedSlotId === "agent" &&
      !AGENT_PROVIDER_TYPES.includes(providerType) &&
      providerType !== "local_qwen"
    ) {
      return {
        ok: false,
        code: "agent_provider_invalid",
        reason: "Agent must use OpenAI, Azure OpenAI, or the local Qwen WebGPU runtime.",
      };
    }

    if (normalizedSlotId === "vision" && !supportsVisionModel(providerType, normalizedModelName)) {
      return {
        ok: false,
        code: "vision_model_invalid",
        reason: "Choose a vision-capable model for browser automation.",
      };
    }

    return {
      ok: true,
      code: "ok",
      reason: "",
    };
  }

  function getSlotSetupStatus(slotId, slots, providers) {
    const def = getSlotDefinition(slotId) || {
      id: slotId,
      label: slotId,
      purpose: "",
      requiredOnFirstStart: false,
    };
    const slot = slots?.[slotId] || {};
    const providerId = asText(slot.providerId);
    const modelName = asText(slot.modelName);
    const provider = providerId ? providers?.[providerId] || null : null;

    const base = {
      slotId: def.id,
      label: def.label,
      purpose: def.purpose || "",
      requiredOnFirstStart: !!def.requiredOnFirstStart,
      providerId,
      modelName,
      providerName: asText(provider?.name),
      modelRef: buildModelRef(providerId, modelName),
      ok: false,
      code: "missing_provider",
      reason: "Choose a provider first.",
    };

    if (!providerId) {
      return base;
    }

    if (!provider) {
      return {
        ...base,
        code: "provider_unknown",
        reason: "The selected provider no longer exists.",
      };
    }

    if (provider.enabled === false && provider.type !== "local_qwen") {
      return {
        ...base,
        code: "provider_disabled",
        reason: "Enable this provider before using this role.",
      };
    }

    if (!asText(provider.apiKey) && provider.type !== "local_qwen" && provider.type !== "ollama") {
      return {
        ...base,
        code: "provider_api_key_missing",
        reason: "Add an API key for the selected provider.",
      };
    }

    const selectionCheck = validateSlotSelection(def.id, provider, modelName);
    if (!selectionCheck.ok) {
      return {
        ...base,
        code: selectionCheck.code,
        reason: selectionCheck.reason,
      };
    }

    return {
      ...base,
      ok: true,
      code: "ok",
      reason: "Ready.",
    };
  }

  function getRequiredSetupStatus(slots, providers) {
    const items = MODEL_SLOT_DEFS
      .filter((def) => def.requiredOnFirstStart)
      .map((def) => getSlotSetupStatus(def.id, slots, providers));
    const missingItems = items.filter((item) => !item.ok);

    return {
      ready: missingItems.length === 0,
      items,
      missingItems,
      missingCount: missingItems.length,
      requiredSlotIds: FIRST_RUN_REQUIRED_SLOT_IDS.slice(),
    };
  }

  globalScope.SinepanelModelConfig = {
    PROVIDER_STORAGE_KEY,
    MODEL_SLOT_STORAGE_KEY,
    MODEL_REF_SEPARATOR,
    PROVIDER_CACHE_KEY,
    MODEL_SLOT_CACHE_KEY,
    CONFIG_READY_CACHE_KEY,
    FIRST_RUN_REQUIRED_SLOT_IDS,
    REASONING_EFFORT_VALUES,
    MODEL_SLOT_DEFS,
    BUILTIN_PROVIDER_CATALOG,
    createDefaultProviders,
    createDefaultModelSlots,
    normalizeProviders,
    normalizeModelSlots,
    normalizeModelNames,
    modelNamesToTextarea,
    textareaToModelNames,
    normalizeReasoningEffort,
    collectProviderChoices,
    collectModelOptions,
    supportsVisionModel,
    readProviders,
    writeProviders,
    readModelSlots,
    writeModelSlots,
    buildModelRef,
    parseModelRef,
    getModelRefForSlot,
    getReasoningEffortForSlot,
    getSlotDefinition,
    validateSlotSelection,
    getSlotSetupStatus,
    getRequiredSetupStatus,
    patchProviderRecord,
    patchModelSlot,
    createCustomProvider,
    orderProviderIds,
    getSlotDisplayLabel,
  };
})(globalThis);
