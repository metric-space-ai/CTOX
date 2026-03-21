// bg/ml-inference.js - final version with Qwen embeddings + Florence2

import "../shared/craft-sync.js";
import { createPeerComputeBus } from "../shared/peer_compute_bus.mjs";
import {
  pipeline,
  env,
  Florence2ForConditionalGeneration,
  AutoProcessor,
  AutoTokenizer,
  Qwen3_5ForConditionalGeneration,
  RawImage,
  Tensor,
} from "../vendor/transformers-esm.mjs";
import {
  buildLocalQwenGenerationConfig,
  collectLocalQwenImageSources,
  createLocalQwenMissingBrowserTrainingManifestError,
  describeLocalQwenBrowserTrainingPackaging,
  getLocalQwenModelRepoUrl,
  getLocalQwenReasoningMode,
  getLocalQwenRuntimePlan,
  getSupportedLocalQwenModels,
  getLocalQwenVisionPreprocessPolicy,
  hasLocalQwenVisionInputs,
  prepareLocalQwenMessages,
  rewriteLocalQwenRemoteUrl,
  stripLocalQwenThinkingTrace,
} from "../shared/local_qwen_runtime.mjs";
import {
  estimateFixedTrainingWorkSamples,
  LOCAL_QWEN_FIXED_TRAINING_CONFIG,
  trainFixedLocalQwenAdapter,
} from "../shared/local_qwen_training.mjs";
import {
  getTransformerLoraTrainingSupportIssue,
} from "../shared/local_qwen_training_guardrails.mjs";
import {
  LOCAL_QWEN_PIPELINE_SMOKE_CONFIG,
  createLocalQwenPipelineSmokeDatasetPayload,
  estimateLocalQwenPipelineSmokeWork,
  runLocalQwenPipelineSmoke,
} from "../shared/local_qwen_pipeline_smoke.mjs";
import {
  BUNDLE_SCHEMA_VERSION,
  WEIGHTS_ARTIFACT_KIND,
  float32ToDataUrl,
  getWeightsArtifactId,
} from "../shared/capability-bundle.mjs";
import {
  REMOTE_COMPUTE_MAX_CONCURRENT_JOBS,
  buildComputeJobId,
  buildComputeRequestId,
  buildComputeWorkerId,
  normalizeRemoteComputeSettings,
} from "../shared/remote_compute.mjs";

env.allowLocalModels = true;
env.localModelPath = chrome.runtime.getURL("models/");
env.allowRemoteModels = true;
env.useBrowserCache = false;

if (env.backends?.onnx?.wasm) {
  env.backends.onnx.wasm.wasmPaths = chrome.runtime.getURL("vendor/wasm/");
  env.backends.onnx.wasm.proxy = false;
}

// Lower the ONNX log level so warnings stay quiet and only errors remain.
if (env.backends?.onnx) {
  // ONNX.env.logLevel -> 'verbose' | 'info' | 'warning' | 'error' | 'fatal'
  env.backends.onnx.logLevel = "error";
}

const craftSync = globalThis.SinepanelCraftSync || null;

const localQwenRuntimePromiseCache = new Map();
const localQwenPreferredExecutionPlan = new Map();
const localQwenTrainingRuns = new Map();
const localQwenTrainingRunObservers = new Map();
const localQwenInferenceAdapterCache = new Map();
const localQwenLoraManifestCache = new Map();
const localQwenTrainingManifestCache = new Map();
const peerComputePeersByDeviceId = new Map();
const peerComputePeerMeta = new WeakMap();
const peerComputePendingRequests = new Map();
const peerComputeLocalActiveJobs = new Map();
const peerComputeState = {
  bus: null,
  busKey: "",
  initPromise: null,
  settingsUnsubscribe: null,
  workerHeartbeatTimer: 0,
};
const LOCAL_QWEN_FETCH_TRACE_LIMIT = 24;
const LOCAL_QWEN_PIPELINE_IDLE_TTL_MS = 45_000;
const EMBED_PIPELINE_IDLE_TTL_MS = 45_000;
const FLORENCE_RUNTIME_IDLE_TTL_MS = 45_000;
const localQwenRecentFetches = [];
const LOCAL_QWEN_REFERENCE_LORA_TARGET_MODULES = Object.freeze([
  "q_proj",
  "k_proj",
  "v_proj",
  "o_proj",
  "gate_proj",
  "up_proj",
  "down_proj",
  "in_proj_qkv",
  "in_proj_z",
  "in_proj_b",
  "in_proj_a",
  "out_proj",
]);

function pickFetchTraceHeaders(headers) {
  if (!headers || typeof headers.get !== "function") {
    return null;
  }
  const selectedHeaders = [
    "content-length",
    "content-type",
    "location",
    "x-request-id",
    "x-hub-cache",
    "x-cache",
    "x-linked-size",
    "x-xet-hash",
  ];
  const out = {};
  for (const header of selectedHeaders) {
    const value = headers.get(header);
    if (value != null && value !== "") {
      out[header] = value;
    }
  }
  return Object.keys(out).length > 0 ? out : null;
}

function getFetchTraceUrl(input) {
  if (typeof input === "string") return input;
  if (input instanceof URL) return input.toString();
  if (input && typeof input === "object") {
    if ("url" in input) {
      return getFetchTraceUrl(input.url);
    }
    if ("href" in input) {
      return getFetchTraceUrl(input.href);
    }
  }
  return "";
}

function rewriteLocalQwenFetchInput(input) {
  const requestedUrl = getFetchTraceUrl(input);
  const rewrittenUrl = rewriteLocalQwenRemoteUrl(requestedUrl);
  if (!rewrittenUrl || rewrittenUrl === requestedUrl) {
    return {
      input,
      requestedUrl,
      rewrittenUrl: "",
    };
  }
  if (typeof Request !== "undefined" && input instanceof Request) {
    return {
      input: new Request(rewrittenUrl, input),
      requestedUrl,
      rewrittenUrl,
    };
  }
  if (input instanceof URL) {
    return {
      input: new URL(rewrittenUrl),
      requestedUrl,
      rewrittenUrl,
    };
  }
  return {
    input: rewrittenUrl,
    requestedUrl,
    rewrittenUrl,
  };
}

function isBlockedStrictLocalQwenRemoteRequest(url) {
  const source = String(url || "").trim();
  return /^https:\/\/huggingface\.co\/local-extension\/Qwen3\.5-[^/]+-ONNX\/resolve\//.test(source);
}

function recordLocalQwenFetchTrace(entry) {
  localQwenRecentFetches.push(entry);
  if (localQwenRecentFetches.length > LOCAL_QWEN_FETCH_TRACE_LIMIT) {
    localQwenRecentFetches.splice(0, localQwenRecentFetches.length - LOCAL_QWEN_FETCH_TRACE_LIMIT);
  }
}

function getRecentLocalQwenFetchTrace(limit = 10) {
  return localQwenRecentFetches.slice(-Math.max(0, Number(limit) || 0));
}

const defaultEnvFetch =
  typeof env.fetch === "function"
    ? env.fetch.bind(env)
    : typeof globalThis.fetch === "function"
      ? globalThis.fetch.bind(globalThis)
      : null;

if (defaultEnvFetch) {
  env.fetch = async (input, init = undefined) => {
    const startedAt = Date.now();
    const rewrittenRequest = rewriteLocalQwenFetchInput(input);
    const requestedUrl = rewrittenRequest.requestedUrl;
    const requestMethod = String(
      init?.method ||
        (input && typeof input === "object" && "method" in input ? input.method : "") ||
        "GET",
    )
      .trim()
      .toUpperCase() || "GET";
    const trace = {
      requestedUrl,
      method: requestMethod,
      startedAt: new Date(startedAt).toISOString(),
    };
    if (rewrittenRequest.rewrittenUrl) {
      trace.rewrittenUrl = rewrittenRequest.rewrittenUrl;
    }

    try {
      if (
        isBlockedStrictLocalQwenRemoteRequest(rewrittenRequest.requestedUrl)
        && !rewrittenRequest.rewrittenUrl
      ) {
        const error = new Error(
          `Strict local Qwen runtime blocked remote fallback for ${rewrittenRequest.requestedUrl}.`,
        );
        trace.durationMs = Date.now() - startedAt;
        trace.error = serializeError(error);
        trace.blockedStrictLocal = true;
        recordLocalQwenFetchTrace(trace);
        throw error;
      }
      const response = await defaultEnvFetch(rewrittenRequest.input, init);
      trace.durationMs = Date.now() - startedAt;
      trace.status = Number(response?.status || 0) || 0;
      trace.ok = response?.ok === true;
      trace.redirected = response?.redirected === true;
      trace.responseUrl = String(response?.url || requestedUrl);
      trace.headers = pickFetchTraceHeaders(response?.headers);
      recordLocalQwenFetchTrace(trace);
      return response;
    } catch (error) {
      trace.durationMs = Date.now() - startedAt;
      trace.error = serializeError(error);
      recordLocalQwenFetchTrace(trace);
      throw error;
    }
  };
}

function getLocalQwenDtypeKey(dtype) {
  if (!dtype) return "auto";
  if (typeof dtype === "string") return dtype;
  try {
    return JSON.stringify(dtype);
  } catch (_error) {
    return String(dtype);
  }
}

function getLocalQwenExecutionPlanRuntimeModelId(runtimeModelId, executionPlan) {
  const planRuntimeModelId = String(executionPlan?.runtimeModelId || "").trim();
  return planRuntimeModelId || String(runtimeModelId || "").trim();
}

function sameLocalQwenExecutionPlan(left, right) {
  return (
    String(left?.runtimeModelId || "").trim() === String(right?.runtimeModelId || "").trim() &&
    left?.device === right?.device &&
    getLocalQwenDtypeKey(left?.dtype) === getLocalQwenDtypeKey(right?.dtype)
  );
}

function patchLocalQwenForwardParams(model, extraKeys = []) {
  if (!model || !Array.isArray(model.forward_params)) return model;
  const modelType = String(model?.config?.model_type || "").trim();
  if (!["qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"].includes(modelType)) {
    return model;
  }

  const nextForwardParams = [...model.forward_params];
  for (const key of ["position_ids", "rope_deltas", "image_grid_thw", "video_grid_thw", ...extraKeys]) {
    if (!nextForwardParams.includes(key)) {
      nextForwardParams.push(key);
    }
  }
  model.forward_params = nextForwardParams;
  return model;
}

function safeSerialize(value, depth = 0) {
  if (value == null) return value;
  if (depth > 2) return String(value);
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return value;
  }
  if (Array.isArray(value)) {
    return value.slice(0, 24).map((item) => safeSerialize(item, depth + 1));
  }
  if (typeof value === "object") {
    const out = {};
    for (const [key, item] of Object.entries(value)) {
      if (key === "stack") continue;
      try {
        out[key] = safeSerialize(item, depth + 1);
      } catch (_error) {
        out[key] = String(item);
      }
    }
    return out;
  }
  return String(value);
}

function serializeError(error) {
  if (!error || typeof error !== "object") {
    return {
      name: "",
      message: String(error || ""),
      stack: "",
      props: {},
    };
  }

  const props = {};
  for (const key of Object.keys(error)) {
    if (key === "name" || key === "message" || key === "stack" || key === "cause") continue;
    props[key] = safeSerialize(error[key]);
  }

  return {
    name: String(error.name || ""),
    message: String(error.message || error),
    stack: typeof error.stack === "string" ? error.stack : "",
    cause:
      error.cause && error.cause !== error
        ? serializeError(error.cause)
        : null,
    props,
  };
}

const LOCAL_QWEN_VISION_BUDGET_ERROR_CODES = new Set([9468408]);

function parseLocalQwenNumericErrorCode(value) {
  const text = asText(value);
  if (!/^\d{4,}$/.test(text)) return 0;
  const numeric = Number(text);
  return Number.isSafeInteger(numeric) ? numeric : 0;
}

function classifyLocalQwenAttemptFailure({
  errorInfo = null,
  stage = "",
  usesVisionInputs = false,
} = {}) {
  const serializedError =
    errorInfo && typeof errorInfo === "object"
      ? {
          ...errorInfo,
          props:
            errorInfo.props && typeof errorInfo.props === "object"
              ? { ...errorInfo.props }
              : {},
        }
      : serializeError(errorInfo);
  const rawMessage = asText(serializedError?.message);
  const numericCode = parseLocalQwenNumericErrorCode(rawMessage);
  const haystack = [
    rawMessage,
    asText(serializedError?.stack),
    typeof serializedError?.props === "object" ? JSON.stringify(serializedError.props) : "",
  ].join(" ").toLowerCase();
  const isNumericRuntimeFailure = numericCode > 0;
  const isVisionBudgetFailure =
    usesVisionInputs &&
    (
      /memory access out of bounds|out of bounds/.test(haystack) ||
      LOCAL_QWEN_VISION_BUDGET_ERROR_CODES.has(numericCode)
    );
  const reason = isVisionBudgetFailure
    ? "local_qwen_vision_pixel_budget_exceeded"
    : isNumericRuntimeFailure
      ? "local_qwen_webgpu_numeric_runtime_error"
      : "";
  const message = isVisionBudgetFailure
    ? `Local Qwen WebGPU vision generation exceeded the safe browser screenshot budget (${numericCode || "out_of_bounds"}).`
    : isNumericRuntimeFailure
      ? `Local Qwen WebGPU/ONNX generation failed with numeric runtime code ${numericCode}.`
      : rawMessage;

  return {
    reason,
    numericCode,
    isVisionBudgetFailure,
    error: {
      ...serializedError,
      message,
      props: {
        ...(serializedError.props && typeof serializedError.props === "object"
          ? serializedError.props
          : {}),
        ...(rawMessage && rawMessage !== message ? { rawMessage } : {}),
        ...(numericCode > 0 ? { numericCode } : {}),
        ...(reason ? { reason } : {}),
        ...(stage ? { stage } : {}),
      },
    },
  };
}

async function collectLocalQwenDiagnostics() {
  const diagnostics = {
    userAgent: globalThis.navigator?.userAgent || "",
    platform: globalThis.navigator?.platform || "",
    language: globalThis.navigator?.language || "",
    online: globalThis.navigator?.onLine ?? null,
    hardwareConcurrency: globalThis.navigator?.hardwareConcurrency ?? null,
    deviceMemory: globalThis.navigator?.deviceMemory ?? null,
    crossOriginIsolated: globalThis.crossOriginIsolated === true,
    hasNavigatorGpu: Boolean(globalThis.navigator?.gpu),
    cacheApiAvailable: typeof globalThis.caches !== "undefined",
    onnx: {
      logLevel: env.backends?.onnx?.logLevel || "",
      wasm: {
        proxy: env.backends?.onnx?.wasm?.proxy ?? null,
        wasmPaths:
          typeof env.backends?.onnx?.wasm?.wasmPaths === "string"
            ? env.backends.onnx.wasm.wasmPaths
            : safeSerialize(env.backends?.onnx?.wasm?.wasmPaths),
      },
    },
    recentFetches: getRecentLocalQwenFetchTrace(),
  };

  if (!globalThis.navigator?.gpu?.requestAdapter) {
    diagnostics.webgpu = {
      adapterAvailable: false,
      reason: "navigator.gpu.requestAdapter unavailable",
    };
    return diagnostics;
  }

  try {
    const adapter = await globalThis.navigator.gpu.requestAdapter();
    diagnostics.webgpu = {
      adapterAvailable: Boolean(adapter),
      features: adapter ? Array.from(adapter.features || []).sort() : [],
    };

    if (adapter?.limits) {
      diagnostics.webgpu.limits = {
        maxBufferSize: adapter.limits.maxBufferSize ?? null,
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize ?? null,
        maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize ?? null,
        maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup ?? null,
      };
    }

    if (adapter?.info) {
      try {
        diagnostics.webgpu.adapterInfo = safeSerialize(adapter.info);
      } catch (error) {
        diagnostics.webgpu.adapterInfoError = serializeError(error);
      }
    }
  } catch (error) {
    diagnostics.webgpu = {
      adapterAvailable: false,
      adapterRequestError: serializeError(error),
    };
  }

  return diagnostics;
}

function getLocalQwenAttemptKey(runtimeModelId, executionPlan) {
  const resolvedRuntimeModelId = getLocalQwenExecutionPlanRuntimeModelId(
    runtimeModelId,
    executionPlan,
  );
  return `${resolvedRuntimeModelId}::${executionPlan.device}::${getLocalQwenDtypeKey(executionPlan.dtype)}`;
}

function clearLocalQwenPipelineDisposeTimer(entry) {
  const timerId = Number(entry?.disposeTimer || 0);
  if (!Number.isFinite(timerId) || timerId <= 0) return;
  clearTimeout(timerId);
  entry.disposeTimer = 0;
}

async function disposeIfSupported(resource) {
  if (!resource || typeof resource.dispose !== "function") return;
  try {
    await resource.dispose();
  } catch (error) {
    console.warn("[offscreen] local_qwen resource dispose failed", error);
  }
}

async function disposeLocalQwenPipelineBundle(bundle) {
  if (!bundle || typeof bundle !== "object") return;
  const processor =
    bundle.processor ||
    await Promise.resolve(bundle.processorPromise).catch(() => null);
  await disposeIfSupported(processor);
  await disposeIfSupported(bundle.model);
  await disposeIfSupported(bundle.tokenizer);
}

async function dropLocalQwenPipelineCacheEntry(cacheKey) {
  const entry = localQwenRuntimePromiseCache.get(cacheKey);
  if (!entry) return;
  localQwenRuntimePromiseCache.delete(cacheKey);
  clearLocalQwenPipelineDisposeTimer(entry);

  try {
    const bundle = await entry.promise;
    await disposeLocalQwenPipelineBundle(bundle);
  } catch {
    // Failed pipeline loads already clean themselves up via promise rejection.
  }
}

function scheduleLocalQwenPipelineCacheEntryDispose(cacheKey) {
  const entry = localQwenRuntimePromiseCache.get(cacheKey);
  if (!entry || Number(entry.activeUses || 0) > 0) return;
  clearLocalQwenPipelineDisposeTimer(entry);
  entry.disposeTimer = setTimeout(() => {
    const current = localQwenRuntimePromiseCache.get(cacheKey);
    if (current !== entry || Number(current?.activeUses || 0) > 0) {
      return;
    }
    console.log("[offscreen] releasing idle local_qwen pipeline", cacheKey);
    void dropLocalQwenPipelineCacheEntry(cacheKey);
  }, LOCAL_QWEN_PIPELINE_IDLE_TTL_MS);
}

function clampProgress(value) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return 0;
  return Math.max(0, Math.min(1, parsed));
}

function asText(value) {
  return String(value == null ? "" : value).trim();
}

function createLocalQwenTrainingJobId() {
  const randomPart = Math.random().toString(16).slice(2, 10);
  return `local-qwen-train-${Date.now()}-${randomPart}`;
}

function addLocalQwenTrainingRunObserver(jobId, listener) {
  const normalizedJobId = asText(jobId);
  if (!normalizedJobId || typeof listener !== "function") {
    return () => {};
  }
  let observers = localQwenTrainingRunObservers.get(normalizedJobId);
  if (!observers) {
    observers = new Set();
    localQwenTrainingRunObservers.set(normalizedJobId, observers);
  }
  observers.add(listener);
  return () => {
    const current = localQwenTrainingRunObservers.get(normalizedJobId);
    current?.delete(listener);
    if (current && current.size === 0) {
      localQwenTrainingRunObservers.delete(normalizedJobId);
    }
  };
}

function notifyLocalQwenTrainingRunObservers(run) {
  const normalizedJobId = asText(run?.jobId);
  if (!normalizedJobId) return;
  const observers = localQwenTrainingRunObservers.get(normalizedJobId);
  if (!observers?.size) return;
  const snapshot = getPublicTrainingSnapshot(run);
  for (const listener of [...observers]) {
    try {
      listener(snapshot);
    } catch (error) {
      console.warn("[offscreen] training observer failed", error);
    }
  }
}

function roundMetric(value, digits = 6) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return 0;
  const factor = 10 ** digits;
  return Math.round(parsed * factor) / factor;
}

function getCapabilityArtifactWriter() {
  if (typeof globalThis.SinepanelCraftSync?.putLocalArtifact === "function") {
    return globalThis.SinepanelCraftSync.putLocalArtifact.bind(globalThis.SinepanelCraftSync);
  }
  return null;
}

async function readCapabilityArtifact(id, fallback = null) {
  const artifactId = String(id || "").trim();
  if (!artifactId) return fallback;
  if (typeof globalThis.SinepanelCraftSync?.readLocalArtifact === "function") {
    const record = await globalThis.SinepanelCraftSync.readLocalArtifact(artifactId, null);
    if (record && typeof record === "object") return record;
  }
  return fallback;
}

function createInt64Tensor(values, shape) {
  return new Tensor(
    "int64",
    BigInt64Array.from(values, (value) => BigInt(Number(value))),
    shape,
  );
}

function getArtifactTimestampKey(record = null) {
  const iso = String(record?.updatedAt || record?.createdAt || "").trim();
  if (iso) return iso;
  const metaTimestamp = record?.meta?.updatedAt ?? record?.meta?.createdAt ?? 0;
  const numericTimestamp = Number(metaTimestamp);
  return Number.isFinite(numericTimestamp) && numericTimestamp > 0
    ? String(numericTimestamp)
    : "";
}

function decodeFloat32DataUrl(dataUrl) {
  const match = /^data:([^;,]+)?(?:;base64)?,(.*)$/i.exec(String(dataUrl || "").trim());
  if (!match) {
    throw new Error("Invalid Float32 data URL.");
  }
  const base64Payload = String(match[2] || "").trim();
  if (!base64Payload) return new Float32Array(0);
  const binary = globalThis.atob(base64Payload);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  if (bytes.byteLength % Float32Array.BYTES_PER_ELEMENT !== 0) {
    throw new Error("Float32 adapter payload has an invalid byte length.");
  }
  return new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
}

async function getLocalQwenLoraManifest(runtimeModelId = "") {
  const normalizedModelId = String(runtimeModelId || "").trim();
  if (!normalizedModelId) return null;
  if (localQwenLoraManifestCache.has(normalizedModelId)) {
    return localQwenLoraManifestCache.get(normalizedModelId);
  }
  const manifestPromise = (async () => {
    const manifestUrl = getLocalQwenModelRepoUrl(normalizedModelId, "lora_manifest.json");
    if (!manifestUrl) return null;
    try {
      const response = await fetch(manifestUrl, { cache: "no-store" });
      if (!response.ok) return null;
      const payload = await response.json();
      return payload && typeof payload === "object" ? payload : null;
    } catch (_error) {
      return null;
    }
  })().catch((error) => {
    localQwenLoraManifestCache.delete(normalizedModelId);
    throw error;
  });
  localQwenLoraManifestCache.set(normalizedModelId, manifestPromise);
  return await manifestPromise;
}

async function getLocalQwenTrainingManifest(runtimeModelId = "") {
  const normalizedModelId = String(runtimeModelId || "").trim();
  if (!normalizedModelId) return null;
  if (localQwenTrainingManifestCache.has(normalizedModelId)) {
    return localQwenTrainingManifestCache.get(normalizedModelId);
  }
  const manifestPromise = (async () => {
    const manifestUrl = getLocalQwenModelRepoUrl(normalizedModelId, "lora_training_manifest.json");
    if (!manifestUrl) return null;
    try {
      const response = await fetch(manifestUrl, { cache: "no-store" });
      if (!response.ok) return null;
      const payload = await response.json();
      return payload && typeof payload === "object" ? payload : null;
    } catch (_error) {
      return null;
    }
  })().catch((error) => {
    localQwenTrainingManifestCache.delete(normalizedModelId);
    throw error;
  });
  localQwenTrainingManifestCache.set(normalizedModelId, manifestPromise);
  return await manifestPromise;
}

async function assertLocalQwenBrowserTrainingSupported(runtimeModelId = "") {
  const normalizedModelId = String(runtimeModelId || "").trim();
  if (!normalizedModelId) {
    throw new Error("Browser transformer LoRA training requires a valid runtime model id.");
  }
  const packaging = describeLocalQwenBrowserTrainingPackaging(normalizedModelId);
  if (!packaging.packagedBrowserTraining) {
    throw createLocalQwenMissingBrowserTrainingManifestError(normalizedModelId);
  }
  const manifest = await getLocalQwenTrainingManifest(normalizedModelId);
  if (!manifest) {
    throw createLocalQwenMissingBrowserTrainingManifestError(normalizedModelId);
  }
  const trainingSupportIssue = getTransformerLoraTrainingSupportIssue(manifest);
  if (trainingSupportIssue) {
    throw new Error(trainingSupportIssue.message);
  }
  return manifest;
}

async function resolveSupportedLocalQwenTrainingModelName(modelName = "") {
  const requestedModelName = String(modelName || "").trim();
  if (requestedModelName) {
    const requestedRuntimePlan = getLocalQwenRuntimePlan(requestedModelName);
    await assertLocalQwenBrowserTrainingSupported(requestedRuntimePlan.runtimeModelId);
    return requestedRuntimePlan.requestedModelName || requestedModelName;
  }

  for (const entry of getSupportedLocalQwenModels()) {
    if (entry?.packagedBrowserTraining !== true) continue;
    const runtimePlan = getLocalQwenRuntimePlan(entry.vanillaModelName);
    await assertLocalQwenBrowserTrainingSupported(runtimePlan.runtimeModelId);
    return runtimePlan.requestedModelName || entry.vanillaModelName;
  }

  throw createLocalQwenMissingBrowserTrainingManifestError("local-extension/Qwen3.5-0.8B-ONNX");
}

function getLocalQwenLoraManifestModuleMap(manifest = null) {
  const modules = Array.isArray(manifest?.modules) ? manifest.modules : [];
  const out = new Map();
  for (const entry of modules) {
    const modulePath = String(entry?.modulePath || "").trim();
    if (!modulePath) continue;
    out.set(modulePath, entry);
  }
  return out;
}

function normalizeTokenId(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? Math.trunc(numeric) : -1;
}

function normalizeTokenIdList(value) {
  if (value == null) return [];
  if (Array.isArray(value)) {
    return value.flat(Infinity).map((entry) => normalizeTokenId(entry)).filter((entry) => entry >= 0);
  }
  const normalized = normalizeTokenId(value);
  return normalized >= 0 ? [normalized] : [];
}

function copyLocalQwenTokenRow(tensorLike) {
  if (!tensorLike) return [];
  if (typeof tensorLike.tolist === "function") {
    const rows = tensorLike.tolist();
    if (Array.isArray(rows) && rows.length) {
      const firstRow = Array.isArray(rows[0]) ? rows[0] : rows;
      return firstRow.map((value) => normalizeTokenId(value)).filter((value) => value >= 0);
    }
  }
  if (ArrayBuffer.isView(tensorLike)) {
    return Array.from(tensorLike, (value) => normalizeTokenId(value)).filter((value) => value >= 0);
  }
  if (tensorLike?.data && ArrayBuffer.isView(tensorLike.data)) {
    return Array.from(tensorLike.data, (value) => normalizeTokenId(value)).filter((value) => value >= 0);
  }
  return [];
}

async function getLocalQwenInferenceAdapter(craftId = "", runtimeModelId = "") {
  const normalizedCraftId = String(craftId || "").trim();
  if (!normalizedCraftId) return null;
  const artifactId = getWeightsArtifactId(normalizedCraftId);
  const record = await readCapabilityArtifact(artifactId, null);
  if (!record || typeof record !== "object") return null;

  const payload = record?.payload && typeof record.payload === "object" ? record.payload : {};
  const status = String(payload?.status || "base_model_only").trim() || "base_model_only";
  const adapterPayload = payload?.adapter && typeof payload.adapter === "object" ? payload.adapter : null;
  const manifest = await getLocalQwenLoraManifest(runtimeModelId);
  const cacheKey = `${artifactId}::${getArtifactTimestampKey(record)}::${String(runtimeModelId || "").trim()}`;
  if (localQwenInferenceAdapterCache.has(cacheKey)) {
    return localQwenInferenceAdapterCache.get(cacheKey);
  }

  const baseDescriptor = {
    applied: false,
    craftId: normalizedCraftId,
    artifactId,
    updatedAt: String(record?.updatedAt || ""),
    status,
    modelName: String(payload?.modelName || "").trim(),
    targetSlot: String(payload?.targetSlot || "").trim(),
    runtimeModelId: String(runtimeModelId || "").trim(),
  };

  if (!adapterPayload || typeof adapterPayload !== "object") {
    if (status === "trained_adapter") {
      throw new Error(`Craft ${normalizedCraftId} is marked as trained_adapter but has no serialized adapter weights.`);
    }
    localQwenInferenceAdapterCache.set(cacheKey, baseDescriptor);
    return baseDescriptor;
  }

  const adapterType = String(adapterPayload.adapterType || "").trim().toLowerCase();
  const modulePayloads = Array.isArray(adapterPayload.modules) ? adapterPayload.modules : [];
  if (adapterType === "transformer_lora" || modulePayloads.length > 0) {
    if (!manifest) {
      throw new Error(`Craft ${normalizedCraftId} uses transformer_lora, but the local runtime model ${runtimeModelId} has no LoRA manifest.`);
    }
    const manifestModuleMap = getLocalQwenLoraManifestModuleMap(manifest);
    const scale = Number(adapterPayload.scale || 1);
    const modules = modulePayloads.map((modulePayload) => {
      const modulePath = String(modulePayload?.modulePath || "").trim();
      const manifestModule = manifestModuleMap.get(modulePath);
      if (!manifestModule) {
        throw new Error(`Craft ${normalizedCraftId} references unknown LoRA module ${modulePath}.`);
      }
      const inFeatures = Number(manifestModule.inFeatures || 0);
      const outFeatures = Number(manifestModule.outFeatures || 0);
      const rank = Number(manifestModule.rank || adapterPayload.rank || manifest?.graph?.rank || 0);
      const loraA = decodeFloat32DataUrl(modulePayload?.loraADataUrl);
      const loraB = decodeFloat32DataUrl(modulePayload?.loraBDataUrl);
      if (loraA.length !== inFeatures * rank) {
        throw new Error(`Craft ${normalizedCraftId} LoRA A for ${modulePath} has invalid length.`);
      }
      if (loraB.length !== rank * outFeatures) {
        throw new Error(`Craft ${normalizedCraftId} LoRA B for ${modulePath} has invalid length.`);
      }
      return {
        ...manifestModule,
        modulePath,
        scale: Number(modulePayload?.scale || scale || 1),
        loraA,
        loraB,
      };
    });
    const descriptor = {
      ...baseDescriptor,
      applied: modules.length > 0,
      kind: "transformer_lora",
      adapterType: "transformer_lora",
      scale,
      rank: Number(adapterPayload.rank || manifest?.graph?.rank || 0),
      manifest,
      modules,
    };
    localQwenInferenceAdapterCache.set(cacheKey, descriptor);
    return descriptor;
  }
  throw new Error(
    `Craft ${normalizedCraftId} uses unsupported legacy adapter type "${adapterType || "unknown"}". Only transformer_lora is supported.`,
  );
}

function buildSerializableTrainingConfig(config = {}) {
  return {
    profile: String(config.profile || "").trim(),
    comparisonPurpose: String(config.comparisonPurpose || "").trim(),
    compareBy: String(config.compareBy || "").trim(),
    resourceProfile: String(config.resourceProfile || "").trim(),
    maxTrainPairs: Number(config.maxTrainPairs || 0),
    maxValidationPairs: Number(config.maxValidationPairs || 0),
    maxTestPairs: Number(config.maxTestPairs || 0),
    rank: Number(config.rank || 0),
    alpha: Number(config.alpha || 0),
    epochs: Number(config.epochs || 0),
    learningRate: Number(config.learningRate || 0),
    optimizer: String(config.optimizer || "").trim(),
    spsaEpsilon: Number(config.spsaEpsilon || 0),
    spsaSamples: Number(config.spsaSamples || 0),
    spsaGradientClip: Number(config.spsaGradientClip || 0),
    maxSeqLen: Number(config.maxSeqLen || 0),
    batchTokens: Number(config.batchTokens || 0),
    evaluationMode: String(config.evaluationMode || "").trim(),
    reasoningMode: String(config.reasoningMode || "").trim(),
    seed: Number(config.seed || 0),
    trainMatrixA: config.trainMatrixA !== false,
    trainMatrixB: config.trainMatrixB !== false,
  };
}

function mergeTrainingConfig(baseConfig = {}, overrides = {}) {
  const next = {
    ...baseConfig,
  };
  for (const [key, value] of Object.entries(overrides || {})) {
    if (value == null) continue;
    next[key] = value;
  }
  return next;
}

function buildSerializableTrainingHistory(history) {
  if (!Array.isArray(history)) return [];
  return history.map((entry) => ({
    epoch: Number(entry?.epoch || 0),
    trainLoss: roundMetric(entry?.train_loss),
    trainAcc: roundMetric(entry?.train_acc),
    validationLoss: roundMetric(entry?.validation_loss),
    validationAcc: roundMetric(entry?.validation_acc),
    testLoss: roundMetric(entry?.test_loss),
    testAcc: roundMetric(entry?.test_acc),
  }));
}

function countTransformerLoraAdapterBytes(modules = []) {
  let total = 0;
  for (const module of Array.isArray(modules) ? modules : []) {
    const loraA = module?.loraA instanceof Float32Array ? module.loraA : null;
    const loraB = module?.loraB instanceof Float32Array ? module.loraB : null;
    if (loraA) total += loraA.byteLength;
    if (loraB) total += loraB.byteLength;
  }
  return total;
}

function serializeLocalQwenAdapterPayload(adapter = null) {
  if (!adapter || typeof adapter !== "object") {
    throw new Error("Local Qwen training finished without a serializable adapter.");
  }

  const moduleEntries = Array.isArray(adapter.modules) ? adapter.modules : [];
  if (adapter.kind === "transformer_lora" || moduleEntries.length > 0) {
    const modules = moduleEntries
      .map((module) => {
        const modulePath = String(module?.modulePath || "").trim();
        const loraA = module?.loraA instanceof Float32Array ? module.loraA : null;
        const loraB = module?.loraB instanceof Float32Array ? module.loraB : null;
        if (!modulePath || !loraA || !loraB) return null;
        return {
          modulePath,
          rank: Number(module?.rank || adapter.rank || 0),
          scale: roundMetric(module?.scale ?? adapter.scale ?? 1) ?? 1,
          loraADataUrl: float32ToDataUrl(loraA),
          loraBDataUrl: float32ToDataUrl(loraB),
        };
      })
      .filter(Boolean);
    if (!modules.length) {
      throw new Error("Transformer LoRA adapter is missing serializable module weights.");
    }
    const targetModules = Array.from(
      new Set(
        modules
          .map((entry) => String(entry.modulePath || "").trim().split(".").pop() || "")
          .filter(Boolean),
      ),
    );
    return {
      payload: {
        format: "float32_data_url_v1",
        adapterType: "transformer_lora",
        rank: Number(adapter.rank || 0),
        scale: roundMetric(adapter.scale ?? 1) ?? 1,
        targetModules,
        candidateTargetModules: LOCAL_QWEN_REFERENCE_LORA_TARGET_MODULES,
        modules,
      },
      sizeBytes: countTransformerLoraAdapterBytes(moduleEntries),
    };
  }

  throw new Error("Local Qwen training produced an unsupported non-transformer adapter payload.");
}

function estimateRemainingTrainingMs(run) {
  if (!run || ["completed", "failed"].includes(String(run.status || "").toLowerCase())) {
    return 0;
  }
  const samplesPerSecond = Number(run.samplesPerSecond || 0);
  if (!(samplesPerSecond > 0)) return 0;
  const totalSamples = Number(run.totalSamples || 0);
  const completedSamples = Number(run.completedSamples || 0);
  const remainingSamples = Math.max(0, totalSamples - completedSamples);
  if (remainingSamples <= 0) return 0;
  return Math.max(0, Math.round((remainingSamples / samplesPerSecond) * 1000));
}

async function persistLocalQwenCapabilityWeights({
  craftId = "",
  shardId = "",
  modelName = "",
  config = {},
  jobId = "",
  result = null,
} = {}) {
  const normalizedCraftId = String(craftId || "").trim();
  if (!normalizedCraftId) {
    throw new Error("Local Qwen training requires a craft id so the full capability bundle can be stored.");
  }

  const writer = getCapabilityArtifactWriter();
  if (!writer) {
    throw new Error("Local capability artifact storage is unavailable.");
  }

  const serializedAdapter = serializeLocalQwenAdapterPayload(
    result?.adapter && typeof result.adapter === "object" ? result.adapter : null,
  );

  const updatedAt = Date.now();
  const adapterSizeMb =
    Number(result?.adapterSizeMb || 0) ||
    (Number(serializedAdapter.sizeBytes || 0) / 1_000_000);
  const normalizedModelName =
    String(modelName || result?.runtime?.runtimeModelId || "").trim() ||
    "unsloth/Qwen3.5-0.8B";

  const payload = {
    schemaVersion: BUNDLE_SCHEMA_VERSION,
    status: "trained_adapter",
    targetSlot: "target",
    modelName: normalizedModelName,
    run: {
      jobId: String(jobId || "").trim(),
      shardId: String(shardId || "").trim(),
      selectedEpoch: Number(result?.selectedEpoch || 0),
      completedAt: new Date(updatedAt).toISOString(),
    },
    config: buildSerializableTrainingConfig(config),
    dataset:
      result?.dataset && typeof result.dataset === "object"
        ? {
            path: String(result.dataset.path || "").trim(),
            trainPairs: Number(result.dataset.trainPairs || 0),
            validationPairs: Number(result.dataset.validationPairs || 0),
            testPairs: Number(result.dataset.testPairs || 0),
            tokenStats:
              result.dataset.tokenStats && typeof result.dataset.tokenStats === "object"
                ? safeSerialize(result.dataset.tokenStats)
                : null,
            meta:
              result.dataset.meta && typeof result.dataset.meta === "object"
                ? { ...result.dataset.meta }
                : {},
          }
        : null,
    runtime:
      result?.runtime && typeof result.runtime === "object"
        ? {
            ...result.runtime,
          }
        : null,
    metrics: {
      baseTrainAcc: roundMetric(result?.baseTrainAcc),
      baseValidationAcc: roundMetric(result?.baseValidationAcc),
      baseTestAcc: roundMetric(result?.baseTestAcc),
      trainEval:
        result?.trainEval && typeof result.trainEval === "object"
          ? {
              loss: roundMetric(result.trainEval.loss),
              accuracy: roundMetric(result.trainEval.accuracy),
              rowCount: Number(result.trainEval.rowCount || 0),
            }
          : null,
      validationEval:
        result?.validationEval && typeof result.validationEval === "object"
          ? {
              loss: roundMetric(result.validationEval.loss),
              accuracy: roundMetric(result.validationEval.accuracy),
              rowCount: Number(result.validationEval.rowCount || 0),
            }
          : null,
      testEval:
        result?.testEval && typeof result.testEval === "object"
          ? {
              loss: roundMetric(result.testEval.loss),
              accuracy: roundMetric(result.testEval.accuracy),
              rowCount: Number(result.testEval.rowCount || 0),
            }
          : null,
      throughput:
        result?.throughput && typeof result.throughput === "object"
          ? {
              compareBy: String(result.throughput.compareBy || "").trim(),
              comparisonPurpose: String(result.throughput.comparisonPurpose || "").trim(),
              resourceProfile: String(result.throughput.resourceProfile || "").trim(),
              totalForwardTokens: Number(result.throughput.totalForwardTokens || 0),
              totalSupervisedTokens: Number(result.throughput.totalSupervisedTokens || 0),
              trainStepForwardTokens: Number(result.throughput.trainStepForwardTokens || 0),
              evalForwardTokens: Number(result.throughput.evalForwardTokens || 0),
              overallElapsedMs: Number(result.throughput.overallElapsedMs || 0),
              totalMeasuredForwardMs: Number(result.throughput.totalMeasuredForwardMs || 0),
              trainStepForwardMs: Number(result.throughput.trainStepForwardMs || 0),
              evalForwardMs: Number(result.throughput.evalForwardMs || 0),
              nonForwardOverheadMs: Number(result.throughput.nonForwardOverheadMs || 0),
              overallForwardTokensPerSecond: roundMetric(result.throughput.overallForwardTokensPerSecond),
              overallSupervisedTokensPerSecond: roundMetric(result.throughput.overallSupervisedTokensPerSecond),
              measuredForwardTokensPerSecond: roundMetric(result.throughput.measuredForwardTokensPerSecond),
              trainStepForwardTokensPerSecond: roundMetric(result.throughput.trainStepForwardTokensPerSecond),
              evalForwardTokensPerSecond: roundMetric(result.throughput.evalForwardTokensPerSecond),
              forwardWorkloadShare: roundMetric(result.throughput.forwardWorkloadShare),
              nonForwardOverheadShare: roundMetric(result.throughput.nonForwardOverheadShare),
            }
          : null,
      history: buildSerializableTrainingHistory(result?.history),
    },
    adapter: serializedAdapter.payload,
  };

  return await writer({
    id: getWeightsArtifactId(normalizedCraftId),
    craftId: normalizedCraftId,
    kind: WEIGHTS_ARTIFACT_KIND,
    payload,
    meta: {
      hasAdapter: true,
      adapterSizeMb: roundMetric(adapterSizeMb),
      selectedEpoch: Number(result?.selectedEpoch || 0),
      updatedAt,
      status: "trained_adapter",
    },
  });
}

function getPublicTrainingSnapshot(run) {
  if (!run) return null;
  return {
    jobId: String(run.jobId || ""),
    craftId: String(run.craftId || ""),
    shardId: String(run.shardId || ""),
    modelName: String(run.modelName || ""),
    status: String(run.status || "idle"),
    phase: String(run.phase || ""),
    phaseLabel: String(run.phaseLabel || ""),
    message: String(run.message || ""),
    error: String(run.error || ""),
    progress: clampProgress(run.progress),
    totalSamples: Number(run.totalSamples || 0),
    completedSamples: Number(run.completedSamples || 0),
    phaseTotalSamples: Number(run.phaseTotalSamples || 0),
    phaseCompletedSamples: Number(run.phaseCompletedSamples || 0),
    phaseUnitLabel: String(run.phaseUnitLabel || ""),
    samplesPerSecond: Number(run.samplesPerSecond || 0),
    estimatedRemainingMs: estimateRemainingTrainingMs(run),
    currentEpoch: Number(run.currentEpoch || 0),
    epochsTotal: Number(run.epochsTotal || 0),
    adapterSizeMb: Number(run.adapterSizeMb || 0),
    selectedEpoch: Number(run.selectedEpoch || 0),
    metrics: run.metrics && typeof run.metrics === "object" ? { ...run.metrics } : {},
    dataset: run.dataset && typeof run.dataset === "object" ? safeSerialize(run.dataset) : null,
    runtime: run.runtime && typeof run.runtime === "object" ? safeSerialize(run.runtime) : null,
    smokeMode: String(run.smokeMode || ""),
    smoke:
      run.smoke && typeof run.smoke === "object"
        ? safeSerialize(run.smoke)
        : null,
    history: Array.isArray(run.history)
      ? run.history.map((entry) => safeSerialize(entry))
      : [],
    createdAt: String(run.createdAt || ""),
    startedAt: String(run.startedAt || ""),
    endedAt: String(run.endedAt || ""),
  };
}

function updateLocalQwenTrainingRun(jobId, patch = {}) {
  const current = localQwenTrainingRuns.get(jobId);
  if (!current) return null;

  const next = {
    ...current,
    ...patch,
    metrics:
      patch.metrics && typeof patch.metrics === "object"
        ? {
            ...(current.metrics && typeof current.metrics === "object" ? current.metrics : {}),
            ...patch.metrics,
          }
        : current.metrics,
  };

  const totalSamples = Number(next.totalSamples || 0);
  const completedSamples = Number(next.completedSamples || 0);
  if (!("progress" in patch) && totalSamples > 0) {
    next.progress = clampProgress(completedSamples / totalSamples);
  } else {
    next.progress = clampProgress(next.progress);
  }

  localQwenTrainingRuns.set(jobId, next);
  notifyLocalQwenTrainingRunObservers(next);
  return next;
}

function getPeerComputeWorkerName(settings, deviceId) {
  const configuredName = asText(settings?.displayName);
  return configuredName || `Peer ${asText(deviceId).slice(0, 6)}`;
}

function getPeerComputeBusKey(settings = null) {
  const normalized = normalizeRemoteComputeSettings(settings);
  return [
    asText(settings?.mode),
    asText(settings?.signalingUrls),
    asText(settings?.token),
    normalized.computeOfferEnabled ? "offer" : "no-offer",
    normalized.remoteExecutionEnabled ? "use-remote" : "local-only",
  ].join("::");
}

function rememberPeerComputePeer(peer, message = {}) {
  const deviceId = asText(message?.deviceId);
  if (!deviceId) return;
  const meta = {
    deviceId,
    name: asText(message?.name),
    modelName: asText(message?.computeModelName || message?.modelName),
    computeOfferEnabled: message?.computeOfferEnabled === true,
    remoteExecutionEnabled: message?.remoteExecutionEnabled === true,
  };
  peerComputePeersByDeviceId.set(deviceId, peer);
  peerComputePeerMeta.set(peer, meta);
}

function forgetPeerComputePeer(peer) {
  const meta = peerComputePeerMeta.get(peer);
  if (meta?.deviceId && peerComputePeersByDeviceId.get(meta.deviceId) === peer) {
    peerComputePeersByDeviceId.delete(meta.deviceId);
  }
  peerComputePeerMeta.delete(peer);
}

function rejectAllPeerComputeRequests(reason = "peer-compute-stopped") {
  for (const [requestId, pending] of [...peerComputePendingRequests.entries()]) {
    globalThis.clearTimeout(pending.timeoutId);
    peerComputePendingRequests.delete(requestId);
    pending.reject(new Error(reason));
  }
}

async function upsertPeerComputeJob(record = {}) {
  if (typeof craftSync?.upsertComputeJob !== "function") return null;
  try {
    return await craftSync.upsertComputeJob(record);
  } catch (error) {
    console.warn("[offscreen] compute job upsert failed", error);
    return null;
  }
}

async function publishLocalPeerComputeWorkerState({ forceRemove = false } = {}) {
  if (!craftSync?.getDeviceId) return null;
  const settings = await craftSync.readSettings();
  const normalizedCompute = normalizeRemoteComputeSettings(settings);
  const deviceId = await craftSync.getDeviceId();
  const workerId = buildComputeWorkerId(deviceId);

  if (
    forceRemove ||
    settings?.mode !== "continuous" ||
    normalizedCompute.computeOfferEnabled !== true ||
    !peerComputeState.bus
  ) {
    if (typeof craftSync.removeComputeWorker === "function" && workerId) {
      await craftSync.removeComputeWorker(workerId);
    }
    return null;
  }

  const activeJobs = [...peerComputeLocalActiveJobs.values()];
  return await craftSync.upsertComputeWorker({
    id: workerId,
    deviceId,
    name: getPeerComputeWorkerName(settings, deviceId),
    modelName: normalizedCompute.computeModelName,
    available: activeJobs.length < REMOTE_COMPUTE_MAX_CONCURRENT_JOBS,
    busy: activeJobs.length > 0,
    activeJobId: asText(activeJobs[0]?.jobId),
    maxConcurrentJobs: REMOTE_COMPUTE_MAX_CONCURRENT_JOBS,
    lastSeen: new Date().toISOString(),
  });
}

async function sendPeerComputeHello(peer) {
  if (!peerComputeState.bus || !craftSync?.getDeviceId) return;
  const settings = await craftSync.readSettings();
  const normalizedCompute = normalizeRemoteComputeSettings(settings);
  const deviceId = await craftSync.getDeviceId();
  await peerComputeState.bus.send(peer, {
    type: "hello",
    deviceId,
    name: getPeerComputeWorkerName(settings, deviceId),
    computeOfferEnabled: normalizedCompute.computeOfferEnabled,
    computeModelName: normalizedCompute.computeModelName,
    remoteExecutionEnabled: normalizedCompute.remoteExecutionEnabled,
    activeJobCount: peerComputeLocalActiveJobs.size,
  });
}

async function stopPeerComputeRuntime() {
  globalThis.clearInterval(peerComputeState.workerHeartbeatTimer);
  peerComputeState.workerHeartbeatTimer = 0;
  rejectAllPeerComputeRequests();
  peerComputePeersByDeviceId.clear();
  if (peerComputeState.bus) {
    const bus = peerComputeState.bus;
    peerComputeState.bus = null;
    peerComputeState.busKey = "";
    try {
      await bus.close();
    } catch (error) {
      console.warn("[offscreen] peer compute bus close failed", error);
    }
  }
  await publishLocalPeerComputeWorkerState({ forceRemove: true });
}

async function runPeerComputeRequest(peer, type, jobId, payload, timeoutMs = 45_000) {
  if (!peerComputeState.bus) {
    throw new Error("peer-compute-not-running");
  }
  const requestId = buildComputeRequestId();
  const settings = await craftSync?.readSettings?.();
  const deviceId = await craftSync?.getDeviceId?.();
  const responsePromise = new Promise((resolve, reject) => {
    const timeoutId = globalThis.setTimeout(() => {
      peerComputePendingRequests.delete(requestId);
      reject(new Error(`peer-compute-timeout:${type}`));
    }, timeoutMs);
    peerComputePendingRequests.set(requestId, { resolve, reject, timeoutId });
  });
  await peerComputeState.bus.send(peer, {
    type,
    requestId,
    deviceId: asText(deviceId),
    name: getPeerComputeWorkerName(settings, deviceId),
    jobId: asText(jobId),
    payload: payload && typeof payload === "object" ? payload : {},
  });
  return await responsePromise;
}

async function selectPeerComputeWorker(modelName = "") {
  if (!peerComputeState.bus || !craftSync?.listComputeWorkers) return null;
  const workers = await craftSync.listComputeWorkers();
  const targetModelName = asText(modelName);
  for (const worker of Array.isArray(workers) ? workers : []) {
    if (worker?.available !== true || worker?.busy === true) continue;
    if (targetModelName && asText(worker?.modelName) !== targetModelName) continue;
    const peer = peerComputePeersByDeviceId.get(asText(worker?.deviceId));
    if (!peer || peer.destroyed) continue;
    return {
      ...worker,
      peer,
    };
  }
  return null;
}

function normalizeRemoteTrainingRunSnapshot(run, worker = null) {
  const snapshot = run && typeof run === "object" ? { ...run } : {};
  snapshot.runtime =
    snapshot.runtime && typeof snapshot.runtime === "object"
      ? { ...snapshot.runtime }
      : {};
  snapshot.runtime.execution = "remote_peer";
  snapshot.runtime.workerDeviceId = asText(worker?.deviceId || snapshot.runtime.workerDeviceId);
  snapshot.runtime.workerName = asText(worker?.name || snapshot.runtime.workerName);
  snapshot.runtime.workerModelName = asText(worker?.modelName || snapshot.runtime.workerModelName);
  return snapshot;
}

async function handlePeerComputeInferenceRequest(peer, message) {
  const requestId = asText(message?.requestId);
  const jobId = asText(message?.jobId) || buildComputeJobId();
  const payload = message?.payload && typeof message.payload === "object" ? message.payload : {};
  const settings = await craftSync?.readSettings?.();
  const normalizedCompute = normalizeRemoteComputeSettings(settings);
  const deviceId = await craftSync?.getDeviceId?.();
  const workerName = getPeerComputeWorkerName(settings, deviceId);

  if (!peerComputeState.bus || normalizedCompute.computeOfferEnabled !== true) {
    await peerComputeState.bus?.send(peer, {
      type: "inference_response",
      requestId,
      jobId,
      ok: false,
      error: "Remote inference is not offered on this peer.",
    });
    return;
  }
  if (asText(payload?.modelName) !== normalizedCompute.computeModelName) {
    await peerComputeState.bus.send(peer, {
      type: "inference_response",
      requestId,
      jobId,
      ok: false,
      error: `Requested model ${asText(payload?.modelName)} does not match offered model ${normalizedCompute.computeModelName}.`,
    });
    return;
  }
  if (peerComputeLocalActiveJobs.size >= REMOTE_COMPUTE_MAX_CONCURRENT_JOBS) {
    await peerComputeState.bus.send(peer, {
      type: "inference_response",
      requestId,
      jobId,
      ok: false,
      error: "This peer is currently busy.",
    });
    return;
  }

  peerComputeLocalActiveJobs.set(jobId, { kind: "inference", jobId });
  await publishLocalPeerComputeWorkerState();
  await upsertPeerComputeJob({
    id: jobId,
    kind: "inference",
    status: "running",
    modelName: normalizedCompute.computeModelName,
    requesterDeviceId: asText(message?.deviceId),
    workerDeviceId: asText(deviceId),
    workerName,
    progress: 0,
    startedAt: new Date().toISOString(),
  });

  try {
    const result = await handleLocalQwenChatLocal(payload);
    await peerComputeState.bus.send(peer, {
      type: "inference_response",
      requestId,
      jobId,
      ok: true,
      text: String(result?.text || ""),
      runtime: result?.runtime || null,
    });
    await upsertPeerComputeJob({
      id: jobId,
      kind: "inference",
      status: "completed",
      modelName: normalizedCompute.computeModelName,
      requesterDeviceId: asText(message?.deviceId),
      workerDeviceId: asText(deviceId),
      workerName,
      progress: 1,
      completedAt: new Date().toISOString(),
    });
  } catch (error) {
    await peerComputeState.bus.send(peer, {
      type: "inference_response",
      requestId,
      jobId,
      ok: false,
      error: String(error?.message || error || "Remote inference failed."),
    });
    await upsertPeerComputeJob({
      id: jobId,
      kind: "inference",
      status: "failed",
      modelName: normalizedCompute.computeModelName,
      requesterDeviceId: asText(message?.deviceId),
      workerDeviceId: asText(deviceId),
      workerName,
      error: String(error?.message || error || "Remote inference failed."),
      completedAt: new Date().toISOString(),
    });
  } finally {
    peerComputeLocalActiveJobs.delete(jobId);
    await publishLocalPeerComputeWorkerState();
  }
}

async function handlePeerComputeTrainingRequest(peer, message) {
  const requestId = asText(message?.requestId);
  const payload = message?.payload && typeof message.payload === "object" ? message.payload : {};
  const jobId = asText(payload?.jobId || message?.jobId) || buildComputeJobId();
  const settings = await craftSync?.readSettings?.();
  const normalizedCompute = normalizeRemoteComputeSettings(settings);
  const deviceId = await craftSync?.getDeviceId?.();
  const workerName = getPeerComputeWorkerName(settings, deviceId);

  if (!peerComputeState.bus || normalizedCompute.computeOfferEnabled !== true) {
    await peerComputeState.bus?.send(peer, {
      type: "training_start_response",
      requestId,
      jobId,
      ok: false,
      error: "Remote training is not offered on this peer.",
    });
    return;
  }
  if (asText(payload?.modelName) !== normalizedCompute.computeModelName) {
    await peerComputeState.bus.send(peer, {
      type: "training_start_response",
      requestId,
      jobId,
      ok: false,
      error: `Requested model ${asText(payload?.modelName)} does not match offered model ${normalizedCompute.computeModelName}.`,
    });
    return;
  }
  if (peerComputeLocalActiveJobs.size >= REMOTE_COMPUTE_MAX_CONCURRENT_JOBS) {
    await peerComputeState.bus.send(peer, {
      type: "training_start_response",
      requestId,
      jobId,
      ok: false,
      error: "This peer is currently busy.",
    });
    return;
  }

  peerComputeLocalActiveJobs.set(jobId, { kind: "training", jobId });
  await publishLocalPeerComputeWorkerState();
  await upsertPeerComputeJob({
    id: jobId,
    kind: "training",
    status: "running",
    modelName: normalizedCompute.computeModelName,
    requesterDeviceId: asText(message?.deviceId),
    workerDeviceId: asText(deviceId),
    workerName,
    progress: 0,
    startedAt: new Date().toISOString(),
  });

  let removeObserver = () => {};
  const finishTrainingRequest = async (run, status, errorText = "") => {
    await upsertPeerComputeJob({
      id: jobId,
      kind: "training",
      status,
      modelName: normalizedCompute.computeModelName,
      requesterDeviceId: asText(message?.deviceId),
      workerDeviceId: asText(deviceId),
      workerName,
      progress: Number(run?.progress || (status === "completed" ? 1 : 0)),
      error: errorText,
      completedAt: new Date().toISOString(),
      summary: {
        phase: asText(run?.phase),
        phaseLabel: asText(run?.phaseLabel),
        metrics: run?.metrics || null,
      },
    });
    peerComputeLocalActiveJobs.delete(jobId);
    await publishLocalPeerComputeWorkerState();
    removeObserver();
  };

  try {
    removeObserver = addLocalQwenTrainingRunObserver(jobId, (run) => {
      void (async () => {
        if (!peerComputeState.bus) return;
        await peerComputeState.bus.send(peer, {
          type: "training_progress",
          jobId,
          run,
        });
        await upsertPeerComputeJob({
          id: jobId,
          kind: "training",
          status: asText(run?.status || "running"),
          modelName: normalizedCompute.computeModelName,
          requesterDeviceId: asText(message?.deviceId),
          workerDeviceId: asText(deviceId),
          workerName,
          progress: Number(run?.progress || 0),
          summary: {
            phase: asText(run?.phase),
            phaseLabel: asText(run?.phaseLabel),
            metrics: run?.metrics || null,
          },
        });
        if (run?.status === "completed") {
          await peerComputeState.bus.send(peer, {
            type: "training_result",
            jobId,
            run,
          });
          await finishTrainingRequest(run, "completed");
        } else if (run?.status === "failed") {
          await peerComputeState.bus.send(peer, {
            type: "training_error",
            jobId,
            error: asText(run?.error),
            run,
          });
          await finishTrainingRequest(run, "failed", asText(run?.error));
        }
      })();
    });

    const initialRun = await startLocalQwenTrainingRunLocal({
      ...payload,
      jobId,
    });
    await peerComputeState.bus.send(peer, {
      type: "training_start_response",
      requestId,
      jobId,
      ok: true,
      run: initialRun,
    });
  } catch (error) {
    removeObserver();
    peerComputeLocalActiveJobs.delete(jobId);
    await publishLocalPeerComputeWorkerState();
    await peerComputeState.bus.send(peer, {
      type: "training_start_response",
      requestId,
      jobId,
      ok: false,
      error: String(error?.message || error || "Remote training failed to start."),
    });
    await upsertPeerComputeJob({
      id: jobId,
      kind: "training",
      status: "failed",
      modelName: normalizedCompute.computeModelName,
      requesterDeviceId: asText(message?.deviceId),
      workerDeviceId: asText(deviceId),
      workerName,
      error: String(error?.message || error || "Remote training failed to start."),
      completedAt: new Date().toISOString(),
    });
  }
}

async function handlePeerComputeBusMessage(event) {
  const message = event?.message && typeof event.message === "object"
    ? event.message
    : event?.response && typeof event.response === "object"
      ? event.response
      : null;
  const peer = event?.peer || null;
  const type = asText(message?.type);
  if (!message || !type) return;

  if (/_response$/.test(type) && message?.requestId && peerComputePendingRequests.has(message.requestId)) {
    const pending = peerComputePendingRequests.get(message.requestId);
    globalThis.clearTimeout(pending.timeoutId);
    peerComputePendingRequests.delete(message.requestId);
    pending.resolve(message);
    return;
  }

  if (type === "hello") {
    rememberPeerComputePeer(peer, message);
    return;
  }

  if (type === "training_progress" || type === "training_result" || type === "training_error") {
    const jobId = asText(message?.jobId);
    if (!jobId) return;
    const peerMeta = peer ? peerComputePeerMeta.get(peer) : null;
    const nextRun = normalizeRemoteTrainingRunSnapshot(message?.run, peerMeta);
    if (localQwenTrainingRuns.has(jobId)) {
      const patch = {
        ...nextRun,
        status:
          type === "training_result"
            ? "completed"
            : type === "training_error"
              ? "failed"
              : asText(nextRun.status || "running"),
        error: type === "training_error" ? asText(message?.error || nextRun.error) : asText(nextRun.error),
      };
      updateLocalQwenTrainingRun(jobId, patch);
      await upsertPeerComputeJob({
        id: jobId,
        kind: "training",
        status: patch.status,
        modelName: asText(nextRun?.modelName),
        requesterDeviceId: await craftSync?.getDeviceId?.(),
        workerDeviceId: asText(peerMeta?.deviceId),
        workerName: asText(peerMeta?.name),
        progress: Number(nextRun?.progress || 0),
        error: asText(patch.error),
        startedAt: nextRun?.startedAt,
        completedAt: patch.status === "completed" || patch.status === "failed" ? (nextRun?.endedAt || new Date().toISOString()) : "",
        summary: {
          phase: asText(nextRun?.phase),
          phaseLabel: asText(nextRun?.phaseLabel),
          metrics: nextRun?.metrics || null,
        },
      });
    }
    return;
  }

  if (type === "inference_request") {
    await handlePeerComputeInferenceRequest(peer, message);
    return;
  }

  if (type === "training_start_request") {
    await handlePeerComputeTrainingRequest(peer, message);
  }
}

async function syncPeerComputeRuntime(settings = null) {
  if (!craftSync?.readSettings) return null;
  const resolvedSettings =
    settings && typeof settings === "object" ? settings : await craftSync.readSettings();
  const normalizedCompute = normalizeRemoteComputeSettings(resolvedSettings);
  const shouldRun =
    resolvedSettings?.mode === "continuous" &&
    (normalizedCompute.computeOfferEnabled === true || normalizedCompute.remoteExecutionEnabled === true);
  const nextBusKey = shouldRun ? getPeerComputeBusKey(resolvedSettings) : "";

  if (!shouldRun) {
    await stopPeerComputeRuntime();
    return null;
  }

  await craftSync.ensureStartedFromSettings?.({ pageName: "ML Offscreen" });
  if (peerComputeState.bus && peerComputeState.busKey === nextBusKey) {
    await publishLocalPeerComputeWorkerState();
    return peerComputeState.bus;
  }

  await stopPeerComputeRuntime();
  peerComputeState.bus = await createPeerComputeBus({
    settings: resolvedSettings,
    onConnect: (peer) => {
      void sendPeerComputeHello(peer);
    },
    onDisconnect: (peer) => {
      forgetPeerComputePeer(peer);
    },
    onMessage: (event) => {
      void handlePeerComputeBusMessage(event);
    },
    onError: (error) => {
      console.warn("[offscreen] peer compute bus error", error);
    },
  });
  peerComputeState.busKey = nextBusKey;
  await publishLocalPeerComputeWorkerState();
  peerComputeState.workerHeartbeatTimer = globalThis.setInterval(() => {
    void publishLocalPeerComputeWorkerState();
  }, 8_000);
  return peerComputeState.bus;
}

async function initPeerComputeRuntime() {
  if (!craftSync?.readSettings) return null;
  if (peerComputeState.initPromise) return await peerComputeState.initPromise;

  peerComputeState.initPromise = (async () => {
    if (!peerComputeState.settingsUnsubscribe && typeof craftSync.subscribe === "function") {
      peerComputeState.settingsUnsubscribe = craftSync.subscribe((snapshot) => {
        const nextSettings = snapshot?.settings && typeof snapshot.settings === "object"
          ? snapshot.settings
          : null;
        if (!nextSettings) return;
        void syncPeerComputeRuntime(nextSettings);
      });
    }
    const settings = await craftSync.readSettings();
    return await syncPeerComputeRuntime(settings);
  })().catch((error) => {
    peerComputeState.initPromise = null;
    throw error;
  });

  return await peerComputeState.initPromise;
}

async function maybeHandleRemoteLocalQwenChat(args = {}) {
  if (!craftSync?.readSettings || !craftSync?.getDeviceId) return null;
  const settings = await craftSync.readSettings();
  const normalizedCompute = normalizeRemoteComputeSettings(settings);
  if (normalizedCompute.remoteExecutionEnabled !== true) return null;

  await initPeerComputeRuntime();
  const worker = await selectPeerComputeWorker(args.modelName);
  if (!worker) return null;

  const requesterDeviceId = await craftSync.getDeviceId();
  const jobId = buildComputeJobId();
  await upsertPeerComputeJob({
    id: jobId,
    kind: "inference",
    status: "queued",
    modelName: asText(args.modelName),
    requesterDeviceId,
    requesterName: getPeerComputeWorkerName(settings, requesterDeviceId),
    workerDeviceId: asText(worker.deviceId),
    workerName: asText(worker.name),
    progress: 0,
    createdAt: new Date().toISOString(),
    summary: {
      mode: "remote_peer",
    },
  });

  try {
    const response = await runPeerComputeRequest(worker.peer, "inference_request", jobId, args, 90_000);
    if (response?.ok !== true) {
      throw new Error(asText(response?.error) || "Remote peer inference failed.");
    }
    await upsertPeerComputeJob({
      id: jobId,
      kind: "inference",
      status: "completed",
      modelName: asText(args.modelName),
      requesterDeviceId,
      workerDeviceId: asText(worker.deviceId),
      workerName: asText(worker.name),
      progress: 1,
      completedAt: new Date().toISOString(),
      summary: {
        textLength: String(response?.text || "").length,
        runtime: response?.runtime || null,
      },
    });
    return {
      text: String(response?.text || ""),
      runtime: {
        ...(response?.runtime && typeof response.runtime === "object" ? response.runtime : {}),
        execution: "remote_peer",
        workerDeviceId: asText(worker.deviceId),
        workerName: asText(worker.name),
        workerModelName: asText(worker.modelName),
      },
    };
  } catch (error) {
    await upsertPeerComputeJob({
      id: jobId,
      kind: "inference",
      status: "failed",
      modelName: asText(args.modelName),
      requesterDeviceId,
      workerDeviceId: asText(worker.deviceId),
      workerName: asText(worker.name),
      error: String(error?.message || error || "Remote peer inference failed."),
      completedAt: new Date().toISOString(),
    });
    console.warn("[offscreen] remote peer inference failed, falling back local", error);
    return null;
  }
}

async function maybeStartRemoteLocalQwenTraining(args = {}) {
  if (!craftSync?.readSettings || !craftSync?.getDeviceId) return null;
  const settings = await craftSync.readSettings();
  const normalizedCompute = normalizeRemoteComputeSettings(settings);
  if (normalizedCompute.remoteExecutionEnabled !== true) return null;

  await initPeerComputeRuntime();
  const worker = await selectPeerComputeWorker(args.modelName);
  if (!worker) return null;

  const requesterDeviceId = await craftSync.getDeviceId();
  const requesterName = getPeerComputeWorkerName(settings, requesterDeviceId);
  const jobId = buildComputeJobId();
  const initialRun = {
    jobId,
    craftId: asText(args.craftId),
    shardId: asText(args.shardId),
    modelName: asText(args.modelName),
    status: "queued",
    phase: "queued_remote",
    phaseLabel: "Queued on remote worker",
    message: `Waiting for ${asText(worker.name || worker.deviceId)} to accept the training job.`,
    error: "",
    progress: 0,
    totalSamples: 0,
    completedSamples: 0,
    phaseTotalSamples: 0,
    phaseCompletedSamples: 0,
    phaseUnitLabel: "job steps",
    samplesPerSecond: 0,
    currentEpoch: 0,
    epochsTotal: Number(args?.configOverrides?.epochs || 0),
    adapterSizeMb: 0,
    selectedEpoch: 0,
    metrics: {},
    dataset: null,
    runtime: {
      execution: "remote_peer",
      workerDeviceId: asText(worker.deviceId),
      workerName: asText(worker.name),
      workerModelName: asText(worker.modelName),
    },
    smokeMode: asText(args.smokeMode),
    smoke: null,
    history: [],
    createdAt: new Date().toISOString(),
    startedAt: "",
    endedAt: "",
  };
  localQwenTrainingRuns.set(jobId, initialRun);
  notifyLocalQwenTrainingRunObservers(initialRun);

  await upsertPeerComputeJob({
    id: jobId,
    kind: "training",
    status: "queued",
    modelName: asText(args.modelName),
    requesterDeviceId,
    requesterName,
    workerDeviceId: asText(worker.deviceId),
    workerName: asText(worker.name),
    progress: 0,
    createdAt: initialRun.createdAt,
    summary: {
      mode: "remote_peer",
      craftId: asText(args.craftId),
    },
  });

  try {
    const response = await runPeerComputeRequest(worker.peer, "training_start_request", jobId, {
      ...args,
      jobId,
    }, 30_000);
    if (response?.ok !== true) {
      throw new Error(asText(response?.error) || "Remote peer training start failed.");
    }
    const run = normalizeRemoteTrainingRunSnapshot(response?.run, worker);
    updateLocalQwenTrainingRun(jobId, {
      ...run,
      status: asText(run.status || "running"),
      startedAt: asText(run.startedAt || new Date().toISOString()),
    });
    await upsertPeerComputeJob({
      id: jobId,
      kind: "training",
      status: asText(run.status || "running"),
      modelName: asText(args.modelName),
      requesterDeviceId,
      workerDeviceId: asText(worker.deviceId),
      workerName: asText(worker.name),
      progress: Number(run.progress || 0),
      startedAt: run.startedAt,
      summary: {
        phase: asText(run.phase),
        phaseLabel: asText(run.phaseLabel),
      },
    });
    return getLocalQwenTrainingRunStatus(jobId);
  } catch (error) {
    localQwenTrainingRuns.delete(jobId);
    await upsertPeerComputeJob({
      id: jobId,
      kind: "training",
      status: "failed",
      modelName: asText(args.modelName),
      requesterDeviceId,
      workerDeviceId: asText(worker.deviceId),
      workerName: asText(worker.name),
      error: String(error?.message || error || "Remote peer training failed."),
      completedAt: new Date().toISOString(),
    });
    console.warn("[offscreen] remote peer training failed, falling back local", error);
    return null;
  }
}

async function startLocalQwenTrainingRunLocal({
  jobId = "",
  craftId = "",
  shardId = "",
  modelName = "",
  datasetPayload = null,
  persistBundle = true,
  smokeMode = "",
  configOverrides = null,
} = {}) {
  const resolvedModelName = await resolveSupportedLocalQwenTrainingModelName(modelName);
  const resolvedRuntimePlan = getLocalQwenRuntimePlan(resolvedModelName);
  const resolvedSmokeMode = String(smokeMode || "").trim();
  const resolvedDatasetPayload =
    datasetPayload && typeof datasetPayload === "object"
      ? datasetPayload
      : resolvedSmokeMode === "pipeline_e2e"
        ? createLocalQwenPipelineSmokeDatasetPayload()
        : null;
  const baseConfig =
    resolvedSmokeMode === "pipeline_e2e"
      ? LOCAL_QWEN_PIPELINE_SMOKE_CONFIG
      : LOCAL_QWEN_FIXED_TRAINING_CONFIG;
  const config =
    resolvedDatasetPayload
      ? mergeTrainingConfig(baseConfig, {
          datasetPayload: resolvedDatasetPayload,
          maxTrainPairs: Array.isArray(resolvedDatasetPayload?.train)
            ? resolvedDatasetPayload.train.length
            : baseConfig.maxTrainPairs,
          maxValidationPairs: Array.isArray(resolvedDatasetPayload?.validation)
            ? resolvedDatasetPayload.validation.length
            : (baseConfig.maxValidationPairs || baseConfig.maxTestPairs),
          maxTestPairs: Array.isArray(resolvedDatasetPayload?.test)
            ? resolvedDatasetPayload.test.length
            : baseConfig.maxTestPairs,
        })
      : mergeTrainingConfig(baseConfig, {});
  const resolvedConfig = configOverrides && typeof configOverrides === "object"
    ? mergeTrainingConfig(config, configOverrides)
    : config;
  await assertLocalQwenBrowserTrainingSupported(resolvedRuntimePlan.runtimeModelId);
  const trainingWorkSamples = estimateFixedTrainingWorkSamples(resolvedConfig);
  const pipelineSmokeWork =
    resolvedSmokeMode === "pipeline_e2e"
      ? estimateLocalQwenPipelineSmokeWork(resolvedDatasetPayload)
      : 0;
  const persistWork = persistBundle ? 1 : 0;
  const totalSamples = pipelineSmokeWork + trainingWorkSamples + persistWork;
  const resolvedJobId = asText(jobId) || createLocalQwenTrainingJobId();
  const createdAt = new Date().toISOString();

  const initialRun = {
    jobId: resolvedJobId,
    craftId: String(craftId || ""),
    shardId: String(shardId || ""),
    modelName: resolvedModelName,
    status: "queued",
    phase: "queued",
    phaseLabel: "Queued",
    message: "Queued native browser training in the offscreen WebGPU runtime.",
    error: "",
    totalSamples,
    completedSamples: 0,
    phaseTotalSamples: 0,
    phaseCompletedSamples: 0,
    phaseUnitLabel: "job steps",
    progress: 0,
    samplesPerSecond: 0,
    currentEpoch: 0,
    epochsTotal: Number(resolvedConfig.epochs || 0),
    adapterSizeMb: 0,
    selectedEpoch: 0,
    metrics: {},
    dataset: null,
    runtime: null,
    smokeMode: resolvedSmokeMode,
    smoke: null,
    history: [],
    createdAt,
    startedAt: "",
    endedAt: "",
  };
  localQwenTrainingRuns.set(resolvedJobId, initialRun);
  notifyLocalQwenTrainingRunObservers(initialRun);

  void (async () => {
    const startedAtMs = Date.now();
    let completedBeforeTraining = 0;
    let smokeSummary = null;
    updateLocalQwenTrainingRun(resolvedJobId, {
      status: "running",
      phase: "initializing",
      phaseLabel: "Initializing WebGPU training",
      message: "Loading the native q4f16 runtime in the offscreen document.",
      startedAt: new Date(startedAtMs).toISOString(),
    });

    try {
      if (resolvedSmokeMode === "pipeline_e2e") {
        smokeSummary = await runLocalQwenPipelineSmoke({
          datasetPayload: resolvedDatasetPayload,
          onStatus: (payload = {}) => {
            updateLocalQwenTrainingRun(resolvedJobId, {
              status: "running",
              phase: String(payload.phase || ""),
              phaseLabel: String(payload.phaseLabel || payload.phase || "Pipeline smoke"),
              message: String(payload.message || ""),
              completedSamples: Number(payload.completedSamples || 0),
              totalSamples,
              phaseCompletedSamples: Number(payload.phaseCompletedSamples || 0),
              phaseTotalSamples: Number(payload.phaseTotalSamples || 0),
              phaseUnitLabel: String(payload.phaseUnitLabel || "pipeline steps"),
              samplesPerSecond: Number(payload.samplesPerSecond || 0),
              epochsTotal: Number(resolvedConfig.epochs || 0),
              smoke: {
                mode: resolvedSmokeMode,
                stage: "preflight",
              },
            });
          },
        });
        completedBeforeTraining = pipelineSmokeWork;
        if (!smokeSummary?.preferenceDataset?.length || smokeSummary?.rl?.dpoReady !== true || smokeSummary?.rl?.grpoReady !== true) {
          throw new Error("Pipeline smoke validation failed before finetuning.");
        }
        updateLocalQwenTrainingRun(resolvedJobId, {
          smoke: {
            mode: resolvedSmokeMode,
            stage: "validated",
            validatedStages: [
              "dataset_validation",
              "policy_bundle",
              "rollouts",
              "rewards",
              "preferences",
              "dpo_contracts",
              "grpo_contracts",
            ],
            stats: smokeSummary.stats,
            rl: smokeSummary.rl,
          },
        });
      }

      const result = await trainFixedLocalQwenAdapter({
        modelName: resolvedModelName,
        config: resolvedConfig,
        onStatus: (payload = {}) => {
          const completed = Number(payload.completedSamples || 0);
          updateLocalQwenTrainingRun(resolvedJobId, {
            status: "running",
            phase: String(payload.phase || ""),
            phaseLabel: String(payload.phaseLabel || payload.phase || "Training"),
            message: String(payload.message || ""),
            completedSamples: completedBeforeTraining + completed,
            totalSamples,
            phaseCompletedSamples: Number(payload.phaseCompletedSamples || 0),
            phaseTotalSamples: Number(payload.phaseTotalSamples || 0),
            phaseUnitLabel: String(payload.phaseUnitLabel || ""),
            samplesPerSecond: Number(payload.samplesPerSecond || 0),
            currentEpoch: Number(payload.currentEpoch || 0),
            epochsTotal: Number(resolvedConfig.epochs || 0),
            dataset: payload.dataset || undefined,
            runtime: payload.runtime || undefined,
            metrics:
              payload.metrics && typeof payload.metrics === "object"
                ? {
                    ...payload.metrics,
                    baseValidationAcc: Number(payload.metrics.baseValidationAcc ?? 0) || 0,
                    baseTestAcc: Number(payload.metrics.baseTestAcc ?? 0) || 0,
                    adaptValidationAcc: Number(payload.metrics.adaptValidationAcc ?? 0) || 0,
                    adaptTestAcc: Number(payload.metrics.adaptTestAcc ?? 0) || 0,
                    trainEvalAcc: Number(payload.metrics.trainEvalAcc ?? 0) || 0,
                  }
                : undefined,
          });
        },
      });

      if (persistBundle) {
        updateLocalQwenTrainingRun(resolvedJobId, {
          status: "running",
          phase: "persisting_bundle",
          phaseLabel: "Persisting capability bundle",
          message: "Saving trained weights so this craft can be shared as one complete bundle.",
          completedSamples: completedBeforeTraining + trainingWorkSamples,
          totalSamples,
          phaseCompletedSamples: 0,
          phaseTotalSamples: 1,
          phaseUnitLabel: "bundle writes",
        });

        await persistLocalQwenCapabilityWeights({
          craftId,
          shardId,
          modelName: resolvedModelName,
          config: resolvedConfig,
          jobId: resolvedJobId,
          result,
        });
      }

      updateLocalQwenTrainingRun(resolvedJobId, {
        status: "completed",
        phase: "completed",
        phaseLabel: "Completed",
        message: persistBundle
          ? "Native browser q4f16 training completed, pipeline smoke passed, and the full capability bundle was saved."
          : resolvedSmokeMode === "pipeline_e2e"
            ? "Native browser q4f16 debug training completed and the staged pipeline smoke passed."
            : "Native browser q4f16 debug training completed without persisting a full capability bundle.",
        completedSamples: totalSamples,
        totalSamples,
        progress: 1,
        phaseCompletedSamples: persistBundle ? 1 : Number(result.phaseCompletedSamples || 0),
        phaseTotalSamples: persistBundle ? 1 : Number(result.phaseTotalSamples || 0),
        phaseUnitLabel: persistBundle ? "bundle writes" : String(result.phaseUnitLabel || ""),
        samplesPerSecond: Number(result.samplesPerSecond || 0),
        currentEpoch: Number(resolvedConfig.epochs || 0),
        epochsTotal: Number(resolvedConfig.epochs || 0),
        adapterSizeMb: Number(result.adapterSizeMb || 0),
        selectedEpoch: Number(result.selectedEpoch || 0),
        metrics: {
          baseTrainAcc: Number(result.baseTrainAcc || 0) || 0,
          baseValidationAcc: Number(result.baseValidationAcc || 0) || 0,
          baseTestAcc: Number(result.baseTestAcc || 0) || 0,
          trainEvalAcc: Number(result.trainEval?.accuracy || 0) || 0,
          validationEvalAcc: Number(result.validationEval?.accuracy || 0) || 0,
          adaptTestAcc: Number(result.testEval?.accuracy || 0) || 0,
          comparisonPurpose: String(result.throughput?.comparisonPurpose || "").trim(),
          compareBy: String(result.throughput?.compareBy || "").trim(),
          resourceProfile: String(result.throughput?.resourceProfile || "").trim(),
          totalForwardTokens: Number(result.throughput?.totalForwardTokens || 0) || 0,
          totalSupervisedTokens: Number(result.throughput?.totalSupervisedTokens || 0) || 0,
          trainStepForwardTokens: Number(result.throughput?.trainStepForwardTokens || 0) || 0,
          evalForwardTokens: Number(result.throughput?.evalForwardTokens || 0) || 0,
          overallElapsedMs: Number(result.throughput?.overallElapsedMs || 0) || 0,
          totalMeasuredForwardMs: Number(result.throughput?.totalMeasuredForwardMs || 0) || 0,
          trainStepForwardMs: Number(result.throughput?.trainStepForwardMs || 0) || 0,
          evalForwardMs: Number(result.throughput?.evalForwardMs || 0) || 0,
          nonForwardOverheadMs: Number(result.throughput?.nonForwardOverheadMs || 0) || 0,
          overallForwardTokensPerSecond: Number(result.throughput?.overallForwardTokensPerSecond || 0) || 0,
          overallSupervisedTokensPerSecond: Number(result.throughput?.overallSupervisedTokensPerSecond || 0) || 0,
          measuredForwardTokensPerSecond: Number(result.throughput?.measuredForwardTokensPerSecond || 0) || 0,
          trainStepForwardTokensPerSecond: Number(result.throughput?.trainStepForwardTokensPerSecond || 0) || 0,
          evalForwardTokensPerSecond: Number(result.throughput?.evalForwardTokensPerSecond || 0) || 0,
          forwardWorkloadShare: Number(result.throughput?.forwardWorkloadShare || 0) || 0,
          nonForwardOverheadShare: Number(result.throughput?.nonForwardOverheadShare || 0) || 0,
        },
        dataset: result.dataset,
        runtime: result.runtime,
        smoke:
          resolvedSmokeMode === "pipeline_e2e"
            ? {
                mode: resolvedSmokeMode,
                stage: "completed",
                validatedStages: [
                  "dataset_validation",
                  "policy_bundle",
                  "rollouts",
                  "rewards",
                  "preferences",
                  "dpo_contracts",
                  "grpo_contracts",
                  "sft_training",
                ],
                stats: smokeSummary?.stats || null,
                rl: smokeSummary?.rl || null,
              }
            : null,
        history: Array.isArray(result.history) ? result.history : [],
        endedAt: new Date().toISOString(),
      });
    } catch (error) {
      updateLocalQwenTrainingRun(resolvedJobId, {
        status: "failed",
        phase: "failed",
        phaseLabel: "Training failed",
        message: "The native browser training run failed.",
        error: String(error?.message || error || "Unknown training failure"),
        endedAt: new Date().toISOString(),
      });
    }
  })();

  return getPublicTrainingSnapshot(localQwenTrainingRuns.get(resolvedJobId));
}

async function startLocalQwenTrainingRun(options = {}) {
  const remoteRun = await maybeStartRemoteLocalQwenTraining(options);
  if (remoteRun) {
    return remoteRun;
  }
  return await startLocalQwenTrainingRunLocal(options);
}

function getLocalQwenTrainingRunStatus(jobId) {
  return getPublicTrainingSnapshot(localQwenTrainingRuns.get(String(jobId || "").trim()));
}

function prioritizeLocalQwenExecutionPlans(runtimePlan) {
  const preferred = localQwenPreferredExecutionPlan.get(runtimePlan.runtimeModelId);
  if (!preferred) return runtimePlan.executionPlans;
  return [
    preferred,
    ...runtimePlan.executionPlans.filter(
      (plan) => !sameLocalQwenExecutionPlan(plan, preferred),
    ),
  ];
}

async function getOrCreateLocalQwenPipelineEntry(runtimeModelId, executionPlan) {
  const cacheKey = getLocalQwenAttemptKey(runtimeModelId, executionPlan);
  if (!localQwenRuntimePromiseCache.has(cacheKey)) {
    const runtimeOptions = {
      device: executionPlan.device,
    };
    if (executionPlan.dtype) {
      runtimeOptions.dtype = executionPlan.dtype;
    }
    const loadPipelineBundle = async () => {
      const loraManifest = await getLocalQwenLoraManifest(runtimeModelId);
      const loraInputNames = Array.isArray(loraManifest?.modules)
        ? loraManifest.modules.flatMap((entry) => [
            String(entry?.loraInputA || "").trim(),
            String(entry?.loraInputB || "").trim(),
          ]).filter(Boolean)
        : [];
      // Keep the text-only runtime path off the vision bootstrap unless an image is actually present.
      const tokenizer = await AutoTokenizer.from_pretrained(runtimeModelId);
      const model = await Qwen3_5ForConditionalGeneration.from_pretrained(runtimeModelId, runtimeOptions);
      return {
        processor: null,
        processorPromise: null,
        tokenizer,
        model: patchLocalQwenForwardParams(model, loraInputNames),
        loraManifest,
        visionPreprocess: null,
      };
    };
    const attemptPromise = loadPipelineBundle()
      .catch(async (error) => {
        const message = String(error?.message || error || "");
        if (!/failed to fetch/i.test(message)) {
          throw error;
        }
        console.warn("[offscreen] retrying local_qwen pipeline load after fetch failure", {
          runtimeModelId,
          device: executionPlan.device,
          dtype: getLocalQwenDtypeKey(executionPlan.dtype),
          error: serializeError(error),
          recentFetches: getRecentLocalQwenFetchTrace(6),
        });
        return await loadPipelineBundle();
      })
      .catch((error) => {
        localQwenRuntimePromiseCache.delete(cacheKey);
        throw error;
      });
    localQwenRuntimePromiseCache.set(cacheKey, {
      promise: attemptPromise,
      activeUses: 0,
      lastUsedAt: Date.now(),
      disposeTimer: 0,
    });
  }
  return localQwenRuntimePromiseCache.get(cacheKey);
}

async function withLocalQwenPipeline(runtimeModelId, executionPlan, callback) {
  const cacheKey = getLocalQwenAttemptKey(runtimeModelId, executionPlan);
  const entry = await getOrCreateLocalQwenPipelineEntry(runtimeModelId, executionPlan);
  entry.activeUses = Number(entry.activeUses || 0) + 1;
  entry.lastUsedAt = Date.now();
  clearLocalQwenPipelineDisposeTimer(entry);
  try {
    const pipeline = await entry.promise;
    return await callback(pipeline);
  } finally {
    entry.activeUses = Math.max(0, Number(entry.activeUses || 0) - 1);
    entry.lastUsedAt = Date.now();
    if (localQwenRuntimePromiseCache.get(cacheKey) === entry && entry.activeUses === 0) {
      scheduleLocalQwenPipelineCacheEntryDispose(cacheKey);
    }
  }
}

async function ensureLocalQwenVisionProcessor(pipeline, runtimeModelId, executionPlan) {
  if (!pipeline || typeof pipeline !== "object") {
    throw new Error("Local Qwen vision processor is unavailable.");
  }
  if (pipeline.processor) {
    return {
      processor: pipeline.processor,
      visionPreprocess: pipeline.visionPreprocess || null,
    };
  }
  if (!pipeline.processorPromise) {
    pipeline.processorPromise = (async () => {
      const processor = await AutoProcessor.from_pretrained(runtimeModelId);
      const visionPreprocess = applyLocalQwenVisionPreprocessPolicy(
        processor,
        runtimeModelId,
        executionPlan,
      );
      pipeline.processor = processor;
      pipeline.visionPreprocess = visionPreprocess || null;
      return processor;
    })().catch((error) => {
      pipeline.processorPromise = null;
      throw error;
    });
  }
  const processor = await pipeline.processorPromise;
  return {
    processor,
    visionPreprocess: pipeline.visionPreprocess || null,
  };
}

function asLocalQwenText(value) {
  return String(value == null ? "" : value).trim();
}

function decodeLocalQwenGeneratedText(tokenizer, promptInputs, generatedIds) {
  const decodedOutputs = tokenizer.batch_decode(generatedIds, {
    skip_special_tokens: true,
  });
  let text = asLocalQwenText(Array.isArray(decodedOutputs) ? decodedOutputs[0] : decodedOutputs);

  if (!text) return text;

  try {
    const decodedPrompts = tokenizer.batch_decode(promptInputs.input_ids, {
      skip_special_tokens: true,
    });
    const promptText = asLocalQwenText(Array.isArray(decodedPrompts) ? decodedPrompts[0] : decodedPrompts);
    if (promptText && text.startsWith(promptText)) {
      text = text.slice(promptText.length).trim();
    }
  } catch (_error) {
    // If prompt decoding differs from generated decoding, keep the full decoded text.
  }

  return text;
}

function modelNeedsLocalQwenRopeInputs(model) {
  const modelType = String(model?.config?.model_type || "").trim();
  return (
    modelType === "qwen3_5" ||
    modelType === "qwen3_5_text" ||
    modelType === "qwen3_5_moe" ||
    modelType === "qwen3_5_moe_text"
  );
}

function getLocalQwenInputSequences(inputIds) {
  if (!inputIds) return null;
  if (typeof inputIds.tolist === "function") {
    const sequences = inputIds.tolist();
    if (Array.isArray(sequences)) return sequences;
  }
  if (ArrayBuffer.isView(inputIds)) {
    return [Array.from(inputIds, (value) => BigInt(Number(value)))];
  }
  return null;
}

function hasUsableLocalQwenPositionIds(positionIds) {
  return (
    Array.isArray(positionIds?.dims) &&
    positionIds.dims.length === 3
  );
}

function prepareLocalQwenModelInputs(model, promptInputs) {
  if (!model || !promptInputs || typeof promptInputs !== "object") {
    return promptInputs;
  }

  const inputSequences = getLocalQwenInputSequences(promptInputs.input_ids);
  if (typeof model?.prepare_inputs_for_generation === "function" && inputSequences) {
    try {
      const preparedInputs = model.prepare_inputs_for_generation(
        inputSequences,
        { ...promptInputs },
        {},
      );
      if (hasUsableLocalQwenPositionIds(preparedInputs?.position_ids)) {
        return {
          ...promptInputs,
          ...preparedInputs,
        };
      }
      if (preparedInputs) {
        promptInputs = {
          ...promptInputs,
          ...preparedInputs,
        };
      }
    } catch (_error) {
      // Fall through to the explicit rope-input path below.
    }
  }

  return withLocalQwenRopeInputs(model, promptInputs);
}

function withLocalQwenRopeInputs(model, promptInputs) {
  if (
    !modelNeedsLocalQwenRopeInputs(model) ||
    typeof model?.get_rope_index !== "function" ||
    !promptInputs?.input_ids ||
    !promptInputs?.attention_mask ||
    hasUsableLocalQwenPositionIds(promptInputs?.position_ids)
  ) {
    return promptInputs;
  }

  const [positionIds, ropeDeltas] = model.get_rope_index(
    promptInputs.input_ids,
    promptInputs.image_grid_thw,
    promptInputs.video_grid_thw,
    promptInputs.attention_mask,
  );

  return {
    ...promptInputs,
    position_ids: positionIds,
    ...(ropeDeltas ? { rope_deltas: ropeDeltas } : {}),
  };
}

function getLocalQwenVisionImageProcessor(processor) {
  const imageProcessor = processor?.image_processor;
  return imageProcessor && typeof imageProcessor === "object"
    ? imageProcessor
    : null;
}

function readAppliedLocalQwenVisionPreprocessPolicy(processor) {
  const imageProcessor = getLocalQwenVisionImageProcessor(processor);
  const policy = imageProcessor?.__sinepanelVisionPreprocessPolicy;
  return policy && typeof policy === "object"
    ? { ...policy }
    : null;
}

function getLocalQwenVisionLongestEdgeFromPixelBudget(pixelBudget) {
  const maxPixels = Number(pixelBudget);
  if (!Number.isFinite(maxPixels) || maxPixels <= 0) return 0;
  return Math.max(1, Math.floor(Math.sqrt(maxPixels)));
}

function applyLocalQwenVisionPreprocessPolicy(processor, runtimeModelId, executionPlan) {
  const policy = getLocalQwenVisionPreprocessPolicy(runtimeModelId, executionPlan);
  const imageProcessor = getLocalQwenVisionImageProcessor(processor);
  if (!policy || !imageProcessor) return null;

  const currentLongestEdge = Number(imageProcessor.size?.longest_edge || 0);
  const currentMaxPixels = Number(
    imageProcessor.max_pixels ||
      imageProcessor.maxPixels ||
      (currentLongestEdge > 0 ? currentLongestEdge ** 2 : 0) ||
      0,
  );
  const appliedMaxPixels =
    currentMaxPixels > 0
      ? Math.min(currentMaxPixels, policy.maxPixels)
      : policy.maxPixels;
  const cappedLongestEdge = getLocalQwenVisionLongestEdgeFromPixelBudget(appliedMaxPixels);
  const appliedLongestEdge =
    currentLongestEdge > 0 && cappedLongestEdge > 0
      ? Math.min(currentLongestEdge, cappedLongestEdge)
      : cappedLongestEdge;
  if (!(appliedMaxPixels > 0)) return null;

  imageProcessor.max_pixels = appliedMaxPixels;
  if ("maxPixels" in imageProcessor) {
    imageProcessor.maxPixels = appliedMaxPixels;
  }
  if (imageProcessor.size && typeof imageProcessor.size === "object") {
    imageProcessor.size = {
      ...imageProcessor.size,
      longest_edge: appliedLongestEdge,
    };
  }

  imageProcessor.__sinepanelVisionPreprocessPolicy = {
    ...policy,
    originalMaxPixels: currentMaxPixels > 0 ? currentMaxPixels : null,
    appliedMaxPixels,
    originalLongestEdge: currentLongestEdge > 0 ? currentLongestEdge : null,
    appliedLongestEdge: appliedLongestEdge > 0 ? appliedLongestEdge : null,
  };
  return readAppliedLocalQwenVisionPreprocessPolicy(processor);
}

async function loadLocalQwenRawImage(imageSource) {
  const source = asLocalQwenText(imageSource);
  if (!source) {
    throw new Error("Local Qwen vision input is missing an image source.");
  }
  if (source.startsWith("data:")) {
    return await RawImage.fromBlob(dataUrlToBlob(source));
  }
  return await RawImage.fromURL(source);
}

async function buildLocalQwenVisionInputs(processor, normalizedMessages, promptTextOverride = "") {
  if (!processor || typeof processor.apply_chat_template !== "function") {
    throw new Error("Local Qwen vision processor is unavailable.");
  }

  const promptText = asLocalQwenText(promptTextOverride) || processor.apply_chat_template(normalizedMessages, {
    tokenize: false,
    add_generation_prompt: true,
  });
  const imageSources = collectLocalQwenImageSources(normalizedMessages);
  const images = await Promise.all(imageSources.map((imageSource) => loadLocalQwenRawImage(imageSource)));
  const processorInputs = await processor(
    promptText,
    images.length === 1 ? images[0] : images,
  );

  return {
    promptInputs: processorInputs,
    inputMode: "processor_vision",
    imageCount: images.length,
    promptInputKeys: Object.keys(processorInputs || {}).sort(),
    visionPreprocess: readAppliedLocalQwenVisionPreprocessPolicy(processor),
  };
}

function buildLocalQwenTextInputs(tokenizer, normalizedMessages, promptTextOverride = "") {
  const promptText = asLocalQwenText(promptTextOverride);
  if (promptText) {
    return {
      promptInputs: tokenizer(promptText, {
        add_special_tokens: false,
        return_attention_mask: true,
        return_tensor: true,
      }),
      inputMode: "prompt_text",
      imageCount: 0,
      promptInputKeys: ["attention_mask", "input_ids"],
    };
  }
  return {
    promptInputs: tokenizer.apply_chat_template(normalizedMessages, {
      add_generation_prompt: true,
      tokenize: true,
      return_dict: true,
      return_tensor: true,
    }),
    inputMode: "tokenizer_text",
    imageCount: 0,
    promptInputKeys: ["attention_mask", "input_ids"],
  };
}

function countLocalQwenPromptTokens(promptInputs) {
  const attentionMask = copyLocalQwenTokenRow(promptInputs?.attention_mask);
  if (attentionMask.length) {
    let total = 0;
    for (const value of attentionMask) {
      if (Number(value) > 0) total += 1;
    }
    return total;
  }
  return copyLocalQwenTokenRow(promptInputs?.input_ids).length;
}

function nowMs() {
  if (globalThis.performance && typeof globalThis.performance.now === "function") {
    return globalThis.performance.now();
  }
  return Date.now();
}

function summarizeTimingSeries(values = []) {
  const safeValues = (Array.isArray(values) ? values : [])
    .map((value) => Number(value))
    .filter((value) => Number.isFinite(value) && value >= 0);
  if (!safeValues.length) {
    return {
      meanMs: 0,
      minMs: 0,
      maxMs: 0,
      medianMs: 0,
    };
  }
  const sorted = [...safeValues].sort((left, right) => left - right);
  const midpoint = Math.floor(sorted.length / 2);
  const medianMs = sorted.length % 2 === 0
    ? (sorted[midpoint - 1] + sorted[midpoint]) / 2
    : sorted[midpoint];
  const total = safeValues.reduce((sum, value) => sum + value, 0);
  return {
    meanMs: total / safeValues.length,
    minMs: sorted[0],
    maxMs: sorted[sorted.length - 1],
    medianMs,
  };
}

function notifyLocalQwenForwardBenchmarkProgress(event = {}) {
  const benchmarkId = asText(event?.benchmarkId);
  if (!benchmarkId || !globalThis.chrome?.runtime?.sendMessage) return;
  try {
    const maybePromise = globalThis.chrome.runtime.sendMessage({
      type: "local_qwen_forward_benchmark_progress",
      benchmarkId,
      phase: asText(event?.phase),
      completedUnits: Number(event?.completedUnits || 0),
      totalUnits: Number(event?.totalUnits || 0),
      progress: clampProgress(event?.progress),
      promptTokenCount: Number(event?.promptTokenCount || 0),
      measuredIterations: Number(event?.measuredIterations || 0),
      warmupIterations: Number(event?.warmupIterations || 0),
    });
    if (maybePromise && typeof maybePromise.catch === "function") {
      maybePromise.catch(() => {});
    }
  } catch (_error) {
    // Ignore missing listeners. The final benchmark response is still authoritative.
  }
}

async function benchmarkLocalQwenForwardOnPipeline({
  pipeline,
  runtimeModelId,
  executionPlan,
  normalizedMessages,
  benchmarkId = "",
  promptText = "",
  iterations = 6,
  warmupIterations = 1,
}) {
  const usesVisionProcessor = hasLocalQwenVisionInputs(normalizedMessages);
  const rawChatInputs = usesVisionProcessor
    ? await (async () => {
        const { processor, visionPreprocess } = await ensureLocalQwenVisionProcessor(
          pipeline,
          runtimeModelId,
          executionPlan,
        );
        const inputs = await buildLocalQwenVisionInputs(processor, normalizedMessages, promptText);
        if (!inputs.visionPreprocess && visionPreprocess) {
          inputs.visionPreprocess = visionPreprocess;
        }
        return inputs;
      })()
    : buildLocalQwenTextInputs(pipeline.tokenizer, normalizedMessages, promptText);
  const chatInputs = prepareLocalQwenModelInputs(pipeline.model, rawChatInputs.promptInputs);
  const promptTokenCount = countLocalQwenPromptTokens(chatInputs);
  if (!(promptTokenCount > 0)) {
    throw new Error("Local Qwen forward benchmark prompt encoded to zero tokens.");
  }

  const safeWarmupIterations = Math.max(0, Math.trunc(Number(warmupIterations || 0)));
  const safeIterations = Math.max(1, Math.trunc(Number(iterations || 1)));
  const totalUnits = safeWarmupIterations + safeIterations;
  const emitProgress = (phase, completedUnits) => {
    notifyLocalQwenForwardBenchmarkProgress({
      benchmarkId,
      phase,
      completedUnits,
      totalUnits,
      progress: totalUnits > 0 ? completedUnits / totalUnits : 0,
      promptTokenCount,
      measuredIterations: safeIterations,
      warmupIterations: safeWarmupIterations,
    });
  };
  emitProgress("starting", 0);
  for (let warmupIndex = 0; warmupIndex < safeWarmupIterations; warmupIndex += 1) {
    const outputs = await pipeline.model(chatInputs);
    if (!Array.isArray(outputs?.logits?.dims)) {
      throw new Error("Local Qwen forward benchmark warmup returned no logits tensor.");
    }
    emitProgress("warmup", warmupIndex + 1);
  }

  const iterationDurationsMs = [];
  let logitsShape = [];
  const benchmarkStartedAt = nowMs();
  for (let iterationIndex = 0; iterationIndex < safeIterations; iterationIndex += 1) {
    const iterationStartedAt = nowMs();
    const outputs = await pipeline.model(chatInputs);
    const logitsTensor = outputs?.logits;
    if (!Array.isArray(logitsTensor?.dims)) {
      throw new Error("Local Qwen forward benchmark returned no logits tensor.");
    }
    logitsShape = Array.from(logitsTensor.dims, Number);
    iterationDurationsMs.push(Math.max(0, nowMs() - iterationStartedAt));
    emitProgress("measure", safeWarmupIterations + iterationIndex + 1);
  }
  const totalElapsedMs = Math.max(0, nowMs() - benchmarkStartedAt);
  const totalForwardTokens = promptTokenCount * safeIterations;
  const timing = summarizeTimingSeries(iterationDurationsMs);

  return {
    mode: "forward_only",
    iterations: safeIterations,
    warmupIterations: safeWarmupIterations,
    promptTokenCount,
    totalForwardTokens,
    totalElapsedMs,
    forwardTokensPerSecond:
      totalForwardTokens > 0
        ? totalForwardTokens / Math.max(totalElapsedMs / 1000, 1e-6)
        : 0,
    inputMode: rawChatInputs.inputMode,
    imageCount: rawChatInputs.imageCount,
    promptInputKeys: rawChatInputs.promptInputKeys,
    logitsShape,
    ...timing,
  };
}

function buildLocalQwenPromptInputs(basePromptInputs, tokenIds, attentionMask) {
  const {
    input_ids: _inputIds,
    attention_mask: _attentionMask,
    position_ids: _positionIds,
    rope_deltas: _ropeDeltas,
    ...rest
  } = basePromptInputs && typeof basePromptInputs === "object" ? basePromptInputs : {};
  const sequenceLength = Math.max(1, tokenIds.length);
  return {
    ...rest,
    input_ids: createInt64Tensor(tokenIds, [1, sequenceLength]),
    attention_mask: createInt64Tensor(attentionMask, [1, sequenceLength]),
  };
}

function extractLastLogitsRow(logitsTensor) {
  const dims = Array.isArray(logitsTensor?.dims) ? logitsTensor.dims : [];
  const hasBatchAxis = dims.length === 3;
  const batchSize = Number(hasBatchAxis ? dims[0] : 1);
  const seqLength = Number(dims[hasBatchAxis ? 1 : 0] || 0);
  const vocabSize = Number(dims[hasBatchAxis ? 2 : 1] || 0);
  if (!logitsTensor?.data || seqLength <= 0 || vocabSize <= 0) {
    throw new Error("Local Qwen generation returned no usable logits.");
  }
  if (batchSize <= 0) {
    throw new Error("Local Qwen logits tensor has an invalid batch size.");
  }
  const rowOffset = (batchSize - 1) * seqLength * vocabSize + (seqLength - 1) * vocabSize;
  return new Float32Array(logitsTensor.data.subarray(rowOffset, rowOffset + vocabSize));
}

function buildLocalQwenTransformerLoraInputs(adapterInfo = null) {
  if (!adapterInfo?.applied || adapterInfo?.kind !== "transformer_lora") {
    return {};
  }
  if (adapterInfo.tensorInputs && typeof adapterInfo.tensorInputs === "object") {
    return adapterInfo.tensorInputs;
  }
  const out = {};
  for (const module of Array.isArray(adapterInfo.modules) ? adapterInfo.modules : []) {
    const inFeatures = Number(module?.inFeatures || 0);
    const outFeatures = Number(module?.outFeatures || 0);
    const rank = Number(module?.rank || 0);
    if (inFeatures <= 0 || outFeatures <= 0 || rank <= 0) continue;
    const loraA = module.loraA instanceof Float32Array ? module.loraA : new Float32Array(module.loraA || []);
    const rawB = module.loraB instanceof Float32Array ? module.loraB : new Float32Array(module.loraB || []);
    const scale = Number(module?.scale || adapterInfo.scale || 1);
    const scaledB =
      scale === 1
        ? rawB
        : Float32Array.from(rawB, (value) => Math.fround(value * scale));
    out[module.loraInputA] = new Tensor("float32", loraA, [inFeatures, rank]);
    out[module.loraInputB] = new Tensor("float32", scaledB, [rank, outFeatures]);
  }
  adapterInfo.tensorInputs = out;
  return out;
}

function applyRepetitionPenalty(logitsRow, seenTokenIds, penalty) {
  const adjusted = new Float32Array(logitsRow);
  const safePenalty = Number(penalty);
  if (!(safePenalty > 0) || safePenalty === 1) {
    return adjusted;
  }
  const uniqueIds = new Set(normalizeTokenIdList(seenTokenIds));
  for (const tokenId of uniqueIds) {
    if (tokenId < 0 || tokenId >= adjusted.length) continue;
    const value = adjusted[tokenId];
    adjusted[tokenId] = value < 0 ? value * safePenalty : value / safePenalty;
  }
  return adjusted;
}

function argmaxLogitIndex(logitsRow) {
  let bestIndex = 0;
  let bestValue = -Infinity;
  for (let index = 0; index < logitsRow.length; index += 1) {
    const value = logitsRow[index];
    if (value > bestValue) {
      bestValue = value;
      bestIndex = index;
    }
  }
  return bestIndex;
}

function sampleFromNormalizedEntries(entries) {
  let total = 0;
  for (const entry of entries) total += entry.probability;
  if (!(total > 0)) {
    return entries.length ? entries[0].index : 0;
  }
  const draw = Math.random() * total;
  let running = 0;
  for (const entry of entries) {
    running += entry.probability;
    if (draw <= running) return entry.index;
  }
  return entries.length ? entries[entries.length - 1].index : 0;
}

function sampleLocalQwenNextToken(logitsRow, generationConfig, generatedTokenIds) {
  const adjustedForPenalty = applyRepetitionPenalty(
    logitsRow,
    generatedTokenIds,
    generationConfig?.repetition_penalty,
  );
  if (generationConfig?.do_sample !== true) {
    return argmaxLogitIndex(adjustedForPenalty);
  }

  const temperature = Number(generationConfig?.temperature);
  const safeTemperature = Number.isFinite(temperature) && temperature > 0 ? temperature : 1;
  const sortedEntries = [];
  for (let index = 0; index < adjustedForPenalty.length; index += 1) {
    const rawLogit = adjustedForPenalty[index];
    if (!Number.isFinite(rawLogit)) continue;
    sortedEntries.push({
      index,
      logit: rawLogit / safeTemperature,
    });
  }
  if (!sortedEntries.length) {
    return argmaxLogitIndex(logitsRow);
  }

  sortedEntries.sort((left, right) => right.logit - left.logit);

  const topK = Number(generationConfig?.top_k);
  const limitedByTopK =
    Number.isFinite(topK) && topK > 0 && topK < sortedEntries.length
      ? sortedEntries.slice(0, Math.trunc(topK))
      : sortedEntries;

  const maxLogit = limitedByTopK[0]?.logit ?? 0;
  const weightedEntries = limitedByTopK.map((entry) => ({
    index: entry.index,
    probability: Math.exp(entry.logit - maxLogit),
  }));

  const topP = Number(generationConfig?.top_p);
  if (Number.isFinite(topP) && topP > 0 && topP < 1) {
    const total = weightedEntries.reduce((sum, entry) => sum + entry.probability, 0);
    if (total > 0) {
      let cumulative = 0;
      const kept = [];
      for (const entry of weightedEntries) {
        kept.push(entry);
        cumulative += entry.probability / total;
        if (cumulative >= topP) break;
      }
      return sampleFromNormalizedEntries(kept);
    }
  }

  return sampleFromNormalizedEntries(weightedEntries);
}

function getLocalQwenStopTokenIds(pipeline, generationConfig) {
  const ids = new Set([
    ...normalizeTokenIdList(generationConfig?.eos_token_id),
    ...normalizeTokenIdList(pipeline?.model?.generation_config?.eos_token_id),
    ...normalizeTokenIdList(pipeline?.tokenizer?.eos_token_id),
  ]);
  return [...ids];
}

async function runLocalQwenGenerationWithAdapter({
  pipeline,
  rawChatInputs,
  generationConfig,
  adapterInfo = null,
}) {
  const promptTokenIds = copyLocalQwenTokenRow(rawChatInputs?.promptInputs?.input_ids);
  if (!promptTokenIds.length) {
    throw new Error("Local Qwen prompt encoding returned no input_ids.");
  }
  const encodedAttentionMask = copyLocalQwenTokenRow(rawChatInputs?.promptInputs?.attention_mask);
  const promptAttentionMask =
    encodedAttentionMask.length === promptTokenIds.length
      ? encodedAttentionMask
      : new Array(promptTokenIds.length).fill(1);

  const generatedTokenIds = [...promptTokenIds];
  const attentionMask = [...promptAttentionMask];
  const stopTokenIds = new Set(getLocalQwenStopTokenIds(pipeline, generationConfig));
  const maxNewTokens = Math.max(1, Number(generationConfig?.max_new_tokens || 1));
  const loraInputs = buildLocalQwenTransformerLoraInputs(adapterInfo);

  for (let step = 0; step < maxNewTokens; step += 1) {
    const currentPromptInputs = buildLocalQwenPromptInputs(
      rawChatInputs?.promptInputs,
      generatedTokenIds,
      attentionMask,
    );
    const chatInputs = prepareLocalQwenModelInputs(pipeline.model, {
      ...currentPromptInputs,
      ...loraInputs,
    });
    const outputs = await pipeline.model(chatInputs);
    const finalLogits = extractLastLogitsRow(outputs?.logits);
    const nextTokenId = sampleLocalQwenNextToken(finalLogits, generationConfig, generatedTokenIds);
    generatedTokenIds.push(nextTokenId);
    attentionMask.push(1);
    if (stopTokenIds.has(nextTokenId)) {
      break;
    }
  }

  return {
    text: decodeLocalQwenGeneratedText(
      pipeline.tokenizer,
      rawChatInputs.promptInputs,
      [generatedTokenIds.map((value) => BigInt(value))],
    ),
    inputMode: rawChatInputs.inputMode,
    imageCount: rawChatInputs.imageCount,
    promptInputKeys: rawChatInputs.promptInputKeys,
    visionPreprocess: rawChatInputs.visionPreprocess || pipeline.visionPreprocess || null,
    adapter: adapterInfo && typeof adapterInfo === "object"
      ? {
          applied: adapterInfo.applied === true,
          kind: String(adapterInfo.kind || ""),
          craftId: String(adapterInfo.craftId || ""),
          artifactId: String(adapterInfo.artifactId || ""),
          updatedAt: String(adapterInfo.updatedAt || ""),
          status: String(adapterInfo.status || ""),
          modelName: String(adapterInfo.modelName || ""),
        }
      : null,
  };
}

async function runLocalQwenGeneration(
  pipeline,
  runtimeModelId,
  executionPlan,
  normalizedMessages,
  generationConfig,
  options = {},
) {
  const usesVisionProcessor = hasLocalQwenVisionInputs(normalizedMessages);
  const rawChatInputs = usesVisionProcessor
    ? await (async () => {
        const { processor, visionPreprocess } = await ensureLocalQwenVisionProcessor(
          pipeline,
          runtimeModelId,
          executionPlan,
        );
        const inputs = await buildLocalQwenVisionInputs(
          processor,
          normalizedMessages,
          options?.promptText,
        );
        if (!inputs.visionPreprocess && visionPreprocess) {
          inputs.visionPreprocess = visionPreprocess;
        }
        return inputs;
      })()
    : buildLocalQwenTextInputs(pipeline.tokenizer, normalizedMessages, options?.promptText);
  const adapterInfo = options?.adapterInfo?.applied ? options.adapterInfo : null;
  return await runLocalQwenGenerationWithAdapter({
    pipeline,
    rawChatInputs,
    generationConfig,
    adapterInfo,
  });
}

async function handleLocalQwenChatLocal({ modelName, messages, parameters, craftId = "", promptText = "" }) {
  const runtimePlan = getLocalQwenRuntimePlan(modelName);
  const normalizedMessages = prepareLocalQwenMessages(messages, parameters);
  if (!normalizedMessages.length && !asLocalQwenText(promptText)) {
    throw new Error("Local Qwen erwartet mindestens eine Textnachricht.");
  }

  const generationConfig = buildLocalQwenGenerationConfig(parameters);
  const reasoningMode = getLocalQwenReasoningMode(parameters);
  const usesVisionInputs = hasLocalQwenVisionInputs(normalizedMessages);
  const adapterInfo = await getLocalQwenInferenceAdapter(craftId, runtimePlan.runtimeModelId);
  const failures = [];
  const failureDetails = [];

  for (const executionPlan of prioritizeLocalQwenExecutionPlans(runtimePlan)) {
    const attemptRuntimeModelId = getLocalQwenExecutionPlanRuntimeModelId(
      runtimePlan.runtimeModelId,
      executionPlan,
    );
    const visionPreprocessPolicy = usesVisionInputs
      ? getLocalQwenVisionPreprocessPolicy(attemptRuntimeModelId, executionPlan)
      : null;
    if (adapterInfo?.applied === true && attemptRuntimeModelId !== runtimePlan.runtimeModelId) {
      const message = `Reference fallback skipped because trained adapter weights target ${runtimePlan.runtimeModelId}.`;
      failures.push(`${executionPlan.label}: ${message}`);
      failureDetails.push({
        stage: "adapter_guard",
        runtimeModelId: attemptRuntimeModelId,
        executionPlan,
        visionPreprocessPolicy,
        error: {
          name: "AdapterGuardError",
          message,
          stack: "",
          cause: null,
          props: {},
        },
      });
      continue;
    }
    const attemptAdapterInfo =
      attemptRuntimeModelId === runtimePlan.runtimeModelId ? adapterInfo : null;
    const cacheKey = getLocalQwenAttemptKey(runtimePlan.runtimeModelId, executionPlan);
    let stage = "pipeline_init";
    try {
      console.log(
        "[offscreen] local_qwen generation attempt",
        attemptRuntimeModelId,
        executionPlan.device,
        getLocalQwenDtypeKey(executionPlan.dtype),
      );
      stage = "generation";
      const generation = await withLocalQwenPipeline(
        attemptRuntimeModelId,
        executionPlan,
        async (pipeline) => await runLocalQwenGeneration(
          pipeline,
          attemptRuntimeModelId,
          executionPlan,
          normalizedMessages,
          generationConfig,
          {
            adapterInfo: attemptAdapterInfo,
            promptText: asLocalQwenText(promptText),
          },
        ),
      );
      const text = stripLocalQwenThinkingTrace(generation.text);
      localQwenPreferredExecutionPlan.set(runtimePlan.runtimeModelId, executionPlan);
      return {
        text,
        runtime: {
          ...runtimePlan,
          runtimeModelId: attemptRuntimeModelId,
          runtimeDisplayName:
            attemptRuntimeModelId === runtimePlan.runtimeModelId
              ? runtimePlan.runtimeDisplayName
              : attemptRuntimeModelId,
          executionPlan,
          normalizedMessageCount: normalizedMessages.length,
          reasoningMode,
          inputMode: generation.inputMode,
          imageCount: generation.imageCount,
          promptInputKeys: generation.promptInputKeys,
          visionPreprocess: generation.visionPreprocess || visionPreprocessPolicy,
          adapter: generation.adapter || (
            attemptAdapterInfo && typeof attemptAdapterInfo === "object"
              ? {
                  applied: attemptAdapterInfo.applied === true,
                  craftId: String(attemptAdapterInfo.craftId || ""),
                  artifactId: String(attemptAdapterInfo.artifactId || ""),
                  updatedAt: String(attemptAdapterInfo.updatedAt || ""),
                  status: String(attemptAdapterInfo.status || ""),
                  modelName: String(attemptAdapterInfo.modelName || ""),
                }
              : null
          ),
        },
      };
    } catch (error) {
      await dropLocalQwenPipelineCacheEntry(cacheKey);
      const preferred = localQwenPreferredExecutionPlan.get(runtimePlan.runtimeModelId);
      if (preferred && sameLocalQwenExecutionPlan(preferred, executionPlan)) {
        localQwenPreferredExecutionPlan.delete(runtimePlan.runtimeModelId);
      }
      const classifiedFailure = classifyLocalQwenAttemptFailure({
        errorInfo: serializeError(error),
        stage,
        usesVisionInputs,
      });
      const diagnostics = await collectLocalQwenDiagnostics();
      const message = classifiedFailure.error.message || String(error || "");
      const detail = {
        stage,
        runtimeModelId: attemptRuntimeModelId,
        executionPlan,
        visionPreprocessPolicy,
        reason: classifiedFailure.reason,
        numericCode: classifiedFailure.numericCode || null,
        usesVisionInputs,
        error: classifiedFailure.error,
        diagnostics,
      };
      console.warn("[offscreen] local_qwen attempt failed", detail);
      failures.push(`${executionPlan.label}: ${message}`);
      failureDetails.push(detail);
    }
  }

  const finalError = new Error(
    `Local Qwen runtime failed for ${runtimePlan.runtimeModelId}. ${failures.join(" | ")}`,
  );
  finalError.detail = {
    runtimePlan,
    normalizedMessageCount: normalizedMessages.length,
    generationConfig,
    reasoningMode,
    usesVisionInputs,
    craftId: String(craftId || "").trim(),
    adapter: adapterInfo && typeof adapterInfo === "object"
      ? {
          applied: adapterInfo.applied === true,
          artifactId: String(adapterInfo.artifactId || ""),
          updatedAt: String(adapterInfo.updatedAt || ""),
          status: String(adapterInfo.status || ""),
          modelName: String(adapterInfo.modelName || ""),
        }
      : null,
    reason: asText(failureDetails[0]?.reason),
    numericCode: Number(failureDetails[0]?.numericCode || 0) || null,
    failures: failureDetails,
    finalDiagnostics: await collectLocalQwenDiagnostics(),
  };
  throw finalError;
}

async function handleLocalQwenChat(args = {}) {
  const remoteResult = await maybeHandleRemoteLocalQwenChat(args);
  if (remoteResult) {
    return remoteResult;
  }
  return await handleLocalQwenChatLocal(args);
}

async function handleLocalQwenForwardBenchmark({
  modelName,
  messages,
  parameters = {},
  benchmarkId = "",
  promptText = "",
  iterations = 6,
  warmupIterations = 1,
} = {}) {
  const runtimePlan = getLocalQwenRuntimePlan(modelName);
  const normalizedMessages = prepareLocalQwenMessages(messages, parameters);
  const failures = [];
  const failureDetails = [];

  for (const executionPlan of prioritizeLocalQwenExecutionPlans(runtimePlan)) {
    const attemptRuntimeModelId = getLocalQwenExecutionPlanRuntimeModelId(
      runtimePlan.runtimeModelId,
      executionPlan,
    );
    const cacheKey = getLocalQwenAttemptKey(runtimePlan.runtimeModelId, executionPlan);
    let stage = "pipeline_init";
    try {
      stage = "forward_benchmark";
      const benchmark = await withLocalQwenPipeline(
        attemptRuntimeModelId,
        executionPlan,
        async (pipeline) => await benchmarkLocalQwenForwardOnPipeline({
          pipeline,
          runtimeModelId: attemptRuntimeModelId,
          executionPlan,
          normalizedMessages,
          benchmarkId: asText(benchmarkId),
          promptText: asLocalQwenText(promptText),
          iterations,
          warmupIterations,
        }),
      );
      localQwenPreferredExecutionPlan.set(runtimePlan.runtimeModelId, executionPlan);
      return {
        benchmark,
        runtime: {
          ...runtimePlan,
          runtimeModelId: attemptRuntimeModelId,
          runtimeDisplayName:
            attemptRuntimeModelId === runtimePlan.runtimeModelId
              ? runtimePlan.runtimeDisplayName
              : attemptRuntimeModelId,
          executionPlan,
          normalizedMessageCount: normalizedMessages.length,
          benchmarkMode: benchmark.mode,
          inputMode: benchmark.inputMode,
          imageCount: benchmark.imageCount,
          promptInputKeys: benchmark.promptInputKeys,
        },
      };
    } catch (error) {
      await dropLocalQwenPipelineCacheEntry(cacheKey);
      const preferred = localQwenPreferredExecutionPlan.get(runtimePlan.runtimeModelId);
      if (preferred && sameLocalQwenExecutionPlan(preferred, executionPlan)) {
        localQwenPreferredExecutionPlan.delete(runtimePlan.runtimeModelId);
      }
      const classifiedFailure = classifyLocalQwenAttemptFailure({
        errorInfo: serializeError(error),
        stage,
        usesVisionInputs: hasLocalQwenVisionInputs(normalizedMessages),
      });
      const diagnostics = await collectLocalQwenDiagnostics();
      const message = classifiedFailure.error.message || String(error || "");
      const detail = {
        stage,
        runtimeModelId: attemptRuntimeModelId,
        executionPlan,
        reason: classifiedFailure.reason,
        numericCode: classifiedFailure.numericCode || null,
        error: classifiedFailure.error,
        diagnostics,
      };
      failures.push(`${executionPlan.label}: ${message}`);
      failureDetails.push(detail);
      console.warn("[offscreen] local_qwen forward benchmark attempt failed", detail);
    }
  }

  const combinedError = failures.join(" | ") || "Local Qwen forward benchmark failed.";
  const error = new Error(combinedError);
  error.detail = {
    runtimePlan,
    failures: failureDetails,
  };
  throw error;
}

// ---------- 1) Embedding-Pipeline (Qwen) ----------

const embedPipelineState = {
  promise: null,
  activeUses: 0,
  disposeTimer: 0,
};

function clearEmbedPipelineDisposeTimer() {
  const timerId = Number(embedPipelineState.disposeTimer || 0);
  if (!Number.isFinite(timerId) || timerId <= 0) return;
  clearTimeout(timerId);
  embedPipelineState.disposeTimer = 0;
}

async function disposeEmbedPipeline() {
  clearEmbedPipelineDisposeTimer();
  const pipelinePromise = embedPipelineState.promise;
  embedPipelineState.promise = null;
  if (!pipelinePromise) return;
  const pipe = await Promise.resolve(pipelinePromise).catch(() => null);
  await disposeIfSupported(pipe);
}

function scheduleEmbedPipelineDispose() {
  clearEmbedPipelineDisposeTimer();
  if (embedPipelineState.activeUses > 0 || !embedPipelineState.promise) return;
  embedPipelineState.disposeTimer = globalThis.setTimeout(() => {
    embedPipelineState.disposeTimer = 0;
    void disposeEmbedPipeline();
  }, EMBED_PIPELINE_IDLE_TTL_MS);
}

async function getEmbedPipeline() {
  clearEmbedPipelineDisposeTimer();
  if (!embedPipelineState.promise) {
    console.log("[offscreen] loading Qwen3-Embedding-0.6B-ONNX pipeline...");
    embedPipelineState.promise = pipeline(
      "feature-extraction",
      "onnx-community/Qwen3-Embedding-0.6B-ONNX",
      {
        device: "wasm",
        dtype: "q8",
      }
    ).catch((error) => {
      embedPipelineState.promise = null;
      throw error;
    });
  }
  return await embedPipelineState.promise;
}

async function withEmbedPipeline(callback) {
  embedPipelineState.activeUses += 1;
  try {
    const pipe = await getEmbedPipeline();
    return await callback(pipe);
  } finally {
    embedPipelineState.activeUses = Math.max(0, embedPipelineState.activeUses - 1);
    if (embedPipelineState.activeUses === 0) {
      scheduleEmbedPipelineDispose();
    }
  }
}

async function handleEmbedText(textOrTexts) {
  return await withEmbedPipeline(async (pipe) => {
    const output = await pipe(textOrTexts, {
      pooling: "mean",
      normalize: true,
    });

    if (Array.isArray(output)) {
      const tensors = output;
      const embeddings = tensors.map((t, idx) => {
        const data = t.data ?? t;
        const arr = Array.from(data);
        console.log(`[offscreen] embedding[${idx}] length:`, arr.length);
        return arr;
      });

      console.log("[offscreen] embeddings batch size:", embeddings.length);
      return embeddings;
    }

    const tensor = output;
    const data = tensor.data ?? tensor;
    const embedding = Array.from(data);

    console.log("[offscreen] embedding tensor dims:", tensor.dims);
    console.log("[offscreen] embedding length:", embedding.length);

    return embedding;
  });
}



// ---------- 2) Florence2-Pipeline ----------

const FLORENCE_MODEL_ID = "onnx-community/Florence-2-base-ft";

const florenceRuntimeState = {
  loadPromise: null,
  modelPromise: null,
  processorPromise: null,
  tokenizerPromise: null,
  activeUses: 0,
  disposeTimer: 0,
};

function clearFlorenceDisposeTimer() {
  const timerId = Number(florenceRuntimeState.disposeTimer || 0);
  if (!Number.isFinite(timerId) || timerId <= 0) return;
  clearTimeout(timerId);
  florenceRuntimeState.disposeTimer = 0;
}

async function disposeFlorenceRuntime() {
  clearFlorenceDisposeTimer();
  const modelPromise = florenceRuntimeState.modelPromise;
  const processorPromise = florenceRuntimeState.processorPromise;
  const tokenizerPromise = florenceRuntimeState.tokenizerPromise;
  florenceRuntimeState.loadPromise = null;
  florenceRuntimeState.modelPromise = null;
  florenceRuntimeState.processorPromise = null;
  florenceRuntimeState.tokenizerPromise = null;
  const settled = await Promise.allSettled([
    Promise.resolve(modelPromise).catch(() => null),
    Promise.resolve(processorPromise).catch(() => null),
    Promise.resolve(tokenizerPromise).catch(() => null),
  ]);
  for (const entry of settled) {
    if (entry.status !== "fulfilled") continue;
    await disposeIfSupported(entry.value);
  }
}

function scheduleFlorenceRuntimeDispose() {
  clearFlorenceDisposeTimer();
  if (florenceRuntimeState.activeUses > 0 || !florenceRuntimeState.loadPromise) return;
  florenceRuntimeState.disposeTimer = globalThis.setTimeout(() => {
    florenceRuntimeState.disposeTimer = 0;
    void disposeFlorenceRuntime();
  }, FLORENCE_RUNTIME_IDLE_TTL_MS);
}

async function loadFlorence() {
  clearFlorenceDisposeTimer();
  if (!florenceRuntimeState.loadPromise) {
    console.log("[offscreen] loading Florence-2 model...");
    florenceRuntimeState.modelPromise = Florence2ForConditionalGeneration.from_pretrained(
      FLORENCE_MODEL_ID,
      {
        dtype: {
          embed_tokens: "fp16",
          vision_encoder: "fp16",
          encoder_model: "q4",
          decoder_model_merged: "q4",
        },
        device: "webgpu",
      }
    );
    florenceRuntimeState.processorPromise = AutoProcessor.from_pretrained(FLORENCE_MODEL_ID);
    florenceRuntimeState.tokenizerPromise = AutoTokenizer.from_pretrained(FLORENCE_MODEL_ID);
    florenceRuntimeState.loadPromise = Promise.all([
      florenceRuntimeState.modelPromise,
      florenceRuntimeState.processorPromise,
      florenceRuntimeState.tokenizerPromise,
    ]).then(([model, processor, tokenizer]) => ({ model, processor, tokenizer })).catch((error) => {
      florenceRuntimeState.loadPromise = null;
      florenceRuntimeState.modelPromise = null;
      florenceRuntimeState.processorPromise = null;
      florenceRuntimeState.tokenizerPromise = null;
      throw error;
    });
  }
  return await florenceRuntimeState.loadPromise;
}

async function withFlorenceRuntime(callback) {
  florenceRuntimeState.activeUses += 1;
  try {
    const runtime = await loadFlorence();
    return await callback(runtime);
  } finally {
    florenceRuntimeState.activeUses = Math.max(0, florenceRuntimeState.activeUses - 1);
    if (florenceRuntimeState.activeUses === 0) {
      scheduleFlorenceRuntimeDispose();
    }
  }
}

function dataUrlToBlob(dataUrl) {
  const s = String(dataUrl || "");
  const m = s.match(/^data:([^;,]+)?(;base64)?,(.*)$/i);
  if (!m) throw new Error("Invalid data URL");
  const mime = m[1] || "application/octet-stream";
  const isBase64 = !!m[2];
  const payload = m[3] || "";

  let bytes;
  if (isBase64) {
    const bin = atob(payload);
    const arr = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i += 1) arr[i] = bin.charCodeAt(i);
    bytes = arr;
  } else {
    const decoded = decodeURIComponent(payload);
    const arr = new Uint8Array(decoded.length);
    for (let i = 0; i < decoded.length; i += 1) arr[i] = decoded.charCodeAt(i);
    bytes = arr;
  }
  return new Blob([bytes], { type: mime });
}

async function handleFlorenceTask({ imageUrl, task }) {
  return await withFlorenceRuntime(async ({ model, processor, tokenizer }) => {
    let image = null;
    if (String(imageUrl || "").startsWith("data:")) {
      const blob = dataUrlToBlob(imageUrl);
      image = await RawImage.fromBlob(blob);
    } else {
      image = await RawImage.fromURL(imageUrl);
    }
    const visionInputs = await processor(image);

    const prompts = processor.construct_prompts(task);
    const textInputs = tokenizer(prompts);

    const generatedIds = await model.generate({
      ...textInputs,
      ...visionInputs,
      max_new_tokens: 128,
    });

    const generatedText = tokenizer.batch_decode(generatedIds, {
      skip_special_tokens: false,
    })[0];

    const result = processor.post_process_generation(
      generatedText,
      task,
      image.size
    );

    return result;
  });
}

// ---------- 3) Message-Handling im Offscreen-Dokument ----------

void initPeerComputeRuntime().catch((error) => {
  console.warn("[offscreen] peer compute runtime bootstrap failed", error);
});

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === "OFFSCREEN_REMOTE_COMPUTE_BOOTSTRAP") {
    (async () => {
      try {
        await initPeerComputeRuntime();
        const settings = await craftSync?.readSettings?.();
        sendResponse({
          ok: true,
          remoteCompute: {
            running: Boolean(peerComputeState.bus),
            mode: asText(settings?.mode),
            computeOfferEnabled: normalizeRemoteComputeSettings(settings).computeOfferEnabled,
            remoteExecutionEnabled: normalizeRemoteComputeSettings(settings).remoteExecutionEnabled,
            peerCount: peerComputePeersByDeviceId.size,
            activeJobCount: peerComputeLocalActiveJobs.size,
          },
        });
      } catch (err) {
        sendResponse({
          ok: false,
          error: String(err?.message || err || "Remote compute bootstrap failed."),
        });
      }
    })();
    return true;
  }

  // Embeddings via OFFSCREEN_EMBED_TEXT (vom Service Worker geroutet)
if (msg.type === "OFFSCREEN_EMBED_TEXT") {
  (async () => {
    try {
      console.log("[offscreen] OFFSCREEN_EMBED_TEXT received:", msg.text);
      const embedding = await handleEmbedText(msg.text || "");
      console.log("[offscreen] embedding length:", embedding?.length);

      // Convert to a plain array before sending it across the message API.
      const serializableEmbedding =
        embedding && typeof embedding.length === "number"
          ? Array.from(embedding)
          : embedding;

      sendResponse({ ok: true, embedding: serializableEmbedding });
    } catch (err) {
      console.error("[offscreen] embed error:", err);
      sendResponse({ ok: false, error: String(err) });
    }
  })();
  return true;
}


  // Florence2-Tasks
  if (msg.type === "OFFSCREEN_FLORENCE2_RUN") {
    (async () => {
      try {
        console.log("[offscreen] handling OFFSCREEN_FLORENCE2_RUN");
        const result = await handleFlorenceTask({
          imageUrl: msg.imageUrl,
          task: msg.task || "<MORE_DETAILED_CAPTION>",
        });
        sendResponse({ ok: true, result });
      } catch (err) {
        console.error("[offscreen] florence2 error:", err);
        sendResponse({ ok: false, error: String(err) });
      }
    })();
    return true;
  }

  if (msg.type === "OFFSCREEN_LOCAL_QWEN_CHAT") {
    (async () => {
      try {
        const result = await handleLocalQwenChat({
          modelName: msg.modelName,
          messages: Array.isArray(msg.messages) ? msg.messages : [],
          parameters: msg.parameters && typeof msg.parameters === "object" ? msg.parameters : {},
          craftId: String(msg.craftId || "").trim(),
          promptText: String(msg.promptText || "").trim(),
        });
        sendResponse({ ok: true, ...result });
      } catch (err) {
        console.error("[offscreen] local_qwen error:", err);
        sendResponse({
          ok: false,
          error: String(err?.message || err),
          errorDetail: err?.detail || serializeError(err),
        });
      }
    })();
    return true;
  }

  if (msg.type === "OFFSCREEN_LOCAL_QWEN_FORWARD_BENCHMARK") {
    (async () => {
      try {
        const result = await handleLocalQwenForwardBenchmark({
          modelName: msg.modelName,
          messages: Array.isArray(msg.messages) ? msg.messages : [],
          parameters: msg.parameters && typeof msg.parameters === "object" ? msg.parameters : {},
          benchmarkId: String(msg.benchmarkId || "").trim(),
          promptText: String(msg.promptText || "").trim(),
          iterations: Number(msg.iterations || 0),
          warmupIterations: Number(msg.warmupIterations || 0),
        });
        sendResponse({ ok: true, ...result });
      } catch (err) {
        console.error("[offscreen] local_qwen forward benchmark error:", err);
        sendResponse({
          ok: false,
          error: String(err?.message || err),
          errorDetail: err?.detail || serializeError(err),
        });
      }
    })();
    return true;
  }

  if (msg.type === "OFFSCREEN_LOCAL_QWEN_TRAINING_START") {
    (async () => {
      try {
        const run = await startLocalQwenTrainingRun({
          craftId: msg.craftId,
          shardId: msg.shardId,
          modelName: msg.modelName,
          datasetPayload: msg.datasetPayload,
          persistBundle: msg.persistBundle !== false,
          smokeMode: msg.smokeMode,
          configOverrides: msg.configOverrides,
        });
        sendResponse({ ok: true, run });
      } catch (err) {
        console.error("[offscreen] local_qwen training start error:", err);
        sendResponse({
          ok: false,
          error: String(err?.message || err),
          errorDetail: serializeError(err),
        });
      }
    })();
    return true;
  }

  if (msg.type === "OFFSCREEN_LOCAL_QWEN_TRAINING_STATUS") {
    (async () => {
      try {
        const run = getLocalQwenTrainingRunStatus(msg.jobId);
        if (!run) {
          throw new Error(`Unknown local_qwen training job: ${String(msg.jobId || "").trim()}`);
        }
        sendResponse({ ok: true, run });
      } catch (err) {
        console.error("[offscreen] local_qwen training status error:", err);
        sendResponse({
          ok: false,
          error: String(err?.message || err),
          errorDetail: serializeError(err),
        });
      }
    })();
    return true;
  }

  // andere Message-Typen ignorieren
});
