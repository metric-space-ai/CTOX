import {
  AutoProcessor,
  AutoTokenizer,
  Qwen3_5ForConditionalGeneration,
  RawImage,
  Tensor,
} from "../vendor/transformers-esm.mjs";
import {
  cloneTransformerLoraAdapter,
  createTransformerLoraAdapter,
  createTransformerLoraAdamState,
  applyTransformerLoraSpsaAdamStep,
  getTransformerLoraMutationVersion,
  perturbTransformerLoraAdapter,
  trainTransformerLoraModuleFromGradients,
} from "./transformer_lora_math.mjs";
import {
  createLocalQwenMissingBrowserTrainingManifestError,
  getLocalQwenModelRepoUrl,
  getLocalQwenRuntimePlan,
} from "./local_qwen_runtime.mjs";
import {
  buildEncodedTrainingExample as buildEncodedTrainingExampleContract,
  buildTrainingMessages as buildTrainingMessagesContract,
  collectTrainingImageSources as collectTrainingImageSourcesContract,
  datasetHasVisionPairs as datasetHasVisionPairsContract,
  normalizeTrainingPairRow as normalizeTrainingPairRowContract,
} from "./local_qwen_training_contract.mjs";
import {
  getTransformerLoraTrainingSupportIssue,
} from "./local_qwen_training_guardrails.mjs";
import {
  summarizeBrowserTrainingThroughputMetrics,
} from "./local_qwen_training_metrics.mjs";

export const LOCAL_QWEN_FIXED_TRAINING_CONFIG = Object.freeze({
  profile: "debug_fast",
  datasetPath: "assets/training/solutionfinding1_instruction_dataset.json",
  maxTrainPairs: 8,
  maxValidationPairs: 2,
  maxTestPairs: 4,
  maxSeqLen: 24,
  rank: 16,
  alpha: 32,
  epochs: 1,
  batchTokens: 64,
  modelBatchSize: 4,
  learningRate: 0.0001,
  optimizer: "full_transformer_lora_spsa",
  spsaEpsilon: 0.001,
  spsaSamples: 1,
  spsaGradientClip: 1,
  reasoningMode: "no_think",
  seed: 42,
  trainMatrixA: true,
  trainMatrixB: true,
});

const fixedDatasetPromises = new Map();
const transformerLoraInputCache = new WeakMap();

function asText(value) {
  return String(value == null ? "" : value).trim();
}

async function disposeLocalQwenTrainingResource(resource, label = "resource") {
  if (!resource) return;
  try {
    if (typeof resource.dispose === "function") {
      await resource.dispose();
      return;
    }
    if (typeof resource.release === "function") {
      await resource.release();
    }
  } catch (error) {
    console.warn(`[local-qwen-training] failed to dispose ${asText(label) || "resource"}`, error);
  }
}

async function settleLocalQwenTrainingPromise(promiseLike) {
  if (!promiseLike || typeof promiseLike.then !== "function") return promiseLike || null;
  try {
    return await promiseLike;
  } catch (_error) {
    return null;
  }
}

async function disposeLocalQwenTrainingRuntime(runtime = null) {
  if (!runtime || typeof runtime !== "object") return;

  const lossHelperRuntime = await settleLocalQwenTrainingPromise(runtime.browserTrainingLossPromise);
  const exactTrainingRuntime = await settleLocalQwenTrainingPromise(runtime.exactTrainingPromise);
  const currentDecoderSession = runtime?.model?.sessions?.decoder_model_merged || null;
  const modelHasOwnDispose =
    typeof runtime?.model?.dispose === "function" || typeof runtime?.model?.release === "function";

  if (lossHelperRuntime?.session) {
    await disposeLocalQwenTrainingResource(lossHelperRuntime.session, "browser training loss session");
    lossHelperRuntime.session = null;
  }

  if (exactTrainingRuntime?.lmHeadBackpropSession) {
    await disposeLocalQwenTrainingResource(
      exactTrainingRuntime.lmHeadBackpropSession,
      "lm-head backprop session",
    );
    exactTrainingRuntime.lmHeadBackpropSession = null;
  }

  if (modelHasOwnDispose) {
    await disposeLocalQwenTrainingResource(runtime.model, "training runtime model");
  } else if (exactTrainingRuntime?.trainingSession) {
    await disposeLocalQwenTrainingResource(exactTrainingRuntime.trainingSession, "exact training session");
  } else if (currentDecoderSession) {
    await disposeLocalQwenTrainingResource(currentDecoderSession, "decoder session");
  }

  await disposeLocalQwenTrainingResource(runtime.processor, "training processor");
  await disposeLocalQwenTrainingResource(runtime.tokenizer, "training tokenizer");

  runtime.browserTrainingManifestPromise = null;
  runtime.browserTrainingLossPromise = null;
  runtime.exactTrainingPromise = null;
  runtime.browserTrainingLossIssue = "";
  if (runtime?.model?.sessions && currentDecoderSession) {
    runtime.model.sessions.decoder_model_merged = null;
  }
  runtime.processor = null;
  runtime.tokenizer = null;
  runtime.model = null;
}

function patchLocalQwenTrainingForwardParams(model, extraKeys = []) {
  if (!model || !Array.isArray(model.forward_params)) return model;
  const modelType = String(model?.config?.model_type || "").trim();
  if (!["qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"].includes(modelType)) {
    return model;
  }
  const nextForwardParams = [...model.forward_params];
  for (const key of extraKeys) {
    const normalizedKey = asText(key);
    if (normalizedKey && !nextForwardParams.includes(normalizedKey)) {
      nextForwardParams.push(normalizedKey);
    }
  }
  model.forward_params = nextForwardParams;
  return model;
}

function fileNameFromPath(pathLike = "") {
  return asText(pathLike).split("/").filter(Boolean).pop() || "";
}

function normalizeTransformersExternalDataFile(pathLike = "") {
  const normalizedPath = asText(pathLike);
  if (!normalizedPath) return "";
  if (normalizedPath.endsWith(".onnx.data")) {
    return `${normalizedPath.slice(0, -5)}_data`;
  }
  return normalizedPath;
}

async function fetchJsonNoStore(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Failed to load JSON from ${url}: ${response.status}`);
  }
  return await response.json();
}

async function fetchUint8ArrayNoStore(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Failed to load binary model artifact from ${url}: ${response.status}`);
  }
  return new Uint8Array(await response.arrayBuffer());
}

function decodeFloat32DataUrl(dataUrl) {
  const match = /^data:([^;,]+)?(?:;base64)?,(.*)$/i.exec(String(dataUrl || ""));
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

function tensorDataToFloat32Array(tensorLike) {
  const data = tensorLike?.data ?? tensorLike?.cpuData ?? tensorLike?.ort_tensor?.data;
  if (!data || !ArrayBuffer.isView(data)) {
    return new Float32Array(0);
  }
  if (data instanceof Float32Array) {
    return new Float32Array(data);
  }
  const out = new Float32Array(data.length);
  for (let index = 0; index < data.length; index += 1) {
    out[index] = Number(data[index]);
  }
  return out;
}

function syncScaledTransformerLoraWeights(source, target, scale) {
  const safeScale = Math.fround(Number(scale || 1));
  for (let index = 0; index < target.length; index += 1) {
    target[index] = Math.fround(source[index] * safeScale);
  }
  return target;
}

function createLocalQwenTransformerLoraModuleTensorCache(module, scale) {
  const inFeatures = Number(module?.inFeatures || 0);
  const outFeatures = Number(module?.outFeatures || 0);
  const rank = Number(module?.rank || 0);
  const loraA = module?.loraA instanceof Float32Array ? module.loraA : new Float32Array(module?.loraA || []);
  const rawB = module?.loraB instanceof Float32Array ? module.loraB : new Float32Array(module?.loraB || []);
  const useScaledB = Number(scale || 1) !== 1;
  const scaledB = useScaledB ? new Float32Array(rawB.length) : null;
  if (scaledB) {
    syncScaledTransformerLoraWeights(rawB, scaledB, scale);
  }
  return {
    modulePath: asText(module?.modulePath),
    loraInputA: asText(module?.loraInputA),
    loraInputB: asText(module?.loraInputB),
    loraA,
    rawB,
    scaledB,
    scale: Number(scale || 1),
    tensorA: new Tensor("float32", loraA, [inFeatures, rank]),
    tensorB: new Tensor("float32", scaledB || rawB, [rank, outFeatures]),
  };
}

function releaseLocalQwenTransformerLoraInputs(adapter = null) {
  if (adapter && typeof adapter === "object") {
    transformerLoraInputCache.delete(adapter);
  }
}

function buildLocalQwenTransformerLoraInputs(adapter = null) {
  if (!adapter || String(adapter?.kind || "").trim() !== "transformer_lora") {
    return {};
  }
  const modules = Array.isArray(adapter.modules) ? adapter.modules : [];
  const version = getTransformerLoraMutationVersion(adapter);
  let cached = transformerLoraInputCache.get(adapter);
  const needsFullRebuild =
    !cached ||
    cached.moduleCount !== modules.length;

  if (needsFullRebuild) {
    cached = {
      version: -1,
      moduleCount: modules.length,
      inputs: {},
      modules: new Array(modules.length),
    };
    transformerLoraInputCache.set(adapter, cached);
  }

  for (let moduleIndex = 0; moduleIndex < modules.length; moduleIndex += 1) {
    const module = modules[moduleIndex];
    const inFeatures = Number(module?.inFeatures || 0);
    const outFeatures = Number(module?.outFeatures || 0);
    const rank = Number(module?.rank || 0);
    if (inFeatures <= 0 || outFeatures <= 0 || rank <= 0) continue;
    const scale = Number(module?.scale || adapter.scale || 1);
    const currentLoraA = module?.loraA instanceof Float32Array ? module.loraA : new Float32Array(module?.loraA || []);
    const currentLoraB = module?.loraB instanceof Float32Array ? module.loraB : new Float32Array(module?.loraB || []);
    const cachedModule = cached.modules[moduleIndex];
    const requiresModuleRebuild =
      !cachedModule ||
      cachedModule.modulePath !== asText(module?.modulePath) ||
      cachedModule.loraInputA !== asText(module?.loraInputA) ||
      cachedModule.loraInputB !== asText(module?.loraInputB) ||
      cachedModule.loraA !== currentLoraA ||
      cachedModule.rawB !== currentLoraB ||
      cachedModule.scale !== scale;

    if (requiresModuleRebuild) {
      if (cachedModule) {
        delete cached.inputs[cachedModule.loraInputA];
        delete cached.inputs[cachedModule.loraInputB];
      }
      const nextModuleCache = createLocalQwenTransformerLoraModuleTensorCache(module, scale);
      cached.modules[moduleIndex] = nextModuleCache;
      cached.inputs[nextModuleCache.loraInputA] = nextModuleCache.tensorA;
      cached.inputs[nextModuleCache.loraInputB] = nextModuleCache.tensorB;
      continue;
    }

    if (cached.version !== version && cachedModule.scaledB) {
      syncScaledTransformerLoraWeights(cachedModule.rawB, cachedModule.scaledB, cachedModule.scale);
    }
  }
  cached.version = version;
  return cached.inputs;
}

function countTransformerLoraAdapterBytes(adapter = null) {
  let total = 0;
  for (const module of Array.isArray(adapter?.modules) ? adapter.modules : []) {
    if (module?.loraA instanceof Float32Array) total += module.loraA.byteLength;
    if (module?.loraB instanceof Float32Array) total += module.loraB.byteLength;
  }
  return total;
}

function getExecutionPlanDtypeKey(dtype) {
  if (!dtype) return "auto";
  if (typeof dtype === "string") return dtype;
  try {
    return JSON.stringify(dtype);
  } catch (_error) {
    return String(dtype);
  }
}

function createInt64Tensor(values, shape) {
  return new Tensor(
    "int64",
    BigInt64Array.from(values, (value) => BigInt(Number(value))),
    shape,
  );
}

function getOrtTensor(tensorLike) {
  return tensorLike?.ort_tensor ?? tensorLike ?? null;
}

function normalizeOrtTensorElementType(rawType = "") {
  const normalized = asText(rawType).toLowerCase();
  if (!normalized) return "";
  if (normalized.includes("bfloat16")) return "bfloat16";
  if (normalized.includes("float16")) return "float16";
  if (
    normalized === "float" ||
    normalized.includes("float32") ||
    normalized.includes("tensor(float)")
  ) {
    return "float32";
  }
  if (normalized.includes("int64")) return "int64";
  if (normalized.includes("int32")) return "int32";
  return normalized;
}

function inferTensorElementType(tensorLike = null) {
  for (const candidate of [
    tensorLike?.type,
    tensorLike?.ort_tensor?.type,
  ]) {
    const normalized = normalizeOrtTensorElementType(candidate);
    if (normalized) return normalized;
  }

  const data = tensorLike?.data ?? tensorLike?.cpuData ?? tensorLike?.ort_tensor?.data;
  if (!data || !ArrayBuffer.isView(data)) return "";
  if (data instanceof Float32Array) return "float32";
  if (typeof Float16Array !== "undefined" && data instanceof Float16Array) return "float16";
  if (data instanceof Int32Array) return "int32";
  if (data instanceof BigInt64Array) return "int64";
  return "";
}

function getTensorLikeDims(tensorLike = null) {
  const dims = Array.isArray(tensorLike?.dims)
    ? tensorLike.dims
    : Array.isArray(tensorLike?.ort_tensor?.dims)
      ? tensorLike.ort_tensor.dims
      : [];
  return dims.map((value) => Number(value));
}

function getSessionInputElementType(session = null, inputName = "") {
  const normalizedInputName =
    asText(inputName) ||
    (Array.isArray(session?.inputNames) && session.inputNames.length
      ? asText(session.inputNames[0])
      : "");
  if (!normalizedInputName) return "";

  const metadata =
    session?.inputMetadata && typeof session.inputMetadata === "object"
      ? session.inputMetadata[normalizedInputName]
      : null;
  for (const candidate of [
    metadata?.type,
    metadata?.tensorType,
    metadata?.dataType,
    metadata?.elementType,
  ]) {
    const normalized = normalizeOrtTensorElementType(candidate);
    if (normalized) return normalized;
  }

  const inputNames = Array.isArray(session?.inputNames)
    ? session.inputNames.map((name) => asText(name))
    : [];
  const inputIndex = inputNames.indexOf(normalizedInputName);
  if (inputIndex >= 0 && Array.isArray(session?.inputTypes)) {
    return normalizeOrtTensorElementType(session.inputTypes[inputIndex]);
  }
  return "";
}

function coerceTensorLikeForSessionInput(session = null, inputName = "", tensorLike = null) {
  const expectedInputType = getSessionInputElementType(session, inputName);
  const actualType = inferTensorElementType(tensorLike);
  if (expectedInputType !== "float32" || actualType !== "float16") {
    return getOrtTensor(tensorLike);
  }

  const dims = getTensorLikeDims(tensorLike);
  if (!dims.length) {
    return getOrtTensor(tensorLike);
  }

  return new Tensor("float32", tensorDataToFloat32Array(tensorLike), dims).ort_tensor;
}

function tensorLikeToScalarNumber(tensorLike, fallback = 0) {
  if (tensorLike == null) return Number(fallback || 0);
  if (typeof tensorLike === "number") {
    return Number.isFinite(tensorLike) ? tensorLike : Number(fallback || 0);
  }
  const data = tensorLike?.data;
  if (ArrayBuffer.isView(data) && data.length) {
    return Number(data[0]);
  }
  if (Array.isArray(data) && data.length) {
    return Number(data[0]);
  }
  if (typeof tensorLike?.item === "function") {
    return Number(tensorLike.item());
  }
  const numericValue = Number(tensorLike);
  return Number.isFinite(numericValue) ? numericValue : Number(fallback || 0);
}

function sum(values) {
  let total = 0;
  for (const value of values) total += Number(value) || 0;
  return total;
}

function dataUrlToBlob(dataUrl) {
  const match = /^data:([^;,]+)?(?:;base64)?,(.*)$/i.exec(String(dataUrl || ""));
  if (!match) {
    throw new Error("Invalid image data URL for local Qwen training.");
  }
  const mimeType = match[1] || "application/octet-stream";
  const payload = match[2] || "";
  const bytes = Uint8Array.from(atob(payload), (char) => char.charCodeAt(0));
  return new Blob([bytes], { type: mimeType });
}

async function loadTrainingRawImage(imageSource) {
  const source = asText(imageSource);
  if (!source) {
    throw new Error("Local Qwen training vision sample is missing an image source.");
  }
  if (source.startsWith("data:")) {
    return await RawImage.fromBlob(dataUrlToBlob(source));
  }
  return await RawImage.fromURL(source);
}

function normalizeIdSequence(value) {
  if (Array.isArray(value)) return value.flat(Infinity).map((entry) => Number(entry));
  if (ArrayBuffer.isView(value)) return Array.from(value, Number);
  if (value && typeof value.tolist === "function") {
    return normalizeIdSequence(value.tolist());
  }
  if (value?.data && ArrayBuffer.isView(value.data)) {
    return Array.from(value.data, Number);
  }
  return [];
}

function tokenizeToIds(tokenizer, text, options = {}) {
  const result = tokenizer(text, options);
  return normalizeIdSequence(result?.input_ids ?? result);
}

function uniqTextList(values) {
  const out = [];
  const seen = new Set();
  for (const value of Array.isArray(values) ? values : [values]) {
    const text = asText(value);
    if (!text || seen.has(text)) continue;
    seen.add(text);
    out.push(text);
  }
  return out;
}

export function collectTrainingImageSources(row, messages = null) {
  return collectTrainingImageSourcesContract(row, messages);
}

export function buildTrainingMessages(row, prompt = "") {
  return buildTrainingMessagesContract(row, prompt);
}

export function estimateFixedTrainingWorkSamples(config = LOCAL_QWEN_FIXED_TRAINING_CONFIG) {
  const trainCount = Number(config.maxTrainPairs || 0);
  const validationCount = Number(config.maxValidationPairs || config.maxTestPairs || 0);
  const testCount = Number(config.maxTestPairs || 0);
  const epochs = Number(config.epochs || 1);
  const evaluationMode = String(config.evaluationMode || "full").trim().toLowerCase();
  const encodeWork = trainCount + validationCount + testCount;
  const baseEvalWork = (evaluationMode === "holdout_only" ? 0 : trainCount) + validationCount + testCount;
  const epochTrainingWork = epochs * trainCount;
  const epochEvalWork = epochs * ((evaluationMode === "holdout_only" ? 0 : trainCount) + validationCount);
  const finalEvalWork = (evaluationMode === "holdout_only" ? 0 : trainCount) + validationCount + testCount;
  return encodeWork + baseEvalWork + epochTrainingWork + epochEvalWork + finalEvalWork;
}

function serializeTrainingTarget(value) {
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value);
  } catch (_error) {
    return String(value == null ? "" : value);
  }
}

export function normalizeTrainingPairRow(row, index) {
  return normalizeTrainingPairRowContract(row, index);
}

export function datasetHasVisionPairs(pairs = []) {
  return datasetHasVisionPairsContract(pairs);
}

export async function loadFixedTrainingPairs(config = LOCAL_QWEN_FIXED_TRAINING_CONFIG) {
  let dataset = null;
  let datasetSource = "inline";

  if (config?.datasetPayload && typeof config.datasetPayload === "object") {
    dataset = config.datasetPayload;
  } else {
    const datasetUrl = chrome.runtime.getURL(config.datasetPath);
    datasetSource = datasetUrl;
    if (!fixedDatasetPromises.has(datasetUrl)) {
      fixedDatasetPromises.set(
        datasetUrl,
        fetch(datasetUrl).then(async (response) => {
          if (!response.ok) {
            throw new Error(`Failed to load packaged training dataset: ${response.status}`);
          }
          return await response.json();
        }),
      );
    }
    dataset = await fixedDatasetPromises.get(datasetUrl);
  }

  const flatRows = Array.isArray(dataset)
    ? dataset
    : Array.isArray(dataset?.rows)
      ? dataset.rows
      : [];
  const maxTrainPairs = Number(config.maxTrainPairs || LOCAL_QWEN_FIXED_TRAINING_CONFIG.maxTrainPairs || 0);
  const maxValidationPairs = Number(
    config.maxValidationPairs || config.maxTestPairs || LOCAL_QWEN_FIXED_TRAINING_CONFIG.maxValidationPairs || 1,
  );
  const maxTestPairs = Number(config.maxTestPairs || LOCAL_QWEN_FIXED_TRAINING_CONFIG.maxTestPairs || 0);

  const takeTailRows = (rows, desiredCount = 0) => {
    const pool = Array.isArray(rows) ? rows : [];
    const maxRemovable = Math.max(0, pool.length - 1);
    const count = Math.max(0, Math.min(Number(desiredCount || 0), maxRemovable));
    if (!count) return [];
    return pool.splice(pool.length - count, count);
  };

  let trainRows = Array.isArray(dataset?.train) ? dataset.train.slice(0, maxTrainPairs) : [];
  let validationRows = Array.isArray(dataset?.validation) ? dataset.validation.slice(0, maxValidationPairs) : [];
  let testRows = Array.isArray(dataset?.test) ? dataset.test.slice(0, maxTestPairs) : [];

  if (!trainRows.length && flatRows.length) {
    const initialPool = flatRows.slice();
    const defaultTestCount = Math.max(1, Math.min(Math.floor(initialPool.length * 0.2), Math.max(1, initialPool.length - 1)));
    const defaultValidationCount = Math.max(
      1,
      Math.min(Math.floor(initialPool.length * 0.15), Math.max(1, initialPool.length - defaultTestCount - 1)),
    );
    if (!testRows.length) testRows = takeTailRows(initialPool, defaultTestCount).slice(0, maxTestPairs);
    if (!validationRows.length) validationRows = takeTailRows(initialPool, defaultValidationCount).slice(0, maxValidationPairs);
    trainRows = initialPool.slice(0, maxTrainPairs);
  }

  if (!validationRows.length) {
    validationRows = takeTailRows(trainRows, Math.max(1, Math.min(maxValidationPairs, Math.floor(trainRows.length * 0.2) || 1)));
  }
  if (!testRows.length) {
    testRows = takeTailRows(trainRows, Math.max(1, Math.min(maxTestPairs, Math.floor(trainRows.length * 0.2) || 1)));
  }

  if (!trainRows.length && flatRows.length) {
    trainRows = flatRows.slice(0, Math.max(1, Math.min(maxTrainPairs, flatRows.length)));
  }
  if (!trainRows.length && (validationRows.length || testRows.length)) {
    trainRows = (validationRows.length ? validationRows : testRows).slice(0, 1);
  }
  if (!validationRows.length) {
    validationRows = testRows.length ? testRows.slice(0, Math.max(1, maxValidationPairs)) : trainRows.slice(-1);
  }
  if (!testRows.length) {
    testRows = validationRows.length ? validationRows.slice(0, Math.max(1, maxTestPairs)) : trainRows.slice(-1);
  }
  return {
    datasetMeta: dataset?.meta || {},
    datasetSource,
    trainPairs: trainRows.map((row, index) => normalizeTrainingPairRow(row, index)),
    validationPairs: validationRows.map((row, index) =>
      normalizeTrainingPairRow(row, trainRows.length + index)
    ),
    testPairs: testRows.map((row, index) =>
      normalizeTrainingPairRow(row, trainRows.length + validationRows.length + index)
    ),
  };
}

export async function loadLocalQwenTrainingRuntime(modelName, { requiresVision = false } = {}) {
  const runtimePlan = getLocalQwenRuntimePlan(modelName);
  const executionPlan = runtimePlan.executionPlans[0];
  const modelOptions = {
    device: executionPlan.device,
  };
  if (executionPlan.dtype) {
    modelOptions.dtype = executionPlan.dtype;
  }
  const [processor, tokenizer, model] = await Promise.all([
    requiresVision ? AutoProcessor.from_pretrained(runtimePlan.runtimeModelId) : Promise.resolve(null),
    AutoTokenizer.from_pretrained(runtimePlan.runtimeModelId),
    Qwen3_5ForConditionalGeneration.from_pretrained(runtimePlan.runtimeModelId, modelOptions),
  ]);
  return {
    processor,
    tokenizer,
    model,
    runtimePlan,
    executionPlan,
  };
}

async function ensureLocalQwenBrowserTrainingManifest(runtime) {
  if (runtime?.browserTrainingManifestPromise) {
    return await runtime.browserTrainingManifestPromise;
  }
  const loadPromise = (async () => {
    const runtimeModelId = asText(runtime?.runtimePlan?.runtimeModelId);
    if (!runtimeModelId) {
      throw new Error("Browser transformer LoRA training requires a valid runtime model id.");
    }
    const manifestUrl = getLocalQwenModelRepoUrl(runtimeModelId, "lora_training_manifest.json");
    if (!manifestUrl) {
      throw createLocalQwenMissingBrowserTrainingManifestError(runtimeModelId);
    }
    let manifest = null;
    try {
      manifest = await fetchJsonNoStore(manifestUrl);
    } catch (error) {
      if (error instanceof Error && /: 404$/.test(asText(error.message))) {
        throw createLocalQwenMissingBrowserTrainingManifestError(runtimeModelId);
      }
      throw error;
    }
    if (!manifest || !Array.isArray(manifest.modules) || !manifest.modules.length) {
      throw new Error(`Browser transformer LoRA training is unavailable for ${runtimeModelId}: the packaged training manifest exposes no LoRA modules.`);
    }
    return manifest;
  })().catch((error) => {
    runtime.browserTrainingManifestPromise = null;
    throw error;
  });
  runtime.browserTrainingManifestPromise = loadPromise;
  return await loadPromise;
}

async function ensureLocalQwenBrowserTrainingLossRuntime(runtime, manifest = null) {
  if (runtime?.browserTrainingLossPromise) {
    return await runtime.browserTrainingLossPromise;
  }
  const loadPromise = (async () => {
    const runtimeModelId = asText(runtime?.runtimePlan?.runtimeModelId);
    if (!runtimeModelId) return null;
    const browserTrainingManifest =
      manifest && typeof manifest === "object"
        ? manifest
        : await ensureLocalQwenBrowserTrainingManifest(runtime);
    const helperModelFile = asText(browserTrainingManifest?.browserTraining?.lossHelperModelFile);
    if (!helperModelFile) return null;

    const sessionClass = runtime?.model?.sessions?.decoder_model_merged?.constructor;
    if (!sessionClass || typeof sessionClass.create !== "function") {
      runtime.browserTrainingLossIssue =
        "The local Qwen runtime does not expose a reusable ONNX session constructor for browser loss evaluation.";
      return null;
    }

    try {
      const helperModelBytes = await fetchUint8ArrayNoStore(
        getLocalQwenModelRepoUrl(runtimeModelId, helperModelFile),
      );
      const executionProvider = asText(runtime?.executionPlan?.device) || "webgpu";
      const session = await sessionClass.create(helperModelBytes, {
        executionProviders: [executionProvider],
        logSeverityLevel: 3,
        logVerbosityLevel: 0,
      });
      runtime.browserTrainingLossIssue = "";
      return {
        session,
        helperModelFile,
        disabled: false,
        failureReason: "",
      };
    } catch (error) {
      runtime.browserTrainingLossIssue = error instanceof Error ? error.message : String(error);
      return null;
    }
  })().catch((error) => {
    runtime.browserTrainingLossPromise = null;
    throw error;
  });
  runtime.browserTrainingLossPromise = loadPromise;
  return await loadPromise;
}

async function ensureLocalQwenExactTrainingRuntime(runtime) {
  if (runtime?.exactTrainingPromise) {
    return await runtime.exactTrainingPromise;
  }
  const loadPromise = (async () => {
    const runtimeModelId = asText(runtime?.runtimePlan?.runtimeModelId);
    if (!runtimeModelId) return null;
    const manifestUrl = getLocalQwenModelRepoUrl(runtimeModelId, "lora_training_manifest.json");
    if (!manifestUrl) return null;

    let manifest = null;
    try {
      manifest = await fetchJsonNoStore(manifestUrl);
    } catch (_error) {
      return null;
    }
    if (!manifest || manifest.exactTraining?.enabled !== true) {
      return null;
    }

    const exactTraining = manifest.exactTraining;
    const module = (Array.isArray(manifest.modules) ? manifest.modules : []).find(
      (entry) => asText(entry?.modulePath) === asText(exactTraining?.modulePath),
    );
    if (!module) {
      throw new Error(`Exact training module ${asText(exactTraining?.modulePath)} is missing from the training manifest.`);
    }

    const sessionClass = runtime?.model?.sessions?.decoder_model_merged?.constructor;
    if (!sessionClass || typeof sessionClass.create !== "function") {
      throw new Error("The local Qwen runtime does not expose a reusable ONNX session constructor for training.");
    }

    const trainingModelFile = "onnx/decoder_model_merged_q4f16_training.onnx";
    const trainingDataFile = normalizeTransformersExternalDataFile(
      "onnx/decoder_model_merged_q4f16_training.onnx_data",
    );
    const lmHeadModelFile = asText(exactTraining?.lmHead?.helperModelFile) || "onnx/lm_head_backprop_q4.onnx";
    const lmHeadDataFile = normalizeTransformersExternalDataFile(
      asText(exactTraining?.lmHead?.helperDataFile) || "onnx/lm_head_backprop_q4.onnx_data",
    );
    const [
      trainingModelBytes,
      trainingDataBytes,
      lmHeadModelBytes,
      lmHeadDataBytes,
    ] = await Promise.all([
      fetchUint8ArrayNoStore(getLocalQwenModelRepoUrl(runtimeModelId, trainingModelFile)),
      fetchUint8ArrayNoStore(getLocalQwenModelRepoUrl(runtimeModelId, trainingDataFile)),
      fetchUint8ArrayNoStore(getLocalQwenModelRepoUrl(runtimeModelId, lmHeadModelFile)),
      fetchUint8ArrayNoStore(getLocalQwenModelRepoUrl(runtimeModelId, lmHeadDataFile)),
    ]);

    const executionProvider = asText(runtime?.executionPlan?.device) || "webgpu";
    const sessionOptions = {
      executionProviders: [executionProvider],
      // The LoRA training graph intentionally exposes adapter tensors as overridable
      // graph inputs. ORT warns about lost const-folding otherwise, which is expected
      // here and drowns out real runtime errors in the extension console.
      logSeverityLevel: 3,
      logVerbosityLevel: 0,
    };
    const previousDecoderSession = runtime.model.sessions.decoder_model_merged;
    const trainingSession = await sessionClass.create(trainingModelBytes, {
      ...sessionOptions,
      externalData: [
        {
          path: fileNameFromPath(trainingDataFile),
          data: trainingDataBytes,
        },
      ],
    });
    const lmHeadBackpropSession = await sessionClass.create(lmHeadModelBytes, {
      ...sessionOptions,
      externalData: [
        {
          path: fileNameFromPath(lmHeadDataFile),
          data: lmHeadDataBytes,
        },
      ],
    });
    trainingSession.config = {
      ...(previousDecoderSession?.config && typeof previousDecoderSession.config === "object"
        ? previousDecoderSession.config
        : {}),
      kv_cache_dtype:
        asText(previousDecoderSession?.config?.kv_cache_dtype) || "float16",
    };

    runtime.model.sessions.decoder_model_merged = trainingSession;
    patchLocalQwenTrainingForwardParams(runtime.model, [
      asText(module?.loraInputA),
      asText(module?.loraInputB),
    ]);
    if (previousDecoderSession && previousDecoderSession !== trainingSession) {
      try {
        await previousDecoderSession.release?.();
      } catch (_error) {
        // The training session is already installed; a leaked inference session is acceptable here.
      }
    }

    return {
      manifest,
      module,
      exactTraining,
      trainingSession,
      lmHeadBackpropSession,
      finalNormWeight: decodeFloat32DataUrl(exactTraining?.finalNormWeightDataUrl),
    };
  })().catch((error) => {
    runtime.exactTrainingPromise = null;
    throw error;
  });
  runtime.exactTrainingPromise = loadPromise;
  return await loadPromise;
}

export function buildEncodedTrainingExample({
  promptIds,
  promptAttentionMask = null,
  targetIds,
  padId,
  maxLength,
  prompt = "",
  target = "",
  usesVision = false,
  imageSources = [],
  extraInputs = null,
}) {
  return buildEncodedTrainingExampleContract({
    promptIds,
    promptAttentionMask,
    targetIds,
    padId,
    maxLength,
    prompt,
    target,
    usesVision,
    imageSources,
    extraInputs,
  });
}

export function encodePromptTargetExample(
  tokenizer,
  prompt,
  target,
  padId,
  maxLength,
) {
  return buildEncodedTrainingExample({
    promptIds: tokenizeToIds(tokenizer, prompt, { add_special_tokens: true }),
    targetIds: tokenizeToIds(tokenizer, target, { add_special_tokens: false }),
    padId,
    maxLength,
    prompt,
    target,
  });
}

export async function encodeTrainingPairExample({
  tokenizer,
  processor = null,
  pair,
  padId,
  maxLength,
}) {
  if (pair?.usesVision) {
    if (!processor) {
      throw new Error("Local Qwen vision training requires an AutoProcessor.");
    }
    const promptMessages = Array.isArray(pair?.messages) && pair.messages.length
      ? pair.messages
      : buildTrainingMessages(pair, pair?.prompt || "");
    const imageSources = Array.isArray(pair?.imageSources) ? pair.imageSources : collectTrainingImageSources(pair, promptMessages);
    if (!imageSources.length) {
      throw new Error("Vision training row declared vision usage without image sources.");
    }
    const promptText = pair?.renderMode === "qwen_native_multiturn"
      ? String(pair?.prompt || "")
      : typeof processor.apply_chat_template === "function"
        ? processor.apply_chat_template(promptMessages, {
            tokenize: false,
            add_generation_prompt: true,
          })
        : String(pair?.prompt || "");
    const images = await Promise.all(imageSources.map((imageSource) => loadTrainingRawImage(imageSource)));
    const processorInputs = await processor(promptText, images.length === 1 ? images[0] : images);
    const extraInputs = {};
    for (const [key, value] of Object.entries(processorInputs || {})) {
      if (key === "input_ids" || key === "attention_mask") continue;
      extraInputs[key] = value;
    }
    return buildEncodedTrainingExample({
      promptIds: processorInputs?.input_ids,
      promptAttentionMask: processorInputs?.attention_mask,
      targetIds: tokenizeToIds(tokenizer, pair.target, { add_special_tokens: false }),
      padId,
      maxLength,
      prompt: pair.prompt,
      target: pair.target,
      usesVision: true,
      imageSources,
      extraInputs,
    });
  }
  return encodePromptTargetExample(tokenizer, pair.prompt, pair.target, padId, maxLength);
}

function buildPositionIds(attentionMask) {
  const output = new Int32Array(attentionMask.length);
  let running = 0;
  for (let index = 0; index < attentionMask.length; index += 1) {
    if (attentionMask[index] > 0) {
      output[index] = running;
      running += 1;
    } else {
      output[index] = 0;
    }
  }
  return output;
}

function modelNeedsQwen35RopeInputs(model) {
  const modelType = String(model?.config?.model_type || "").trim();
  return (
    modelType === "qwen3_5" ||
    modelType === "qwen3_5_text" ||
    modelType === "qwen3_5_moe" ||
    modelType === "qwen3_5_moe_text"
  );
}

function getInputIdSequences(inputIdsTensor, fallbackInputIds) {
  if (typeof inputIdsTensor?.tolist === "function") {
    const sequences = inputIdsTensor.tolist();
    if (Array.isArray(sequences)) return sequences;
  }
  if (Array.isArray(fallbackInputIds)) {
    return fallbackInputIds.map((sequence) => Array.from(sequence, (value) => BigInt(Number(value))));
  }
  return [Array.from(fallbackInputIds, (value) => BigInt(Number(value)))];
}

function buildBatchedInt64Tensor(encodedExamples, key, seqLength) {
  const flatValues = [];
  for (const example of encodedExamples) {
    flatValues.push(...example[key]);
  }
  return createInt64Tensor(flatValues, [encodedExamples.length, seqLength]);
}

function buildModelForwardInputs(model, encodedExamplesInput, extraModelInputs = null) {
  const encodedExamples = Array.isArray(encodedExamplesInput)
    ? encodedExamplesInput
    : [encodedExamplesInput];
  if (!encodedExamples.length) {
    throw new Error("Expected at least one encoded example for local Qwen training.");
  }
  const seqLength = encodedExamples[0].inputIds.length;
  const inputIds = buildBatchedInt64Tensor(encodedExamples, "inputIds", seqLength);
  const attentionMask = buildBatchedInt64Tensor(encodedExamples, "attentionMask", seqLength);
  const inputs = {
    input_ids: inputIds,
    attention_mask: attentionMask,
  };
  const multimodalExamples = encodedExamples.filter(
    (example) => example?.extraInputs && Object.keys(example.extraInputs).length,
  );
  const multimodalExample = multimodalExamples[0] || null;
  if (multimodalExample) {
    if (encodedExamples.length !== 1) {
      throw new Error("Local Qwen vision training currently supports one multimodal example per forward pass.");
    }
    Object.assign(inputs, multimodalExample.extraInputs);
  }

  if (typeof model?.prepare_inputs_for_generation === "function") {
    try {
      const preparedInputs = model.prepare_inputs_for_generation(
        getInputIdSequences(
          inputIds,
          encodedExamples.map((example) => example.inputIds),
        ),
        { ...inputs },
        {},
      );
      if (
        preparedInputs?.position_ids &&
        Array.isArray(preparedInputs.position_ids.dims) &&
        preparedInputs.position_ids.dims.length === 3
      ) {
        if (extraModelInputs && typeof extraModelInputs === "object") {
          Object.assign(preparedInputs, extraModelInputs);
        }
        return preparedInputs;
      }
      if (preparedInputs) {
        Object.assign(inputs, preparedInputs);
      }
    } catch (_error) {
      // Fall back to the explicit rope path below.
    }
  }

  if (modelNeedsQwen35RopeInputs(model) && typeof model?.get_rope_index === "function") {
    const [positionIds, ropeDeltas] = model.get_rope_index(
      inputIds,
      multimodalExample?.extraInputs?.image_grid_thw ?? null,
      multimodalExample?.extraInputs?.video_grid_thw ?? null,
      attentionMask,
    );
    inputs.position_ids = positionIds;
    if (ropeDeltas) {
      inputs.rope_deltas = ropeDeltas;
    }
    if (extraModelInputs && typeof extraModelInputs === "object") {
      Object.assign(inputs, extraModelInputs);
    }
    return inputs;
  }

  inputs.position_ids = createInt64Tensor(
    encodedExamples.flatMap((example) => Array.from(buildPositionIds(example.attentionMask))),
    [encodedExamples.length, seqLength],
  );
  if (extraModelInputs && typeof extraModelInputs === "object") {
    Object.assign(inputs, extraModelInputs);
  }
  return inputs;
}

async function runModelForwardOutputs(model, encodedExamples, extraModelInputs = null) {
  const inputs = buildModelForwardInputs(model, encodedExamples, extraModelInputs);
  return await model(inputs);
}

async function runModelForward(model, encodedExamples, extraModelInputs = null) {
  const outputs = await runModelForwardOutputs(model, encodedExamples, extraModelInputs);
  const logitsTensor = outputs?.logits;
  if (!logitsTensor?.data || !Array.isArray(logitsTensor?.dims)) {
    throw new Error("Local Qwen training forward pass returned no logits tensor.");
  }
  return logitsTensor;
}

function collectSupervisedTokenIndices(encodedExample, seqLength) {
  const shiftedLabels = encodedExample.labels.slice(1);
  const indices = [];
  for (let tokenIndex = 0; tokenIndex < shiftedLabels.length && tokenIndex < seqLength - 1; tokenIndex += 1) {
    if (shiftedLabels[tokenIndex] !== -100) {
      indices.push(tokenIndex);
    }
  }
  return indices;
}

export function extractSupervisedLogitRows(logitsTensor, encodedExample, batchIndex = 0) {
  const dims = Array.isArray(logitsTensor?.dims) ? logitsTensor.dims : [];
  const hasBatchAxis = dims.length === 3;
  const batchSize = Number(hasBatchAxis ? dims[0] : 1);
  const seqLength = Number(dims[hasBatchAxis ? 1 : 0] || 0);
  const vocabSize = Number(dims[hasBatchAxis ? 2 : 1] || 0);
  if (seqLength <= 1 || vocabSize <= 0) {
    return {
      baseRows: new Float32Array(0),
      labels: new Int32Array(0),
      rowCount: 0,
      vocabSize,
    };
  }

  const tokenIndices = collectSupervisedTokenIndices(encodedExample, seqLength);
  const supervisedCount = tokenIndices.length;
  if (!supervisedCount) {
    return {
      baseRows: new Float32Array(0),
      labels: new Int32Array(0),
      rowCount: 0,
      tokenIndices: new Int32Array(0),
      vocabSize,
    };
  }

  const logitsData = logitsTensor.data;
  const baseRows = new Float32Array(supervisedCount * vocabSize);
  const labels = new Int32Array(supervisedCount);
  const batchOffset = hasBatchAxis ? batchIndex * seqLength * vocabSize : 0;
  if (batchIndex < 0 || batchIndex >= batchSize) {
    throw new Error(`Batch index ${batchIndex} is out of range for logits batch size ${batchSize}.`);
  }

  for (let rowIndex = 0; rowIndex < tokenIndices.length; rowIndex += 1) {
    const tokenIndex = tokenIndices[rowIndex];
    const label = encodedExample.labels[tokenIndex + 1];
    const sourceOffset = batchOffset + tokenIndex * vocabSize;
    baseRows.set(logitsData.subarray(sourceOffset, sourceOffset + vocabSize), rowIndex * vocabSize);
    labels[rowIndex] = label;
  }

  return {
    baseRows,
    labels,
    rowCount: supervisedCount,
    tokenIndices: Int32Array.from(tokenIndices),
    vocabSize,
  };
}

function extractSupervisedRowsFromTensor(tensor, encodedExample, batchIndex = 0, tokenIndicesInput = null) {
  const dims = Array.isArray(tensor?.dims) ? tensor.dims : [];
  const hasBatchAxis = dims.length === 3;
  const batchSize = Number(hasBatchAxis ? dims[0] : 1);
  const seqLength = Number(dims[hasBatchAxis ? 1 : 0] || 0);
  const featureSize = Number(dims[hasBatchAxis ? 2 : 1] || 0);
  const tokenIndices = Array.isArray(tokenIndicesInput)
    ? tokenIndicesInput
    : ArrayBuffer.isView(tokenIndicesInput)
      ? Array.from(tokenIndicesInput, Number)
      : collectSupervisedTokenIndices(encodedExample, seqLength);
  if (!tensor?.data || seqLength <= 0 || featureSize <= 0 || !tokenIndices.length) {
    return {
      rows: new Float32Array(0),
      rowCount: 0,
      featureSize,
      tokenIndices: new Int32Array(0),
    };
  }
  if (batchIndex < 0 || batchIndex >= batchSize) {
    throw new Error(`Batch index ${batchIndex} is out of range for tensor batch size ${batchSize}.`);
  }
  const data = tensor.data;
  const out = new Float32Array(tokenIndices.length * featureSize);
  const batchOffset = hasBatchAxis ? batchIndex * seqLength * featureSize : 0;
  for (let rowIndex = 0; rowIndex < tokenIndices.length; rowIndex += 1) {
    const tokenIndex = tokenIndices[rowIndex];
    const sourceOffset = batchOffset + tokenIndex * featureSize;
    const targetOffset = rowIndex * featureSize;
    for (let featureIndex = 0; featureIndex < featureSize; featureIndex += 1) {
      out[targetOffset + featureIndex] = Number(data[sourceOffset + featureIndex]);
    }
  }
  return {
    rows: out,
    rowCount: tokenIndices.length,
    featureSize,
    tokenIndices: Int32Array.from(tokenIndices),
  };
}

function* iterateExampleBatches(examples, batchSize) {
  const safeBatchSize = Math.max(1, Number(batchSize || 1));
  for (let start = 0; start < examples.length; start += safeBatchSize) {
    yield examples.slice(start, start + safeBatchSize);
  }
}

function countExampleInputTokens(example = null) {
  const attentionMask = example?.attentionMask;
  if (Array.isArray(attentionMask) || ArrayBuffer.isView(attentionMask)) {
    let total = 0;
    for (const value of attentionMask) {
      if (Number(value) > 0) total += 1;
    }
    return total;
  }
  const inputIds = example?.inputIds;
  if (Array.isArray(inputIds) || ArrayBuffer.isView(inputIds)) {
    return inputIds.length;
  }
  return 0;
}

function countExampleSupervisedTokens(example = null) {
  const labels = example?.labels;
  if (!(Array.isArray(labels) || ArrayBuffer.isView(labels))) {
    return 0;
  }
  let total = 0;
  for (const label of labels) {
    if (Number(label) !== -100) total += 1;
  }
  return total;
}

function countExamplesInputTokens(examples = []) {
  let total = 0;
  for (const example of Array.isArray(examples) ? examples : []) {
    total += countExampleInputTokens(example);
  }
  return total;
}

function countExamplesSupervisedTokens(examples = []) {
  let total = 0;
  for (const example of Array.isArray(examples) ? examples : []) {
    total += countExampleSupervisedTokens(example);
  }
  return total;
}

function summarizeEncodedExamplesTokenStats(examples = []) {
  const safeExamples = Array.isArray(examples) ? examples : [];
  let inputTokens = 0;
  let supervisedTokens = 0;
  let maxInputTokens = 0;
  let maxSupervisedTokens = 0;
  for (const example of safeExamples) {
    const exampleInputTokens = countExampleInputTokens(example);
    const exampleSupervisedTokens = countExampleSupervisedTokens(example);
    inputTokens += exampleInputTokens;
    supervisedTokens += exampleSupervisedTokens;
    if (exampleInputTokens > maxInputTokens) maxInputTokens = exampleInputTokens;
    if (exampleSupervisedTokens > maxSupervisedTokens) maxSupervisedTokens = exampleSupervisedTokens;
  }
  const exampleCount = safeExamples.length;
  return {
    examples: exampleCount,
    inputTokens,
    supervisedTokens,
    meanInputTokens: exampleCount > 0 ? inputTokens / exampleCount : 0,
    meanSupervisedTokens: exampleCount > 0 ? supervisedTokens / exampleCount : 0,
    maxInputTokens,
    maxSupervisedTokens,
  };
}

function createTrainingLcg(seed) {
  let state = (Number(seed) >>> 0) || 1;
  return function next() {
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

function buildEpochBatches({
  rowCount = 0,
  batchSize = 1,
  epochs = 1,
  seed = 42,
} = {}) {
  const safeRowCount = Math.max(0, Number(rowCount || 0));
  const safeBatchSize = Math.max(1, Number(batchSize || 1));
  const safeEpochs = Math.max(0, Number(epochs || 0));
  const batchesByEpoch = [];

  for (let epoch = 0; epoch < safeEpochs; epoch += 1) {
    const epochSeed = ((Number(seed) >>> 0) + (((epoch + 1) * 2654435761) >>> 0)) >>> 0;
    const nextRandom = createTrainingLcg(epochSeed);
    const shuffledIndices = Array.from({ length: safeRowCount }, (_value, index) => index);
    for (let index = shuffledIndices.length - 1; index > 0; index -= 1) {
      const swapIndex = Math.floor(nextRandom() * (index + 1));
      const value = shuffledIndices[index];
      shuffledIndices[index] = shuffledIndices[swapIndex];
      shuffledIndices[swapIndex] = value;
    }
    const epochBatches = [];
    for (let start = 0; start < shuffledIndices.length; start += safeBatchSize) {
      epochBatches.push(shuffledIndices.slice(start, start + safeBatchSize));
    }
    batchesByEpoch.push(epochBatches);
  }

  return batchesByEpoch;
}

function computeSoftmaxCrossEntropyGradients(logitsRows, labels, rowCount, vocabSize) {
  const safeRowCount = Math.max(0, Number(rowCount || 0));
  const gradLogits = new Float32Array(safeRowCount * vocabSize);
  let totalLoss = 0;
  let correctCount = 0;

  for (let row = 0; row < safeRowCount; row += 1) {
    const rowOffset = row * vocabSize;
    let maxLogit = -Infinity;
    let bestIndex = 0;
    let bestValue = -Infinity;
    for (let vocab = 0; vocab < vocabSize; vocab += 1) {
      const value = Number(logitsRows[rowOffset + vocab]);
      if (value > maxLogit) maxLogit = value;
      if (value > bestValue) {
        bestValue = value;
        bestIndex = vocab;
      }
    }
    const label = Number(labels[row]);
    if (bestIndex === label) correctCount += 1;

    let expSum = 0;
    for (let vocab = 0; vocab < vocabSize; vocab += 1) {
      const expValue = Math.exp(Number(logitsRows[rowOffset + vocab]) - maxLogit);
      gradLogits[rowOffset + vocab] = expValue;
      expSum += expValue;
    }
    const normalizer = expSum > 0 ? expSum : 1;
    for (let vocab = 0; vocab < vocabSize; vocab += 1) {
      gradLogits[rowOffset + vocab] /= normalizer;
    }
    const labelProb = Math.max(1e-12, Number(gradLogits[rowOffset + label] || 0));
    totalLoss -= Math.log(labelProb);
    gradLogits[rowOffset + label] -= 1;
  }

  return {
    gradLogits,
    loss: safeRowCount > 0 ? totalLoss / safeRowCount : 0,
    accuracy: safeRowCount > 0 ? correctCount / safeRowCount : 0,
    correctCount,
  };
}

async function backpropLmHeadGradients(session, gradLogits, rowCount, vocabSize) {
  const outputs = await session.run({
    grad_logits: new Tensor("float32", gradLogits, [rowCount, vocabSize]).ort_tensor,
  });
  const outputName = Array.isArray(session?.outputNames) && session.outputNames.length
    ? session.outputNames[0]
    : "grad_hidden";
  const gradHidden = outputs?.[outputName];
  if (!gradHidden?.data) {
    throw new Error("The lm_head backprop helper returned no hidden-state gradients.");
  }
  return tensorDataToFloat32Array(gradHidden);
}

function rmsNormBackwardRows(inputsRows, gradOutputs, weight, epsilon = 1e-6) {
  const hiddenSize = Number(weight?.length || 0);
  if (!(hiddenSize > 0)) {
    throw new Error("RMSNorm backward requires a non-empty weight vector.");
  }
  if (inputsRows.length !== gradOutputs.length) {
    throw new Error("RMSNorm backward requires matching input and gradient row buffers.");
  }
  const rowCount = Math.max(0, Math.floor(inputsRows.length / hiddenSize));
  const gradInputs = new Float32Array(rowCount * hiddenSize);

  for (let row = 0; row < rowCount; row += 1) {
    const rowOffset = row * hiddenSize;
    let sumSquares = 0;
    for (let feature = 0; feature < hiddenSize; feature += 1) {
      const value = Number(inputsRows[rowOffset + feature]);
      sumSquares += value * value;
    }
    const invRms = 1 / Math.sqrt((sumSquares / hiddenSize) + Number(epsilon || 0));
    const invRmsCubed = invRms * invRms * invRms;

    let meanWeightedDot = 0;
    for (let feature = 0; feature < hiddenSize; feature += 1) {
      const weightedGrad = Number(gradOutputs[rowOffset + feature]) * Number(weight[feature]);
      gradInputs[rowOffset + feature] = weightedGrad;
      meanWeightedDot += Number(inputsRows[rowOffset + feature]) * weightedGrad;
    }
    meanWeightedDot /= hiddenSize;

    for (let feature = 0; feature < hiddenSize; feature += 1) {
      const inputValue = Number(inputsRows[rowOffset + feature]);
      const weightedGrad = gradInputs[rowOffset + feature];
      gradInputs[rowOffset + feature] = Math.fround(
        (weightedGrad * invRms) - (inputValue * meanWeightedDot * invRmsCubed),
      );
    }
  }

  return gradInputs;
}

async function evaluateTransformerLoraLossOnDevice({
  lossHelperRuntime = null,
  logitsTensor = null,
  examples = [],
} = {}) {
  if (!lossHelperRuntime?.session || lossHelperRuntime.disabled) {
    return null;
  }
  if (!Array.isArray(examples) || !examples.length) {
    return { loss: 0, accuracy: 0, rowCount: 0, correctCount: 0 };
  }
  const seqLength = Number(examples[0]?.labels?.length || examples[0]?.inputIds?.length || 0);
  if (!(seqLength > 0)) {
    return { loss: 0, accuracy: 0, rowCount: 0, correctCount: 0 };
  }

  try {
    const labelsTensor = buildBatchedInt64Tensor(examples, "labels", seqLength);
    const outputs = await lossHelperRuntime.session.run({
      logits: coerceTensorLikeForSessionInput(lossHelperRuntime.session, "logits", logitsTensor),
      labels: labelsTensor.ort_tensor,
    });
    const loss = tensorLikeToScalarNumber(outputs?.loss, 0);
    const correctCount = Math.max(0, Math.round(tensorLikeToScalarNumber(outputs?.correct_count, 0)));
    const rowCount = Math.max(0, Math.round(tensorLikeToScalarNumber(outputs?.row_count, 0)));
    return {
      loss: rowCount > 0 ? loss : 0,
      accuracy: rowCount > 0 ? correctCount / rowCount : 0,
      rowCount,
      correctCount,
    };
  } catch (error) {
    lossHelperRuntime.disabled = true;
    lossHelperRuntime.failureReason = error instanceof Error ? error.message : String(error);
    return null;
  }
}

async function evaluateTransformerLoraStreamingExact({
  model,
  examples,
  adapter = null,
  modelBatchSize = 1,
  lossHelperRuntime = null,
  onExample = null,
  onBatch = null,
}) {
  let totalLoss = 0;
  let totalCorrect = 0;
  let totalRows = 0;

  for (const exampleBatch of iterateExampleBatches(examples, modelBatchSize)) {
    const forwardStartedAtMs = Date.now();
    const outputs = await runModelForwardOutputs(
      model,
      exampleBatch,
      buildLocalQwenTransformerLoraInputs(adapter),
    );
    const forwardDurationMs = Math.max(0, Date.now() - forwardStartedAtMs);
    const logitsTensor = outputs?.logits;
    if (!Array.isArray(logitsTensor?.dims)) {
      throw new Error("Exact transformer LoRA evaluation received no logits tensor.");
    }
    const helperMetrics = await evaluateTransformerLoraLossOnDevice({
      lossHelperRuntime,
      logitsTensor,
      examples: exampleBatch,
    });
    if (helperMetrics) {
      totalLoss += helperMetrics.loss * helperMetrics.rowCount;
      totalCorrect += helperMetrics.correctCount;
      totalRows += helperMetrics.rowCount;
      if (onExample) {
        for (const example of exampleBatch) {
          onExample(example, null);
        }
      }
      if (onBatch) {
        onBatch(exampleBatch, {
          forwardDurationMs,
          helperMetrics,
        });
      }
      continue;
    }
    if (!logitsTensor?.data) {
      throw new Error("Exact transformer LoRA evaluation could not materialize logits for the JS fallback path.");
    }
    for (let batchIndex = 0; batchIndex < exampleBatch.length; batchIndex += 1) {
      const example = exampleBatch[batchIndex];
      const extracted = extractSupervisedLogitRows(logitsTensor, example, batchIndex);
      if (onExample) onExample(example, extracted);
      if (!extracted.rowCount) continue;
      const metrics = computeSoftmaxCrossEntropyGradients(
        extracted.baseRows,
        extracted.labels,
        extracted.rowCount,
        extracted.vocabSize,
      );
      totalLoss += metrics.loss * extracted.rowCount;
      totalCorrect += metrics.correctCount;
      totalRows += extracted.rowCount;
    }
    if (onBatch) {
      onBatch(exampleBatch, {
        forwardDurationMs,
      });
    }
  }

  if (!totalRows) {
    return { loss: 0, accuracy: 0, rowCount: 0 };
  }
  return {
    loss: totalLoss / totalRows,
    accuracy: totalCorrect / totalRows,
    rowCount: totalRows,
  };
}

async function evaluateTransformerLoraBatchExact({
  model,
  examples,
  adapter = null,
  lossHelperRuntime = null,
} = {}) {
  if (!Array.isArray(examples) || !examples.length) {
    return { loss: 0, accuracy: 0, rowCount: 0, correctCount: 0, forwardDurationMs: 0 };
  }
  const forwardStartedAtMs = Date.now();
  const outputs = await runModelForwardOutputs(
    model,
    examples,
    buildLocalQwenTransformerLoraInputs(adapter),
  );
  const forwardDurationMs = Math.max(0, Date.now() - forwardStartedAtMs);
  const logitsTensor = outputs?.logits;
  if (!Array.isArray(logitsTensor?.dims)) {
    throw new Error("Browser transformer LoRA batch evaluation received no logits tensor.");
  }
  const helperMetrics = await evaluateTransformerLoraLossOnDevice({
    lossHelperRuntime,
    logitsTensor,
    examples,
  });
  if (helperMetrics) {
    return {
      ...helperMetrics,
      forwardDurationMs,
    };
  }
  if (!logitsTensor?.data) {
    throw new Error("Browser transformer LoRA batch evaluation could not materialize logits for the JS fallback path.");
  }

  let totalLoss = 0;
  let totalCorrect = 0;
  let totalRows = 0;
  for (let batchIndex = 0; batchIndex < examples.length; batchIndex += 1) {
    const extracted = extractSupervisedLogitRows(logitsTensor, examples[batchIndex], batchIndex);
    if (!extracted.rowCount) continue;
    const metrics = computeSoftmaxCrossEntropyGradients(
      extracted.baseRows,
      extracted.labels,
      extracted.rowCount,
      extracted.vocabSize,
    );
    totalLoss += metrics.loss * extracted.rowCount;
    totalCorrect += metrics.correctCount;
    totalRows += extracted.rowCount;
  }

  if (!totalRows) {
    return { loss: 0, accuracy: 0, rowCount: 0, correctCount: 0 };
  }
  return {
    loss: totalLoss / totalRows,
    accuracy: totalCorrect / totalRows,
    rowCount: totalRows,
    correctCount: totalCorrect,
    forwardDurationMs,
  };
}

function getBrowserTrainingStrategy(manifest, config = {}) {
  const manifestStrategy = asText(manifest?.browserTraining?.strategy);
  const configStrategy = asText(config?.optimizer);
  if (configStrategy) return configStrategy;
  if (manifestStrategy) return manifestStrategy;
  return "full_transformer_lora_spsa";
}

function makeBrowserSpsaSeed(baseSeed, epochIndex, batchIndex, sampleIndex) {
  let state = Number(baseSeed) >>> 0;
  state = (state + (((epochIndex + 1) * 2654435761) >>> 0)) >>> 0;
  state = (state + (((batchIndex + 1) * 2246822519) >>> 0)) >>> 0;
  state = (state + (((sampleIndex + 1) * 3266489917) >>> 0)) >>> 0;
  return state >>> 0;
}

function resolveBrowserSpsaConfig(config = {}) {
  const fallback = LOCAL_QWEN_FIXED_TRAINING_CONFIG;
  const learningRate = Number(config.learningRate || fallback.learningRate || 0.0001);
  const epsilon = Number(config.spsaEpsilon || fallback.spsaEpsilon || 0.001);
  const samples = Math.max(1, Number(config.spsaSamples || fallback.spsaSamples || 1));
  const gradientClip = Number(config.spsaGradientClip || fallback.spsaGradientClip || 0);
  return {
    learningRate,
    epsilon,
    samples,
    gradientClip,
  };
}

export async function trainFixedLocalQwenAdapter({
  modelName,
  config = LOCAL_QWEN_FIXED_TRAINING_CONFIG,
  onStatus = null,
}) {
  const trainingRunStartedAtMs = Date.now();
  const { trainPairs, validationPairs, testPairs, datasetMeta, datasetSource } = await loadFixedTrainingPairs(config);
  const containsVisionSamples =
    datasetHasVisionPairs(trainPairs) || datasetHasVisionPairs(validationPairs) || datasetHasVisionPairs(testPairs);
  const evaluationMode = String(config.evaluationMode || "full").trim().toLowerCase();
  const runtime = await loadLocalQwenTrainingRuntime(modelName, { requiresVision: containsVisionSamples });
  const padId = runtime.tokenizer.pad_token_id ?? runtime.tokenizer.eos_token_id ?? 0;
  const totalSamples = estimateFixedTrainingWorkSamples(config);
  let completedSamples = 0;
  let phaseStartedAtMs = Date.now();
  let phaseCompletedSamples = 0;
  let phaseTotalSamples = 0;
  let phaseUnitLabel = "job steps";
  let phaseForwardTokens = 0;
  let phaseSupervisedTokens = 0;
  let totalForwardTokens = 0;
  let totalSupervisedTokens = 0;
  let trainStepForwardTokens = 0;
  let evalForwardTokens = 0;
  let trainStepForwardMs = 0;
  let evalForwardMs = 0;
  let datasetTokenStats = null;
  let lossHelperRuntime = null;
  let adapter = null;
  let bestAdapter = null;
  let selectedAdapter = null;
  const getLossEvaluationMode = () => (
    lossHelperRuntime?.session && !lossHelperRuntime.disabled
      ? "onnx_scalar_helper"
      : "host_js_logits"
  );
  const comparisonPurpose =
    asText(config.comparisonPurpose)
    || (String(config.profile || "").includes("proofread")
      ? "browser_text_training_benchmark"
      : String(config.profile || "").includes("pipeline")
        ? "pipeline_correctness_smoke"
        : "custom");
  const compareBy = asText(config.compareBy) || "forward_tokens_per_second";
  const resourceProfile = asText(config.resourceProfile) || "memory_conservative";
  const status = (fields = {}) => {
    if (typeof onStatus === "function") {
      const phaseElapsedS = Math.max((Date.now() - phaseStartedAtMs) / 1000, 1e-6);
      const overallElapsedMs = Math.max(0, Date.now() - trainingRunStartedAtMs);
      const throughputMetrics = summarizeBrowserTrainingThroughputMetrics({
        totalForwardTokens,
        totalSupervisedTokens,
        trainStepForwardTokens,
        evalForwardTokens,
        trainStepForwardMs,
        evalForwardMs,
        overallElapsedMs,
      });
      const providedMetrics =
        fields?.metrics && typeof fields.metrics === "object"
          ? fields.metrics
          : {};
      const restFields = {
        ...(fields && typeof fields === "object" ? fields : {}),
      };
      delete restFields.metrics;
      delete restFields.dataset;
      delete restFields.runtime;
      onStatus({
        totalSamples,
        completedSamples,
        phaseCompletedSamples,
        phaseTotalSamples,
        phaseUnitLabel,
        samplesPerSecond: phaseCompletedSamples > 0 ? phaseCompletedSamples / phaseElapsedS : 0,
        dataset: {
          path: datasetSource,
          trainPairs: trainPairs.length,
          validationPairs: validationPairs.length,
          testPairs: testPairs.length,
          tokenStats: datasetTokenStats,
          meta: {
            ...(datasetMeta && typeof datasetMeta === "object" ? datasetMeta : {}),
            containsVisionSamples,
          },
        },
        runtime: {
          ...runtime.runtimePlan,
          executionPlan: runtime.executionPlan,
          reasoningMode: config.reasoningMode,
          multimodal: containsVisionSamples,
          lossEvaluation: getLossEvaluationMode(),
          lossEvaluationIssue: asText(
            lossHelperRuntime?.failureReason || runtime?.browserTrainingLossIssue,
          ) || undefined,
          trainingComparison: {
            purpose: comparisonPurpose,
            compareBy,
            resourceProfile,
          },
        },
        metrics: {
          ...providedMetrics,
          phaseForwardTokens,
          phaseSupervisedTokens,
          phaseForwardTokensPerSecond: phaseForwardTokens > 0 ? phaseForwardTokens / phaseElapsedS : 0,
          phaseSupervisedTokensPerSecond: phaseSupervisedTokens > 0 ? phaseSupervisedTokens / phaseElapsedS : 0,
          ...throughputMetrics,
        },
        ...restFields,
      });
    }
  };

  const beginPhase = ({
    phase,
    phaseLabel,
    message,
    phaseTotal = 0,
    unitLabel = "job steps",
    ...fields
  }) => {
    phaseStartedAtMs = Date.now();
    phaseCompletedSamples = 0;
    phaseTotalSamples = Math.max(0, Number(phaseTotal || 0));
    phaseUnitLabel = String(unitLabel || "job steps");
    phaseForwardTokens = 0;
    phaseSupervisedTokens = 0;
    status({
      phase,
      phaseLabel,
      message,
      ...fields,
    });
  };

  const advancePhase = (delta = 1, fields = {}, work = {}) => {
    const safeDelta = Math.max(0, Number(delta || 0));
    completedSamples += safeDelta;
    phaseCompletedSamples += safeDelta;
    const forwardTokens = Math.max(0, Number(work.forwardTokens || 0));
    const supervisedTokens = Math.max(0, Number(work.supervisedTokens || 0));
    const forwardDurationMs = Math.max(0, Number(work.forwardDurationMs || 0));
    const workloadKind = asText(work.workloadKind);
    phaseForwardTokens += forwardTokens;
    phaseSupervisedTokens += supervisedTokens;
    totalForwardTokens += forwardTokens;
    totalSupervisedTokens += supervisedTokens;
    if (workloadKind === "train_step") {
      trainStepForwardTokens += forwardTokens;
      trainStepForwardMs += forwardDurationMs;
    } else if (workloadKind === "eval") {
      evalForwardTokens += forwardTokens;
      evalForwardMs += forwardDurationMs;
    }
    status(fields);
  };
  const advanceEvalBatch = (exampleBatch, { forwardDurationMs = 0 } = {}) => {
    advancePhase(Array.isArray(exampleBatch) ? exampleBatch.length : 0, {}, {
      forwardTokens: countExamplesInputTokens(exampleBatch),
      supervisedTokens: countExamplesSupervisedTokens(exampleBatch),
      forwardDurationMs,
      workloadKind: "eval",
    });
  };

  try {
  beginPhase({
    phase: "encoding_dataset",
    phaseLabel: "Encoding dataset",
    message:
      datasetSource === "inline"
        ? containsVisionSamples
          ? "Encoding the craft dataset, including multimodal vision samples."
          : "Encoding the craft dataset from the sidepanel artifact."
        : containsVisionSamples
          ? "Encoding the fixed packaged multimodal training split."
          : "Encoding the fixed packaged training split.",
    phaseTotal: trainPairs.length + validationPairs.length + testPairs.length,
    unitLabel: "encoded samples",
  });

  const trainExamples = [];
  for (const pair of trainPairs) {
    trainExamples.push(
      await encodeTrainingPairExample({
        tokenizer: runtime.tokenizer,
        processor: runtime.processor,
        pair,
        padId,
        maxLength: config.maxSeqLen,
      }),
    );
    advancePhase(1);
  }
  const testExamples = [];
  for (const pair of testPairs) {
    testExamples.push(
      await encodeTrainingPairExample({
        tokenizer: runtime.tokenizer,
        processor: runtime.processor,
        pair,
        padId,
        maxLength: config.maxSeqLen,
      }),
    );
    advancePhase(1);
  }
  const validationExamplesRaw = [];
  for (const pair of validationPairs) {
    validationExamplesRaw.push(
      await encodeTrainingPairExample({
        tokenizer: runtime.tokenizer,
        processor: runtime.processor,
        pair,
        padId,
        maxLength: config.maxSeqLen,
      }),
    );
    advancePhase(1);
  }
  datasetTokenStats = {
    train: summarizeEncodedExamplesTokenStats(trainExamples),
    validation: summarizeEncodedExamplesTokenStats(validationExamplesRaw),
    test: summarizeEncodedExamplesTokenStats(testExamples),
  };

  const modelBatchSize = containsVisionSamples ? 1 : Math.max(1, Number(config.modelBatchSize || 1));
  const evaluateTrainSplit = evaluationMode !== "holdout_only";
  const validationExamples = validationExamplesRaw.length ? validationExamplesRaw : (testExamples.length ? testExamples : trainExamples);
  const validationLabel = validationExamplesRaw.length ? "validation" : (testExamples.length ? "test" : "train");
  const finalTestExamples = testExamples.length ? testExamples : validationExamples;
  const finalTestLabel = testExamples.length ? "test" : validationLabel;
  const trainingManifest = await ensureLocalQwenBrowserTrainingManifest(runtime);
  const trainingStrategy = getBrowserTrainingStrategy(trainingManifest, config);
  const spsaConfig = resolveBrowserSpsaConfig(config);
  const trainingSupportIssue = getTransformerLoraTrainingSupportIssue(trainingManifest);
  if (trainingSupportIssue) {
    throw new Error(trainingSupportIssue.message);
  }
  if (trainingStrategy !== "full_transformer_lora_spsa") {
    throw new Error(`Unsupported browser transformer LoRA training strategy: ${trainingStrategy || "<empty>"}`);
  }
  if (!(spsaConfig.epsilon > 0)) {
    throw new Error("Browser transformer LoRA training requires a positive SPSA epsilon.");
  }
  lossHelperRuntime = await ensureLocalQwenBrowserTrainingLossRuntime(runtime, trainingManifest);

  const effectiveRank = Math.max(1, Number(trainingManifest?.graph?.rank || config.rank || 0));
  const scale = config.alpha / effectiveRank;
  adapter = createTransformerLoraAdapter({
    manifest: trainingManifest,
    seed: config.seed,
    trainA: config.trainMatrixA !== false,
    trainB: config.trainMatrixB !== false,
  });
  if (!adapter.modules.length) {
    throw new Error("Browser transformer LoRA training found no trainable LoRA modules in the packaged manifest.");
  }
  adapter.rank = effectiveRank;
  adapter.runtimeModelId = asText(runtime.runtimePlan.runtimeModelId);
  adapter.scale = scale;
  for (const module of adapter.modules) {
    module.scale = scale;
  }
  const adam = createTransformerLoraAdamState(adapter);
  const history = [];
  const epochBatches = buildEpochBatches({
    rowCount: trainExamples.length,
    batchSize: modelBatchSize,
    epochs: config.epochs,
    seed: config.seed,
  });
  let bestMetrics = null;

  beginPhase({
    phase: "base_metrics",
    phaseLabel: "Evaluating base model",
    message: evaluateTrainSplit
      ? "Computing base accuracy on the train, validation, and test splits."
      : `Computing base accuracy on the ${validationLabel} and ${finalTestLabel} splits.`,
    phaseTotal:
      (evaluateTrainSplit ? trainExamples.length : 0) + validationExamples.length + finalTestExamples.length,
    unitLabel: "eval samples",
  });
  const baseTrainEval = evaluateTrainSplit
    ? await evaluateTransformerLoraStreamingExact({
        model: runtime.model,
        examples: trainExamples,
        modelBatchSize,
        lossHelperRuntime,
        onBatch: advanceEvalBatch,
      })
    : { loss: 0, accuracy: 0, rowCount: 0 };
  const baseValidationEval = await evaluateTransformerLoraStreamingExact({
    model: runtime.model,
    examples: validationExamples,
    modelBatchSize,
    lossHelperRuntime,
    onBatch: advanceEvalBatch,
  });
  const baseTestEval = await evaluateTransformerLoraStreamingExact({
    model: runtime.model,
    examples: finalTestExamples,
    modelBatchSize,
    lossHelperRuntime,
    onBatch: advanceEvalBatch,
  });

  for (let epoch = 0; epoch < config.epochs; epoch += 1) {
    beginPhase({
      phase: "training",
      phaseLabel: `Training epoch ${epoch + 1}/${config.epochs}`,
      message: `Training full browser LoRA epoch ${epoch + 1} of ${config.epochs} with a frozen forward pass.`,
      phaseTotal: trainExamples.length,
      unitLabel: "train examples",
      currentEpoch: epoch + 1,
      metrics: {
        baseValidationAcc: baseValidationEval.accuracy,
        baseTestAcc: baseTestEval.accuracy,
      },
    });

    for (let batchIndex = 0; batchIndex < epochBatches[epoch].length; batchIndex += 1) {
      const batchIndices = epochBatches[epoch][batchIndex];
      const exampleBatch = batchIndices.map((exampleIndex) => trainExamples[exampleIndex]);
      if (!exampleBatch.length) continue;

      const spsaSamples = spsaConfig.samples;
      const batchInputTokens = countExamplesInputTokens(exampleBatch);
      const batchSupervisedTokens = countExamplesSupervisedTokens(exampleBatch);
      let estimatedBatchLoss = 0;
      let batchForwardTokensProcessed = 0;
      let batchSupervisedTokensProcessed = 0;
      let batchForwardMsProcessed = 0;
      for (let sampleIndex = 0; sampleIndex < spsaSamples; sampleIndex += 1) {
        const spsaSeed = makeBrowserSpsaSeed(config.seed, epoch, batchIndex, sampleIndex);
        perturbTransformerLoraAdapter({
          adapter,
          seed: spsaSeed,
          epsilon: spsaConfig.epsilon,
          multiplier: 1,
        });
        const positiveEval = await evaluateTransformerLoraBatchExact({
          model: runtime.model,
          examples: exampleBatch,
          adapter,
          lossHelperRuntime,
        });

        perturbTransformerLoraAdapter({
          adapter,
          seed: spsaSeed,
          epsilon: spsaConfig.epsilon,
          multiplier: -2,
        });
        const negativeEval = await evaluateTransformerLoraBatchExact({
          model: runtime.model,
          examples: exampleBatch,
          adapter,
          lossHelperRuntime,
        });

        perturbTransformerLoraAdapter({
          adapter,
          seed: spsaSeed,
          epsilon: spsaConfig.epsilon,
          multiplier: 1,
        });

        estimatedBatchLoss += (positiveEval.loss + negativeEval.loss) / 2;
        batchForwardTokensProcessed += batchInputTokens * 2;
        batchSupervisedTokensProcessed += batchSupervisedTokens * 2;
        batchForwardMsProcessed += Number(positiveEval.forwardDurationMs || 0) + Number(negativeEval.forwardDurationMs || 0);
        applyTransformerLoraSpsaAdamStep({
          adapter,
          adam,
          seed: spsaSeed,
          epsilon: spsaConfig.epsilon,
          objectiveDiff: positiveEval.loss - negativeEval.loss,
          learningRate: spsaConfig.learningRate,
          gradientClip: spsaConfig.gradientClip,
        });
      }

      advancePhase(exampleBatch.length, {
        metrics: {
          baseValidationAcc: baseValidationEval.accuracy,
          baseTestAcc: baseTestEval.accuracy,
          estimatedTrainLoss: estimatedBatchLoss / Math.max(1, Number(config.spsaSamples || 1)),
        },
      }, {
        forwardTokens: batchForwardTokensProcessed,
        supervisedTokens: batchSupervisedTokensProcessed,
        forwardDurationMs: batchForwardMsProcessed,
        workloadKind: "train_step",
      });
    }

    beginPhase({
      phase: "epoch_eval",
      phaseLabel: `Evaluating epoch ${epoch + 1}/${config.epochs}`,
      message: evaluateTrainSplit
        ? `Evaluating validation metrics after epoch ${epoch + 1}.`
        : `Evaluating ${validationLabel} metrics after epoch ${epoch + 1}.`,
      phaseTotal: (evaluateTrainSplit ? trainExamples.length : 0) + validationExamples.length,
      unitLabel: "eval samples",
      currentEpoch: epoch + 1,
    });
    const trainEval = evaluateTrainSplit
      ? await evaluateTransformerLoraStreamingExact({
          model: runtime.model,
          examples: trainExamples,
          adapter,
          modelBatchSize,
          lossHelperRuntime,
          onBatch: advanceEvalBatch,
        })
      : { loss: 0, accuracy: 0, rowCount: 0 };
    const validationEval = await evaluateTransformerLoraStreamingExact({
      model: runtime.model,
      examples: validationExamples,
      adapter,
      modelBatchSize,
      lossHelperRuntime,
      onBatch: advanceEvalBatch,
    });
    history.push({
      epoch: epoch + 1,
      train_loss: trainEval.loss,
      train_acc: trainEval.accuracy,
      validation_loss: validationEval.loss,
      validation_acc: validationEval.accuracy,
    });
    status({
      metrics: {
        baseValidationAcc: baseValidationEval.accuracy,
        baseTestAcc: baseTestEval.accuracy,
        adaptValidationAcc: validationEval.accuracy,
        trainEvalAcc: trainEval.accuracy,
      },
    });
    if (
      !bestMetrics ||
      validationEval.accuracy > bestMetrics.validation_acc ||
      (validationEval.accuracy === bestMetrics.validation_acc && validationEval.loss < bestMetrics.validation_loss)
    ) {
      bestMetrics = {
        epoch: epoch + 1,
        train_loss: trainEval.loss,
        train_acc: trainEval.accuracy,
        validation_loss: validationEval.loss,
        validation_acc: validationEval.accuracy,
      };
      bestAdapter = cloneTransformerLoraAdapter(adapter);
      bestAdapter.scale = scale;
      for (const module of bestAdapter.modules) {
        module.scale = scale;
      }
    }
  }

  selectedAdapter = bestAdapter || cloneTransformerLoraAdapter(adapter);
  selectedAdapter.scale = scale;
  for (const module of selectedAdapter.modules) {
    module.scale = scale;
  }
  beginPhase({
    phase: "final_eval",
    phaseLabel: "Final evaluation",
    message: evaluateTrainSplit
      ? "Evaluating the selected adapter checkpoint on the train, validation, and test splits."
      : `Evaluating the selected adapter on the ${validationLabel} and ${finalTestLabel} splits.`,
    phaseTotal:
      (evaluateTrainSplit ? trainExamples.length : 0) + validationExamples.length + finalTestExamples.length,
    unitLabel: "eval samples",
  });
  const finalTrainEval = evaluateTrainSplit
    ? await evaluateTransformerLoraStreamingExact({
        model: runtime.model,
        examples: trainExamples,
        adapter: selectedAdapter,
        modelBatchSize,
        lossHelperRuntime,
        onBatch: advanceEvalBatch,
      })
    : { loss: 0, accuracy: 0, rowCount: 0 };
  const finalValidationEval = await evaluateTransformerLoraStreamingExact({
    model: runtime.model,
    examples: validationExamples,
    adapter: selectedAdapter,
    modelBatchSize,
    lossHelperRuntime,
    onBatch: advanceEvalBatch,
  });
  const finalTestEval = await evaluateTransformerLoraStreamingExact({
    model: runtime.model,
    examples: finalTestExamples,
    adapter: selectedAdapter,
    modelBatchSize,
    lossHelperRuntime,
    onBatch: advanceEvalBatch,
  });
  const throughput = summarizeBrowserTrainingThroughputMetrics({
    totalForwardTokens,
    totalSupervisedTokens,
    trainStepForwardTokens,
    evalForwardTokens,
    trainStepForwardMs,
    evalForwardMs,
    overallElapsedMs: Math.max(0, Date.now() - trainingRunStartedAtMs),
  });

  return {
    runtime: {
      ...runtime.runtimePlan,
      executionPlan: runtime.executionPlan,
      reasoningMode: config.reasoningMode,
      trainingStrategy,
      lossEvaluation: getLossEvaluationMode(),
      lossEvaluationIssue: asText(
        lossHelperRuntime?.failureReason || runtime?.browserTrainingLossIssue,
      ) || undefined,
      trainingComparison: {
        purpose: comparisonPurpose,
        compareBy,
        resourceProfile,
      },
    },
    dataset: {
      path: datasetSource,
      trainPairs: trainPairs.length,
      validationPairs: validationPairs.length,
      testPairs: testPairs.length,
      tokenStats: datasetTokenStats,
      meta: {
        ...(datasetMeta && typeof datasetMeta === "object" ? datasetMeta : {}),
        containsVisionSamples,
      },
    },
    config: {
      ...config,
      rank: effectiveRank,
    },
    totalSamples,
    completedSamples,
    phaseCompletedSamples,
    phaseTotalSamples,
    phaseUnitLabel,
    samplesPerSecond: phaseCompletedSamples > 0
      ? phaseCompletedSamples / Math.max((Date.now() - phaseStartedAtMs) / 1000, 1e-6)
      : 0,
    history,
    selectedEpoch: bestMetrics?.epoch || config.epochs,
    adapterSizeMb: countTransformerLoraAdapterBytes(selectedAdapter) / 1_000_000,
    throughput: {
      compareBy,
      comparisonPurpose,
      resourceProfile,
      ...throughput,
    },
    adapter: selectedAdapter,
    baseTrainAcc: baseTrainEval.accuracy,
    baseValidationAcc: baseValidationEval.accuracy,
    baseTestAcc: baseTestEval.accuracy,
    trainEval: finalTrainEval,
    validationEval: finalValidationEval,
    testEval: finalTestEval,
  };
  } finally {
    releaseLocalQwenTransformerLoraInputs(adapter);
    if (selectedAdapter && selectedAdapter !== adapter) {
      releaseLocalQwenTransformerLoraInputs(selectedAdapter);
    }
    if (bestAdapter && bestAdapter !== adapter && bestAdapter !== selectedAdapter) {
      releaseLocalQwenTransformerLoraInputs(bestAdapter);
    }
    await disposeLocalQwenTrainingRuntime(runtime);
  }
  throw new Error(
    "Browser transformer LoRA training requires a supported packaged training runtime. No supported runtime was found.",
  );
}
