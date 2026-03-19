function asText(value) {
  return String(value == null ? "" : value).trim();
}

const LOCAL_QWEN_PRIMARY_EXECUTION_PLAN = Object.freeze({
  device: "webgpu",
  dtype: {
    embed_tokens: "q4f16",
    vision_encoder: "fp16",
    decoder_model_merged: "q4f16",
  },
  label: "WebGPU Qwen3.5 q4f16/fp16",
});

const LOCAL_QWEN_REMOTE_REFERENCE_EXECUTION_PLAN = Object.freeze({
  device: "webgpu",
  dtype: {
    embed_tokens: "q4f16",
    vision_encoder: "fp16",
    decoder_model_merged: "q4f16",
  },
  label: "WebGPU Qwen3.5 reference q4f16/fp16",
});

const LOCAL_QWEN_WASM_FALLBACK_EXECUTION_PLAN = Object.freeze({
  device: "wasm",
  dtype: "q8",
  label: "WASM Qwen3.5 reference q8",
});

const LOCAL_QWEN_LOCAL_BROWSER_VISION_PREPROCESS_POLICY = Object.freeze({
  maxPixels: 1_048_576,
  reason: "Clamp visible browser screenshots to a safer local WebGPU vision pixel budget.",
});

const LOCAL_QWEN_BROWSER_TRAINING_REQUIRED_FILES = Object.freeze([
  "lora_training_manifest.json",
  "onnx/decoder_model_merged_q4f16_training.onnx",
  "onnx/decoder_model_merged_q4f16_training.onnx_data",
  "onnx/lm_head_backprop_q4.onnx",
  "onnx/lm_head_backprop_q4.onnx_data",
  "onnx/training_loss_helper.onnx",
]);

const SUPPORTED_LOCAL_QWEN_MODELS = Object.freeze([
  Object.freeze({
    sizeLabel: "0.8B",
    canonicalModelName: "unsloth/Qwen3.5-0.8B",
    vanillaModelName: "unsloth/Qwen3.5-0.8B (Vanilla)",
    runtimeModelId: "local-extension/Qwen3.5-0.8B-ONNX",
    remoteRuntimeModelId: "metricspace/Qwen3.5-0.8B-ONNX-browser-agent",
    packagedLocalRuntime: true,
    packagedBrowserTraining: true,
  }),
  Object.freeze({
    sizeLabel: "2B",
    canonicalModelName: "unsloth/Qwen3.5-2B",
    vanillaModelName: "unsloth/Qwen3.5-2B (Vanilla)",
    runtimeModelId: "local-extension/Qwen3.5-2B-ONNX",
    remoteRuntimeModelId: "onnx-community/Qwen3.5-2B-ONNX",
    packagedLocalRuntime: false,
    packagedBrowserTraining: false,
  }),
  Object.freeze({
    sizeLabel: "4B",
    canonicalModelName: "unsloth/Qwen3.5-4B",
    vanillaModelName: "unsloth/Qwen3.5-4B (Vanilla)",
    runtimeModelId: "local-extension/Qwen3.5-4B-ONNX",
    remoteRuntimeModelId: "onnx-community/Qwen3.5-4B-ONNX",
    packagedLocalRuntime: false,
    packagedBrowserTraining: false,
  }),
]);

const DEFAULT_REQUESTED_MODEL = SUPPORTED_LOCAL_QWEN_MODELS[0].vanillaModelName;
const NO_THINK_DIRECTIVE = "/no_think";
const THINK_BLOCK_PATTERN = /^\s*<think>[\s\S]*?<\/think>\s*/i;
const LOCAL_QWEN_MODEL_LOOKUP = new Map();
const LOCAL_QWEN_RUNTIME_MODEL_LOOKUP = new Map();

for (const entry of SUPPORTED_LOCAL_QWEN_MODELS) {
  LOCAL_QWEN_RUNTIME_MODEL_LOOKUP.set(entry.runtimeModelId, entry);
  for (const alias of [
    entry.canonicalModelName,
    entry.canonicalModelName.replace(/^unsloth\//, "Qwen/"),
    entry.vanillaModelName,
    entry.runtimeModelId,
    entry.remoteRuntimeModelId,
  ]) {
    LOCAL_QWEN_MODEL_LOOKUP.set(alias, entry);
  }
}

function getLocalQwenPackagedModelEntry(runtimeModelId = "") {
  return LOCAL_QWEN_RUNTIME_MODEL_LOOKUP.get(asText(runtimeModelId)) || null;
}

export function getLocalQwenModelRepoRelativePath(runtimeModelId = "", fileName = "") {
  const modelId = asText(runtimeModelId);
  const relativeFile = asText(fileName).replace(/^\/+/, "");
  if (!modelId || !relativeFile) return "";
  return `models/${modelId}/${relativeFile}`;
}

function buildLocalQwenHubResolveUrl(repoId = "", fileName = "") {
  const normalizedRepoId = asText(repoId);
  const relativeFile = asText(fileName).replace(/^\/+/, "");
  if (!normalizedRepoId || !relativeFile) return "";
  return `https://huggingface.co/${normalizedRepoId}/resolve/main/${relativeFile}`;
}

export function getLocalQwenModelRepoUrl(runtimeModelId = "", fileName = "") {
  const modelId = asText(runtimeModelId);
  const relativeFile = asText(fileName).replace(/^\/+/, "");
  const entry = LOCAL_QWEN_RUNTIME_MODEL_LOOKUP.get(modelId);
  if (entry?.remoteRuntimeModelId && entry.remoteRuntimeModelId !== modelId) {
    return buildLocalQwenHubResolveUrl(entry.remoteRuntimeModelId, relativeFile);
  }
  const relativePath = getLocalQwenModelRepoRelativePath(runtimeModelId, fileName);
  if (!relativePath) return "";
  const runtimeApi = globalThis.chrome?.runtime || null;
  if (typeof runtimeApi?.getURL === "function") {
    return runtimeApi.getURL(relativePath);
  }
  return relativePath;
}

export function describeLocalQwenBrowserTrainingPackaging(runtimeModelId = "") {
  const normalizedRuntimeModelId = asText(runtimeModelId);
  const entry = getLocalQwenPackagedModelEntry(normalizedRuntimeModelId);
  const manifestRelativePath = getLocalQwenModelRepoRelativePath(
    normalizedRuntimeModelId,
    "lora_training_manifest.json",
  );
  return {
    runtimeModelId: normalizedRuntimeModelId,
    packagedLocalRuntime: entry?.packagedLocalRuntime === true,
    packagedBrowserTraining: entry?.packagedBrowserTraining === true,
    repoId:
      asText(entry?.remoteRuntimeModelId) || normalizedRuntimeModelId,
    manifestRelativePath,
    manifestUrl: getLocalQwenModelRepoUrl(normalizedRuntimeModelId, "lora_training_manifest.json"),
    requiredRelativePaths:
      entry?.packagedBrowserTraining === true
        ? LOCAL_QWEN_BROWSER_TRAINING_REQUIRED_FILES.map((fileName) =>
            getLocalQwenModelRepoRelativePath(normalizedRuntimeModelId, fileName)
          )
        : manifestRelativePath
          ? [manifestRelativePath]
          : [],
    availableRuntimeModelIds: SUPPORTED_LOCAL_QWEN_MODELS
      .filter((model) => model.packagedBrowserTraining === true)
      .map((model) => model.runtimeModelId),
  };
}

export function createLocalQwenMissingBrowserTrainingManifestError(runtimeModelId = "") {
  const packaging = describeLocalQwenBrowserTrainingPackaging(runtimeModelId);
  const error = new Error(
    `Browser transformer LoRA training is unavailable for ${packaging.runtimeModelId}: `
    + "missing curated lora_training_manifest.json.",
  );
  error.detail = {
    reason: "local_qwen_browser_training_manifest_missing",
    runtimeModelId: packaging.runtimeModelId,
    repoId: packaging.repoId,
    expectedRelativePath: packaging.manifestRelativePath,
    requiredRelativePaths: packaging.requiredRelativePaths,
    packagedLocalRuntime: packaging.packagedLocalRuntime,
    packagedBrowserTraining: packaging.packagedBrowserTraining,
    availableRuntimeModelIds: packaging.availableRuntimeModelIds,
  };
  return error;
}

function getExecutionPlanDtypeKey(dtype) {
  if (!dtype) return "auto";
  if (typeof dtype === "string") return dtype;
  try {
    return JSON.stringify(dtype);
  } catch {
    return String(dtype);
  }
}

function sameExecutionPlan(left, right) {
  return (
    String(left?.runtimeModelId || "").trim() === String(right?.runtimeModelId || "").trim() &&
    left?.device === right?.device &&
    getExecutionPlanDtypeKey(left?.dtype) === getExecutionPlanDtypeKey(right?.dtype)
  );
}

function cloneExecutionPlan(plan, runtimeModelId) {
  return {
    ...plan,
    runtimeModelId,
    dtype:
      plan?.dtype && typeof plan.dtype === "object" && !Array.isArray(plan.dtype)
        ? { ...plan.dtype }
        : plan?.dtype,
  };
}

function buildLocalQwenExecutionPlans(requestedInfo) {
  const primaryRuntimeModelId = String(requestedInfo?.runtimeModelId || "").trim();
  const strictLocal = requestedInfo?.strictLocal !== false;
  const remoteRuntimeModelId = String(
    requestedInfo?.remoteRuntimeModelId || requestedInfo?.runtimeModelId || "",
  ).trim();
  const plans = [
    cloneExecutionPlan(LOCAL_QWEN_PRIMARY_EXECUTION_PLAN, primaryRuntimeModelId),
  ];

  if (!strictLocal && remoteRuntimeModelId && remoteRuntimeModelId !== primaryRuntimeModelId) {
    plans.push(cloneExecutionPlan(LOCAL_QWEN_REMOTE_REFERENCE_EXECUTION_PLAN, remoteRuntimeModelId));
  }

  if (!strictLocal && remoteRuntimeModelId) {
    plans.push(cloneExecutionPlan(LOCAL_QWEN_WASM_FALLBACK_EXECUTION_PLAN, remoteRuntimeModelId));
  }

  return plans.filter(
    (plan, index) => plans.findIndex((entry) => sameExecutionPlan(entry, plan)) === index,
  );
}

function listSupportedRequestedModelNames() {
  return SUPPORTED_LOCAL_QWEN_MODELS.map((entry) => entry.vanillaModelName).join(", ");
}

export function getPreferredLocalQwenVanillaModelName(
  requestedModelName = "",
  fallbackModelName = DEFAULT_REQUESTED_MODEL,
) {
  for (const candidate of [requestedModelName, fallbackModelName, DEFAULT_REQUESTED_MODEL]) {
    const entry = LOCAL_QWEN_MODEL_LOOKUP.get(asText(candidate));
    if (entry?.vanillaModelName) {
      return entry.vanillaModelName;
    }
  }
  return DEFAULT_REQUESTED_MODEL;
}

function normalizeRequestedModel(requestedModelName = "") {
  const requested = asText(requestedModelName) || DEFAULT_REQUESTED_MODEL;
  const entry = LOCAL_QWEN_MODEL_LOOKUP.get(requested);
  if (!entry) {
    throw new Error(
      `Only supported local Qwen3.5 browser models are allowed (${listSupportedRequestedModelNames()}). Received: ${requested || "<empty>"}`,
    );
  }

  let requestedVariant = "default";
  if (requested === entry.vanillaModelName) requestedVariant = "vanilla";
  if (requested === entry.runtimeModelId || requested === entry.remoteRuntimeModelId) {
    requestedVariant = "onnx";
  }

  const useRemoteRuntime = requested === entry.remoteRuntimeModelId;

  return {
    requested,
    canonicalRequested: entry.canonicalModelName,
    requestedVariant,
    sizeLabel: entry.sizeLabel,
    runtimeModelId: useRemoteRuntime ? entry.remoteRuntimeModelId : entry.runtimeModelId,
    remoteRuntimeModelId: entry.remoteRuntimeModelId,
    strictLocal: !useRemoteRuntime,
  };
}

export function getSupportedLocalQwenModels() {
  return SUPPORTED_LOCAL_QWEN_MODELS.map((entry) => ({
    ...entry,
    browserTrainingPackaging: describeLocalQwenBrowserTrainingPackaging(entry.runtimeModelId),
    executionPlan: cloneExecutionPlan(LOCAL_QWEN_PRIMARY_EXECUTION_PLAN, entry.runtimeModelId),
    executionPlans: buildLocalQwenExecutionPlans(entry),
  }));
}

export function getLocalQwenVisionPreprocessPolicy(runtimeModelId = "", executionPlan = null) {
  const resolvedRuntimeModelId = asText(executionPlan?.runtimeModelId || runtimeModelId);
  const resolvedDevice = asText(executionPlan?.device).toLowerCase();
  if (!resolvedRuntimeModelId.startsWith("local-extension/")) return null;
  if (resolvedDevice !== "webgpu") return null;
  return {
    ...LOCAL_QWEN_LOCAL_BROWSER_VISION_PREPROCESS_POLICY,
    runtimeModelId: resolvedRuntimeModelId,
    device: resolvedDevice,
  };
}

function normalizeImageSourceValue(value) {
  if (typeof value === "string") return asText(value);
  if (value instanceof URL) return value.toString();
  if (value && typeof value === "object") {
    if ("url" in value) return normalizeImageSourceValue(value.url);
    if ("image" in value) return normalizeImageSourceValue(value.image);
    if ("image_url" in value) return normalizeImageSourceValue(value.image_url);
  }
  return "";
}

function normalizeImagePart(part) {
  const image = normalizeImageSourceValue(part?.image ?? part?.image_url);
  if (!image) return null;

  const normalized = {
    type: "image",
    image,
  };
  if (part?.providerOptions && typeof part.providerOptions === "object") {
    normalized.providerOptions = { ...part.providerOptions };
  }
  return normalized;
}

function normalizeContentParts(content) {
  if (content == null) return [];
  const source = Array.isArray(content) ? content : [content];
  const out = [];

  for (const part of source) {
    if (typeof part === "string") {
      const text = asText(part);
      if (text) out.push({ type: "text", text });
      continue;
    }

    if (!part || typeof part !== "object") continue;
    if (part.type === "text") {
      const text = asText(part.text);
      if (text) out.push({ type: "text", text });
      continue;
    }
    if (part.type === "image" || part.type === "image_url") {
      const normalizedImage = normalizeImagePart(part);
      if (normalizedImage) out.push(normalizedImage);
      continue;
    }

    const fallbackText = asText(part.content ?? part.text);
    if (fallbackText) {
      out.push({ type: "text", text: fallbackText });
    }
  }

  return out;
}

function normalizeMessageContent(content) {
  if (content == null) return "";
  if (typeof content === "string") return content;

  const parts = normalizeContentParts(content);
  if (!parts.length) return "";
  if (parts.some((part) => part.type === "image")) return parts;

  return parts
    .map((part) => (part.type === "text" ? asText(part.text) : ""))
    .filter(Boolean)
    .join("\n");
}

function parseToolArguments(value) {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    return value;
  }
  const text = asText(value);
  if (!text) return {};
  try {
    const parsed = JSON.parse(text);
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

function normalizeToolCalls(value) {
  return (Array.isArray(value) ? value : [])
    .map((toolCall) => {
      const source = toolCall?.function && typeof toolCall.function === "object"
        ? toolCall.function
        : toolCall;
      const name = asText(source?.name);
      if (!name) return null;
      return {
        id: asText(toolCall?.id) || `call_${name.replace(/[^a-z0-9]+/gi, "_").toLowerCase() || "1"}`,
        type: "function",
        function: {
          name,
          arguments: parseToolArguments(source?.arguments),
        },
      };
    })
    .filter(Boolean);
}

export function getLocalQwenRuntimePlan(requestedModelName = "") {
  const requestedInfo = normalizeRequestedModel(requestedModelName);
  const requestedArchitecture = "qwen3.5";
  const runtimeArchitecture = "qwen3.5";
  const executionPlans = buildLocalQwenExecutionPlans(requestedInfo);
  const browserTrainingPackaging = describeLocalQwenBrowserTrainingPackaging(
    requestedInfo.runtimeModelId,
  );
  const runtimeFallbackReason =
    requestedInfo.strictLocal
      ? "Use only the curated browser runtime artifact set. Missing model files must fail loudly."
      : requestedInfo.runtimeModelId === requestedInfo.remoteRuntimeModelId
      ? "Retry the ONNX reference model in WASM q8 when the primary WebGPU plan fails."
      : "Retry the packaged q4f16 runtime against the ONNX reference model, then fall back to WASM q8.";

  return {
    requestedModelName: requestedInfo.requested,
    requestedModelCanonicalName: requestedInfo.canonicalRequested,
    requestedVariant: requestedInfo.requestedVariant,
    requestedSizeLabel: requestedInfo.sizeLabel,
    requestedArchitecture,
    runtimeModelId: requestedInfo.runtimeModelId,
    remoteRuntimeModelId: requestedInfo.remoteRuntimeModelId,
    strictLocal: requestedInfo.strictLocal,
    runtimeDisplayName: requestedInfo.runtimeModelId,
    runtimeArchitecture,
    runtimeDtypeSummary: "embed q4f16 · decoder q4f16 · vision fp16",
    runtimeFallbackReason,
    browserTrainingPackaging,
    executionPlans,
  };
}

export function rewriteLocalQwenRemoteUrl(url) {
  const source = asText(url);
  if (!source.startsWith("https://huggingface.co/")) return source;
  const marker = "/resolve/";
  const withoutHost = source.slice("https://huggingface.co/".length);
  const markerIndex = withoutHost.indexOf(marker);
  if (markerIndex <= 0) return source;
  const runtimeModelId = withoutHost.slice(0, markerIndex);
  const suffix = withoutHost.slice(markerIndex);
  const entry = LOCAL_QWEN_MODEL_LOOKUP.get(runtimeModelId);
  if (entry?.runtimeModelId === runtimeModelId && entry.remoteRuntimeModelId !== runtimeModelId) {
    return `https://huggingface.co/${entry.remoteRuntimeModelId}${suffix}`;
  }
  if (!entry?.remoteRuntimeModelId || entry.remoteRuntimeModelId === runtimeModelId) {
    return source;
  }
  return `https://huggingface.co/${entry.remoteRuntimeModelId}${suffix}`;
}

export function normalizeLocalQwenMessages(messages = []) {
  const normalized = (messages || [])
    .filter((message) => message && /^(system|user|assistant|tool)$/.test(message.role))
    .map((message) => {
      const base = {
        role: message.role,
        content: normalizeMessageContent(message.content),
      };
      if (message.role === "assistant") {
        const toolCalls = normalizeToolCalls(message.tool_calls);
        if (toolCalls.length) base.tool_calls = toolCalls;
        if (typeof message.reasoning_content === "string") {
          base.reasoning_content = message.reasoning_content;
        }
      }
      if (message.role === "tool") {
        base.tool_call_id = asText(message.tool_call_id);
        base.name = asText(message.name);
      }
      return base;
    })
    .filter((message) => {
      if (message.role === "assistant" && Array.isArray(message.tool_calls) && message.tool_calls.length) {
        return true;
      }
      if (message.role === "tool") {
        return Boolean(asText(message.content) || asText(message.tool_call_id) || asText(message.name));
      }
      return Array.isArray(message.content) ? message.content.length > 0 : Boolean(asText(message.content));
    });

  const systemMessages = [];
  const conversationMessages = [];
  for (const message of normalized) {
    if (message.role === "system") {
      systemMessages.push(message);
    } else {
      conversationMessages.push(message);
    }
  }

  return [...systemMessages, ...conversationMessages];
}

export function collectLocalQwenImageSources(messages = []) {
  const normalizedMessages = normalizeLocalQwenMessages(messages);
  const images = [];

  for (const message of normalizedMessages) {
    const content = Array.isArray(message.content) ? message.content : [];
    for (const part of content) {
      if (part?.type === "image") {
        const image = normalizeImageSourceValue(part.image);
        if (image) images.push(image);
      }
    }
  }

  return images;
}

export function hasLocalQwenVisionInputs(messages = []) {
  return collectLocalQwenImageSources(messages).length > 0;
}

export function getLocalQwenReasoningMode(parameters = {}) {
  return parameters?.enableThinking === true ? "think" : "no_think";
}

export function prepareLocalQwenMessages(messages = [], parameters = {}) {
  const normalized = normalizeLocalQwenMessages(messages);
  if (getLocalQwenReasoningMode(parameters) !== "no_think") return normalized;

  return normalized.map((message, index) => {
    if (message.role !== "user") return message;
    if (Array.isArray(message.content)) {
      let injected = false;
      const nextContent = message.content.map((part) => {
        if (part?.type !== "text" || injected) return part;
        injected = true;
        return {
          ...part,
          text: `${NO_THINK_DIRECTIVE}\n${asText(part.text)}`.trim(),
        };
      });
      if (injected) return { ...message, content: nextContent };
      if (index === normalized.length - 1) {
        return {
          ...message,
          content: [{ type: "text", text: NO_THINK_DIRECTIVE }],
        };
      }
      return message;
    }

    const text = asText(message.content);
    return {
      ...message,
      content: text ? `${NO_THINK_DIRECTIVE}\n${text}` : NO_THINK_DIRECTIVE,
    };
  });
}

export function buildLocalQwenGenerationConfig(parameters = {}) {
  const rawMaxNewTokens = Number(parameters?.maxTokens ?? parameters?.max_new_tokens ?? 256);
  const maxNewTokens = Math.max(
    1,
    Math.min(512, Number.isFinite(rawMaxNewTokens) ? rawMaxNewTokens : 256),
  );
  const rawTemperature = Number(parameters?.temperature ?? 0.7);
  const temperature = Math.max(0, Math.min(2, Number.isFinite(rawTemperature) ? rawTemperature : 0.7));
  const rawTopP = Number(parameters?.topP ?? parameters?.top_p ?? 0.95);
  const topP = Math.max(0, Math.min(1, Number.isFinite(rawTopP) ? rawTopP : 0.95));
  const rawTopK = Number(parameters?.topK ?? parameters?.top_k ?? 50);
  const topK = Math.max(0, Math.min(200, Math.trunc(Number.isFinite(rawTopK) ? rawTopK : 50)));
  const rawRepetitionPenalty = Number(parameters?.repetitionPenalty ?? parameters?.repetition_penalty ?? 1.05);
  const repetitionPenalty = Math.max(
    0.8,
    Math.min(2, Number.isFinite(rawRepetitionPenalty) ? rawRepetitionPenalty : 1.05),
  );
  const doSample = temperature > 0;

  return {
    max_new_tokens: maxNewTokens,
    temperature,
    top_p: topP,
    top_k: topK,
    repetition_penalty: repetitionPenalty,
    do_sample: doSample,
    return_full_text: false,
  };
}

export function extractLocalQwenText(output) {
  if (typeof output === "string") return output;
  if (Array.isArray(output)) {
    const first = output[0];
    if (typeof first === "string") return first;
    if (Array.isArray(first?.generated_text)) {
      const assistantTurn = [...first.generated_text]
        .reverse()
        .find((entry) => asText(entry?.role).toLowerCase() === "assistant");
      return asText(assistantTurn?.content);
    }
    if (typeof first?.generated_text === "string") return first.generated_text;
    if (typeof first?.text === "string") return first.text;
  }
  if (typeof output?.generated_text === "string") return output.generated_text;
  if (typeof output?.text === "string") return output.text;
  return "";
}

export function stripLocalQwenThinkingTrace(text) {
  return asText(text).replace(THINK_BLOCK_PATTERN, "");
}
