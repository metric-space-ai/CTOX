import "../shared/craft-sync.js";
import "../shared/model-config.js";
import {
  formatCraftingAgentToolLabels,
  normalizeCraftingAgentToolingPayload,
} from "../shared/crafting-agent-tooling.mjs";
import {
  parseTrainingSampleOpsText,
  TRAINING_SAMPLE_OPS_SCHEMA,
  TRAINING_SAMPLE_SPLITS,
  TRAINING_SAMPLE_STATUSES,
} from "../shared/training-sample-ops.mjs";
import { sendMessageToOffscreen } from "./offscreen-bridge.reference.js";
import {
  AiSdkAnthropicProvider,
  AiSdkAzureProvider,
  AiSdkCerebrasProvider,
  AiSdkDeepSeekProvider,
  AiSdkGroqProvider,
  AiSdkOpenAIProvider,
  generateObject,
  generateText,
  streamText,
} from "../vendor/agent-bundle.mjs";

const configApi = globalThis.SinepanelModelConfig;
const SUPPORTED_SLOT_IDS = new Set(
  (configApi?.MODEL_SLOT_DEFS || []).map((slot) => String(slot?.id || "").trim()).filter(Boolean),
);

const OPENAI_COMPATIBLE_PROVIDER_TYPES = new Set([
  "openai",
  "custom_openai",
  "openrouter",
  "ollama",
]);
const LOCAL_QWEN_PROVIDER_ALIASES = new Set([
  "qwen",
  "local_qwen",
  "local-qwen",
  "local_qwen_vision",
  "local-qwen-vision",
]);
const AI_SDK_SCHEMA_SYMBOL = Symbol.for("vercel.ai.schema");
const AI_SDK_VALIDATOR_SYMBOL = Symbol.for("vercel.ai.validator");

function asText(value) {
  return String(value == null ? "" : value).trim();
}

function normalizeTextList(values, limit = 12) {
  return (Array.isArray(values) ? values : [])
    .map((value) => asText(value))
    .filter(Boolean)
    .slice(0, limit);
}

function canonicalizeProviderId(providerId = "") {
  const raw = asText(providerId);
  const normalized = raw.toLowerCase();
  if (LOCAL_QWEN_PROVIDER_ALIASES.has(normalized)) {
    return "local_qwen";
  }
  return raw;
}

function getDefaultModelNameForProvider(provider = null, slotConfig = null) {
  const providerId = canonicalizeProviderId(provider?.id);
  const slotProviderId = canonicalizeProviderId(slotConfig?.providerId);
  if (providerId && slotProviderId === providerId && asText(slotConfig?.modelName)) {
    return asText(slotConfig.modelName);
  }
  const modelNames = Array.isArray(provider?.modelNames)
    ? provider.modelNames.map((item) => asText(item)).filter(Boolean)
    : [];
  return modelNames[0] || "";
}

function isOpenAiReasoningModel(modelName) {
  const name = asText(modelName);
  if (!name) return false;
  return name.startsWith("o") || (name.startsWith("gpt-5") && !name.startsWith("gpt-5-chat"));
}

function sanitizeProvider(provider) {
  if (!provider || typeof provider !== "object") return null;
  return {
    id: asText(provider.id),
    type: asText(provider.type),
    name: asText(provider.name),
    enabled: provider.enabled !== false,
    baseUrl: asText(provider.baseUrl),
    modelNames: Array.isArray(provider.modelNames) ? provider.modelNames.map((item) => asText(item)).filter(Boolean) : [],
    hasApiKey: Boolean(asText(provider.apiKey)),
  };
}

function summarizeResolvedConfig(resolved) {
  return {
    slotId: asText(resolved?.slotId),
    modelRef: asText(resolved?.modelRef),
    providerId: asText(resolved?.providerId),
    modelName: asText(resolved?.modelName),
    provider: sanitizeProvider({
      ...(resolved?.provider && typeof resolved.provider === "object" ? resolved.provider : {}),
      id: resolved?.providerId,
    }),
    parameters:
      resolved?.parameters && typeof resolved.parameters === "object"
        ? { ...resolved.parameters }
        : {},
    reasoningEffort: asText(resolved?.reasoningEffort),
  };
}

function decodeBase64ToUint8Array(base64) {
  const normalized = asText(base64).replace(/\s+/g, "");
  if (!normalized) return null;

  try {
    if (typeof globalThis.atob === "function") {
      const binary = globalThis.atob(normalized);
      const bytes = new Uint8Array(binary.length);
      for (let index = 0; index < binary.length; index += 1) {
        bytes[index] = binary.charCodeAt(index);
      }
      return bytes;
    }
  } catch {
    return null;
  }

  try {
    if (typeof globalThis.Buffer?.from === "function") {
      return new Uint8Array(globalThis.Buffer.from(normalized, "base64"));
    }
  } catch {
    return null;
  }

  return null;
}

function parseInlineDataUrl(value) {
  const text = asText(value);
  const match = /^data:([^;,]+)?(?:;charset=[^;,]+)?(;base64)?,([\s\S]*)$/i.exec(text);
  if (!match) return null;

  const mediaType = asText(match[1]) || "application/octet-stream";
  const body = match[3] || "";
  if (match[2]) {
    const data = decodeBase64ToUint8Array(body);
    if (!data) return null;
    return { data, mediaType };
  }

  try {
    if (typeof globalThis.TextEncoder === "function") {
      return {
        data: new globalThis.TextEncoder().encode(decodeURIComponent(body)),
        mediaType,
      };
    }
  } catch {
    return null;
  }

  return null;
}

function coerceMessageImageSource(value) {
  if (typeof value === "string") return value;
  if (value instanceof URL) return value.toString();
  if (value && typeof value === "object" && "url" in value) {
    return coerceMessageImageSource(value.url);
  }
  return value;
}

function normalizeMessageContent(content, options = {}) {
  const inlineImageMode = options?.inlineImageMode === "preserve" ? "preserve" : "decode";
  const normalizeImagePart = (part) => {
    const providerOptions = part?.providerOptions && typeof part.providerOptions === "object"
      ? { ...part.providerOptions }
      : {};
    const detail = asText(
      part?.detail ||
        part?.imageDetail ||
        part?.providerOptions?.openai?.imageDetail,
    ).toLowerCase();
    if (detail === "low" || detail === "high" || detail === "auto") {
      providerOptions.openai = {
        ...(providerOptions.openai && typeof providerOptions.openai === "object"
          ? providerOptions.openai
          : {}),
        imageDetail: detail,
      };
    }
    const image = coerceMessageImageSource(part?.image ?? part?.image_url?.url ?? part?.image_url ?? "");
    if (!image) return null;
    const inlineDataUrl =
      inlineImageMode === "preserve" || typeof image !== "string"
        ? null
        : parseInlineDataUrl(image);
    const out = {
      type: "image",
      image: inlineDataUrl?.data || image,
    };
    if (inlineDataUrl?.mediaType) out.mediaType = inlineDataUrl.mediaType;
    if (Object.keys(providerOptions).length) out.providerOptions = providerOptions;
    return out;
  };

  if (content == null) return "";
  if (Array.isArray(content)) {
    const out = [];
    for (const part of content) {
      if (!part || typeof part !== "object") continue;
      if (part.type === "text") {
        out.push({ type: "text", text: String(part.text || "") });
        continue;
      }
      if (part.type === "image" || part.type === "image_url") {
        const imagePart = normalizeImagePart(part);
        if (imagePart) out.push(imagePart);
      }
    }
    return out.length ? out : "";
  }
  if (typeof content === "object") {
    if (content.type === "text") return [{ type: "text", text: String(content.text || "") }];
    if (content.type === "image" || content.type === "image_url") {
      const imagePart = normalizeImagePart(content);
      return imagePart ? [imagePart] : "";
    }
    return String(content.text || content.content || "");
  }
  return String(content);
}

function parseToolArguments(value) {
  if (value && typeof value === "object" && !Array.isArray(value)) return value;
  const text = asText(value);
  if (!text) return {};
  try {
    const parsed = JSON.parse(text);
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

function normalizeMessageToolCalls(value) {
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

function normalizeMessages(messages, options = {}) {
  return (messages || [])
    .filter((message) => message && /^(system|user|assistant|tool)$/.test(message.role))
    .map((message) => {
      const base = {
        role: message.role,
        content: normalizeMessageContent(message.content, options),
      };
      if (message.role === "assistant") {
        const toolCalls = normalizeMessageToolCalls(message.tool_calls);
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
    });
}

function parseJsonLoose(raw) {
  const text = asText(raw);
  if (!text) return null;

  try {
    return JSON.parse(text);
  } catch {}

  const fenced = text.match(/```(?:json)?\s*([\s\S]*?)\s*```/i);
  if (fenced?.[1]) {
    try {
      return JSON.parse(fenced[1]);
    } catch {}
  }

  const objectMatch = text.match(/\{[\s\S]*\}/);
  if (objectMatch?.[0]) {
    try {
      return JSON.parse(objectMatch[0]);
    } catch {}
  }

  return null;
}

function stringifySchemaPreview(schemaJson) {
  if (!schemaJson || typeof schemaJson !== "object") return "";
  try {
    return JSON.stringify(schemaJson, null, 2).slice(0, 12_000);
  } catch {
    return "";
  }
}

async function validateStructuredOutputValue(schema, value) {
  if (!schema) {
    return { success: true, value };
  }
  if (schema && typeof schema === "object" && AI_SDK_SCHEMA_SYMBOL in schema && typeof schema.validate === "function") {
    return schema.validate(value);
  }
  if (typeof schema.safeParseAsync === "function") {
    const result = await schema.safeParseAsync(value);
    return result.success
      ? { success: true, value: result.data }
      : { success: false, error: result.error };
  }
  if (typeof schema.safeParse === "function") {
    const result = schema.safeParse(value);
    return result.success
      ? { success: true, value: result.data }
      : { success: false, error: result.error };
  }
  if (typeof schema.parseAsync === "function") {
    try {
      return { success: true, value: await schema.parseAsync(value) };
    } catch (error) {
      return { success: false, error };
    }
  }
  if (typeof schema.parse === "function") {
    try {
      return { success: true, value: schema.parse(value) };
    } catch (error) {
      return { success: false, error };
    }
  }
  return { success: true, value };
}

function buildStructuredOutputSchema(schema, jsonSchema) {
  if (!jsonSchema || typeof jsonSchema !== "object") {
    return schema;
  }
  return {
    [AI_SDK_SCHEMA_SYMBOL]: true,
    [AI_SDK_VALIDATOR_SYMBOL]: true,
    _type: undefined,
    get jsonSchema() {
      return jsonSchema;
    },
    validate(value) {
      return validateStructuredOutputValue(schema, value);
    },
  };
}

function collapseLocalQwenSystemMessages(messages = [], injectedSystemContent = "") {
  const normalized = normalizeMessages(messages, { inlineImageMode: "preserve" });
  const systemChunks = [];
  const nonSystemMessages = [];

  const injected = asText(injectedSystemContent);
  if (injected) {
    systemChunks.push(injected);
  }

  for (const message of normalized) {
    if (message?.role === "system") {
      const content = normalizeMessageContent(message.content, { inlineImageMode: "preserve" });
      if (typeof content === "string" && asText(content)) {
        systemChunks.push(asText(content));
      } else if (Array.isArray(content) && content.length) {
        const flattened = content
          .map((part) => (part?.type === "text" ? asText(part.text) : ""))
          .filter(Boolean)
          .join("\n");
        if (flattened) systemChunks.push(flattened);
      }
      continue;
    }
    nonSystemMessages.push(message);
  }

  if (!systemChunks.length) {
    return nonSystemMessages;
  }

  return [
    {
      role: "system",
      content: systemChunks.join("\n\n"),
    },
    ...nonSystemMessages,
  ];
}

function parseStructuredObjectText(rawText, schema = null, schemaName = "structured_output") {
  const parsed = parseJsonLoose(rawText);
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error(`Model output for ${schemaName} was not a valid JSON object.`);
  }
  if (schema && typeof schema.safeParse === "function") {
    const validation = schema.safeParse(parsed);
    if (!validation.success) {
      const error = new Error(`Model output for ${schemaName} did not satisfy the schema.`);
      error.detail = validation.error?.issues || null;
      throw error;
    }
    return validation.data;
  }
  return parsed;
}

function resolveStructuredOutputMaxTokens(resolved = null, parameters = {}) {
  return Math.max(
    160,
    Math.min(
      800,
      Number(
        resolved?.parameters?.maxTokens ||
          resolved?.parameters?.max_new_tokens ||
          parameters?.maxTokens ||
          parameters?.max_new_tokens ||
          420,
      ) || 420,
    ),
  );
}

function buildStructuredOutputSystemContent({
  schemaName = "structured_output",
  schemaDescription = "",
  schemaPreview = "",
  mode = "generate",
} = {}) {
  const isRepair = mode === "repair";
  return [
    isRepair
      ? `Repair invalid model output into one valid JSON object for ${asText(schemaName) || "structured_output"}.`
      : `You are a strict JSON generator for ${asText(schemaName) || "structured_output"}.`,
    "Return exactly one valid JSON object and nothing else.",
    "Never emit markdown fences, comments, bullet points, or prose.",
    "Use only double-quoted JSON keys and string values.",
    isRepair ? "Do not add wrapper keys, explanations, or markdown." : "",
    schemaDescription ? `Schema notes: ${schemaDescription}` : "",
    schemaPreview ? `Target JSON schema:\n${schemaPreview}` : "",
  ].filter(Boolean).join("\n");
}

function buildStructuredOutputRepairError({
  schemaName = "structured_output",
  schemaPreview = "",
  initialText = "",
  repairedText = "",
  initialParseError = null,
  repairParseError = null,
  resolved = null,
} = {}) {
  const error = new Error(`Model output was not valid JSON for ${schemaName}.`);
  error.detail = {
    schemaName: asText(schemaName),
    schemaPreview,
    initialText: String(initialText || ""),
    repairedText: String(repairedText || ""),
    initialParseError,
    repairParseError,
    resolved: resolved || null,
  };
  return error;
}

async function recoverStructuredOutputViaRawText({
  resolved = null,
  safeMessages = [],
  schema = null,
  schemaName = "structured_output",
  schemaDescription = "",
  schemaPreview = "",
  signal,
  parameters = {},
  localContext = {},
} = {}) {
  const maxTokens = resolveStructuredOutputMaxTokens(resolved, parameters);
  const initialResponse = await llmChat({
    slotId: resolved?.slotId || "agent",
    modelRef: resolved?.modelRef || "",
    parameters: {
      ...(resolved?.parameters && typeof resolved.parameters === "object" ? resolved.parameters : {}),
      ...(parameters && typeof parameters === "object" ? parameters : {}),
      temperature: 0,
      maxTokens,
    },
    reasoningEffort: asText(resolved?.reasoningEffort),
    signal,
    localContext,
    messages: collapseLocalQwenSystemMessages(
      safeMessages,
      buildStructuredOutputSystemContent({
        schemaName,
        schemaDescription,
        schemaPreview,
        mode: "generate",
      }),
    ),
  });

  try {
    return {
      object: parseStructuredObjectText(initialResponse?.text || "", schema, schemaName),
      usage: initialResponse?.usage || null,
      finishReason: initialResponse?.finishReason || "",
      resolved: initialResponse?.resolved || summarizeResolvedConfig(resolved),
    };
  } catch (parseError) {
    const repairResponse = await llmChat({
      slotId: resolved?.slotId || "agent",
      modelRef: resolved?.modelRef || "",
      parameters: {
        ...(resolved?.parameters && typeof resolved.parameters === "object" ? resolved.parameters : {}),
        ...(parameters && typeof parameters === "object" ? parameters : {}),
        temperature: 0,
        maxTokens,
      },
      reasoningEffort: asText(resolved?.reasoningEffort),
      signal,
      localContext,
      messages: collapseLocalQwenSystemMessages(
        [
          {
            role: "user",
            content: JSON.stringify(
              {
                candidateText: String(initialResponse?.text || ""),
              },
              null,
              2,
            ),
          },
        ],
        buildStructuredOutputSystemContent({
          schemaName,
          schemaDescription,
          schemaPreview,
          mode: "repair",
        }),
      ),
    });

    try {
      return {
        object: parseStructuredObjectText(repairResponse?.text || "", schema, schemaName),
        usage: repairResponse?.usage || initialResponse?.usage || null,
        finishReason: repairResponse?.finishReason || initialResponse?.finishReason || "",
        resolved: repairResponse?.resolved || initialResponse?.resolved || summarizeResolvedConfig(resolved),
      };
    } catch (repairError) {
      throw buildStructuredOutputRepairError({
        schemaName,
        schemaPreview,
        initialText: String(initialResponse?.text || ""),
        repairedText: String(repairResponse?.text || ""),
        initialParseError:
          parseError && typeof parseError === "object" && "detail" in parseError
            ? parseError.detail || null
            : parseError instanceof Error
              ? parseError.message
              : String(parseError || ""),
        repairParseError:
          repairError && typeof repairError === "object" && "detail" in repairError
            ? repairError.detail || null
            : repairError instanceof Error
              ? repairError.message
              : String(repairError || ""),
        resolved: initialResponse?.resolved || summarizeResolvedConfig(resolved),
      });
    }
  }
}

function buildLocalTrainingSampleOpsInput(craft = null, brief = "", currentSamples = []) {
  const safeCraft = craft && typeof craft === "object" ? craft : {};
  const safeSamples = Array.isArray(currentSamples) ? currentSamples : [];
  return {
    craft: {
      id: asText(safeCraft.id),
      name: asText(safeCraft.name),
      summary: asText(safeCraft.summary),
      stage: asText(safeCraft.stage),
      inputMode: asText(safeCraft.inputMode),
      inputHint: asText(safeCraft.inputHint),
      actionLabel: asText(safeCraft.actionLabel),
      inputExamples: normalizeTextList(safeCraft.inputExamples, 6),
      accuracy: asText(safeCraft.accuracy),
      seedRows: asText(safeCraft.seedRows),
      datasetRows: asText(safeCraft.datasetRows),
      coverageGaps: asText(safeCraft.coverageGaps),
      tools: normalizeTextList(safeCraft.tools, 12),
      agentTooling: normalizeCraftingAgentToolingPayload(safeCraft.agentTooling),
      targetSlot: asText(safeCraft.targetSlot),
    },
    brief: String(brief || "").trim(),
    currentSamples: safeSamples.slice(0, 6).map((sample) => ({
      sampleId: asText(sample?.sampleId || sample?.id),
      mode: "multiturn",
      promptText: asText(sample?.promptText),
      targetTurnSummary: asText(sample?.targetTurnSummary || sample?.targetText),
      messages: Array.isArray(sample?.messages) && sample.messages.length ? sample.messages : null,
      tools: Array.isArray(sample?.tools) && sample.tools.length ? sample.tools : null,
      targetTurnIndex: Number.isInteger(sample?.targetTurnIndex) ? sample.targetTurnIndex : null,
      split: asText(sample?.split),
      status: asText(sample?.status),
      source: asText(sample?.source),
    })),
    allowedSplits: TRAINING_SAMPLE_SPLITS,
    allowedStatuses: TRAINING_SAMPLE_STATUSES,
    hardLimits: {
      maxOperations: 8,
      deleteRequiresSampleId: true,
      updateRequiresSampleId: true,
    },
  };
}

function buildLocalTrainingSampleOpsShape() {
  return {
    summary: "<short summary string>",
    rationale: "<short rationale string>",
    report: {
      objective: "<what this pass is trying to improve>",
      currentState: "<what the agent matched in the current dataset>",
      nextAction: "<what should happen next>",
      matchingSignals: ["<short matching signal>"],
    },
    openQuestions: [
      {
        question: "<question for the user>",
        reason: "<why the agent needs it>",
      },
    ],
    provenance: [
      {
        title: "<short provenance title>",
        detail: "<what was matched or why an operation was planned>",
        kind: "<match|constraint|operation|sample>",
        sampleId: "<optional sample id>",
        operationType: "<optional add|update|delete>",
      },
    ],
    useSurface: {
      inputMode: "<free_text|mixed|selection|current_tab|context_only>",
      inputHint: "<how the finished ability gathers input>",
      inputPlaceholder: "<placeholder when typed input is needed>",
      actionLabel: "<button label for the finished ability>",
      inputExamples: ["<example execution or prompt>"],
    },
    operations: [
      {
        type: "<add|update|delete>",
        sampleId: "<required only for update/delete>",
        reason: "<why this edit is needed>",
        fields: {
          promptText: "<optional short summary or source prompt>",
          messages: [
            { "role": "user", "content": "<user turn>" },
            {
              "role": "assistant",
              "content": "",
              "tool_calls": [{ "id": "call_1", "type": "function", "function": { "name": "<tool>", "arguments": "{\"query\":\"Berlin Mitte\"}" } }]
            },
            { "role": "tool", "tool_call_id": "call_1", "name": "<tool>", "content": "{\"ok\":true,\"data\":{}}" },
            { "role": "assistant", "content": "<final supervised assistant turn>" }
          ],
          tools: [
            { "type": "function", "function": { "name": "<tool>", "description": "<tool description>", "parameters": { "type": "object" } } }
          ],
          targetTurnIndex: 3,
          split: "<train|validation|test>",
          status: "<draft|review|ready|blocked>",
          source: "<source label>",
        },
      },
    ],
  };
}

function createAiSdkProvider(providerRecord) {
  const type = asText(providerRecord?.type).toLowerCase();
  const apiKey = String(providerRecord?.apiKey || "");
  const baseURL = asText(providerRecord?.baseUrl) || undefined;

  if (type === "local_qwen") {
    throw new Error("Provider-Typ local_qwen laeuft nicht ueber den Remote-LLM-Background.");
  }

  if (OPENAI_COMPATIBLE_PROVIDER_TYPES.has(type)) {
    return AiSdkOpenAIProvider({
      apiKey,
      baseURL,
    });
  }

  switch (type) {
    case "azure_openai":
      return AiSdkAzureProvider({
        apiKey,
        baseURL,
        apiVersion: asText(providerRecord?.azureApiVersion) || "2024-02-15-preview",
      });
    case "anthropic":
      return AiSdkAnthropicProvider({
        apiKey,
        baseURL,
      });
    case "deepseek":
      return AiSdkDeepSeekProvider({
        apiKey,
        baseURL,
      });
    case "groq":
      return AiSdkGroqProvider({
        apiKey,
        baseURL,
      });
    case "cerebras":
      return AiSdkCerebrasProvider({
        apiKey,
        baseURL,
      });
    default:
      throw new Error(`Provider-Typ nicht angebunden: ${type || "unknown"}`);
  }
}

function createLanguageModel({ provider, modelName, parameters, reasoningEffort }) {
  const providerFactory = createAiSdkProvider(provider);
  const options = {
    ...(parameters && typeof parameters === "object" ? parameters : {}),
  };

  if (isOpenAiReasoningModel(modelName) && asText(reasoningEffort)) {
    options.reasoningEffort = asText(reasoningEffort);
  }

  return providerFactory(modelName, options);
}

function assertSupportedSlot(slotId) {
  const normalized = asText(slotId) || "agent";
  if (!SUPPORTED_SLOT_IDS.has(normalized)) {
    throw new Error(`Unbekannter Slot: ${normalized}`);
  }
  return normalized;
}

export async function loadProviders() {
  return configApi.readProviders();
}

export async function loadModelSlots(providers) {
  return configApi.readModelSlots(providers);
}

export async function getConfigSnapshot() {
  const providers = await loadProviders();
  const slots = await loadModelSlots(providers);
  const sanitizedProviders = Object.fromEntries(
    Object.entries(providers || {}).map(([id, provider]) => [
      id,
      sanitizeProvider({ ...provider, id }),
    ]),
  );
  const slotRefs = Object.fromEntries(
    Object.keys(slots || {}).map((slotId) => [slotId, configApi.getModelRefForSlot(slotId, slots)]),
  );

  return {
    providers: sanitizedProviders,
    slots,
    slotRefs,
  };
}

export async function resolveProviderAndModel({
  slotId = "agent",
  modelRef = "",
  providerId = "",
  modelName = "",
  parameters = {},
  reasoningEffort = "",
} = {}) {
  const providers = await loadProviders();
  const slots = await loadModelSlots(providers);
  const normalizedSlotId = assertSupportedSlot(slotId);
  const slotConfig = slots?.[normalizedSlotId] || {};

  let resolvedProviderId = "";
  let resolvedModelName = "";

  if (asText(modelRef)) {
    const parsed = configApi.parseModelRef(modelRef);
    resolvedProviderId = canonicalizeProviderId(parsed.providerId);
    resolvedModelName = asText(parsed.modelName);
  } else if (asText(providerId) && asText(modelName)) {
    resolvedProviderId = canonicalizeProviderId(providerId);
    resolvedModelName = asText(modelName);
  } else if (asText(providerId)) {
    resolvedProviderId = canonicalizeProviderId(providerId);
    resolvedModelName = asText(modelName);
  } else {
    resolvedProviderId = canonicalizeProviderId(slots?.[normalizedSlotId]?.providerId);
    resolvedModelName = asText(slots?.[normalizedSlotId]?.modelName);
  }

  if (!resolvedProviderId) {
    throw new Error(`Kein Modell fuer Slot ${normalizedSlotId} konfiguriert.`);
  }

  const provider = providers?.[resolvedProviderId];
  if (!provider) {
    throw new Error(`Provider nicht gefunden: ${resolvedProviderId}`);
  }
  if (!resolvedModelName) {
    resolvedModelName = getDefaultModelNameForProvider(
      {
        ...provider,
        id: resolvedProviderId,
      },
      slotConfig,
    );
  }
  if (!resolvedModelName) {
    throw new Error(`Kein Modell fuer Slot ${normalizedSlotId} konfiguriert.`);
  }
  if (provider.enabled === false && provider.type !== "local_qwen") {
    throw new Error(`Provider ist deaktiviert: ${resolvedProviderId}`);
  }
  if (!asText(provider.apiKey) && provider.type !== "local_qwen" && provider.type !== "ollama") {
    throw new Error(`API-Key fehlt fuer Provider: ${resolvedProviderId}`);
  }

  const slotValidation = configApi.validateSlotSelection(
    normalizedSlotId,
    provider,
    resolvedModelName,
  );
  if (!slotValidation.ok) {
    throw new Error(slotValidation.reason || `Slot ${normalizedSlotId} ist ungueltig konfiguriert.`);
  }

  return {
    slotId: normalizedSlotId,
    modelRef: configApi.buildModelRef(resolvedProviderId, resolvedModelName),
    providerId: resolvedProviderId,
    modelName: resolvedModelName,
    provider,
    parameters: parameters && typeof parameters === "object" ? parameters : {},
    reasoningEffort:
      asText(reasoningEffort) ||
      configApi.getReasoningEffortForSlot(normalizedSlotId, slots) ||
      asText(slotConfig?.reasoningEffort),
  };
}

export async function getLanguageModelForSlot(options = {}) {
  const resolved = await resolveProviderAndModel(options);
  return createLanguageModel(resolved);
}

export async function describeResolvedModel(options = {}) {
  const resolved = await resolveProviderAndModel(options);
  return summarizeResolvedConfig(resolved);
}

export async function llmChat({
  slotId = "agent",
  modelRef = "",
  providerId = "",
  modelName = "",
  messages = [],
  tools = [],
  promptText = "",
  stream = false,
  onToken,
  signal,
  parameters = {},
  reasoningEffort = "",
  localContext = {},
} = {}) {
  const resolved = await resolveProviderAndModel({
    slotId,
    modelRef,
    providerId,
    modelName,
    parameters,
    reasoningEffort,
  });
  const providerType = asText(resolved?.provider?.type).toLowerCase();
  const safeMessages = normalizeMessages(messages, {
    inlineImageMode: providerType === "local_qwen" ? "preserve" : "decode",
  });

  if (providerType === "local_qwen") {
    const response = await sendMessageToOffscreen("OFFSCREEN_LOCAL_QWEN_CHAT", {
      modelName: resolved.modelName,
      messages: safeMessages,
      tools: Array.isArray(tools) ? tools : [],
      promptText: asText(promptText),
      parameters: resolved.parameters,
      craftId: asText(localContext?.craftId),
    });
    if (!response?.ok) {
      const error = new Error(
        asText(response?.error) || "Local-Qwen-Offscreen-Call fehlgeschlagen.",
      );
      if (response?.errorDetail != null || response?.runtime != null) {
        const runtime =
          response?.runtime && typeof response.runtime === "object"
            ? { ...response.runtime }
            : null;
        if (response?.errorDetail && typeof response.errorDetail === "object" && !Array.isArray(response.errorDetail)) {
          error.detail = {
            ...response.errorDetail,
            ...(runtime ? { runtime } : {}),
          };
        } else {
          error.detail = {
            errorDetail: response?.errorDetail ?? null,
            ...(runtime ? { runtime } : {}),
          };
        }
      }
      throw error;
    }
    const text = String(response?.text || "");
    if (stream && typeof onToken === "function" && text) {
      onToken(text);
    }
    return {
      text,
      usage: null,
      finishReason: "stop",
      runtime: response?.runtime || null,
      resolved: {
        ...summarizeResolvedConfig(resolved),
        localRuntime: response?.runtime || null,
      },
    };
  }

  const model = createLanguageModel(resolved);

  if (stream) {
    const { textStream } = await streamText({
      model,
      messages: safeMessages,
      abortSignal: signal,
    });
    let text = "";
    for await (const chunk of textStream) {
      text += String(chunk || "");
      if (typeof onToken === "function") onToken(chunk);
    }
    return {
      text,
      resolved: summarizeResolvedConfig(resolved),
    };
  }

  const response = await generateText({
    model,
    messages: safeMessages,
    abortSignal: signal,
  });

  return {
    text: String(response?.text || ""),
    usage: response?.usage || null,
    finishReason: response?.finishReason || "",
    resolved: summarizeResolvedConfig(resolved),
  };
}

export async function generateStructuredOutput({
  slotId = "agent",
  modelRef = "",
  providerId = "",
  modelName = "",
  messages = [],
  schema = null,
  schemaName = "structured_output",
  jsonSchema = null,
  schemaDescription = "",
  signal,
  parameters = {},
  reasoningEffort = "",
  localContext = {},
} = {}) {
  const resolved = await resolveProviderAndModel({
    slotId,
    modelRef,
    providerId,
    modelName,
    parameters,
    reasoningEffort,
  });
  const providerType = asText(resolved?.provider?.type).toLowerCase();
  const safeMessages = normalizeMessages(messages, {
    inlineImageMode: providerType === "local_qwen" ? "preserve" : "decode",
  });
  const schemaPreview = stringifySchemaPreview(jsonSchema);

  if (providerType === "local_qwen") {
    return await recoverStructuredOutputViaRawText({
      resolved,
      safeMessages,
      schema,
      schemaName,
      schemaDescription,
      schemaPreview,
      signal,
      parameters,
      localContext,
    });
  }

  if ((!schema || typeof schema !== "object") && (!jsonSchema || typeof jsonSchema !== "object")) {
    throw new Error(`Structured output for ${schemaName} requires a schema.`);
  }
  const model = createLanguageModel(resolved);
  const structuredSchema = buildStructuredOutputSchema(schema, jsonSchema);
  try {
    const response = await generateObject({
      model,
      schema: structuredSchema,
      schemaName: asText(schemaName) || "structured_output",
      messages: safeMessages,
      abortSignal: signal,
    });

    return {
      object: response?.object || null,
      usage: response?.usage || null,
      finishReason: response?.finishReason || "",
      resolved: summarizeResolvedConfig(resolved),
    };
  } catch (generateObjectError) {
    try {
      return await recoverStructuredOutputViaRawText({
        resolved,
        safeMessages,
        schema,
        schemaName,
        schemaDescription,
        schemaPreview,
        signal,
        parameters,
        localContext,
      });
    } catch (repairError) {
      if (repairError && typeof repairError === "object") {
        const detail =
          repairError.detail && typeof repairError.detail === "object" && !Array.isArray(repairError.detail)
            ? repairError.detail
            : {};
        repairError.detail = {
          ...detail,
          generateObjectError:
            generateObjectError && typeof generateObjectError === "object" && "detail" in generateObjectError
              ? generateObjectError.detail || null
              : generateObjectError instanceof Error
                ? generateObjectError.message
                : String(generateObjectError || ""),
        };
      }
      throw repairError;
    }
  }
}

export async function planTrainingSampleOps({
  slotId = "agent",
  modelRef = "",
  providerId = "",
  modelName = "",
  craft = null,
  brief = "",
  currentSamples = [],
  parameters = {},
  reasoningEffort = "",
} = {}) {
  const resolved = await resolveProviderAndModel({
    slotId,
    modelRef,
    providerId,
    modelName,
    parameters,
    reasoningEffort,
  });
  const agentTooling = normalizeCraftingAgentToolingPayload(craft?.agentTooling);
  const agentToolingLabel = formatCraftingAgentToolLabels(agentTooling) || "Web Search + Browser Inspect + Browser Action + Browser Tabs + Playwright CTX";

  if (asText(resolved?.provider?.type).toLowerCase() === "local_qwen") {
    const localRequest = buildLocalTrainingSampleOpsInput(craft, brief, currentSamples);
    const localShape = buildLocalTrainingSampleOpsShape();
    const maxTokens = Math.max(
      160,
      Math.min(
        640,
        Number(
          resolved?.parameters?.maxTokens ||
            resolved?.parameters?.max_new_tokens ||
            420,
        ) || 420,
      ),
    );
    const response = await llmChat({
      slotId: resolved.slotId,
      modelRef: resolved.modelRef,
      parameters: {
        ...(resolved.parameters && typeof resolved.parameters === "object" ? resolved.parameters : {}),
        temperature: 0,
        maxTokens,
      },
      reasoningEffort: resolved.reasoningEffort,
      messages: [
        {
          role: "system",
          content: [
            "You are a strict JSON generator for training-sample CRUD plans.",
            "Return exactly one valid JSON object and nothing else.",
            "Never use markdown fences, bullet points, comments, or prose.",
            "Use only double-quoted JSON strings and keys.",
            "The top-level keys must be summary, rationale, report, openQuestions, provenance, useSurface, and operations.",
            "Do not return wrapper keys like task, outputShape, plan, result, or data.",
            "Never echo the request payload back to the user.",
            "Do not copy placeholder strings from outputShape.",
            `The crafting supervisor always has these fixed tools available: ${agentToolingLabel}.`,
            "Do not ask the user to choose tools before the run; that chooser is intentionally hidden in the sidepanel.",
            "Use web_search first for URLs and references, then use browser_tabs/browser_inspect/browser_action for visible browser work, and playwright_ctx for deterministic in-tab DOM work.",
            "Treat browser_inspect as the reviewed local Qwen vision path for visible browser state.",
            "operations must be an array of add, update, or delete objects.",
            "report should explain objective, matched state, next action, and up to a few matchingSignals.",
            "openQuestions should only be present when critical task details are missing.",
            "provenance should capture milestone highlights such as applied edits, grounded browser evidence, or explicit user decisions.",
            "Write summary, report, and provenance in plain German for the sidepanel UI.",
            "Avoid internal jargon like bootstrap, starter row, contract, schema, reviewed pattern, or lexical variety.",
            "Use provenance only for milestone highlights such as applied edits, grounded browser evidence, or explicit user decisions.",
            "useSurface should describe how the finished ability should be executed in the sidepanel.",
            "Set useSurface.inputMode to selection, current_tab, or context_only when the finished ability should run without an extra text field.",
            "Set useSurface.inputMode to free_text or mixed only when typed input is genuinely needed.",
            "useSurface.inputExamples should contain short realistic example prompts or execution cases.",
            "For add operations omit sampleId.",
            "For update and delete operations include sampleId.",
            "If fields is present, use only a native Qwen multi-turn training row with messages, tools, and targetTurnIndex.",
            "If fields is present, it may contain promptText, messages, tools, targetTurnIndex, split, status, and source.",
            "For multi-turn agent rows, targetTurnIndex must point at the assistant turn that should be supervised next.",
            "For multi-turn agent rows, include the full reviewed transcript before that target turn, including every earlier tool response message.",
            "Do not stop at naming a tool call when the next assistant step depends on tool output; persist the matching tool response turn in messages.",
            "When the reviewed next action depends on screenshot-visible UI state, keep the row multimodal with text and image content parts.",
            "Do not rewrite screenshot-only evidence into text-only prompts when the row is meant to train or repair the vision path.",
            "Do not emit flat promptText plus expectedJson training rows for the local_qwen target model.",
            "Prefer 0 to 2 precise operations over broad speculative rewrites.",
          ].join("\n"),
        },
        {
          role: "user",
          content: JSON.stringify(
            {
              task: localRequest,
              outputShape: localShape,
            },
            null,
            2,
          ),
        },
      ],
    });

    try {
      return {
        object: parseTrainingSampleOpsText(response?.text || ""),
        usage: response?.usage || null,
        finishReason: response?.finishReason || "",
        resolved: response?.resolved || summarizeResolvedConfig(resolved),
      };
    } catch (parseError) {
      const repairResponse = await llmChat({
        slotId: resolved.slotId,
        modelRef: resolved.modelRef,
        parameters: {
          ...(resolved.parameters && typeof resolved.parameters === "object" ? resolved.parameters : {}),
          temperature: 0,
          maxTokens: Math.max(160, Math.min(512, maxTokens)),
        },
        reasoningEffort: resolved.reasoningEffort,
        messages: [
          {
            role: "system",
            content: [
              "You repair invalid model output into valid JSON.",
              "Return exactly one valid JSON object and nothing else.",
              "Use the same schema as the provided outputShape.",
              "Do not return wrapper keys like task, outputShape, plan, result, or data.",
              "Do not copy placeholder strings from outputShape.",
              "Preserve report, openQuestions, provenance, useSurface, and operations when they are recoverable.",
              "Do not add markdown, comments, or explanations.",
            ].join("\n"),
          },
          {
            role: "user",
            content: JSON.stringify(
              {
                outputShape: localShape,
                task: localRequest,
                candidateText: String(response?.text || ""),
              },
              null,
              2,
            ),
          },
        ],
      });

      try {
        return {
          object: parseTrainingSampleOpsText(repairResponse?.text || ""),
          usage: repairResponse?.usage || response?.usage || null,
          finishReason: repairResponse?.finishReason || response?.finishReason || "",
          resolved: repairResponse?.resolved || response?.resolved || summarizeResolvedConfig(resolved),
        };
      } catch (repairError) {
        const error = new Error("Local Qwen agent output was not valid JSON for training sample ops.");
        error.detail = {
          request: localRequest,
          outputShape: localShape,
          initialText: String(response?.text || ""),
          repairedText: String(repairResponse?.text || ""),
          initialParseError:
            parseError && typeof parseError === "object" && "detail" in parseError
              ? parseError.detail || null
              : parseError instanceof Error
                ? parseError.message
                : String(parseError || ""),
          repairParseError:
            repairError && typeof repairError === "object" && "detail" in repairError
              ? repairError.detail || null
              : repairError instanceof Error
                ? repairError.message
                : String(repairError || ""),
          resolved: response?.resolved || summarizeResolvedConfig(resolved),
        };
        throw error;
      }
    }

  }

  const model = createLanguageModel(resolved);
  const response = await generateObject({
    model,
    schema: TRAINING_SAMPLE_OPS_SCHEMA,
    schemaName: "training_sample_ops",
    messages: normalizeMessages([
      {
        role: "system",
        content: [
          "You supervise a browser-first small-model training factory.",
          "Return only structured operations for CRUD changes on training samples.",
          "Use add when a missing seed example should be created.",
          "Use update when an existing sample should be improved or normalized.",
          "Use delete only for duplicates, malformed rows, or rows that clearly harm the dataset.",
          "Keep operations seed-grounded and task-generic. The first task may be grammar/style correction, but do not hardcode that assumption.",
          "Every new or updated sample must preserve a stable supervised shape.",
          "For the local_qwen target model use only messages + tools + targetTurnIndex rows.",
          "For agentic tool-use tasks persist the full reviewed transcript before the supervised assistant turn, including every earlier tool response turn.",
          "Do not emit incomplete multi-turn rows that contain a prior assistant tool call without its matching tool response.",
          "Do not emit flat promptText + expectedJson rows for the local_qwen target model.",
          `The crafting supervisor always has these fixed tools available: ${agentToolingLabel}.`,
          "Do not ask the user to choose tools before the run; that chooser is intentionally hidden in the sidepanel.",
          "Use web_search for quick source discovery, browser_tabs/browser_inspect/browser_action for visible browser work, and playwright_ctx for deterministic in-tab DOM work.",
          "Treat browser_inspect as the reviewed local Qwen vision path for visible browser state.",
          "Use report to summarize the matched dataset state, the objective of this pass, and the next useful action.",
          "Use openQuestions only for missing information that blocks high-confidence sample generation.",
          "Use provenance only for milestone highlights such as applied edits, grounded browser evidence, or explicit user decisions.",
          "Write summary, report, and provenance in plain German for the sidepanel UI.",
          "Avoid internal jargon like bootstrap, starter row, contract, schema, reviewed pattern, or lexical variety.",
          "Prefer provenance entries that read like milestone highlights, not raw planning notes.",
          "Use useSurface to describe how the finished ability should be executed in the sidepanel.",
          "If the ability should run directly on browser selection, the current tab, or fixed context, do not require a free text input field.",
          "Provide a few realistic useSurface.inputExamples for hover help when possible.",
          "When the reviewed next action depends on screenshot-visible UI state, keep the row multimodal with text and image content parts.",
          "Do not rewrite screenshot-only evidence into text-only prompts when the row is meant to train or repair the vision path.",
          "Prefer a small high-confidence batch of edits over broad speculative rewrites.",
        ].join("\n"),
      },
      {
        role: "user",
        content: JSON.stringify({
          craft:
            craft && typeof craft === "object"
              ? {
                  ...craft,
                  agentTooling,
                }
              : {
                  agentTooling,
                },
          brief: String(brief || "").trim(),
          currentSamples: Array.isArray(currentSamples) ? currentSamples : [],
          allowedSplits: TRAINING_SAMPLE_SPLITS,
          allowedStatuses: TRAINING_SAMPLE_STATUSES,
          hardLimits: {
            maxOperations: 8,
            deleteRequiresSampleId: true,
            updateRequiresSampleId: true,
          },
        }, null, 2),
      },
    ]),
  });

  return {
    object: response?.object || null,
    usage: response?.usage || null,
    finishReason: response?.finishReason || "",
    resolved: summarizeResolvedConfig(resolved),
  };
}

export async function testModelConnection({
  slotId = "agent",
  modelRef = "",
  prompt = 'Return a single short JSON object like {"ok":true}.',
  parameters = {},
  reasoningEffort = "",
} = {}) {
  const result = await llmChat({
    slotId,
    modelRef,
    parameters,
    reasoningEffort,
    messages: [
      {
        role: "system",
        content: "Return only the requested answer. Keep it short.",
      },
      {
        role: "user",
        content: String(prompt || "").trim(),
      },
    ],
  });

  return {
    ok: true,
    text: result.text,
    usage: result.usage || null,
    runtime: result.runtime || null,
    modelRef: result.resolved.modelRef,
    providerId: result.resolved.providerId,
    modelName: result.resolved.modelName,
    slotId: result.resolved.slotId,
  };
}

async function resolveLocalQwenDiagnosticModel({
  slotId = "target",
  modelRef = "",
  parameters = {},
  reasoningEffort = "",
} = {}) {
  const resolved = await resolveProviderAndModel({
    slotId,
    modelRef,
    parameters,
    reasoningEffort,
  });

  if (asText(resolved?.provider?.type).toLowerCase() !== "local_qwen") {
    throw new Error(`Slot ${resolved.slotId} ist kein local_qwen-Modell.`);
  }
  return resolved;
}

export async function testLocalQwenDiagnostic({
  slotId = "target",
  modelRef = "",
  prompt = 'Return a single short JSON object like {"ok":true}.',
  parameters = {},
  reasoningEffort = "",
  craftId = "",
  benchmark = null,
} = {}) {
  const resolved = await resolveLocalQwenDiagnosticModel({
    slotId,
    modelRef,
    parameters,
    reasoningEffort,
  });

  const offscreen = await sendMessageToOffscreen("OFFSCREEN_LOCAL_QWEN_CHAT", {
    modelName: resolved.modelName,
    messages: normalizeMessages([
      {
        role: "system",
        content: "Return only the requested answer. Keep it short.",
      },
      {
        role: "user",
        content: String(prompt || "").trim(),
      },
    ]),
    parameters: resolved.parameters,
    craftId: asText(craftId),
  });

  let benchmarkResult = null;
  const benchmarkConfig =
    benchmark && typeof benchmark === "object"
      ? benchmark
      : null;
  const benchmarkRequested = Boolean(benchmarkConfig);
  if (offscreen?.ok && benchmarkConfig) {
    const benchmarkPrompt = asText(benchmarkConfig.promptText);
    const benchmarkResponse = await sendMessageToOffscreen("OFFSCREEN_LOCAL_QWEN_FORWARD_BENCHMARK", {
      modelName: resolved.modelName,
      messages: [],
      parameters: resolved.parameters,
      promptText: benchmarkPrompt,
      iterations: Number(benchmarkConfig.iterations || 0),
      warmupIterations: Number(benchmarkConfig.warmupIterations || 0),
    });
    benchmarkResult = benchmarkResponse && typeof benchmarkResponse === "object"
      ? { ...benchmarkResponse }
      : benchmarkResponse;
  }
  if (offscreen?.ok && benchmarkRequested && !benchmarkResult) {
    benchmarkResult = {
      ok: false,
      error:
        "The local forward benchmark response was missing. Reload the extension so the updated service worker and offscreen runtime are active, then retry.",
    };
  }

  return {
    ok: Boolean(offscreen?.ok),
    resolved: summarizeResolvedConfig(resolved),
    offscreen: offscreen && typeof offscreen === "object" ? { ...offscreen } : offscreen,
    benchmarkRequested,
    benchmark: benchmarkResult,
  };
}

export async function benchmarkLocalQwenForward({
  slotId = "target",
  modelRef = "",
  parameters = {},
  reasoningEffort = "",
  benchmarkId = "",
  promptText = "",
  iterations = 6,
  warmupIterations = 1,
} = {}) {
  const resolved = await resolveLocalQwenDiagnosticModel({
    slotId,
    modelRef,
    parameters,
    reasoningEffort,
  });

  const benchmarkResponse = await sendMessageToOffscreen("OFFSCREEN_LOCAL_QWEN_FORWARD_BENCHMARK", {
    modelName: resolved.modelName,
    messages: [],
    parameters: resolved.parameters,
    benchmarkId: asText(benchmarkId),
    promptText: asText(promptText),
    iterations: Number(iterations || 0),
    warmupIterations: Number(warmupIterations || 0),
  });
  const benchmark =
    benchmarkResponse && typeof benchmarkResponse === "object"
      ? { ...benchmarkResponse }
      : benchmarkResponse;

  if (!benchmark) {
    return {
      ok: false,
      resolved: summarizeResolvedConfig(resolved),
      benchmark: {
        ok: false,
        error:
          "The local forward benchmark response was missing. Reload the extension so the updated service worker and offscreen runtime are active, then retry.",
      },
    };
  }

  return {
    ok: Boolean(benchmark?.ok),
    resolved: summarizeResolvedConfig(resolved),
    benchmark,
  };
}
