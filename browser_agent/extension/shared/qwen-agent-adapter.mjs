import {
  buildCanonicalCallToolAction,
  canonicalToolActionToLegacyPayload,
  extractCanonicalToolName,
  normalizeCanonicalToolAction,
} from "./canonical-tool-action.mjs";
import { z } from "../vendor/agent-bundle.mjs";
import {
  buildBrowserCapabilityToolCatalog,
  normalizeBrowserCapabilityBundlePayload,
  resolveBrowserCapability,
} from "./browser-capability-bundle.mjs";
import {
  getQwen35NativeToolInstructionBlock,
  parseQwen35AssistantRawText,
} from "./qwen35-chat-template.mjs";

export const QWEN_AGENT_ADAPTER_VERSION = 1;
const PORTABLE_ASSISTANT_WRAPPER_KEYS = [
  "assistant_message",
  "message",
  "messages",
  "choices",
  "output",
  "result",
  "object",
  "data",
  "response",
];

function asText(value) {
  return String(value == null ? "" : value).trim();
}

function cloneJson(value, fallback = null) {
  try {
    if (typeof globalThis.structuredClone === "function") {
      return globalThis.structuredClone(value);
    }
    return JSON.parse(JSON.stringify(value));
  } catch {
    return fallback;
  }
}

function normalizeTextList(values, limit = 24) {
  return Array.from(
    new Set(
      (Array.isArray(values) ? values : [])
        .map((entry) => asText(entry))
        .filter(Boolean),
    ),
  ).slice(0, limit);
}

function escapeRegExp(value) {
  return asText(value).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

const PORTABLE_ASSISTANT_TOOL_CALL_SCHEMA = z.object({
  id: z.string().min(1).max(160),
  type: z.literal("function"),
  function: z.object({
    name: z.string().min(1).max(160),
    arguments: z.string().max(24_000),
  }).strict(),
}).strict();

export const PORTABLE_ASSISTANT_TURN_SCHEMA = z.object({
  role: z.literal("assistant"),
  content: z.string().max(24_000).optional(),
  tool_calls: z.array(PORTABLE_ASSISTANT_TOOL_CALL_SCHEMA).max(1).optional(),
}).strict().superRefine((value, context) => {
  const content = asText(value?.content);
  const toolCalls = Array.isArray(value?.tool_calls) ? value.tool_calls : [];
  if (!content && !toolCalls.length) {
    context.addIssue({
      code: z.ZodIssueCode.custom,
      message: "Assistant turn must include content or tool_calls.",
      path: ["content"],
    });
  }
});

function stringifyPortableValue(value) {
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value);
  } catch {
    return String(value == null ? "" : value);
  }
}

function parseJsonObjectText(text, fallback = {}) {
  try {
    const parsed = JSON.parse(asText(text));
    return parsed && typeof parsed === "object" && !Array.isArray(parsed)
      ? parsed
      : fallback;
  } catch {
    return fallback;
  }
}

function normalizePortableToolArguments(value) {
  if (typeof value === "string") {
    return asText(value) || "{}";
  }
  if (value && typeof value === "object" && !Array.isArray(value)) {
    return stringifyPortableValue(value);
  }
  return "{}";
}

function extractLeadingJsonObject(text = "") {
  const source = asText(text).replace(/^[\s:,-]+/, "");
  if (!source.startsWith("{")) return null;
  let depth = 0;
  let inString = false;
  let escaped = false;

  for (let index = 0; index < source.length; index += 1) {
    const char = source[index];
    if (inString) {
      if (escaped) {
        escaped = false;
        continue;
      }
      if (char === "\\") {
        escaped = true;
        continue;
      }
      if (char === "\"") {
        inString = false;
      }
      continue;
    }

    if (char === "\"") {
      inString = true;
      continue;
    }
    if (char === "{") {
      depth += 1;
      continue;
    }
    if (char !== "}") continue;
    depth -= 1;
    if (depth !== 0) continue;
    const candidate = source.slice(0, index + 1);
    try {
      const parsed = JSON.parse(candidate);
      return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : null;
    } catch {
      return null;
    }
  }

  return null;
}

function parseExplicitToolArguments(text = "") {
  const normalized = asText(text)
    .replace(/^[\s:,-]+/, "")
    .replace(/[.!?]\s*$/g, "")
    .trim();
  if (!normalized) return {};

  const withoutWith = normalized.replace(/^with\b\s*/i, "").trim();
  if (
    !withoutWith ||
    /^(?:an?\s+)?empty\s+(?:json\s+)?object$/i.test(withoutWith) ||
    /^(?:an?\s+)?empty\s+arguments?$/i.test(withoutWith) ||
    /^no\s+arguments?$/i.test(withoutWith) ||
    /^\{\s*\}$/i.test(withoutWith)
  ) {
    return {};
  }

  const extractedObject = extractLeadingJsonObject(withoutWith);
  return extractedObject ?? null;
}

function recoverExplicitPlainTextToolCall(text, {
  allowedToolNames = [],
  defaultCallId = "call_1",
} = {}) {
  const cleaned = asText(text);
  const normalizedTools = normalizeTextList(allowedToolNames, 200);
  if (!cleaned || !normalizedTools.length) return null;

  for (const toolName of normalizedTools) {
    const escapedToolName = escapeRegExp(toolName);
    const patterns = [
      new RegExp(`(?:^|[.!?\\n]\\s*)(?:please\\s+)?call\\s+${escapedToolName}\\b([\\s\\S]*)$`, "i"),
      new RegExp(`(?:^|[.!?\\n]\\s*)(?:please\\s+)?use\\s+${escapedToolName}\\b([\\s\\S]*)$`, "i"),
      new RegExp(`(?:^|[.!?\\n]\\s*)(?:please\\s+)?invoke\\s+${escapedToolName}\\b([\\s\\S]*)$`, "i"),
      new RegExp(`(?:^|[.!?\\n]\\s*)(?:please\\s+)?emit\\s+(?:the\\s+)?(?:tool\\s+call\\s+for\\s+)?${escapedToolName}\\b([\\s\\S]*)$`, "i"),
    ];
    for (const pattern of patterns) {
      const match = pattern.exec(cleaned);
      if (!match) continue;
      const argumentsValue = parseExplicitToolArguments(match[1] || "");
      if (argumentsValue == null) continue;
      const prefix = cleaned
        .slice(0, match.index)
        .replace(/[.!?]\s*$/g, "")
        .trim();
      const candidate = {
        role: "assistant",
        ...(prefix ? { content: prefix } : { content: "" }),
        tool_calls: [
          {
            id: defaultCallId,
            type: "function",
            function: {
              name: toolName,
              arguments: stringifyPortableValue(argumentsValue),
            },
          },
        ],
      };
      const parsed = PORTABLE_ASSISTANT_TURN_SCHEMA.safeParse(candidate);
      if (parsed.success) {
        return parsed.data;
      }
    }
  }

  return null;
}

function recoverBareToolStubCall(text, {
  allowedToolNames = [],
  defaultCallId = "call_1",
} = {}) {
  const cleaned = asText(text);
  const normalizedTools = normalizeTextList(allowedToolNames, 200);
  if (!cleaned || !normalizedTools.length) return null;

  for (const toolName of normalizedTools) {
    const escapedToolName = escapeRegExp(toolName);
    const objectMatch = new RegExp(`^${escapedToolName}\\s*:?[\\s]+([\\s\\S]+)$`, "i").exec(cleaned);
    if (objectMatch) {
      const argumentsValue = parseExplicitToolArguments(objectMatch[1] || "");
      if (argumentsValue != null) {
        return PORTABLE_ASSISTANT_TURN_SCHEMA.parse({
          role: "assistant",
          content: "",
          tool_calls: [
            {
              id: defaultCallId,
              type: "function",
              function: {
                name: toolName,
                arguments: stringifyPortableValue(argumentsValue),
              },
            },
          ],
        });
      }
    }

    const parenthesizedMatch = new RegExp(`^${escapedToolName}\\s*\\(([\\s\\S]*)\\)$`, "i").exec(cleaned);
    if (parenthesizedMatch) {
      const inner = asText(parenthesizedMatch[1]);
      const argumentsValue = inner ? parseExplicitToolArguments(inner) : {};
      if (argumentsValue != null) {
        return PORTABLE_ASSISTANT_TURN_SCHEMA.parse({
          role: "assistant",
          content: "",
          tool_calls: [
            {
              id: defaultCallId,
              type: "function",
              function: {
                name: toolName,
                arguments: stringifyPortableValue(argumentsValue),
              },
            },
          ],
        });
      }
    }
  }

  return null;
}

function coercePortableAssistantTurnCandidate(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const toolCalls = (Array.isArray(value.tool_calls) ? value.tool_calls : [])
    .map((entry, index) => {
      const source = entry && typeof entry === "object" ? entry : {};
      const fn = source.function && typeof source.function === "object" ? source.function : {};
      const name = asText(fn.name || source.name);
      if (!name) return null;
      return {
        id: asText(source.id) || `call_${index + 1}`,
        type: "function",
        function: {
          name,
          arguments: normalizePortableToolArguments(fn.arguments),
        },
      };
    })
    .filter(Boolean)
    .slice(0, 1);
  const candidate = {
    role: "assistant",
    content: asText(value.content),
    ...(toolCalls.length ? { tool_calls: toolCalls } : {}),
  };
  const parsed = PORTABLE_ASSISTANT_TURN_SCHEMA.safeParse(candidate);
  return parsed.success ? parsed.data : null;
}

function unwrapPortableAssistantTurn(value, seen = new Set()) {
  if (Array.isArray(value)) {
    for (const entry of value) {
      const nested = unwrapPortableAssistantTurn(entry, seen);
      if (nested) return nested;
    }
    return null;
  }
  if (!value || typeof value !== "object") return null;
  if (seen.has(value)) return null;
  seen.add(value);

  const direct = coercePortableAssistantTurnCandidate(value);
  if (direct) return direct;

  for (const key of PORTABLE_ASSISTANT_WRAPPER_KEYS) {
    const nested = unwrapPortableAssistantTurn(value?.[key], seen);
    if (nested) return nested;
  }

  for (const nestedValue of Object.values(value)) {
    if (!nestedValue || typeof nestedValue !== "object") continue;
    const nested = unwrapPortableAssistantTurn(nestedValue, seen);
    if (nested) return nested;
  }

  return null;
}

function resolveCapabilityBundlePayload(capabilityBundle = null, {
  craft = null,
  toolScriptsPayload = null,
  normalizationOptions = null,
} = {}) {
  return normalizeBrowserCapabilityBundlePayload(
    capabilityBundle,
    craft,
    toolScriptsPayload,
    normalizationOptions,
  );
}

export function getQwenAgentCapabilityProfile({
  providerType = "",
  modelName = "",
  runtimeKind = "",
} = {}) {
  const provider = asText(providerType).toLowerCase();
  const model = asText(modelName).toLowerCase();
  const runtime = asText(runtimeKind).toLowerCase();
  const isLocal = provider === "local_qwen" || runtime === "webgpu_local";
  const isHostedCompatible = ["custom_openai", "openrouter", "ollama", "openai"].includes(provider);
  const shouldPreferSingleActionDecisions = isLocal;
  return {
    adapterVersion: QWEN_AGENT_ADAPTER_VERSION,
    providerType: provider,
    modelName: model,
    runtimeKind: runtime,
    supportsNativeToolCalling: isLocal || isHostedCompatible,
    supportsStrictStructuredOutputs: !isLocal && isHostedCompatible,
    supportsReasoningChannel: model.includes("thinking") || model.includes("reason"),
    requiresCanonicalActionAdapter: false,
    shouldPreferSingleActionDecisions,
  };
}

export function buildQwenCapabilityCatalogPayload(capabilityBundle = null, options = {}) {
  const normalizedBundle = resolveCapabilityBundlePayload(capabilityBundle, options);
  const capabilityCatalog = buildBrowserCapabilityToolCatalog(normalizedBundle, {
    toolScriptsPayload: options?.toolScriptsPayload || null,
    normalizationOptions: options?.normalizationOptions || null,
  });
  return {
    adapterVersion: QWEN_AGENT_ADAPTER_VERSION,
    actionProtocolVersion: normalizedBundle.actionProtocolVersion,
    skills: normalizeTextList(normalizedBundle.skills, 24),
    resources: normalizeTextList(normalizedBundle.resources, 48),
    capabilities: capabilityCatalog.map((entry) => ({
      id: entry.id,
      toolName: entry.toolName,
      displayName: asText(entry.displayName),
      description: entry.description,
      parameterSchema: cloneJson(entry.parameterSchema, {}),
      returnSchema: cloneJson(entry.returnSchema, {}),
      preconditions: normalizeTextList(entry.preconditions, 24),
      skillRef: asText(entry.skillRef),
      resourceRefs: normalizeTextList(entry.resourceRefs, 24),
    })),
  };
}

export function buildQwenCapabilityToolDefinitions(capabilityBundle = null, options = {}) {
  return buildQwenCapabilityCatalogPayload(capabilityBundle, options).capabilities.map((entry) => ({
    type: "function",
    function: {
      name: entry.toolName,
      description: entry.description,
      parameters: cloneJson(entry.parameterSchema, {}),
    },
  }));
}

export function buildQwenCapabilitySkillText(capabilityBundle = null, {
  outputSurface = "portable_json",
  normalizationOptions = null,
  craft = null,
  toolScriptsPayload = null,
} = {}) {
  const catalog = buildQwenCapabilityCatalogPayload(capabilityBundle, {
    normalizationOptions,
    craft,
    toolScriptsPayload,
  });
  const nativeRaw = asText(outputSurface).toLowerCase() === "qwen_native_raw";
  const lines = [
    `Canonical action protocol version: ${catalog.actionProtocolVersion}.`,
    "Return one next assistant turn at a time.",
    nativeRaw
      ? "If a reviewed browser capability should run next, emit exactly one native Qwen3.5 raw tool call."
      : "Use assistant tool_calls to invoke at most one reviewed browser capability.",
    nativeRaw
      ? "If required context is missing, ask the user directly in plain assistant text before any tool call."
      : "If required context is missing, ask the user directly in assistant content.",
    nativeRaw
      ? "If the task is done or blocked, return plain assistant text without any JSON wrapper."
      : "If the task is done or blocked, return the final assistant response in content.",
  ];
  if (nativeRaw) {
    lines.push("Native Qwen3.5 tool-call format:");
    lines.push(getQwen35NativeToolInstructionBlock());
  }
  if (catalog.skills.length) {
    lines.push("Bundle skills:");
    for (const skill of catalog.skills.slice(0, 24)) {
      lines.push(`- ${skill}`);
    }
  }
  if (catalog.resources.length) {
    lines.push(`Bundle resources: ${catalog.resources.join(", ")}.`);
  }
  if (catalog.capabilities.length) {
    lines.push("Reviewed capabilities:");
    for (const capability of catalog.capabilities.slice(0, 24)) {
      const label =
        capability.displayName && capability.displayName !== capability.toolName
          ? `${capability.toolName} (${capability.displayName})`
          : capability.toolName;
      lines.push(
        `- ${label}: ${capability.description || "No description"}`,
      );
    }
  }
  return lines.join("\n");
}

export function buildPortableAssistantTurnJsonSchema({
  allowedToolNames = [],
  maxToolCalls = 1,
} = {}) {
  const normalizedTools = normalizeTextList(allowedToolNames, 200);
  return {
    type: "object",
    additionalProperties: false,
    required: ["role"],
    properties: {
      role: { type: "string", enum: ["assistant"] },
      content: { type: "string" },
      tool_calls: {
        type: "array",
        minItems: 1,
        maxItems: Math.max(1, Number(maxToolCalls || 1)),
        items: {
          type: "object",
          additionalProperties: false,
          required: ["id", "type", "function"],
          properties: {
            id: { type: "string" },
            type: { type: "string", enum: ["function"] },
            function: {
              type: "object",
              additionalProperties: false,
              required: ["name", "arguments"],
              properties: {
                name: normalizedTools.length
                  ? { type: "string", enum: normalizedTools }
                  : { type: "string" },
                arguments: { type: "string" },
              },
            },
          },
        },
      },
    },
    anyOf: [
      { required: ["content"] },
      { required: ["tool_calls"] },
    ],
  };
}

export function buildPortableAssistantTurnFromCanonicalAction(action, {
  callId = "",
} = {}) {
  const normalized = normalizeCanonicalToolAction(action, { passThroughUnknown: false });
  if (!normalized) return null;
  if (normalized.type === "call_tool") {
    return PORTABLE_ASSISTANT_TURN_SCHEMA.parse({
      role: "assistant",
      content: "",
      tool_calls: [{
        id: asText(callId) || `call_${asText(normalized.tool_name).replace(/[^a-z0-9]+/gi, "_").toLowerCase() || "1"}`,
        type: "function",
        function: {
          name: asText(normalized.tool_name),
          arguments: stringifyPortableValue(normalized.arguments || {}),
        },
      }],
    });
  }
  if (normalized.type === "ask_user") {
    const content = normalizeTextList(
      (Array.isArray(normalized.questions) ? normalized.questions : []).map((entry) => entry?.question),
      6,
    ).join("\n");
    return PORTABLE_ASSISTANT_TURN_SCHEMA.parse({
      role: "assistant",
      content: content || "Please provide the missing information.",
    });
  }
  if (normalized.type === "finish") {
    const content =
      asText(normalized?.result?.text) ||
      asText(normalized?.result?.summary) ||
      asText(normalized?.summary) ||
      stringifyPortableValue(normalized.result ?? "");
    return PORTABLE_ASSISTANT_TURN_SCHEMA.parse({
      role: "assistant",
      content: content || "Task completed.",
    });
  }
  return null;
}

export function normalizePortableAssistantTurn(value, {
  allowedToolNames = [],
  allowHeuristicRecovery = true,
} = {}) {
  const normalizedCanonical = normalizeCanonicalToolAction(value, { passThroughUnknown: false });
  if (normalizedCanonical) {
    return buildPortableAssistantTurnFromCanonicalAction(normalizedCanonical);
  }
  if (typeof value === "string") {
    const normalizedTools = normalizeTextList(allowedToolNames, 200);
    const parsedRaw = parseQwen35AssistantRawText(value, {
      allowedToolNames: normalizedTools,
    });
    if (Array.isArray(parsedRaw?.tool_calls) && parsedRaw.tool_calls.length) {
      return parsedRaw;
    }
    if (allowHeuristicRecovery) {
      const recoveredBareStub = recoverBareToolStubCall(value, {
        allowedToolNames: normalizedTools,
      });
      if (recoveredBareStub) return recoveredBareStub;
      const recovered = recoverExplicitPlainTextToolCall(value, {
        allowedToolNames: normalizedTools,
      });
      if (recovered) return recovered;
    }
    return parsedRaw;
  }
  const rawText = asText(value?.text || value?.content);
  if (rawText) {
    const normalizedTools = normalizeTextList(allowedToolNames, 200);
    const parsedRaw = parseQwen35AssistantRawText(rawText, {
      allowedToolNames: normalizedTools,
    });
    if (Array.isArray(parsedRaw?.tool_calls) && parsedRaw.tool_calls.length) {
      return parsedRaw;
    }
    if (allowHeuristicRecovery) {
      const recoveredBareStub = recoverBareToolStubCall(rawText, {
        allowedToolNames: normalizedTools,
      });
      if (recoveredBareStub) return recoveredBareStub;
      const recovered = recoverExplicitPlainTextToolCall(rawText, {
        allowedToolNames: normalizedTools,
      });
      if (recovered) return recovered;
    }
    if (parsedRaw) return parsedRaw;
  }
  const direct = unwrapPortableAssistantTurn(value);
  if (!direct) return null;
  const normalizedTools = new Set(normalizeTextList(allowedToolNames, 200));
  if (normalizedTools.size > 0) {
    for (const toolCall of Array.isArray(direct.tool_calls) ? direct.tool_calls : []) {
      if (!normalizedTools.has(asText(toolCall?.function?.name))) {
        return null;
      }
    }
  }
  return direct;
}

export function convertPortableAssistantTurnToCanonicalAction(turn, {
  defaultBundleRef = "",
  defaultSkillRef = "",
  defaultResourceRefs = [],
} = {}) {
  const normalized = normalizePortableAssistantTurn(turn);
  if (!normalized) return null;
  const toolCall = Array.isArray(normalized.tool_calls) ? normalized.tool_calls[0] : null;
  const toolName = asText(toolCall?.function?.name);
  if (!toolName) return null;
  return buildCanonicalCallToolAction({
    toolName,
    argumentsValue: parseJsonObjectText(toolCall?.function?.arguments, {}),
    bundleRef: defaultBundleRef,
    capabilityRef: toolName,
    skillRef: defaultSkillRef,
    resourceRefs: defaultResourceRefs,
  });
}

export function compileCanonicalActionExecutionPlan(action, capabilityBundle = null, options = {}) {
  const normalizedAction = normalizeCanonicalToolAction(action, { passThroughUnknown: false });
  if (!normalizedAction || normalizedAction.type !== "call_tool") return null;
  const normalizedBundle = resolveCapabilityBundlePayload(capabilityBundle, options);
  const capability = resolveBrowserCapability(normalizedBundle, normalizedAction.tool_name, {
    toolScriptsPayload: options?.toolScriptsPayload || null,
    normalizationOptions: options?.normalizationOptions || null,
  });
  if (!capability) {
    return {
      ok: false,
      error: `Unknown reviewed capability: ${normalizedAction.tool_name}`,
      action: normalizedAction,
      capability: null,
    };
  }
  const fullCapability = normalizedBundle.capabilities.find((entry) => entry.id === capability.id) || null;
  return {
    ok: true,
    action: normalizedAction,
    capability: cloneJson(fullCapability, null),
    invocation: {
      toolName: capability.toolName,
      arguments: cloneJson(normalizedAction.arguments, {}),
      parameterSchema: cloneJson(capability.parameterSchema, {}),
      returnSchema: cloneJson(capability.returnSchema, {}),
      scripts: {
        pre: asText(fullCapability?.scripts?.pre),
        execute: asText(fullCapability?.scripts?.execute),
        post: asText(fullCapability?.scripts?.post),
      },
    },
  };
}

export function convertQwenActionToLegacyToolPayload(action) {
  return canonicalToolActionToLegacyPayload(action);
}

export function buildCanonicalToolTrainingExample({
  promptText = "",
  toolName = "",
  argumentsValue = {},
  bundleRef = "",
  capabilityRef = "",
  skillRef = "",
  resourceRefs = [],
}) {
  const normalizedPrompt = asText(promptText);
  const normalizedToolName = asText(toolName);
  const normalizedArguments = cloneJson(argumentsValue, {});
  return {
    prompt_text: normalizedPrompt,
    messages: [
      {
        role: "system",
        content: "You are a browser-safe agent. Decide the next assistant action using reviewed tools only.",
      },
      {
        role: "user",
        content: normalizedPrompt,
      },
      {
        role: "assistant",
        content: "",
        tool_calls: [
          {
            id: `call_${normalizedToolName.replace(/[^a-z0-9]+/gi, "_").toLowerCase() || "1"}`,
            type: "function",
            function: {
              name: normalizedToolName,
              arguments: JSON.stringify(normalizedArguments),
            },
          },
        ],
      },
    ],
    tools: [
      {
        type: "function",
        function: {
          name: normalizedToolName,
          description: `Reviewed capability ${normalizedToolName}.`,
          parameters: {
            type: "object",
            ...(normalizedArguments && typeof normalizedArguments === "object" && !Array.isArray(normalizedArguments)
              ? {
                  properties: Object.fromEntries(
                    Object.keys(normalizedArguments).map((key) => [key, { type: "string" }]),
                  ),
                }
              : {}),
          },
        },
      },
    ],
    target_turn_index: 2,
    output_mode: "multiturn_tool_agent",
  };
}
