import { z } from "../vendor/agent-bundle.mjs";

export const CANONICAL_TOOL_ACTION_PROTOCOL_VERSION = 1;

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

function parseJsonObjectText(value, fallback = {}) {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    return cloneJson(value, fallback);
  }
  const text = asText(value);
  if (!text) return fallback;
  try {
    const parsed = JSON.parse(text);
    return parsed && typeof parsed === "object" && !Array.isArray(parsed)
      ? parsed
      : fallback;
  } catch {
    return fallback;
  }
}

function normalizeQuestionEntries(value) {
  return (Array.isArray(value) ? value : [])
    .map((entry, index) => {
      const source = entry && typeof entry === "object" ? entry : {};
      const question = asText(source.question || source.prompt || source.text);
      if (!question) return null;
      return {
        id: asText(source.id) || `question-${index + 1}`,
        question,
        reason: asText(source.reason || source.why) || null,
      };
    })
    .filter(Boolean)
    .slice(0, 6);
}

const nullableString = (max) => z.string().max(max).nullable();
const CANONICAL_QUESTION_ENTRY_SCHEMA = z.object({
  id: nullableString(80),
  question: z.string().min(3).max(320),
  reason: nullableString(240),
}).strict();

export const LEGACY_TOOL_CALL_PAYLOAD_SCHEMA = z.object({
  tool_name: z.string().min(1).max(160),
  arguments: z.object({}).catchall(z.unknown()),
}).strict();

export const CANONICAL_CALL_TOOL_ACTION_SCHEMA = z.object({
  protocol_version: z.literal(CANONICAL_TOOL_ACTION_PROTOCOL_VERSION),
  type: z.literal("call_tool"),
  tool_name: z.string().min(1).max(160),
  arguments: z.object({}).catchall(z.unknown()),
  bundle_ref: nullableString(240),
  capability_ref: nullableString(240),
  skill_ref: nullableString(240),
  resource_refs: z.array(z.string().max(240)).max(24),
}).strict();

export const CANONICAL_ASK_USER_ACTION_SCHEMA = z.object({
  protocol_version: z.literal(CANONICAL_TOOL_ACTION_PROTOCOL_VERSION),
  type: z.literal("ask_user"),
  questions: z.array(CANONICAL_QUESTION_ENTRY_SCHEMA).min(1).max(6),
  bundle_ref: nullableString(240),
  skill_ref: nullableString(240),
}).strict();

export const CANONICAL_FINISH_ACTION_SCHEMA = z.object({
  protocol_version: z.literal(CANONICAL_TOOL_ACTION_PROTOCOL_VERSION),
  type: z.literal("finish"),
  result: z.unknown(),
  summary: nullableString(800),
  bundle_ref: nullableString(240),
  skill_ref: nullableString(240),
}).strict();

export const CANONICAL_TOOL_ACTION_SCHEMA = z.object({
  protocol_version: z.literal(CANONICAL_TOOL_ACTION_PROTOCOL_VERSION),
  type: z.enum(["call_tool", "ask_user", "finish"]),
  tool_name: z.string().min(1).max(160).optional(),
  arguments: z.object({}).catchall(z.unknown()).optional(),
  bundle_ref: nullableString(240).optional(),
  capability_ref: nullableString(240).optional(),
  skill_ref: nullableString(240).optional(),
  resource_refs: z.array(z.string().max(240)).max(24).optional(),
  questions: z.array(CANONICAL_QUESTION_ENTRY_SCHEMA).min(1).max(6).optional(),
  result: z.unknown().optional(),
  summary: nullableString(800).optional(),
}).strict();

function parseCanonicalToolAction(value) {
  const parsed = CANONICAL_TOOL_ACTION_SCHEMA.safeParse(value);
  if (!parsed.success) return null;
  const candidate = parsed.data;
  const actionType = asText(candidate.type).toLowerCase();
  if (actionType === "call_tool") {
    if (
      !asText(candidate.tool_name) ||
      !candidate.arguments ||
      typeof candidate.arguments !== "object" ||
      Array.isArray(candidate.arguments) ||
      !Array.isArray(candidate.resource_refs)
    ) {
      return null;
    }
    return CANONICAL_CALL_TOOL_ACTION_SCHEMA.parse({
      protocol_version: candidate.protocol_version,
      type: "call_tool",
      tool_name: candidate.tool_name,
      arguments: candidate.arguments,
      bundle_ref: candidate.bundle_ref ?? null,
      capability_ref: candidate.capability_ref ?? null,
      skill_ref: candidate.skill_ref ?? null,
      resource_refs: candidate.resource_refs,
    });
  }
  if (actionType === "ask_user") {
    if (!Array.isArray(candidate.questions) || candidate.questions.length < 1) {
      return null;
    }
    return CANONICAL_ASK_USER_ACTION_SCHEMA.parse({
      protocol_version: candidate.protocol_version,
      type: "ask_user",
      questions: candidate.questions,
      bundle_ref: candidate.bundle_ref ?? null,
      skill_ref: candidate.skill_ref ?? null,
    });
  }
  if (!Object.prototype.hasOwnProperty.call(candidate, "result")) {
    return null;
  }
  return CANONICAL_FINISH_ACTION_SCHEMA.parse({
    protocol_version: candidate.protocol_version,
    type: "finish",
    result: candidate.result,
    summary: candidate.summary ?? null,
    bundle_ref: candidate.bundle_ref ?? null,
    skill_ref: candidate.skill_ref ?? null,
  });
}

export function buildCanonicalCallToolAction({
  toolName = "",
  argumentsValue = {},
  bundleRef = "",
  capabilityRef = "",
  skillRef = "",
  resourceRefs = [],
} = {}) {
  return CANONICAL_CALL_TOOL_ACTION_SCHEMA.parse({
    protocol_version: CANONICAL_TOOL_ACTION_PROTOCOL_VERSION,
    type: "call_tool",
    tool_name: asText(toolName) || "unknown_tool",
    arguments:
      argumentsValue && typeof argumentsValue === "object" && !Array.isArray(argumentsValue)
        ? cloneJson(argumentsValue, {})
        : {},
    bundle_ref: asText(bundleRef) || null,
    capability_ref: asText(capabilityRef) || null,
    skill_ref: asText(skillRef) || null,
    resource_refs: normalizeTextList(resourceRefs),
  });
}

export function buildCanonicalFinishAction({
  result = {},
  summary = "",
  bundleRef = "",
  skillRef = "",
} = {}) {
  return CANONICAL_FINISH_ACTION_SCHEMA.parse({
    protocol_version: CANONICAL_TOOL_ACTION_PROTOCOL_VERSION,
    type: "finish",
    result: cloneJson(result, result),
    summary: asText(summary) || null,
    bundle_ref: asText(bundleRef) || null,
    skill_ref: asText(skillRef) || null,
  });
}

export function isLegacyToolCallPayload(value) {
  return LEGACY_TOOL_CALL_PAYLOAD_SCHEMA.safeParse(value).success;
}

export function isCanonicalToolAction(value) {
  return Boolean(parseCanonicalToolAction(value));
}

export function normalizeCanonicalToolAction(value, {
  defaultBundleRef = "",
  defaultCapabilityRef = "",
  defaultSkillRef = "",
  defaultResourceRefs = [],
  passThroughUnknown = true,
} = {}) {
  const parsedCanonical = parseCanonicalToolAction(value);
  if (parsedCanonical) {
    return parsedCanonical;
  }
  if (isLegacyToolCallPayload(value)) {
    return buildCanonicalCallToolAction({
      toolName: value.tool_name,
      argumentsValue: value.arguments,
      bundleRef: defaultBundleRef,
      capabilityRef: defaultCapabilityRef || value.tool_name,
      skillRef: defaultSkillRef,
      resourceRefs: defaultResourceRefs,
    });
  }
  if (value && typeof value === "object" && !Array.isArray(value)) {
    const actionType = asText(value.type).toLowerCase();
    if (actionType === "call_tool" && asText(value.tool_name || value.toolName)) {
      return buildCanonicalCallToolAction({
        toolName: value.tool_name || value.toolName,
        argumentsValue: parseJsonObjectText(value.arguments, {}),
        bundleRef: asText(value.bundle_ref) || defaultBundleRef,
        capabilityRef: asText(value.capability_ref) || defaultCapabilityRef || value.tool_name || value.toolName,
        skillRef: asText(value.skill_ref) || defaultSkillRef,
        resourceRefs: Array.isArray(value.resource_refs) ? value.resource_refs : defaultResourceRefs,
      });
    }
    if (actionType === "ask_user") {
      const questions = normalizeQuestionEntries(value.questions);
      if (questions.length) {
        return CANONICAL_ASK_USER_ACTION_SCHEMA.parse({
          protocol_version: CANONICAL_TOOL_ACTION_PROTOCOL_VERSION,
          type: "ask_user",
          questions,
          bundle_ref: asText(value.bundle_ref) || null,
          skill_ref: asText(value.skill_ref) || defaultSkillRef || null,
        });
      }
    }
    if (actionType === "finish" && Object.prototype.hasOwnProperty.call(value, "result")) {
      return CANONICAL_FINISH_ACTION_SCHEMA.parse({
        protocol_version: CANONICAL_TOOL_ACTION_PROTOCOL_VERSION,
        type: "finish",
        result: cloneJson(value.result, value.result),
        summary: asText(value.summary) || null,
        bundle_ref: asText(value.bundle_ref) || defaultBundleRef || null,
        skill_ref: asText(value.skill_ref) || defaultSkillRef || null,
      });
    }
  }
  return passThroughUnknown ? cloneJson(value, value) : null;
}

export function extractCanonicalToolName(value) {
  const normalized = normalizeCanonicalToolAction(value, { passThroughUnknown: false });
  if (!normalized || normalized.type !== "call_tool") return "";
  return asText(normalized.tool_name);
}

export function canonicalToolActionToLegacyPayload(value) {
  const normalized = normalizeCanonicalToolAction(value, { passThroughUnknown: false });
  if (!normalized || normalized.type !== "call_tool") return null;
  return LEGACY_TOOL_CALL_PAYLOAD_SCHEMA.parse({
    tool_name: normalized.tool_name,
    arguments: normalized.arguments,
  });
}

export function buildCanonicalToolActionJsonSchema({
  allowedToolNames = [],
  includeAskUser = true,
  includeFinish = true,
} = {}) {
  const normalizedTools = normalizeTextList(allowedToolNames, 200);
  const properties = {
    protocol_version: { type: "integer", enum: [CANONICAL_TOOL_ACTION_PROTOCOL_VERSION] },
    type: {
      type: "string",
      enum: [
        "call_tool",
        ...(includeAskUser ? ["ask_user"] : []),
        ...(includeFinish ? ["finish"] : []),
      ],
    },
    tool_name: normalizedTools.length
      ? { type: "string", enum: normalizedTools }
      : { type: "string" },
    arguments: {
      type: "object",
      additionalProperties: true,
    },
    bundle_ref: { type: ["string", "null"] },
    capability_ref: { type: ["string", "null"] },
    skill_ref: { type: ["string", "null"] },
    resource_refs: {
      type: "array",
      items: { type: "string" },
    },
    questions: {
      type: "array",
      minItems: 1,
      maxItems: 6,
      items: {
        type: "object",
        additionalProperties: false,
        required: ["question"],
        properties: {
          id: { type: ["string", "null"] },
          question: { type: "string" },
          reason: { type: ["string", "null"] },
        },
      },
    },
    result: {},
    summary: { type: ["string", "null"] },
  };

  return {
    type: "object",
    additionalProperties: false,
    required: ["protocol_version", "type"],
    properties,
  };
}
