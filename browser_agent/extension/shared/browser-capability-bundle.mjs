export const BROWSER_CAPABILITY_BUNDLE_ARTIFACT_KIND = "browser_capability_bundle";
export const BROWSER_CAPABILITY_BUNDLE_SCHEMA_VERSION = 1;
export const BROWSER_CAPABILITY_RUNTIME_PACKAGE_VERSION = 1;
import {
  buildDefaultReviewedToolExecuteScript,
} from "./reviewed-tool-script-runtime.mjs";

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

function normalizeToolScriptsFingerprintSource(toolScriptsPayload = null) {
  if (toolScriptsPayload && typeof toolScriptsPayload === "object") {
    return toolScriptsPayload;
  }
  return {
    schemaVersion: 1,
    scripts: [],
    declaredTools: [],
  };
}

function getPayloadFingerprint(value) {
  try {
    const json = JSON.stringify(value ?? null);
    let hash = 2166136261;
    for (let index = 0; index < json.length; index += 1) {
      hash ^= json.charCodeAt(index);
      hash = Math.imul(hash, 16777619) >>> 0;
    }
    return `${json.length}:${hash.toString(16)}`;
  } catch {
    return "";
  }
}

function normalizeArray(value) {
  return Array.isArray(value) ? value : [];
}

function normalizeTextList(values, limit = 32) {
  return Array.from(
    new Set(
      normalizeArray(values)
        .map((entry) => asText(entry))
        .filter(Boolean),
    ),
  ).slice(0, limit);
}

function normalizeJsonSchema(value) {
  return value && typeof value === "object" && !Array.isArray(value)
    ? cloneJson(value, {})
    : {};
}

function normalizeToolName(value, fallback = "") {
  const normalized = asText(value)
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
  return normalized || asText(fallback);
}

function normalizeSearchableText(value) {
  return asText(value)
    .toLowerCase()
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function normalizeCapabilityBundleOptions(options = null) {
  const source = options && typeof options === "object" ? options : {};
  return {
    allowToolNameInference: source.allowToolNameInference !== false,
    allowSyntheticCapabilities: source.allowSyntheticCapabilities !== false,
    allowFallbackExecuteScript: source.allowFallbackExecuteScript !== false,
    allowBundleFallback: source.allowBundleFallback !== false,
  };
}

const STRICT_PUBLISHED_BROWSER_CAPABILITY_OPTIONS = Object.freeze({
  allowBundleFallback: false,
});

const BUILTIN_RUNTIME_TOOL_NAMES = new Set([
  "read_active_text_target",
  "replace_active_text_target",
  "read_clipboard_text",
  "write_clipboard_text",
  "compose_email",
  "capture_bug_report_context",
]);

function isBuiltinRuntimeToolName(toolName = "") {
  return BUILTIN_RUNTIME_TOOL_NAMES.has(normalizeToolName(toolName));
}

function hasMatchingToolScript(toolName = "", toolScriptsPayload = null) {
  const normalizedToolName = normalizeToolName(toolName);
  if (!normalizedToolName) return false;
  const availableToolNames = new Set([
    ...normalizeArray(toolScriptsPayload?.declaredTools).map((entry) => normalizeToolName(entry)),
    ...normalizeArray(toolScriptsPayload?.scripts).flatMap((entry) => [
      normalizeToolName(entry?.id),
      normalizeToolName(entry?.name),
    ]),
  ]);
  return availableToolNames.has(normalizedToolName);
}

function getPreferredDeclaredToolName(source = {}, slug = "", fallbackName = "", toolScriptsPayload = null) {
  const candidates = Array.from(new Set([
    source.toolName,
    source.tool_name,
    source.functionName,
    source.function_name,
    source.canonicalToolName,
    source.canonical_tool_name,
    source.id,
    source.name,
    slug,
    fallbackName,
  ].map((entry) => normalizeToolName(entry)).filter(Boolean))).slice(0, 24);

  for (const candidate of candidates) {
    if (isBuiltinRuntimeToolName(candidate) || hasMatchingToolScript(candidate, toolScriptsPayload)) {
      return candidate;
    }
  }
  return "";
}

function looksLikeExecutableJsSource(value = "") {
  const text = asText(value);
  if (!text) return false;
  if (/\b(return|await|async|function|const|let|var|if|else|throw|try|catch|module\.exports|exports\.|globalThis|ctx\b|tools\b|page\b|new\s+Promise)\b/.test(text)) {
    return true;
  }
  if (/=>/.test(text)) return true;
  if (/[;{}]/.test(text)) return true;
  return false;
}

function getFallbackExecuteScript(entry = {}, toolScriptsPayload = null, options = null) {
  const normalizedOptions = normalizeCapabilityBundleOptions(options);
  const existingExecute = asText(entry?.scripts?.execute);
  const normalizedToolName = normalizeToolName(entry?.toolName);
  const hasDeterministicFallback =
    normalizedToolName &&
    (isBuiltinRuntimeToolName(normalizedToolName) || hasMatchingToolScript(normalizedToolName, toolScriptsPayload));
  if (existingExecute && (!hasDeterministicFallback || looksLikeExecutableJsSource(existingExecute))) {
    return existingExecute;
  }
  if (!normalizedOptions.allowFallbackExecuteScript) {
    return existingExecute;
  }
  if (!normalizedToolName) return "";
  if (hasDeterministicFallback) {
    return buildDefaultReviewedToolExecuteScript(normalizedToolName);
  }
  return "";
}

function buildCapabilitySearchText(source = {}) {
  return [
    source.id,
    source.name,
    source.toolName,
    source.tool_name,
    source.functionName,
    source.function_name,
    source.canonicalToolName,
    source.canonical_tool_name,
    source.description,
    ...(Array.isArray(source.tags) ? source.tags : []),
    ...(Array.isArray(source.examples) ? source.examples : []),
    ...(Array.isArray(source.preconditions) ? source.preconditions : []),
    ...(Array.isArray(source.readsFrom) ? source.readsFrom : []),
    ...(Array.isArray(source.writesTo) ? source.writesTo : []),
  ]
    .map((entry) => normalizeSearchableText(entry))
    .filter(Boolean)
    .join(" ");
}

function inferKnownCapabilityToolName(source = {}) {
  const haystack = buildCapabilitySearchText(source);
  if (!haystack) return "";

  const mentionsActiveText =
    /(aktiven?\s+text|active\s+text|textziel|selection|selected(?:\s+text)?|auswahl|markiert(?:e|en|er|es)?|eingabefeld|editable|focused(?:\s+field|\s+editable)?|fokussiert(?:e|en|er|es)?|copied(?:\s+text)?|kopiert(?:e|en|er|es)?|zwischenablage|clipboard)/i
      .test(haystack);
  if (mentionsActiveText) {
    const mentionsRead = /\b(read|lesen|inspect|capture|get|collect|extract|resolve|find|locate|ermittle|lies)\b/i.test(haystack);
    const mentionsWrite = /\b(replace|write|overwrite|apply|update|rewrite|schreib|setze)\b/i.test(haystack) ||
      /\bersetz[a-z]*\b/i.test(haystack);

    if (mentionsRead && !mentionsWrite) {
      return "read_active_text_target";
    }
    if (mentionsWrite) {
      return "replace_active_text_target";
    }
  }

  const mentionsClipboard =
    /(clipboard|zwischenablage|copy\b|copied|copiar|paste\b|kopier|kopiere|kopieren|einfu(?:e|h)ren|zwischenablage)/i
      .test(haystack);
  if (mentionsClipboard) {
    const mentionsClipboardRead =
      /\b(read|les(?:e|en)[a-z]*|get|inspect|capture|collect|extract|resolve|show|anzeig[a-z]*|hol[a-z]*)\b/i.test(haystack);
    const mentionsClipboardWrite =
      /\b(write|copy|set|update|save|paste|prefill|schreib[a-z]*|kopier[a-z]*|uebernehm[a-z]*|uebertrag[a-z]*)\b/i.test(haystack);
    if (mentionsClipboardWrite) {
      return "write_clipboard_text";
    }
    if (mentionsClipboardRead) {
      return "read_clipboard_text";
    }
  }

  const mentionsEmail =
    /(email|e-mail|mail\b|mailentwurf|mail draft|mail draft|mailtext|verkaeufer|verk[aä]ufer|interest message|bug report mail|newsletter mail)/i
      .test(haystack);
  const mentionsCompose =
    /\b(compose|draft|prepare|prefill|create|generate|write|schreib[a-z]*|entwurf[a-z]*|vorbereit[a-z]*|verfass[a-z]*|erstell[a-z]*)\b/i.test(haystack);
  if (mentionsEmail && mentionsCompose) {
    return "compose_email";
  }

  const mentionsBugReport =
    /(bug report|bugreport|fehlerbericht|fehlerreport|bugmeldung|issue report|problem report|fehlermeldung|aktuell(?:e|en)? seite|current page|page context|seitenkontext)/i
      .test(haystack);
  const mentionsContextCapture =
    /\b(capture|collect|analy[sz]e|inspect|extract|read|sammel[a-z]*|analysier[a-z]*|erfass[a-z]*|kontext[a-z]*)\b/i.test(haystack);
  if (mentionsBugReport && mentionsContextCapture) {
    return "capture_bug_report_context";
  }

  return "";
}

function inferCapabilityToolName(source = {}, slug = "", fallbackName = "", toolScriptsPayload = null, options = null) {
  const normalizedOptions = normalizeCapabilityBundleOptions(options);
  const explicit = normalizeToolName(
    source.toolName ||
      source.tool_name ||
      source.functionName ||
      source.function_name ||
      source.canonicalToolName ||
      source.canonical_tool_name,
  );
  if (explicit) return explicit;

  const preferredDeclaredToolName = getPreferredDeclaredToolName(
    source,
    slug,
    fallbackName,
    toolScriptsPayload,
  );
  if (preferredDeclaredToolName) {
    return preferredDeclaredToolName;
  }

  if (!normalizedOptions.allowToolNameInference) {
    return "";
  }

  const known = inferKnownCapabilityToolName(source);
  if (known) return known;

  return normalizeToolName(slug || fallbackName, "browser_capability");
}

const ACTIVE_TEXT_READ_PARAMETER_SCHEMA = Object.freeze({
  type: "object",
  properties: {},
  additionalProperties: false,
});

const ACTIVE_TEXT_TARGET_PRIORITY = Object.freeze(["selection", "focused_editable", "clipboard"]);

const ACTIVE_TEXT_READ_RETURN_SCHEMA = Object.freeze({
  type: "object",
  additionalProperties: false,
  properties: {
    ok: { type: "boolean" },
    data: {
      type: "object",
      additionalProperties: false,
      properties: {
        targetType: { type: "string", enum: ACTIVE_TEXT_TARGET_PRIORITY },
        text: { type: "string" },
      },
      required: ["targetType", "text"],
    },
    error: {
      type: "object",
      additionalProperties: false,
      properties: {
        code: { type: "string" },
        message: { type: "string" },
      },
      required: ["code", "message"],
    },
  },
  required: ["ok"],
});

const ACTIVE_TEXT_REPLACE_PARAMETER_SCHEMA = Object.freeze({
  type: "object",
  additionalProperties: false,
  properties: {
    text: { type: "string" },
  },
  required: ["text"],
});

const ACTIVE_TEXT_REPLACE_RETURN_SCHEMA = Object.freeze({
  type: "object",
  additionalProperties: false,
  properties: {
    ok: { type: "boolean" },
    data: {
      type: "object",
      additionalProperties: false,
      properties: {
        targetType: { type: "string", enum: ACTIVE_TEXT_TARGET_PRIORITY },
        text: { type: "string" },
      },
      required: ["targetType", "text"],
    },
    error: {
      type: "object",
      additionalProperties: false,
      properties: {
        code: { type: "string" },
        message: { type: "string" },
      },
      required: ["code", "message"],
    },
  },
  required: ["ok"],
});

const TOOL_ERROR_SCHEMA = Object.freeze({
  type: "object",
  additionalProperties: false,
  properties: {
    code: { type: "string" },
    message: { type: "string" },
  },
  required: ["code", "message"],
});

function buildSimpleToolResultSchema(dataProperties = {}, dataRequired = []) {
  return Object.freeze({
    type: "object",
    additionalProperties: false,
    properties: {
      ok: { type: "boolean" },
      data: {
        type: "object",
        additionalProperties: false,
        properties: cloneJson(dataProperties, {}),
        required: normalizeArray(dataRequired),
      },
      error: cloneJson(TOOL_ERROR_SCHEMA, {}),
    },
    required: ["ok"],
  });
}

const READ_CLIPBOARD_PARAMETER_SCHEMA = Object.freeze({
  type: "object",
  properties: {},
  additionalProperties: false,
});

const READ_CLIPBOARD_RETURN_SCHEMA = buildSimpleToolResultSchema({
  text: { type: "string" },
}, ["text"]);

const WRITE_CLIPBOARD_PARAMETER_SCHEMA = Object.freeze({
  type: "object",
  additionalProperties: false,
  properties: {
    text: { type: "string" },
  },
  required: ["text"],
});

const WRITE_CLIPBOARD_RETURN_SCHEMA = buildSimpleToolResultSchema({
  text: { type: "string" },
}, ["text"]);

const EMAIL_ARRAY_FIELD_SCHEMA = Object.freeze({
  type: "array",
  items: { type: "string" },
});

const COMPOSE_EMAIL_PARAMETER_SCHEMA = Object.freeze({
  type: "object",
  additionalProperties: false,
  properties: {
    recipient: { type: "string" },
    subject: { type: "string" },
    body: { type: "string" },
    cc: cloneJson(EMAIL_ARRAY_FIELD_SCHEMA, {}),
    bcc: cloneJson(EMAIL_ARRAY_FIELD_SCHEMA, {}),
    replyTo: { type: "string" },
  },
  required: ["subject", "body"],
});

const COMPOSE_EMAIL_RETURN_SCHEMA = buildSimpleToolResultSchema({
  recipient: { type: "string" },
  subject: { type: "string" },
  body: { type: "string" },
  cc: cloneJson(EMAIL_ARRAY_FIELD_SCHEMA, {}),
  bcc: cloneJson(EMAIL_ARRAY_FIELD_SCHEMA, {}),
  replyTo: { type: "string" },
  mailtoUrl: { type: "string" },
  clipboardText: { type: "string" },
}, ["recipient", "subject", "body", "cc", "bcc", "replyTo", "mailtoUrl", "clipboardText"]);

const CLASSIFIED_LISTING_SCHEMA = Object.freeze({
  type: "object",
  additionalProperties: false,
  properties: {
    listing_url: { type: "string" },
    title: { type: "string" },
    price: { type: "string" },
    location: { type: "string" },
    snippet: { type: "string" },
    matchedCriteria: cloneJson(EMAIL_ARRAY_FIELD_SCHEMA, {}),
  },
  required: ["listing_url", "title", "matchedCriteria"],
});

const SEARCH_LISTINGS_PARAMETER_SCHEMA = Object.freeze({
  type: "object",
  additionalProperties: false,
  properties: {
    searchTarget: { type: "string" },
    criteria: cloneJson(EMAIL_ARRAY_FIELD_SCHEMA, {}),
    maxResults: { type: "integer", minimum: 1, maximum: 25 },
  },
  required: ["searchTarget", "criteria"],
});

const SEARCH_LISTINGS_RETURN_SCHEMA = (() => {
  const schema = cloneJson(buildSimpleToolResultSchema({
    searchTarget: { type: "string" },
    appliedCriteria: cloneJson(EMAIL_ARRAY_FIELD_SCHEMA, {}),
    results: {
      type: "array",
      items: cloneJson(CLASSIFIED_LISTING_SCHEMA, {}),
    },
  }, ["searchTarget", "appliedCriteria", "results"]), {});
  schema.properties.browserPlan = {};
  return Object.freeze(schema);
})();

const SHORTLIST_CANDIDATE_SCHEMA = Object.freeze({
  type: "object",
  additionalProperties: false,
  properties: {
    listing_url: { type: "string" },
    title: { type: "string" },
    reasons: cloneJson(EMAIL_ARRAY_FIELD_SCHEMA, {}),
    score: { type: "number" },
  },
  required: ["listing_url", "title", "reasons"],
});

const SHORTLIST_CANDIDATES_PARAMETER_SCHEMA = Object.freeze({
  type: "object",
  additionalProperties: false,
  properties: {
    candidates: {
      type: "array",
      items: cloneJson(CLASSIFIED_LISTING_SCHEMA, {}),
    },
    shortlistCriteria: cloneJson(EMAIL_ARRAY_FIELD_SCHEMA, {}),
    maxSelections: { type: "integer", minimum: 1, maximum: 10 },
  },
  required: ["candidates", "shortlistCriteria"],
});

const SHORTLIST_CANDIDATES_RETURN_SCHEMA = buildSimpleToolResultSchema({
  shortlistCriteria: cloneJson(EMAIL_ARRAY_FIELD_SCHEMA, {}),
  selected: {
    type: "array",
    items: cloneJson(SHORTLIST_CANDIDATE_SCHEMA, {}),
  },
}, ["shortlistCriteria", "selected"]);

const COMPOSE_OUTREACH_MESSAGE_PARAMETER_SCHEMA = Object.freeze({
  type: "object",
  additionalProperties: false,
  properties: {
    listing_url: { type: "string" },
    listing_title: { type: "string" },
    seller_name: { type: "string" },
    matchReasons: cloneJson(EMAIL_ARRAY_FIELD_SCHEMA, {}),
    userContext: { type: "string" },
    tone: { type: "string" },
  },
  required: ["listing_url", "listing_title", "matchReasons"],
});

const COMPOSE_OUTREACH_MESSAGE_RETURN_SCHEMA = buildSimpleToolResultSchema({
  listing_url: { type: "string" },
  subject: { type: "string" },
  message: { type: "string" },
  matchReasons: cloneJson(EMAIL_ARRAY_FIELD_SCHEMA, {}),
}, ["listing_url", "message", "matchReasons"]);

const CAPTURE_BUG_REPORT_PARAMETER_SCHEMA = Object.freeze({
  type: "object",
  additionalProperties: false,
  properties: {
    summary: { type: "string" },
    stepsToReproduce: cloneJson(EMAIL_ARRAY_FIELD_SCHEMA, {}),
    expectedBehavior: { type: "string" },
    actualBehavior: { type: "string" },
    maxBodyChars: { type: "integer", minimum: 200, maximum: 4_000 },
  },
});

const CAPTURE_BUG_REPORT_RETURN_SCHEMA = buildSimpleToolResultSchema({
  url: { type: "string" },
  title: { type: "string" },
  summary: { type: "string" },
  selectionText: { type: "string" },
  bodyExcerpt: { type: "string" },
  headings: cloneJson(EMAIL_ARRAY_FIELD_SCHEMA, {}),
  dialogs: cloneJson(EMAIL_ARRAY_FIELD_SCHEMA, {}),
  visibleErrorTexts: cloneJson(EMAIL_ARRAY_FIELD_SCHEMA, {}),
  stepsToReproduce: cloneJson(EMAIL_ARRAY_FIELD_SCHEMA, {}),
  expectedBehavior: { type: "string" },
  actualBehavior: { type: "string" },
  emailSubject: { type: "string" },
  emailBody: { type: "string" },
  activeElement: {
    type: "object",
    additionalProperties: false,
    properties: {
      tagName: { type: "string" },
      inputType: { type: "string" },
      placeholder: { type: "string" },
      name: { type: "string" },
      id: { type: "string" },
      ariaLabel: { type: "string" },
      valuePreview: { type: "string" },
    },
    required: ["tagName", "inputType", "placeholder", "name", "id", "ariaLabel", "valuePreview"],
  },
}, [
  "url",
  "title",
  "summary",
  "selectionText",
  "bodyExcerpt",
  "headings",
  "dialogs",
  "visibleErrorTexts",
  "stepsToReproduce",
  "expectedBehavior",
  "actualBehavior",
  "emailSubject",
  "emailBody",
]);

function mergeUniqueTextList(values = [], additions = [], limit = 32) {
  return normalizeTextList([...(Array.isArray(values) ? values : []), ...(Array.isArray(additions) ? additions : [])], limit);
}

function normalizeActiveTextTargetType(value) {
  const normalized = asText(value)
    .toLowerCase()
    .replace(/[\s-]+/g, "_");
  if (!normalized) return "";
  if (normalized === "editable" || normalized === "focused_field" || normalized === "focused_editable_field") {
    return "focused_editable";
  }
  if (ACTIVE_TEXT_TARGET_PRIORITY.includes(normalized)) {
    return normalized;
  }
  return normalized;
}

export function normalizeActiveTextCapabilityResult(value = null) {
  const normalizedValue = cloneJson(value, value);
  if (!normalizedValue || typeof normalizedValue !== "object" || Array.isArray(normalizedValue)) {
    return normalizedValue;
  }

  const normalizeTargetHolder = (entry) => {
    if (!entry || typeof entry !== "object" || Array.isArray(entry)) return;
    if (!Object.prototype.hasOwnProperty.call(entry, "targetType")) return;
    const normalizedTargetType = normalizeActiveTextTargetType(entry.targetType);
    if (normalizedTargetType) {
      entry.targetType = normalizedTargetType;
    }
  };

  normalizeTargetHolder(normalizedValue);
  normalizeTargetHolder(normalizedValue.data);
  return normalizedValue;
}

function normalizeActiveTextTargetList(values = [], fallback = ACTIVE_TEXT_TARGET_PRIORITY) {
  const normalized = normalizeTextList(
    normalizeArray(values)
      .map((entry) => normalizeActiveTextTargetType(entry))
      .filter(Boolean),
    12,
  ).filter((entry) => ACTIVE_TEXT_TARGET_PRIORITY.includes(entry));
  return normalized.length ? normalized : [...fallback];
}

function normalizeActiveTextReturnSchema(schema = {}, fallbackSchema = ACTIVE_TEXT_READ_RETURN_SCHEMA) {
  const normalized = Object.keys(schema || {}).length
    ? cloneJson(schema, {})
    : cloneJson(fallbackSchema, {});
  const enumValues = normalizeActiveTextTargetList(
    normalized?.properties?.data?.properties?.targetType?.enum,
    ACTIVE_TEXT_TARGET_PRIORITY,
  );
  if (normalized?.properties?.data?.properties?.targetType && enumValues.length) {
    normalized.properties.data.properties.targetType.enum = enumValues;
  }
  return normalized;
}

function inferBundleSkillsFromCapabilities(capabilities = []) {
  const toolNames = new Set(
    normalizeArray(capabilities)
      .map((entry) => asText(entry?.toolName).toLowerCase())
      .filter(Boolean),
  );
  const skills = [];
  if (toolNames.has("read_active_text_target") && toolNames.has("replace_active_text_target")) {
    skills.push(
      "For active text correction, rewriting, or translation, use the reviewed in-place workflow instead of replying with transformed text directly.",
      "First call read_active_text_target with an empty argument object to resolve the active text target in priority order: selection, focused_editable, then clipboard.",
      "After transforming the returned text, call replace_active_text_target with the full replacement text.",
      "Do not finish with only the rewritten text when replace_active_text_target can apply it in place.",
    );
  }
  if (toolNames.has("compose_email")) {
    skills.push(
      "Use compose_email to return a structured email draft with subject and body instead of free-form prose when the task is about contacting someone or preparing a report email.",
    );
  }
  if (toolNames.has("search_listings") && toolNames.has("shortlist_candidates") && toolNames.has("compose_outreach_message")) {
    skills.push(
      "For classifieds outreach, follow the reviewed flow search_listings -> shortlist_candidates -> compose_outreach_message.",
      "Pass explicit non-empty searchTarget and criteria into search_listings, keep shortlist_candidates grounded in concrete candidate results plus shortlistCriteria, and include real listing matchReasons when composing seller outreach.",
      "Do not return empty criteria, an empty shortlist, or a generic seller message when the reviewed classifieds capabilities can keep those fields specific.",
    );
  }
  if (toolNames.has("write_clipboard_text")) {
    skills.push(
      "When the user explicitly asks to copy, prefill, or place prepared content onto the clipboard, call write_clipboard_text instead of only replying with that text.",
    );
  }
  if (toolNames.has("capture_bug_report_context")) {
    skills.push(
      "For bug-report flows, capture the relevant current-page context before composing the report so the draft includes the real URL, title, visible errors, and focused field details.",
    );
  }
  return skills;
}

function buildActiveTextWorkflowHaystack(capabilities = [], craft = null) {
  return [
    asText(craft?.name),
    asText(craft?.summary),
    asText(craft?.agentPrompt),
    asText(craft?.inputHint),
    ...normalizeArray(craft?.inputExamples),
    ...normalizeArray(capabilities).flatMap((entry) => [
      buildCapabilitySearchText(entry),
      asText(entry?.scripts?.pre),
      asText(entry?.scripts?.execute),
      asText(entry?.scripts?.post),
    ]),
  ]
    .map((entry) => normalizeSearchableText(entry))
    .filter(Boolean)
    .join(" ");
}

export function looksLikeTransformingActiveTextWorkflow(capabilities = [], craft = null) {
  const haystack = buildActiveTextWorkflowHaystack(capabilities, craft);
  if (!haystack) return false;
  const mentionsTargets =
    /(selection|selected\s+text|auswahl|markiert(?:e|en|er|es)?|focused\s+field|focused\s+editable|eingabefeld|textfeld|zwischenablage|clipboard|copied\s+text|kopiert(?:e|en|er|es)?)/i
      .test(haystack);
  const mentionsTransform =
    /\b(fix|correct|rewrite|rephrase|translate|improve|grammar|spelling|proofread|summari[sz]e|shorten|expand|extract|korrigier|verbesser|uebersetz|übersetz|grammatik|rechtschreibung|umformulier|zusammenfass|kuerz|kürz|extrahier)\b/i
      .test(haystack);
  return mentionsTargets && mentionsTransform;
}

function createSyntheticActiveTextCapability(toolName = "", craft = null, toolScriptsPayload = null, options = null) {
  const normalizedToolName = normalizeToolName(toolName);
  if (!normalizedToolName) return null;
  const capabilityName =
    normalizedToolName === "read_active_text_target"
      ? "Aktiven Text lesen"
      : asText(craft?.name) || "Aktiven Text ersetzen";
  return normalizeBrowserCapabilityEntry({
    id: normalizedToolName,
    name: capabilityName,
    toolName: normalizedToolName,
    scripts: {},
  }, 0, toolScriptsPayload, options);
}

function looksLikeTransformingActiveTextWrapperCapability(entry = {}, craft = null) {
  const toolName = asText(entry?.toolName);
  if (!toolName || toolName === "read_active_text_target" || toolName === "replace_active_text_target") {
    return false;
  }
  return looksLikeTransformingActiveTextWorkflow([entry], craft);
}

function hasImmediateBlockedPlaceholderExecuteScript(source = "") {
  const text = asText(source);
  if (!text) return false;
  if (!/\breturn\s*\{[\s\S]*\bstatus\s*:\s*["']blocked["']/i.test(text)) {
    return false;
  }
  if (/\b(__callReviewedTool|runtime\.callBuiltin|activeTextTools|tools\.activeText|ctx\.activeText)\b/.test(text)) {
    return false;
  }
  return true;
}

function getPreferredActiveTextCapability(capabilities = [], toolName = "") {
  const normalizedToolName = asText(toolName);
  if (!normalizedToolName) return null;
  const candidates = normalizeArray(capabilities).filter((entry) => asText(entry?.toolName) === normalizedToolName);
  if (!candidates.length) return null;
  return [...candidates].sort((left, right) => {
    const scoreEntry = (entry) => {
      let score = 0;
      if (asText(entry?.id) === normalizedToolName) score += 4;
      if (asText(entry?.scripts?.execute) === buildDefaultReviewedToolExecuteScript(normalizedToolName)) {
        score += 3;
      }
      if (!hasImmediateBlockedPlaceholderExecuteScript(entry?.scripts?.execute)) {
        score += 1;
      }
      return score;
    };
    return scoreEntry(right) - scoreEntry(left);
  })[0];
}

function dedupePreferredActiveTextCapabilities(capabilities = []) {
  const normalizedCapabilities = normalizeArray(capabilities);
  const preferredRead = getPreferredActiveTextCapability(normalizedCapabilities, "read_active_text_target");
  const preferredReplace = getPreferredActiveTextCapability(normalizedCapabilities, "replace_active_text_target");
  if (!preferredRead || !preferredReplace) {
    return normalizedCapabilities;
  }
  return normalizedCapabilities.filter((entry) => {
    const toolName = asText(entry?.toolName);
    if (toolName === "read_active_text_target") return entry === preferredRead;
    if (toolName === "replace_active_text_target") return entry === preferredReplace;
    return true;
  });
}

function pruneAbstractTransformingActiveTextCapabilities(capabilities = [], craft = null, toolScriptsPayload = null) {
  const normalizedCapabilities = normalizeArray(capabilities);
  if (!normalizedCapabilities.length) return normalizedCapabilities;
  if (!looksLikeTransformingActiveTextWorkflow(normalizedCapabilities, craft)) {
    return normalizedCapabilities;
  }
  const toolNames = new Set(
    normalizedCapabilities
      .map((entry) => asText(entry?.toolName))
      .filter(Boolean),
  );
  if (!toolNames.has("read_active_text_target") || !toolNames.has("replace_active_text_target")) {
    return normalizedCapabilities;
  }

  const filtered = normalizedCapabilities.filter((entry) => {
    const toolName = asText(entry?.toolName);
    if (toolName === "read_active_text_target" || toolName === "replace_active_text_target") {
      return true;
    }
    if (isBuiltinRuntimeToolName(toolName) || hasMatchingToolScript(toolName, toolScriptsPayload)) {
      return true;
    }
    if (
      looksLikeTransformingActiveTextWrapperCapability(entry, craft) &&
      hasImmediateBlockedPlaceholderExecuteScript(entry?.scripts?.execute)
    ) {
      return false;
    }
    if (asText(entry?.scripts?.execute)) {
      return true;
    }
    return false;
  });
  const pruned = filtered.length ? filtered : normalizedCapabilities;
  return dedupePreferredActiveTextCapabilities(pruned);
}

function ensureActiveTextCapabilityPair(capabilities = [], craft = null, toolScriptsPayload = null, options = null) {
  const normalizedOptions = normalizeCapabilityBundleOptions(options);
  const normalizedCapabilities = normalizeArray(capabilities);
  const toolNames = new Set(
    normalizedCapabilities
      .map((entry) => asText(entry?.toolName))
      .filter(Boolean),
  );
  const hasRead = toolNames.has("read_active_text_target");
  const hasReplace = toolNames.has("replace_active_text_target");
  if (hasRead && hasReplace) {
    return pruneAbstractTransformingActiveTextCapabilities(normalizedCapabilities, craft, toolScriptsPayload);
  }
  if (!normalizedOptions.allowSyntheticCapabilities) {
    return normalizedCapabilities;
  }
  if (!looksLikeTransformingActiveTextWorkflow(normalizedCapabilities, craft)) {
    return normalizedCapabilities;
  }

  const nextCapabilities = [...normalizedCapabilities];
  if (!hasRead) {
    const syntheticRead = createSyntheticActiveTextCapability("read_active_text_target", craft, toolScriptsPayload, options);
    if (syntheticRead) {
      nextCapabilities.unshift(syntheticRead);
    }
  }
  if (!hasReplace) {
    const syntheticReplace = createSyntheticActiveTextCapability("replace_active_text_target", craft, toolScriptsPayload, options);
    if (syntheticReplace) {
      nextCapabilities.push(syntheticReplace);
    }
  }
  return pruneAbstractTransformingActiveTextCapabilities(nextCapabilities, craft, toolScriptsPayload);
}

function applyKnownCapabilityDefaults(entry = {}, toolScriptsPayload = null, options = null) {
  const scripts = {
    pre: asText(entry?.scripts?.pre),
    execute: getFallbackExecuteScript(entry, toolScriptsPayload, options),
    post: asText(entry?.scripts?.post),
  };
  if (entry.toolName === "read_active_text_target") {
    return {
      ...entry,
      description:
        entry.description ||
        "Read the active text target in priority order: browser selection, focused editable field, then clipboard.",
      parameterSchema: Object.keys(entry.parameterSchema || {}).length
        ? entry.parameterSchema
        : cloneJson(ACTIVE_TEXT_READ_PARAMETER_SCHEMA, {}),
      returnSchema: normalizeActiveTextReturnSchema(entry.returnSchema, ACTIVE_TEXT_READ_RETURN_SCHEMA),
      readsFrom: normalizeActiveTextTargetList(entry.readsFrom, ACTIVE_TEXT_TARGET_PRIORITY),
      scripts,
      tags: mergeUniqueTextList(entry.tags, ["active_text", "read", "in_place_workflow"]),
    };
  }
  if (entry.toolName === "replace_active_text_target") {
    return {
      ...entry,
      description:
        entry.description ||
        "Replace the active text target in priority order: browser selection, focused editable field, then clipboard.",
      parameterSchema: Object.keys(entry.parameterSchema || {}).length
        ? entry.parameterSchema
        : cloneJson(ACTIVE_TEXT_REPLACE_PARAMETER_SCHEMA, {}),
      returnSchema: normalizeActiveTextReturnSchema(entry.returnSchema, ACTIVE_TEXT_REPLACE_RETURN_SCHEMA),
      writesTo: normalizeActiveTextTargetList(entry.writesTo, ACTIVE_TEXT_TARGET_PRIORITY),
      scripts,
      tags: mergeUniqueTextList(entry.tags, ["active_text", "write", "in_place_workflow"]),
    };
  }
  if (entry.toolName === "read_clipboard_text") {
    return {
      ...entry,
      description:
        entry.description ||
        "Read plain text from the clipboard.",
      parameterSchema: Object.keys(entry.parameterSchema || {}).length
        ? entry.parameterSchema
        : cloneJson(READ_CLIPBOARD_PARAMETER_SCHEMA, {}),
      returnSchema: Object.keys(entry.returnSchema || {}).length
        ? entry.returnSchema
        : cloneJson(READ_CLIPBOARD_RETURN_SCHEMA, {}),
      readsFrom: mergeUniqueTextList(entry.readsFrom, ["clipboard"]),
      scripts,
      tags: mergeUniqueTextList(entry.tags, ["clipboard", "read"]),
    };
  }
  if (entry.toolName === "write_clipboard_text") {
    return {
      ...entry,
      description:
        entry.description ||
        "Write plain text to the clipboard.",
      parameterSchema: Object.keys(entry.parameterSchema || {}).length
        ? entry.parameterSchema
        : cloneJson(WRITE_CLIPBOARD_PARAMETER_SCHEMA, {}),
      returnSchema: Object.keys(entry.returnSchema || {}).length
        ? entry.returnSchema
        : cloneJson(WRITE_CLIPBOARD_RETURN_SCHEMA, {}),
      writesTo: mergeUniqueTextList(entry.writesTo, ["clipboard"]),
      scripts,
      tags: mergeUniqueTextList(entry.tags, ["clipboard", "write"]),
    };
  }
  if (entry.toolName === "compose_email") {
    return {
      ...entry,
      description:
        entry.description ||
        "Compose a structured email draft with normalized recipient fields, a mailto URL, and clipboard-friendly text.",
      parameterSchema: Object.keys(entry.parameterSchema || {}).length
        ? entry.parameterSchema
        : cloneJson(COMPOSE_EMAIL_PARAMETER_SCHEMA, {}),
      returnSchema: Object.keys(entry.returnSchema || {}).length
        ? entry.returnSchema
        : cloneJson(COMPOSE_EMAIL_RETURN_SCHEMA, {}),
      scripts,
      tags: mergeUniqueTextList(entry.tags, ["email", "draft", "handoff"]),
    };
  }
  if (entry.toolName === "search_listings") {
    return {
      ...entry,
      description:
        entry.description ||
        "Search classified listings for a concrete search target and explicit matching criteria, then return structured listing candidates.",
      parameterSchema: Object.keys(entry.parameterSchema || {}).length
        ? entry.parameterSchema
        : cloneJson(SEARCH_LISTINGS_PARAMETER_SCHEMA, {}),
      returnSchema: Object.keys(entry.returnSchema || {}).length
        ? entry.returnSchema
        : cloneJson(SEARCH_LISTINGS_RETURN_SCHEMA, {}),
      readsFrom: mergeUniqueTextList(entry.readsFrom, ["classified_listings", "listing_results"]),
      scripts,
      tags: mergeUniqueTextList(entry.tags, ["classifieds", "search", "matching"]),
    };
  }
  if (entry.toolName === "shortlist_candidates") {
    return {
      ...entry,
      description:
        entry.description ||
        "Select the strongest listing opportunities from structured candidates and explain the concrete reasons for each pick.",
      parameterSchema: Object.keys(entry.parameterSchema || {}).length
        ? entry.parameterSchema
        : cloneJson(SHORTLIST_CANDIDATES_PARAMETER_SCHEMA, {}),
      returnSchema: Object.keys(entry.returnSchema || {}).length
        ? entry.returnSchema
        : cloneJson(SHORTLIST_CANDIDATES_RETURN_SCHEMA, {}),
      readsFrom: mergeUniqueTextList(entry.readsFrom, ["listing_results", "match_reasons"]),
      writesTo: mergeUniqueTextList(entry.writesTo, ["shortlist"]),
      scripts,
      tags: mergeUniqueTextList(entry.tags, ["classifieds", "ranking", "shortlist"]),
    };
  }
  if (entry.toolName === "compose_outreach_message") {
    return {
      ...entry,
      description:
        entry.description ||
        "Compose a concise seller outreach message for one selected listing using concrete match reasons instead of a generic interest note.",
      parameterSchema: Object.keys(entry.parameterSchema || {}).length
        ? entry.parameterSchema
        : cloneJson(COMPOSE_OUTREACH_MESSAGE_PARAMETER_SCHEMA, {}),
      returnSchema: Object.keys(entry.returnSchema || {}).length
        ? entry.returnSchema
        : cloneJson(COMPOSE_OUTREACH_MESSAGE_RETURN_SCHEMA, {}),
      readsFrom: mergeUniqueTextList(entry.readsFrom, ["selected_listing", "match_reasons"]),
      writesTo: mergeUniqueTextList(entry.writesTo, ["seller_message"]),
      scripts,
      tags: mergeUniqueTextList(entry.tags, ["classifieds", "outreach", "message"]),
    };
  }
  if (entry.toolName === "capture_bug_report_context") {
    return {
      ...entry,
      description:
        entry.description ||
        "Capture the relevant current-page context and prepare a bug-report-ready summary with email subject and body suggestions.",
      parameterSchema: Object.keys(entry.parameterSchema || {}).length
        ? entry.parameterSchema
        : cloneJson(CAPTURE_BUG_REPORT_PARAMETER_SCHEMA, {}),
      returnSchema: Object.keys(entry.returnSchema || {}).length
        ? entry.returnSchema
        : cloneJson(CAPTURE_BUG_REPORT_RETURN_SCHEMA, {}),
      readsFrom: mergeUniqueTextList(entry.readsFrom, ["current_page", "visible_text", "focused_element"]),
      scripts,
      tags: mergeUniqueTextList(entry.tags, ["bug_report", "page_context", "email_draft"]),
    };
  }
  return {
    ...entry,
    scripts,
  };
}

export function getBrowserCapabilityBundleArtifactId(craftId) {
  return `browser-capabilities:${asText(craftId)}`;
}

export function normalizeBrowserCapabilityEntry(entry, index = 0, toolScriptsPayload = null, options = null) {
  return normalizeBrowserCapabilityEntryWithOptions(entry, index, toolScriptsPayload, options);
}

function normalizeBrowserCapabilityEntryWithOptions(entry, index = 0, toolScriptsPayload = null, options = null) {
  const source = entry && typeof entry === "object" ? entry : {};
  const name = asText(source.name) || `browser_capability_${index + 1}`;
  const slug = asText(source.id || name)
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "") || `browser_capability_${index + 1}`;
  const normalized = {
    id: slug,
    name,
    toolName: inferCapabilityToolName(source, slug, name, toolScriptsPayload, options),
    version: asText(source.version) || "1.0.0",
    description: asText(source.description),
    parameterSchema: normalizeJsonSchema(source.parameterSchema || source.parameters || source.inputSchema),
    returnSchema: normalizeJsonSchema(source.returnSchema || source.outputSchema),
    preconditions: normalizeTextList(source.preconditions, 24),
    readsFrom: normalizeTextList(source.readsFrom, 12),
    writesTo: normalizeTextList(source.writesTo, 12),
    examples: normalizeTextList(source.examples, 12),
    tags: normalizeTextList(source.tags, 16),
    skillRef: asText(source.skillRef || source.skill_ref),
    resourceRefs: normalizeTextList(source.resourceRefs || source.resource_refs, 24),
    scripts: {
      pre: asText(source?.scripts?.pre || source.preScript || source.pre_script),
      execute: asText(source?.scripts?.execute || source.executeScript || source.execute_script || source.source),
      post: asText(source?.scripts?.post || source.postScript || source.post_script),
    },
  };
  return applyKnownCapabilityDefaults(normalized, toolScriptsPayload, options);
}

export function inferBrowserCapabilityBundlePayload(craft = null, toolScriptsPayload = null, options = null) {
  return inferBrowserCapabilityBundlePayloadWithOptions(craft, toolScriptsPayload, options);
}

function inferBrowserCapabilityBundlePayloadWithOptions(craft = null, toolScriptsPayload = null, options = null) {
  const normalizedOptions = normalizeCapabilityBundleOptions(options);
  if (!normalizedOptions.allowBundleFallback) {
    return {
      schemaVersion: BROWSER_CAPABILITY_BUNDLE_SCHEMA_VERSION,
      actionProtocolVersion: 1,
      capabilities: [],
      resources: [],
      skills: [],
    };
  }
  const sourceScripts = Array.isArray(toolScriptsPayload?.scripts) && toolScriptsPayload.scripts.length
    ? toolScriptsPayload.scripts
    : Array.isArray(craft?.tools)
      ? craft.tools.map((toolName) => ({
          id: asText(toolName)
            .toLowerCase()
            .replace(/[^a-z0-9]+/g, "_")
            .replace(/^_+|_+$/g, ""),
          name: asText(toolName),
          description: "",
          source: "",
        }))
      : [];
  const capabilities = sourceScripts.map((script, index) =>
    normalizeBrowserCapabilityEntry({
      id: script.id,
      name: script.name,
      toolName: script.id || script.name,
      description: script.description,
      version: "1.0.0",
      parameterSchema: {},
      returnSchema: {},
      preconditions: [],
      readsFrom: [],
      writesTo: [],
      examples: [],
      tags: ["browser_script"],
      scripts: {
        pre: "",
        execute: "",
        post: "",
      },
      resourceRefs: [],
      skillRef: "",
    }, index, toolScriptsPayload, options),
  );

  return {
    schemaVersion: BROWSER_CAPABILITY_BUNDLE_SCHEMA_VERSION,
    actionProtocolVersion: 1,
    capabilities,
    resources: [],
    skills: [],
  };
}

export function normalizeBrowserCapabilityBundlePayload(payload, craft = null, toolScriptsPayload = null, options = null) {
  return normalizeBrowserCapabilityBundlePayloadWithOptions(payload, craft, toolScriptsPayload, options);
}

export function buildPublishedBrowserCapabilityBundlePayload(
  payload,
  {
    craft = null,
    toolScriptsPayload = null,
    publishedAt = "",
    publishedBy = "",
  } = {},
) {
  const normalized = normalizeBrowserCapabilityBundlePayload(
    payload,
    craft,
    toolScriptsPayload,
    STRICT_PUBLISHED_BROWSER_CAPABILITY_OPTIONS,
  );
  return {
    ...normalized,
    publication: {
      status: "published",
      runtimePackageVersion: BROWSER_CAPABILITY_RUNTIME_PACKAGE_VERSION,
      toolScriptsFingerprint: getPayloadFingerprint(
        normalizeToolScriptsFingerprintSource(toolScriptsPayload),
      ),
      publishedAt: asText(publishedAt) || new Date().toISOString(),
      publishedBy: asText(publishedBy) || "workspace",
      capabilityCount: Array.isArray(normalized?.capabilities) ? normalized.capabilities.length : 0,
    },
  };
}

export function compilePublishedBrowserCapabilityBundlePayload(
  payload,
  {
    craft = null,
    toolScriptsPayload = null,
    publishedAt = "",
    publishedBy = "",
    expectedRuntimePackageVersion = BROWSER_CAPABILITY_RUNTIME_PACKAGE_VERSION,
  } = {},
) {
  const publishedPayload = buildPublishedBrowserCapabilityBundlePayload(payload, {
    craft,
    toolScriptsPayload,
    publishedAt,
    publishedBy,
  });
  const expectedToolScriptsFingerprint = asText(
    publishedPayload?.publication?.toolScriptsFingerprint || publishedPayload?.publication?.tool_scripts_fingerprint,
  );
  const resolution = resolvePublishedBrowserCapabilityBundlePayload(publishedPayload, {
    craft,
    toolScriptsPayload,
    expectedToolScriptsFingerprint,
    expectedRuntimePackageVersion,
  });
  if (!resolution?.ok) {
    return {
      ok: false,
      error: asText(resolution?.error) || "The reviewed capability runtime package is invalid.",
      reason: asText(resolution?.reason) || "runtime_package_invalid",
      payload: publishedPayload,
    };
  }
  return {
    ok: true,
    payload: resolution.payload,
  };
}

function normalizeBrowserCapabilityBundlePayloadWithOptions(payload, craft = null, toolScriptsPayload = null, options = null) {
  const fallback = inferBrowserCapabilityBundlePayloadWithOptions(craft, toolScriptsPayload, options);
  const normalizedCapabilities = normalizeArray(payload?.capabilities).length
    ? normalizeArray(payload.capabilities).map((entry, index) => normalizeBrowserCapabilityEntryWithOptions(entry, index, toolScriptsPayload, options))
    : fallback.capabilities;
  const capabilities = ensureActiveTextCapabilityPair(normalizedCapabilities, craft, toolScriptsPayload, options);
  return {
    schemaVersion: BROWSER_CAPABILITY_BUNDLE_SCHEMA_VERSION,
    actionProtocolVersion: 1,
    capabilities,
    resources: normalizeTextList(payload?.resources, 48),
    skills: normalizeTextList([
      ...normalizeArray(payload?.skills),
      ...inferBundleSkillsFromCapabilities(capabilities),
    ], 24),
    publication:
      payload?.publication && typeof payload.publication === "object"
        ? cloneJson(payload.publication, {})
        : null,
  };
}

export function resolvePublishedBrowserCapabilityBundlePayload(payload, {
  craft = null,
  toolScriptsPayload = null,
  expectedToolScriptsFingerprint = "",
  expectedRuntimePackageVersion = BROWSER_CAPABILITY_RUNTIME_PACKAGE_VERSION,
} = {}) {
  const publication = payload?.publication && typeof payload.publication === "object"
    ? payload.publication
    : null;
  const publicationStatus = asText(publication?.status).toLowerCase();
  if (publicationStatus !== "published") {
    return {
      ok: false,
      error: "Craft runtime requires a published reviewed capability package.",
      reason: "bundle_not_published",
    };
  }

  const runtimePackageVersion = Number(
    publication?.runtimePackageVersion ??
      publication?.runtime_package_version ??
      0,
  );
  if (!Number.isInteger(runtimePackageVersion) || runtimePackageVersion !== Number(expectedRuntimePackageVersion || 0)) {
    return {
      ok: false,
      error: `Craft runtime expected published capability package version ${Number(expectedRuntimePackageVersion || 0)}.`,
      reason: "runtime_package_version_mismatch",
    };
  }

  const declaredToolScriptsFingerprint = asText(
    publication?.toolScriptsFingerprint || publication?.tool_scripts_fingerprint,
  );

  const normalized = normalizeBrowserCapabilityBundlePayloadWithOptions(
    payload,
    craft,
    toolScriptsPayload,
    {
      allowToolNameInference: false,
      allowSyntheticCapabilities: false,
      allowFallbackExecuteScript: false,
      allowBundleFallback: false,
    },
  );
  const capabilities = normalizeArray(normalized?.capabilities);
  const builtinOnlyCapabilities =
    capabilities.length > 0 &&
    capabilities.every((capability) => isBuiltinRuntimeToolName(capability?.toolName));
  if (
    declaredToolScriptsFingerprint &&
    asText(expectedToolScriptsFingerprint) &&
    declaredToolScriptsFingerprint !== asText(expectedToolScriptsFingerprint) &&
    !builtinOnlyCapabilities
  ) {
    return {
      ok: false,
      error: "The published reviewed capability package no longer matches the current tool-script package. Republish the capability package.",
      reason: "tool_scripts_fingerprint_mismatch",
    };
  }
  if (!capabilities.length) {
    return {
      ok: false,
      error: "The published reviewed capability package does not contain executable capabilities.",
      reason: "empty_published_capability_bundle",
    };
  }

  for (const capability of capabilities) {
    const toolName = asText(capability?.toolName);
    const executeScript = asText(capability?.scripts?.execute);
    if (!toolName) {
      return {
        ok: false,
        error: "A published reviewed capability is missing its machine tool name.",
        reason: "missing_tool_name",
      };
    }
    if (!executeScript) {
      return {
        ok: false,
        error: `The published reviewed capability "${toolName}" is missing an execute script.`,
        reason: "missing_execute_script",
      };
    }
    const usesPublishedToolScript =
      executeScript === buildDefaultReviewedToolExecuteScript(toolName);
    if (
      usesPublishedToolScript &&
      !isBuiltinRuntimeToolName(toolName) &&
      !hasMatchingToolScript(toolName, toolScriptsPayload)
    ) {
      return {
        ok: false,
        error: `The published reviewed capability "${toolName}" references a reviewed tool script that is not available in the current runtime package.`,
        reason: "missing_reviewed_tool_script",
      };
    }
  }

  return {
    ok: true,
    payload: normalized,
  };
}

export function buildBrowserCapabilityToolCatalog(payload, {
  toolScriptsPayload = null,
  normalizationOptions = null,
} = {}) {
  return normalizeArray(payload?.capabilities).map((entry, index) => {
    const capability = normalizeBrowserCapabilityEntry(entry, index, toolScriptsPayload, normalizationOptions);
    return {
      id: capability.id,
      toolName: capability.toolName,
      displayName: capability.name,
      description: capability.description,
      parameterSchema: capability.parameterSchema,
      returnSchema: capability.returnSchema,
      preconditions: capability.preconditions,
      skillRef: capability.skillRef,
      resourceRefs: capability.resourceRefs,
    };
  });
}

export function resolveBrowserCapability(payload, toolName = "", {
  toolScriptsPayload = null,
  normalizationOptions = null,
} = {}) {
  const normalizedToolName = asText(toolName).toLowerCase();
  return buildBrowserCapabilityToolCatalog(payload, {
    toolScriptsPayload,
    normalizationOptions,
  }).find((entry) => {
    return (
      asText(entry.toolName).toLowerCase() === normalizedToolName ||
      asText(entry.id).toLowerCase() === normalizedToolName ||
      asText(entry.displayName).toLowerCase() === normalizedToolName
    );
  }) || null;
}
