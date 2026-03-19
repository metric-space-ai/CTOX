import "../shared/craft-sync.js";
import {
  BROWSER_CAPABILITY_BUNDLE_ARTIFACT_KIND,
  buildCapabilityBundle,
  POLICY_BUNDLE_ARTIFACT_KIND,
  TOOL_SCRIPTS_ARTIFACT_KIND,
  TRAINING_DATA_ARTIFACT_KIND,
  WEIGHTS_ARTIFACT_KIND,
} from "../shared/capability-bundle.mjs";
import {
  BROWSER_CAPABILITY_RUNTIME_PACKAGE_VERSION,
  looksLikeTransformingActiveTextWorkflow,
  normalizeActiveTextCapabilityResult,
  resolvePublishedBrowserCapabilityBundlePayload,
} from "../shared/browser-capability-bundle.mjs";
import {
  buildCanonicalCallToolAction,
  buildCanonicalToolActionJsonSchema,
  CANONICAL_TOOL_ACTION_SCHEMA,
  normalizeCanonicalToolAction,
} from "../shared/canonical-tool-action.mjs";
import {
  getQwenAgentCapabilityProfile,
  buildQwenCapabilityCatalogPayload,
  buildQwenCapabilityToolDefinitions,
  buildQwenCapabilitySkillText,
  buildPortableAssistantTurnFromCanonicalAction,
  buildPortableAssistantTurnJsonSchema,
  compileCanonicalActionExecutionPlan,
  convertPortableAssistantTurnToCanonicalAction,
  normalizePortableAssistantTurn,
  PORTABLE_ASSISTANT_TURN_SCHEMA,
} from "../shared/qwen-agent-adapter.mjs";
import { renderQwen35ChatTemplate } from "../shared/qwen35-chat-template.mjs";
import {
  buildDefaultReviewedToolExecuteScript,
  buildReviewedToolScriptsPrelude,
} from "../shared/reviewed-tool-script-runtime.mjs";
import { toolCollectPageState } from "../shared/browserAutomationTools.js";
import { runRestrictedBrowserScript } from "../shared/browserAutomationRuntime.js";
import {
  describeResolvedModel,
  generateStructuredOutput,
  llmChat,
} from "./llm.js";

const DEFAULT_MAX_RUNTIME_TURNS = 4;
const MULTISTEP_MAX_RUNTIME_TURNS = 8;
const STRICT_RUNTIME_TOOL_SCRIPTS_OPTIONS = Object.freeze({
  inferPlaceholderScripts: false,
  allowToolFallback: false,
});
const STRICT_RUNTIME_CAPABILITY_NORMALIZATION_OPTIONS = Object.freeze({
  allowToolNameInference: false,
  allowSyntheticCapabilities: false,
  allowFallbackExecuteScript: false,
  allowBundleFallback: false,
});
const STRICT_RUNTIME_ASSISTANT_TURN_OPTIONS = Object.freeze({
  allowHeuristicRecovery: false,
});
const LOCAL_REVIEWED_TOOL_SCRIPT_DISALLOWED_PATTERN = /\b(document|window|chrome|page|querySelector|getByText|getByRole|locator|mouse|keyboard|activeText|clipboard)\b|\.goto\(|\.click\(|\.fill\(|\.press\(|\.evaluate\(|(?:__callReviewedTool|callReviewedTool|callBuiltin)\(\s*["'](?:read_active_text_target|replace_active_text_target|read_clipboard_text|write_clipboard_text|capture_bug_report_context)["']/i;
const LOCAL_REVIEWED_TOOL_RUNTIME_TIMEOUT_MS = 30_000;
const AUXILIARY_PAGE_STATE_TIMEOUT_MS = 1_500;
// Keep this above the offscreen local-qwen chat timeout so agent fallback can still run.
const ACTIVE_TEXT_REPLACEMENT_TIMEOUT_MS = 40_000;
const ACTIVE_TEXT_CAPABILITY_EXECUTION_TIMEOUT_MS = 20_000;
const REVIEWED_RUNTIME_WEB_SEARCH_ENDPOINT = "https://html.duckduckgo.com/html/";
const REVIEWED_RUNTIME_WEB_SEARCH_RESULT_LIMIT = 8;
const DEFAULT_REVIEWED_RUNTIME_MAX_TOKENS = 700;
const LOCAL_QWEN_REVIEWED_DECISION_MAX_TOKENS = 220;
const LOCAL_REVIEWED_TOOL_BROWSER_BOUND_TOOL_NAMES = new Set([
  "read_active_text_target",
  "replace_active_text_target",
  "read_clipboard_text",
  "write_clipboard_text",
  "capture_bug_report_context",
]);
const ASYNC_FUNCTION_CTOR = Object.getPrototypeOf(async function () {}).constructor;
const SUPPORTS_LOCAL_REVIEWED_TOOL_EVAL = (() => {
  try {
    return typeof new ASYNC_FUNCTION_CTOR("return true;") === "function";
  } catch {
    return false;
  }
})();

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

function trimText(value, max = 320) {
  const text = asText(value).replace(/\s+/g, " ").trim();
  if (text.length <= max) return text;
  return `${text.slice(0, Math.max(1, max - 1)).trimEnd()}...`;
}

function updateCraftUseDebug(stage = "", detail = null) {
  try {
    globalThis.__sinepanelCraftUseDebug = {
      stage: asText(stage),
      detail: detail && typeof detail === "object" ? cloneJson(detail, {}) : detail ?? null,
      updatedAt: new Date().toISOString(),
    };
  } catch {}
}

function shouldRetryDecisionViaCanonicalFallback(error) {
  const detail =
    error && typeof error === "object" && "detail" in error && error.detail && typeof error.detail === "object"
      ? error.detail
      : null;
  const failureTexts = Array.isArray(detail?.failures)
    ? detail.failures.flatMap((entry) => [
        asText(entry?.reason),
        asText(entry?.numericCode),
        asText(entry?.error?.message),
        asText(entry?.error?.props?.rawMessage),
      ])
    : [];
  const haystack = [
    error instanceof Error ? error.message : String(error || ""),
    asText(detail?.reason),
    asText(detail?.numericCode),
    asText(detail?.runtime?.reason),
    asText(detail?.runtime?.numericCode),
    asText(detail?.error?.message),
    asText(detail?.error?.props?.rawMessage),
    ...failureTexts,
  ].join(" ").toLowerCase();
  if (/local qwen runtime failed|local-qwen-offscreen-call fehlgeschlagen|local_qwen|webgpu|onnx|ortrun|offscreen/.test(haystack)) {
    return false;
  }
  return true;
}

function buildCraftLocalContext(craft = null) {
  const craftId = asText(craft?.id);
  return craftId ? { craftId } : {};
}

function buildCraftModelRequest(craft = null, overrides = {}) {
  const runtimeParameters =
    craft?.runtimeParameters && typeof craft.runtimeParameters === "object"
      ? craft.runtimeParameters
      : {};
  const overrideParameters =
    overrides?.parameters && typeof overrides.parameters === "object"
      ? overrides.parameters
      : {};
  return {
    slotId: asText(overrides?.slotId) || asText(craft?.targetSlot) || "target",
    modelRef: asText(overrides?.modelRef) || asText(craft?.modelRef || craft?.runtimeModelRef),
    providerId: asText(overrides?.providerId) || asText(craft?.providerId || craft?.runtimeProviderId),
    modelName: asText(overrides?.modelName) || asText(craft?.modelName || craft?.runtimeModelName),
    reasoningEffort:
      asText(overrides?.reasoningEffort) ||
      asText(craft?.reasoningEffort || craft?.runtimeReasoningEffort),
    localContext: {
      ...buildCraftLocalContext(craft),
      ...(overrides?.localContext && typeof overrides.localContext === "object" ? overrides.localContext : {}),
    },
    ...overrides,
    parameters: {
      ...runtimeParameters,
      ...overrideParameters,
    },
  };
}

function normalizeArray(value) {
  return Array.isArray(value) ? value : [];
}

function normalizeToolName(value) {
  return asText(value)
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
}

function normalizeRuntimeSignalText(value) {
  return asText(value)
    .toLowerCase()
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function findReviewedToolScriptEntry(toolScriptsPayload = null, toolName = "") {
  const normalizedToolName = normalizeToolName(toolName);
  if (!normalizedToolName) return null;
  return normalizeArray(toolScriptsPayload?.scripts).find((entry) => {
    return [
      normalizeToolName(entry?.id),
      normalizeToolName(entry?.name),
    ].includes(normalizedToolName);
  }) || null;
}

function usesDirectReviewedToolExecute(capability = null) {
  const toolName = asText(capability?.toolName);
  if (!toolName) return false;
  return asText(capability?.scripts?.execute) === buildDefaultReviewedToolExecuteScript(toolName);
}

function canExecuteReviewedToolLocally(bundle = null, capability = null) {
  if (!SUPPORTS_LOCAL_REVIEWED_TOOL_EVAL) return false;
  const toolName = normalizeToolName(capability?.toolName || capability?.id || capability?.name);
  if (LOCAL_REVIEWED_TOOL_BROWSER_BOUND_TOOL_NAMES.has(toolName)) return false;
  if (!usesDirectReviewedToolExecute(capability)) return false;
  const toolScriptEntry = findReviewedToolScriptEntry(bundle?.toolScripts?.payload || null, capability?.toolName);
  const source = asText(toolScriptEntry?.source);
  if (!source) return false;
  return !LOCAL_REVIEWED_TOOL_SCRIPT_DISALLOWED_PATTERN.test(source);
}

function createLocalReviewedToolTimeoutError(toolName = "", timeoutMs = LOCAL_REVIEWED_TOOL_RUNTIME_TIMEOUT_MS) {
  const normalizedToolName = normalizeToolName(toolName) || "reviewed_tool";
  const totalMs = Math.max(0, Number(timeoutMs) || 0);
  const durationLabel = totalMs >= 1000 ? `${Math.round(totalMs / 1000)} s` : `${totalMs} ms`;
  const error = new Error(`Local reviewed tool "${normalizedToolName}" timed out after ${durationLabel}.`);
  error.name = "TimeoutError";
  error.detail = {
    reason: "local_reviewed_tool_timeout",
    toolName: normalizedToolName,
    timeoutMs: totalMs,
  };
  return error;
}

async function waitForLocalReviewedToolResult(promise, {
  signal = null,
  toolName = "",
  timeoutMs = LOCAL_REVIEWED_TOOL_RUNTIME_TIMEOUT_MS,
} = {}) {
  const waiters = [Promise.resolve(promise)];
  let timeoutId = 0;
  let detachAbort = () => {};

  if (signal) {
    waiters.push(new Promise((_, reject) => {
      const onAbort = () => {
        reject(signal.reason || new DOMException("Craft runtime aborted.", "AbortError"));
      };
      if (signal.aborted) {
        onAbort();
        return;
      }
      signal.addEventListener("abort", onAbort, { once: true });
      detachAbort = () => {
        signal.removeEventListener("abort", onAbort);
      };
    }));
  }

  if (Number.isFinite(Number(timeoutMs)) && Number(timeoutMs) > 0) {
    waiters.push(new Promise((_, reject) => {
      timeoutId = globalThis.setTimeout(() => {
        reject(createLocalReviewedToolTimeoutError(toolName, timeoutMs));
      }, Math.max(1, Math.floor(Number(timeoutMs) || 0)));
    }));
  }

  try {
    return await Promise.race(waiters);
  } finally {
    detachAbort();
    if (timeoutId) {
      globalThis.clearTimeout(timeoutId);
    }
  }
}

async function collectPageStateSafely(tabId, timeoutMs = AUXILIARY_PAGE_STATE_TIMEOUT_MS) {
  const numericTabId = Number(tabId || 0);
  if (!Number.isFinite(numericTabId) || numericTabId <= 0) return null;
  let timeoutId = 0;
  try {
    return await Promise.race([
      toolCollectPageState(numericTabId).catch(() => null),
      new Promise((resolve) => {
        timeoutId = globalThis.setTimeout(() => resolve(null), Math.max(100, Number(timeoutMs || 0) || AUXILIARY_PAGE_STATE_TIMEOUT_MS));
      }),
    ]);
  } catch {
    return null;
  } finally {
    if (timeoutId) {
      globalThis.clearTimeout(timeoutId);
    }
  }
}

function buildReviewedWebSearchRuntimePrelude() {
  return [
    `const __reviewedWebSearchEndpoint = ${JSON.stringify(REVIEWED_RUNTIME_WEB_SEARCH_ENDPOINT)};`,
    `const __reviewedWebSearchResultLimit = ${Math.max(1, Math.min(8, Number(REVIEWED_RUNTIME_WEB_SEARCH_RESULT_LIMIT) || 8))};`,
    "const __decodeHtmlEntities = (value) => String(value || '')",
    "  .replace(/&amp;/g, '&')",
    "  .replace(/&quot;/g, '\"')",
    "  .replace(/&#39;/g, \"'\")",
    "  .replace(/&lt;/g, '<')",
    "  .replace(/&gt;/g, '>')",
    "  .replace(/&#x2F;/gi, '/')",
    "  .replace(/&#x3D;/gi, '=')",
    "  .replace(/&#x26;/gi, '&');",
    "const __stripHtml = (value) => __decodeHtmlEntities(String(value || '').replace(/<[^>]+>/g, ' ').replace(/\\s+/g, ' ').trim());",
    "const __unwrapDuckDuckGoUrl = (rawUrl) => {",
    "  const value = __asText(rawUrl);",
    "  if (!value) return '';",
    "  try {",
    "    const parsed = new URL(value, __reviewedWebSearchEndpoint);",
    "    const direct = parsed.searchParams.get('uddg');",
    "    return direct ? decodeURIComponent(direct) : parsed.toString();",
    "  } catch {",
    "    return value;",
    "  }",
    "};",
    "const __normalizeSearchDomainList = (value) => __normalizeStringList(value)",
    "  .map((entry) => __asText(entry).toLowerCase())",
    "  .filter(Boolean)",
    "  .slice(0, 8);",
    "const __webSearchResultMatchesAllowedDomains = (url, allowedDomains = []) => {",
    "  if (!Array.isArray(allowedDomains) || !allowedDomains.length) return true;",
    "  try {",
    "    const hostname = new URL(String(url || '')).hostname.toLowerCase();",
    "    return allowedDomains.some((domain) => hostname === domain || hostname.endsWith(`.${domain}`));",
    "  } catch {",
    "    return false;",
    "  }",
    "};",
    "const __extractDuckDuckGoResults = (html, limit = __reviewedWebSearchResultLimit) => {",
    "  const source = String(html || '');",
    "  const blockPattern = /<div[^>]*class=\"[^\"]*result[^\"]*\"[^>]*>([\\s\\S]*?)<\\/div>\\s*<\\/div>/gi;",
    "  const blocks = [];",
    "  for (const match of source.matchAll(blockPattern)) {",
    "    if (!match?.[1]) continue;",
    "    blocks.push(match[1]);",
    "    if (blocks.length >= Math.max(1, Number(limit || 0)) * 3) break;",
    "  }",
    "  const results = [];",
    "  for (const block of blocks) {",
    "    const anchorMatch = block.match(/<a[^>]*class=\"[^\"]*result__a[^\"]*\"[^>]*href=\"([^\"]+)\"[^>]*>([\\s\\S]*?)<\\/a>/i);",
    "    if (!anchorMatch) continue;",
    "    const snippetMatch = block.match(/<a[^>]*class=\"[^\"]*result__snippet[^\"]*\"[^>]*>([\\s\\S]*?)<\\/a>|<div[^>]*class=\"[^\"]*result__snippet[^\"]*\"[^>]*>([\\s\\S]*?)<\\/div>/i);",
    "    const url = __unwrapDuckDuckGoUrl(anchorMatch[1]);",
    "    const title = __stripHtml(anchorMatch[2]);",
    "    const snippet = __stripHtml(snippetMatch?.[1] || snippetMatch?.[2] || '');",
    "    if (!url || !title) continue;",
    "    results.push({ title, url, snippet });",
    "    if (results.length >= Math.max(1, Number(limit || 0))) break;",
    "  }",
    "  return results;",
    "};",
    "const __webSearchTool = async (input = {}) => {",
    "  const params = input && typeof input === 'object' ? input : {};",
    "  const query = [params.query, params.focus]",
    "    .map((entry) => __asText(entry))",
    "    .filter(Boolean)",
    "    .join(' ')",
    "    .replace(/\\s+/g, ' ')",
    "    .trim();",
    "  if (!query) return __buildToolError('MISSING_SEARCH_QUERY', 'web_search requires a query.');",
    "  const maxResults = Math.max(",
    "    1,",
    "    Math.min(",
    "      __reviewedWebSearchResultLimit,",
    "      Math.floor(Number(params.maxResults ?? params.max_results) || 0) || __reviewedWebSearchResultLimit,",
    "    ),",
    "  );",
    "  const allowedDomains = __normalizeSearchDomainList(params.allowedDomains || params.allowed_domains);",
    "  const url = `${__reviewedWebSearchEndpoint}?q=${encodeURIComponent(query)}`;",
    "  let response;",
    "  try {",
    "    response = await fetch(url, {",
    "      method: 'GET',",
    "      headers: {",
    "        accept: 'text/html,application/xhtml+xml',",
    "      },",
    "    });",
    "  } catch (error) {",
    "    return __buildToolError(",
    "      'WEB_SEARCH_REQUEST_FAILED',",
    "      error instanceof Error ? error.message : 'web_search request failed.',",
    "    );",
    "  }",
    "  if (!response?.ok) {",
    "    return __buildToolError(",
    "      'WEB_SEARCH_HTTP_ERROR',",
    "      `web_search returned ${Number(response?.status || 0) || 'an unknown'} status.`,",
    "    );",
    "  }",
    "  const html = await response.text();",
    "  const results = __extractDuckDuckGoResults(html, maxResults * 3)",
    "    .filter((entry) => __webSearchResultMatchesAllowedDomains(entry?.url, allowedDomains))",
    "    .slice(0, maxResults);",
    "  if (!results.length) {",
    "    return __buildToolError('WEB_SEARCH_NO_RESULTS', `web_search returned no results for ${query}.`);",
    "  }",
    "  return __buildToolOk({",
    "    query,",
    "    results,",
    "    backend: 'duckduckgo_html',",
    "  });",
    "};",
  ].join("\n");
}

function buildLocalReviewedToolRuntimePrelude() {
  return [
    "const __asText = (value) => String(value == null ? '' : value).trim();",
    "const __normalizeToolKey = (value) => __asText(value).toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '');",
    "const __normalizeToolInputPayload = (input = args) => {",
    "  if (typeof input === 'string') return { text: input };",
    "  if (input && typeof input === 'object' && !Array.isArray(input)) return input;",
    "  return {};",
    "};",
    "const __normalizeStringList = (value) => {",
    "  if (Array.isArray(value)) return value.map((entry) => __asText(entry)).filter(Boolean);",
    "  const text = __asText(value);",
    "  return text ? text.split(/[\\n,;]+/).map((entry) => __asText(entry)).filter(Boolean) : [];",
    "};",
    "const __buildToolOk = (data = {}) => ({",
    "  ok: true,",
    "  data: data && typeof data === 'object' ? data : {},",
    "});",
    "const __buildToolError = (code, message) => ({",
    "  ok: false,",
    "  error: {",
    "    code: __asText(code || 'TOOL_ERROR'),",
    "    message: __asText(message || 'Tool execution failed.'),",
    "  },",
    "});",
    "const __buildMailtoUrl = ({ recipient = '', subject = '', body = '', cc = [], bcc = [], replyTo = '' } = {}) => {",
    "  const query = new URLSearchParams();",
    "  if (__asText(subject)) query.set('subject', __asText(subject));",
    "  if (__asText(body)) query.set('body', __asText(body));",
    "  if (Array.isArray(cc) && cc.length) query.set('cc', cc.map((entry) => __asText(entry)).filter(Boolean).join(','));",
    "  if (Array.isArray(bcc) && bcc.length) query.set('bcc', bcc.map((entry) => __asText(entry)).filter(Boolean).join(','));",
    "  if (__asText(replyTo)) query.set('replyTo', __asText(replyTo));",
    "  const queryText = query.toString();",
    "  return `mailto:${encodeURIComponent(__asText(recipient))}${queryText ? `?${queryText}` : ''}`;",
    "};",
    "const __buildEmailClipboardText = ({ recipient = '', subject = '', body = '', cc = [], bcc = [], replyTo = '' } = {}) => {",
    "  return [",
    "    __asText(recipient) ? `To: ${__asText(recipient)}` : '',",
    "    Array.isArray(cc) && cc.length ? `Cc: ${cc.join(', ')}` : '',",
    "    Array.isArray(bcc) && bcc.length ? `Bcc: ${bcc.join(', ')}` : '',",
    "    __asText(replyTo) ? `Reply-To: ${__asText(replyTo)}` : '',",
    "    __asText(subject) ? `Subject: ${__asText(subject)}` : '',",
    "    __asText(body),",
    "  ].filter(Boolean).join('\\n\\n');",
    "};",
    "const __composeEmailDraft = async (input = {}) => {",
    "  const params = input && typeof input === 'object' ? input : {};",
    "  const recipient = __asText(params.recipient || params.to);",
    "  const subject = __asText(params.subject);",
    "  const body = __asText(params.body);",
    "  if (!subject || !body) return __buildToolError('MISSING_EMAIL_FIELDS', 'compose_email requires subject and body.');",
    "  const cc = __normalizeStringList(params.cc);",
    "  const bcc = __normalizeStringList(params.bcc);",
    "  const replyTo = __asText(params.replyTo || params.reply_to);",
    "  return __buildToolOk({",
    "    recipient,",
    "    subject,",
    "    body,",
    "    cc,",
    "    bcc,",
    "    replyTo,",
    "    mailtoUrl: __buildMailtoUrl({ recipient, subject, body, cc, bcc, replyTo }),",
    "    clipboardText: __buildEmailClipboardText({ recipient, subject, body, cc, bcc, replyTo }),",
    "  });",
    "};",
    buildReviewedWebSearchRuntimePrelude(),
    "const __reviewedToolRegistry = Object.create(null);",
    "const __builtinReviewedToolRegistry = Object.create(null);",
    "const __registerReviewedTool = (toolName, handler) => {",
    "  if (typeof handler !== 'function') return handler;",
    "  const rawKey = __asText(toolName);",
    "  const normalizedKey = __normalizeToolKey(toolName);",
    "  if (rawKey) __reviewedToolRegistry[rawKey] = handler;",
    "  if (normalizedKey) __reviewedToolRegistry[normalizedKey] = handler;",
    "  try {",
    "    if (rawKey) globalThis[rawKey] = handler;",
    "    if (normalizedKey && normalizedKey !== rawKey) globalThis[normalizedKey] = handler;",
    "  } catch {}",
    "  return handler;",
    "};",
    "const __registerBuiltinReviewedTool = (toolName, handler) => {",
    "  if (typeof handler !== 'function') return handler;",
    "  const rawKey = __asText(toolName);",
    "  const normalizedKey = __normalizeToolKey(toolName);",
    "  if (rawKey) __builtinReviewedToolRegistry[rawKey] = handler;",
    "  if (normalizedKey) __builtinReviewedToolRegistry[normalizedKey] = handler;",
    "  return __registerReviewedTool(toolName, handler);",
    "};",
    "const __resolveBuiltinReviewedTool = (toolName) => {",
    "  const rawKey = __asText(toolName);",
    "  const normalizedKey = __normalizeToolKey(toolName);",
    "  return __builtinReviewedToolRegistry[rawKey] || __builtinReviewedToolRegistry[normalizedKey] || null;",
    "};",
    "const __resolveReviewedTool = (toolName) => {",
    "  const rawKey = __asText(toolName);",
    "  const normalizedKey = __normalizeToolKey(toolName);",
    "  return __reviewedToolRegistry[rawKey] ||",
    "    __reviewedToolRegistry[normalizedKey] ||",
    "    (typeof globalThis[rawKey] === 'function' ? globalThis[rawKey] : null) ||",
    "    (typeof globalThis[normalizedKey] === 'function' ? globalThis[normalizedKey] : null) ||",
    "    __resolveBuiltinReviewedTool(toolName);",
    "};",
    "const __respondToReviewedTool = (value = null) => value;",
    "const __buildToolRuntime = (input = args) => {",
    "  const inputPayload = __normalizeToolInputPayload(input);",
    "  return Object.freeze({",
    "    ...inputPayload,",
    "    input: inputPayload,",
    "    args: inputPayload,",
    "    ctx,",
    "    WAIT,",
    "    tools,",
    "    prompt,",
    "    browserContext,",
    "    state,",
    "    observation,",
    "    modelOutput,",
    "    craft,",
    "    capability,",
    "    action,",
    "    composeEmail: __composeEmailDraft,",
    "    compose_email: __composeEmailDraft,",
    "    webSearch: __webSearchTool,",
    "    web_search: __webSearchTool,",
    "    callReviewedTool: (toolName, nextInput = inputPayload) => __callReviewedTool(toolName, nextInput),",
    "    callBuiltin: (toolName, nextInput = inputPayload) => __callBuiltinReviewedTool(toolName, nextInput),",
    "    respond: (value = null) => __respondToReviewedTool(value),",
    "    ok: (data = {}) => __buildToolOk(data),",
    "    error: (code, message) => __buildToolError(code, message),",
    "  });",
    "};",
    "const __buildToolCallContext = (input = args, runtime = null) => {",
    "  const inputPayload = __normalizeToolInputPayload(input);",
    "  return Object.freeze({",
    "    ...ctx,",
    "    ...inputPayload,",
    "    input: inputPayload,",
    "    args: inputPayload,",
    "    runtime,",
    "    WAIT,",
    "    tools,",
    "    prompt,",
    "    browserContext,",
    "    state,",
    "    observation,",
    "    modelOutput,",
    "    craft,",
    "    capability,",
    "    action,",
    "    composeEmail: __composeEmailDraft,",
    "    compose_email: __composeEmailDraft,",
    "    webSearch: __webSearchTool,",
    "    web_search: __webSearchTool,",
    "    callReviewedTool: (toolName, nextInput = inputPayload) => __callReviewedTool(toolName, nextInput),",
    "    callBuiltin: (toolName, nextInput = inputPayload) => __callBuiltinReviewedTool(toolName, nextInput),",
    "    respond: (value = null) => __respondToReviewedTool(value),",
    "    ok: (data = {}) => __buildToolOk(data),",
    "    error: (code, message) => __buildToolError(code, message),",
    "  });",
    "};",
    "const __invokeToolHandler = async (handler, input = args) => {",
    "  const runtime = __buildToolRuntime(input);",
    "  const callCtx = __buildToolCallContext(input, runtime);",
    "  return await handler(runtime, callCtx, runtime.args);",
    "};",
    "const __callBuiltinReviewedTool = async (toolName, input = args) => {",
    "  const handler = __resolveBuiltinReviewedTool(toolName);",
    "  if (typeof handler !== 'function') {",
    "    throw new ReferenceError(`${__asText(toolName) || 'builtin_reviewed_tool'} is not defined`);",
    "  }",
    "  return await __invokeToolHandler(handler, input);",
    "};",
    "const __callReviewedTool = async (toolName, input = args) => {",
    "  const handler = __resolveReviewedTool(toolName);",
    "  if (typeof handler !== 'function') {",
    "    throw new ReferenceError(`${__asText(toolName) || 'reviewed_tool'} is not defined`);",
    "  }",
    "  return await __invokeToolHandler(handler, input);",
    "};",
    "const compose_email = __registerBuiltinReviewedTool('compose_email', __composeEmailDraft);",
    "const composeEmail = __registerBuiltinReviewedTool('composeEmail', __composeEmailDraft);",
    "const web_search = __registerBuiltinReviewedTool('web_search', __webSearchTool);",
    "const webSearch = __registerBuiltinReviewedTool('webSearch', __webSearchTool);",
    "const tools = Object.freeze({",
    "  WAIT,",
    "  composeEmail: __composeEmailDraft,",
    "  compose_email: __composeEmailDraft,",
    "  webSearch: __webSearchTool,",
    "  web_search: __webSearchTool,",
    "  callReviewedTool: __callReviewedTool,",
    "  callBuiltin: __callBuiltinReviewedTool,",
    "});",
    "const ctx = Object.freeze({",
    "  WAIT,",
    "  tools,",
    "  composeEmail: __composeEmailDraft,",
    "  compose_email: __composeEmailDraft,",
    "  webSearch: __webSearchTool,",
    "  web_search: __webSearchTool,",
    "  callReviewedTool: __callReviewedTool,",
    "  callBuiltin: __callBuiltinReviewedTool,",
    "  args,",
    "  prompt,",
    "  browserContext,",
    "  state,",
    "  observation,",
    "  modelOutput,",
    "  craft,",
    "  capability,",
    "  action,",
    "});",
    "try { globalThis.ctx = ctx; } catch {}",
    "try { globalThis.tools = tools; } catch {}",
  ].join("\n");
}

async function executeReviewedToolLocally({
  bundle = null,
  craft = null,
  capability = null,
  action = null,
  prompt = "",
  browserContext = null,
  state = null,
  observation = null,
  modelOutput = null,
  signal,
} = {}) {
  const toolName = normalizeToolName(capability?.toolName || capability?.id || capability?.name);
  const toolScriptsPayload = bundle?.toolScripts?.payload || null;
  const toolScriptEntry = findReviewedToolScriptEntry(toolScriptsPayload, toolName);
  if (!toolName || !toolScriptEntry || !asText(toolScriptEntry?.source)) {
    throw new Error(`The reviewed tool "${toolName || "tool"}" is not available for local execution.`);
  }
  const runtimePayload = {
    craft:
      craft && typeof craft === "object"
        ? {
            id: asText(craft.id),
            name: asText(craft.name),
            summary: asText(craft.summary),
            inputMode: asText(craft.inputMode),
          }
        : null,
    capability: cloneJson(capability, null),
    action: cloneJson(action, null),
    args:
      action?.arguments && typeof action.arguments === "object"
        ? cloneJson(action.arguments, {})
        : {},
    prompt: asText(prompt),
    browserContext: cloneJson(browserContext, null),
    state: cloneJson(state, {}),
    observation: cloneJson(observation, null),
    modelOutput: cloneJson(modelOutput, null),
  };
  const localRuntimeSource = [
    `const __runtime = ${JSON.stringify(runtimePayload)};`,
    "const craft = __runtime.craft || null;",
    "const capability = __runtime.capability || null;",
    "const action = __runtime.action || null;",
    "const args = __runtime.args || {};",
    "const prompt = __runtime.prompt || '';",
    "const browserContext = __runtime.browserContext || null;",
    "const state = __runtime.state || {};",
    "const observation = __runtime.observation || null;",
    "const modelOutput = __runtime.modelOutput || null;",
    "const WAIT = (ms = 0) => new Promise((resolve) => setTimeout(resolve, Math.max(0, Number(ms) || 0)));",
    buildLocalReviewedToolRuntimePrelude(),
    buildReviewedToolScriptsPrelude(toolScriptsPayload),
    `const __toolName = ${JSON.stringify(toolName)};`,
    "const __handler =",
    "  __resolveReviewedTool(__toolName) ||",
    "  (typeof globalThis[__toolName] === 'function' ? globalThis[__toolName] : null) ||",
    "  (typeof globalThis[__normalizeToolKey(__toolName)] === 'function' ? globalThis[__normalizeToolKey(__toolName)] : null);",
    "if (typeof __handler !== 'function') {",
    "  throw new ReferenceError(`${__toolName || 'reviewed_tool'} is not defined`);",
    "}",
    "const runtime = __buildToolRuntime(args);",
    "const callCtx = __buildToolCallContext(args, runtime);",
    "return await __handler(runtime, callCtx, runtime.args);",
  ].filter((part) => asText(part)).join("\n\n");
  const rawResult = await waitForLocalReviewedToolResult(
    new ASYNC_FUNCTION_CTOR(localRuntimeSource)(),
    {
      signal,
      toolName,
    },
  );
  return normalizeCapabilityExecutionResult("execute", rawResult, capability);
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

function buildCanonicalActionDefaults(bundle = null) {
  return {
    defaultBundleRef: asText(bundle?.browserCapabilities?.artifactId),
    defaultResourceRefs: buildBundleDefaultResourceRefs(bundle),
  };
}

function extractUsage(response) {
  const usage = response?.usage || null;
  if (!usage || typeof usage !== "object") {
    return {
      inputTokens: 0,
      outputTokens: 0,
      totalTokens: 0,
      costUsd: 0,
    };
  }
  const inputTokens = Number(
    usage.inputTokens ??
      usage.promptTokens ??
      usage.input_tokens ??
      usage.prompt_tokens ??
      0,
  );
  const outputTokens = Number(
    usage.outputTokens ??
      usage.completionTokens ??
      usage.output_tokens ??
      usage.completion_tokens ??
      0,
  );
  const totalTokens = Number(usage.totalTokens ?? usage.total_tokens ?? inputTokens + outputTokens);
  const costUsd = Number(usage.costUsd ?? usage.cost_usd ?? 0);
  return {
    inputTokens: Math.max(0, inputTokens),
    outputTokens: Math.max(0, outputTokens),
    totalTokens: Math.max(0, totalTokens),
    costUsd: Math.max(0, costUsd),
  };
}

function addUsage(total, usage) {
  return {
    inputTokens: Number(total?.inputTokens || 0) + Number(usage?.inputTokens || 0),
    outputTokens: Number(total?.outputTokens || 0) + Number(usage?.outputTokens || 0),
    totalTokens: Number(total?.totalTokens || 0) + Number(usage?.totalTokens || 0),
    costUsd: Number(total?.costUsd || 0) + Number(usage?.costUsd || 0),
  };
}

function clipPromptPayload(value, maxChars = 5_000) {
  try {
    const json = JSON.stringify(value ?? null);
    if (json.length <= maxChars) {
      return value ?? null;
    }
    return {
      clipped: true,
      preview: `${json.slice(0, Math.max(1, maxChars - 1)).trimEnd()}...`,
    };
  } catch {
    return trimText(value, maxChars);
  }
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

function hasReviewedRuntimePackageSignals(craft = null, bundle = null) {
  return (
    normalizeArray(craft?.tools).map((entry) => asText(entry)).filter(Boolean).length > 0 ||
    normalizeArray(bundle?.toolScripts?.payload?.scripts).length > 0 ||
    normalizeArray(bundle?.browserCapabilities?.payload?.capabilities).length > 0
  );
}

function buildRuntimeCapabilityCatalog(bundle = null) {
  return buildQwenCapabilityCatalogPayload(bundle?.browserCapabilities?.payload || null, {
    normalizationOptions: STRICT_RUNTIME_CAPABILITY_NORMALIZATION_OPTIONS,
  });
}

function buildRuntimeCapabilityToolDefinitions(capabilityPayload = null) {
  return buildQwenCapabilityToolDefinitions(capabilityPayload, {
    normalizationOptions: STRICT_RUNTIME_CAPABILITY_NORMALIZATION_OPTIONS,
  });
}

function buildRuntimeCapabilitySkillText(capabilityPayload = null, options = {}) {
  return buildQwenCapabilitySkillText(capabilityPayload, {
    ...options,
    normalizationOptions: STRICT_RUNTIME_CAPABILITY_NORMALIZATION_OPTIONS,
  });
}

function compileRuntimeCapabilityExecutionPlan(action, capabilityPayload = null) {
  return compileCanonicalActionExecutionPlan(action, capabilityPayload, {
    normalizationOptions: STRICT_RUNTIME_CAPABILITY_NORMALIZATION_OPTIONS,
  });
}

function resolveStrictRuntimeBundle(craft = null, bundle = null) {
  const resolution = resolvePublishedBrowserCapabilityBundlePayload(
    bundle?.browserCapabilities?.payload || null,
    {
      craft,
      toolScriptsPayload: bundle?.toolScripts?.payload || null,
      expectedToolScriptsFingerprint: getPayloadFingerprint(bundle?.toolScripts?.payload || null),
      expectedRuntimePackageVersion: BROWSER_CAPABILITY_RUNTIME_PACKAGE_VERSION,
    },
  );
  if (!resolution?.ok) {
    return {
      ok: false,
      error: asText(resolution?.error) || "Craft runtime requires a published reviewed capability package.",
      reason: asText(resolution?.reason) || "runtime_package_invalid",
    };
  }
  return {
    ok: true,
    bundle: {
      ...(bundle && typeof bundle === "object" ? bundle : {}),
      browserCapabilities: {
        ...(bundle?.browserCapabilities && typeof bundle.browserCapabilities === "object"
          ? bundle.browserCapabilities
          : {}),
        payload: resolution.payload,
      },
    },
  };
}

function summarizeObservation(execution = null) {
  if (!execution || typeof execution !== "object") {
    return {
      ok: false,
      summary: "Capability execution returned no result.",
    };
  }
  if (execution.ok === false) {
    return {
      ok: false,
      summary: trimText(execution.error || "Capability execution failed.", 280),
      detail: clipPromptPayload(execution, 3_000),
    };
  }
  return {
    ok: true,
    capability: asText(execution?.capability?.name || execution?.capabilityName),
    summary: trimText(execution.summary || "Capability executed.", 280),
    output: clipPromptPayload(execution.finalOutput, 3_000),
    pageState: clipPromptPayload(execution.pageState, 2_000),
  };
}

function buildBundleDefaultResourceRefs(bundle = null) {
  return buildRuntimeCapabilityCatalog(bundle)
    .capabilities
    .flatMap((entry) => normalizeArray(entry.resourceRefs))
    .slice(0, 24);
}

function getCraftCapabilityProfile(craft = null) {
  const fallbackModelName =
    asText(craft?.starterModelName) ||
    asText(craft?.training?.shards?.[0]?.modelName);
  const explicitProviderType = asText(craft?.providerId || craft?.runtimeProviderId);
  const inferredProviderType =
    !explicitProviderType && /qwen/i.test(fallbackModelName)
      ? "local_qwen"
      : explicitProviderType;
  const explicitRuntimeKind = asText(craft?.runtimeKind);
  const inferredRuntimeKind =
    !explicitRuntimeKind && inferredProviderType === "local_qwen"
      ? "webgpu_local"
      : explicitRuntimeKind;
  return getQwenAgentCapabilityProfile({
    providerType: inferredProviderType,
    modelName: asText(craft?.modelName || craft?.runtimeModelName) || fallbackModelName,
    runtimeKind: inferredRuntimeKind,
  });
}

async function resolveCraftCapabilityProfile(craft = null) {
  const fallbackProfile = getCraftCapabilityProfile(craft);
  try {
    const resolved = await describeResolvedModel({
      slotId: asText(craft?.targetSlot) || "target",
      modelRef: asText(craft?.modelRef || craft?.runtimeModelRef),
      providerId: asText(craft?.providerId || craft?.runtimeProviderId),
      modelName: asText(craft?.modelName || craft?.runtimeModelName),
      parameters:
        craft?.runtimeParameters && typeof craft.runtimeParameters === "object"
          ? craft.runtimeParameters
          : {},
      reasoningEffort:
        asText(craft?.reasoningEffort) ||
        asText(craft?.runtimeReasoningEffort),
    });
    const resolvedProviderType = asText(resolved?.provider?.type).toLowerCase();
    const resolvedRuntimeKind =
      resolvedProviderType === "local_qwen"
        ? "webgpu_local"
        : asText(fallbackProfile?.runtimeKind);
    return getQwenAgentCapabilityProfile({
      providerType: resolvedProviderType || fallbackProfile.providerType,
      modelName: asText(resolved?.modelName) || fallbackProfile.modelName,
      runtimeKind: resolvedRuntimeKind,
    });
  } catch {
    return fallbackProfile;
  }
}

function resolveReviewedDecisionMaxTokens(capabilityProfile = null) {
  return asText(capabilityProfile?.providerType).toLowerCase() === "local_qwen"
    ? LOCAL_QWEN_REVIEWED_DECISION_MAX_TOKENS
    : DEFAULT_REVIEWED_RUNTIME_MAX_TOKENS;
}

function resolveCraftRuntimeMaxTurns({
  craft = null,
  bundle = null,
  requestedMaxTurns = null,
} = {}) {
  const explicitTurns = Number(requestedMaxTurns);
  if (Number.isFinite(explicitTurns) && explicitTurns > 0) {
    return Math.max(1, Math.min(MULTISTEP_MAX_RUNTIME_TURNS, Math.floor(explicitTurns)));
  }

  const policySpec = bundle?.policy?.payload?.policySpec && typeof bundle.policy.payload.policySpec === "object"
    ? bundle.policy.payload.policySpec
    : null;
  const policyTurns = Number(policySpec?.runtimeMaxTurns || policySpec?.maxRuntimeTurns);
  if (Number.isFinite(policyTurns) && policyTurns > 0) {
    return Math.max(1, Math.min(MULTISTEP_MAX_RUNTIME_TURNS, Math.floor(policyTurns)));
  }

  const catalog = buildRuntimeCapabilityCatalog(bundle);
  const capabilityToolNames = catalog.capabilities
    .map((entry) => asText(entry.toolName).toLowerCase())
    .filter(Boolean);
  const skillText = normalizeArray(catalog.skills).join("\n").toLowerCase();
  const looksLikeSimpleActiveTextFlow =
    capabilityToolNames.length <= 2 &&
    capabilityToolNames.includes("read_active_text_target") &&
    capabilityToolNames.includes("replace_active_text_target");
  if (looksLikeSimpleActiveTextFlow) {
    return DEFAULT_MAX_RUNTIME_TURNS;
  }

  const multiStepSignals = [
    skillText,
    capabilityToolNames.join("\n"),
    asText(craft?.summary).toLowerCase(),
    asText(craft?.name).toLowerCase(),
  ].join("\n");
  if (
    capabilityToolNames.length > 2 ||
    /search|listing|news|newsletter|restaurant|maps|digest|report|email|clipboard|bug/.test(multiStepSignals)
  ) {
    return MULTISTEP_MAX_RUNTIME_TURNS;
  }

  return DEFAULT_MAX_RUNTIME_TURNS;
}

function getPreferredInitialCapabilityToolName(bundle = null, capabilityProfile = null) {
  const catalog = buildRuntimeCapabilityCatalog(bundle);
  const toolNames = catalog.capabilities
    .map((entry) => asText(entry.toolName))
    .filter(Boolean);
  if (!toolNames.length) return "";

  const policySpec = bundle?.policy?.payload?.policySpec && typeof bundle.policy.payload.policySpec === "object"
    ? bundle.policy.payload.policySpec
    : null;
  const allowedTools = normalizeArray(policySpec?.allowedTools)
    .map((entry) => asText(entry))
    .filter((entry) => entry && toolNames.includes(entry));

  if (capabilityProfile?.shouldPreferSingleActionDecisions && allowedTools.length) {
    return allowedTools[0];
  }
  if (allowedTools.length === 1) {
    return allowedTools[0];
  }
  if (toolNames.length === 1) {
    return toolNames[0];
  }
  return "";
}

function buildForcedCapabilityAction(bundle = null, toolName = "", argumentsValue = {}) {
  const normalizedToolName = asText(toolName);
  if (!normalizedToolName) return null;

  const catalogEntry = buildRuntimeCapabilityCatalog(bundle)
    .capabilities
    .find((entry) => asText(entry.toolName) === normalizedToolName);
  return buildCanonicalCallToolAction({
    toolName: normalizedToolName,
    argumentsValue,
    bundleRef: asText(bundle?.browserCapabilities?.artifactId),
    capabilityRef: asText(catalogEntry?.id) || normalizedToolName,
    skillRef: asText(catalogEntry?.skillRef),
    resourceRefs: normalizeArray(catalogEntry?.resourceRefs),
  });
}

function buildForcedSingleCapabilityAction(bundle = null, toolName = "") {
  return buildForcedCapabilityAction(bundle, toolName, {});
}

function getDeterministicLocalCapabilitySequence(bundle = null) {
  const catalog = buildRuntimeCapabilityCatalog(bundle);
  const toolNames = [];
  for (const entry of normalizeArray(catalog.capabilities)) {
    const toolName = asText(entry?.toolName);
    if (!toolName) return [];
    const action = buildForcedCapabilityAction(bundle, toolName, {});
    const plan = compileRuntimeCapabilityExecutionPlan(action, bundle?.browserCapabilities?.payload || null);
    if (!plan?.ok) return [];
    if (asText(plan.invocation?.scripts?.pre) || asText(plan.invocation?.scripts?.post)) {
      return [];
    }
    const toolScriptEntry = findReviewedToolScriptEntry(bundle?.toolScripts?.payload || null, plan.capability?.toolName);
    const source = asText(toolScriptEntry?.source);
    if (!source || LOCAL_REVIEWED_TOOL_SCRIPT_DISALLOWED_PATTERN.test(source)) {
      return [];
    }
    toolNames.push(toolName);
  }
  return toolNames;
}

function buildDeterministicLocalCapabilityAction(bundle = null, steps = []) {
  const toolNames = getDeterministicLocalCapabilitySequence(bundle);
  if (!toolNames.length) return null;

  const remainingToolName = toolNames.find((toolName) => {
    return !normalizeArray(steps).some((step) => getLastStepToolName([step]) === toolName);
  });
  if (!remainingToolName) return null;
  return buildForcedCapabilityAction(bundle, remainingToolName, {});
}

function getInPlaceActiveTextFlow(bundle = null, craft = null) {
  const catalog = buildRuntimeCapabilityCatalog(bundle);
  const toolNames = new Set(
    normalizeArray(catalog.capabilities)
      .map((entry) => asText(entry.toolName))
      .filter(Boolean),
  );
  const hasRead = toolNames.has("read_active_text_target");
  const hasReplace = toolNames.has("replace_active_text_target");
  const policySpec = bundle?.policy?.payload?.policySpec && typeof bundle.policy.payload.policySpec === "object"
    ? bundle.policy.payload.policySpec
    : null;
  const policyAllowedTools = normalizeArray(policySpec?.allowedTools)
    .map((entry) => normalizeToolName(entry))
    .filter(Boolean);
  const hasDedicatedActiveTextPolicy =
    policyAllowedTools.length === 2 &&
    policyAllowedTools[0] === "read_active_text_target" &&
    policyAllowedTools[1] === "replace_active_text_target";
  const hasOnlyActiveTextCapabilities = toolNames.size === 2 && hasRead && hasReplace;
  const normalizedCatalogSignals = [
    ...normalizeArray(catalog.skills),
    ...normalizeArray(catalog.capabilities).flatMap((entry) => [
      entry?.description,
      entry?.skillRef,
      ...(Array.isArray(entry?.preconditions) ? entry.preconditions : []),
      ...(Array.isArray(entry?.resourceRefs) ? entry.resourceRefs : []),
    ]),
  ]
    .map((entry) => normalizeRuntimeSignalText(entry))
    .filter(Boolean)
    .join("\n");
  const hasNormalizedActiveTextSignals =
    /active text|in place|selection|selected text|focused editable|focused field|clipboard/.test(normalizedCatalogSignals);
  const looksLikeActiveTextCraft =
    asText(craft?.inputMode).toLowerCase() === "context_only" ||
    hasDedicatedActiveTextPolicy ||
    (hasOnlyActiveTextCapabilities &&
      looksLikeTransformingActiveTextWorkflow(
        normalizeArray(bundle?.browserCapabilities?.payload?.capabilities),
        craft,
      )) ||
    (hasOnlyActiveTextCapabilities && hasNormalizedActiveTextSignals);

  return {
    enabled: hasRead && hasReplace && looksLikeActiveTextCraft,
    readToolName: hasRead ? "read_active_text_target" : "",
    replaceToolName: hasReplace ? "replace_active_text_target" : "",
  };
}

function getLastStepToolName(steps = []) {
  const lastStep = normalizeArray(steps)[normalizeArray(steps).length - 1] || null;
  return asText(lastStep?.action?.tool_name || lastStep?.execution?.capability?.id || lastStep?.execution?.capability?.name);
}

function hasCompletedDeterministicLocalCapabilitySequence(bundle = null, steps = []) {
  const toolNames = getDeterministicLocalCapabilitySequence(bundle);
  if (!toolNames.length) return false;
  const seenToolNames = new Set(
    normalizeArray(steps)
      .map((step) => getLastStepToolName([step]))
      .filter(Boolean),
  );
  return toolNames.every((toolName) => seenToolNames.has(toolName));
}

function coerceActiveTextReplyToReplaceAction(bundle = null, steps = [], content = "", flow = null) {
  const normalizedContent = asText(content);
  if (!flow?.enabled || !normalizedContent) return null;
  if (looksLikeClarificationRequest(normalizedContent)) return null;
  if (getLastStepToolName(steps) !== flow.readToolName) return null;
  return buildForcedCapabilityAction(bundle, flow.replaceToolName, {
    text: normalizedContent,
  });
}

function extractActiveTextReadPayload(step = null) {
  const executionData =
    step?.execution?.finalOutput?.data && typeof step.execution.finalOutput.data === "object"
      ? step.execution.finalOutput.data
      : step?.execution?.finalOutput && typeof step.execution.finalOutput === "object"
        ? step.execution.finalOutput
        : step?.observation?.output?.data && typeof step.observation.output.data === "object"
          ? step.observation.output.data
          : null;
  const text = asText(executionData?.text);
  if (!text) return null;
  return {
    text,
    targetType: asText(executionData?.targetType),
  };
}

function normalizeActiveTextReplacement(value = "") {
  const rawText = asText(value);
  if (!rawText) return "";
  const parsed = parseJsonLoose(rawText);
  if (typeof parsed === "string") return asText(parsed);
  if (parsed && typeof parsed === "object" && asText(parsed.text)) {
    return asText(parsed.text);
  }
  const fenced = rawText.match(/```(?:[\w-]+)?\s*([\s\S]*?)\s*```/);
  if (fenced?.[1]) return asText(fenced[1]);
  return rawText;
}

function estimateActiveTextReplacementMaxTokens(sourceText = "") {
  const sourceLength = asText(sourceText).length;
  if (!sourceLength) return 64;
  return Math.max(48, Math.min(160, Math.ceil(sourceLength / 3) + 24));
}

function shouldFallbackActiveTextReplacementViaAgent(craft = null, error = null) {
  const providerType = asText(getCraftCapabilityProfile(craft).providerType).toLowerCase();
  const detail =
    error && typeof error === "object" && "detail" in error && error.detail && typeof error.detail === "object"
      ? error.detail
      : null;
  const failureTexts = Array.isArray(detail?.failures)
    ? detail.failures.flatMap((entry) => [
        asText(entry?.reason),
        asText(entry?.numericCode),
        asText(entry?.error?.message),
        asText(entry?.error?.props?.rawMessage),
      ])
    : [];
  const haystack = [
    error instanceof Error ? error.message : String(error || ""),
    asText(detail?.reason),
    asText(detail?.numericCode),
    asText(detail?.runtime?.reason),
    asText(detail?.runtime?.numericCode),
    asText(detail?.error?.message),
    asText(detail?.error?.props?.rawMessage),
    ...failureTexts,
  ].join(" ").toLowerCase();
  const looksLikeLocalQwenFailure =
    /local qwen runtime failed|local-qwen-offscreen-call fehlgeschlagen|local_qwen|webgpu|onnx|ortrun|offscreen|offscreen_timeout/.test(haystack);
  return looksLikeLocalQwenFailure || (providerType === "local_qwen" && /\btime(?:d)? out|timeout\b/.test(haystack));
}

async function buildDirectActiveTextReplacement({
  craft = null,
  prompt = "",
  sourceText = "",
  targetType = "",
  signal,
} = {}) {
  updateCraftUseDebug("active_text_replacement:start", {
    craftId: asText(craft?.id),
    sourceLength: asText(sourceText).length,
    targetType: asText(targetType),
  });
  const replacementPrompt = asText(prompt) || asText(craft?.summary) || asText(craft?.name);
  if (!replacementPrompt || !asText(sourceText)) return null;
  const maxTokens = estimateActiveTextReplacementMaxTokens(sourceText);
  const request = buildCraftModelRequest(craft, {
    localContext: buildCraftLocalContext(craft),
    messages: [
      {
        role: "system",
        content: [
          `You are the runtime for the craft "${asText(craft?.name) || "craft"}".`,
          asText(craft?.summary) ? `Craft objective: ${asText(craft.summary)}` : "",
          "Transform the provided active text so it satisfies the craft objective.",
          "Return only the full replacement text with no JSON, no markdown fences, and no commentary.",
        ].filter(Boolean).join("\n\n"),
      },
      {
        role: "user",
        content: [
          replacementPrompt ? `Instruction: ${replacementPrompt}` : "",
          targetType ? `Resolved target type: ${targetType}` : "",
          `Source text:\n${asText(sourceText)}`,
          "Return the full replacement text only.",
        ].filter(Boolean).join("\n\n"),
      },
    ],
    parameters: {
      temperature: 0,
      maxTokens,
    },
    signal,
  });
  let response = null;
  try {
    updateCraftUseDebug("active_text_replacement:target_model", {
      slotId: asText(request.slotId),
      modelRef: asText(request.modelRef),
      maxTokens,
    });
    response = await llmChat(request);
  } catch (error) {
    if (!shouldFallbackActiveTextReplacementViaAgent(craft, error)) {
      updateCraftUseDebug("active_text_replacement:error", {
        message: asText(error?.message || error),
      });
      throw error;
    }
    updateCraftUseDebug("active_text_replacement:agent_fallback", {
      maxTokens,
      message: asText(error?.message || error),
    });
    response = await llmChat({
      slotId: "agent",
      messages: request.messages,
      parameters: request.parameters,
      signal,
      localContext: buildCraftLocalContext(craft),
    });
  }
  const text = normalizeActiveTextReplacement(response?.text);
  if (!text) {
    throw new Error("Active-text transform returned no replacement text.");
  }
  updateCraftUseDebug("active_text_replacement:done", {
    textLength: text.length,
    modelRef: asText(response?.resolved?.modelRef),
  });
  return {
    text,
    usage: extractUsage(response),
    modelRef: asText(response?.resolved?.modelRef),
  };
}

function buildStepObservationPayload(step = null) {
  const observation = step?.observation && typeof step.observation === "object" ? step.observation : {};
  const execution = step?.execution && typeof step.execution === "object" ? step.execution : {};
  return {
    ok: observation.ok !== false,
    summary: asText(observation.summary || execution.summary),
    output: clipPromptPayload(observation.output ?? execution.finalOutput ?? null, 2_500),
    pageState: clipPromptPayload(observation.pageState ?? execution.pageState ?? null, 1_800),
    detail: clipPromptPayload(observation.detail ?? execution.error ?? null, 1_800),
  };
}

function buildRuntimeTranscriptMessages({
  craft = null,
  prompt = "",
  browserContext = null,
  steps = [],
  turn = 1,
  maxTurns = DEFAULT_MAX_RUNTIME_TURNS,
} = {}) {
  const transcript = [
    {
      role: "system",
      content: [
        `You are the runtime for the craft "${asText(craft?.name) || "craft"}".`,
        asText(craft?.summary) ? `Craft objective: ${asText(craft.summary)}` : "",
      ].filter(Boolean).join("\n"),
    },
  ];

  if (asText(prompt)) {
    transcript.push({
      role: "user",
      content: asText(prompt),
    });
  }

  transcript.push({
    role: "user",
    content: JSON.stringify(
      {
        browserContext: clipPromptPayload(browserContext, 3_500),
        turn: {
          current: turn,
          max: maxTurns,
        },
      },
      null,
      2,
    ),
  });

  normalizeArray(steps).forEach((step, index) => {
    const assistantTurn =
      normalizePortableAssistantTurn(step?.assistantTurn) ||
      buildPortableAssistantTurnFromCanonicalAction(step?.action, {
        callId: `call_${index + 1}`,
      });
    if (!assistantTurn) return;
    transcript.push(assistantTurn);
    const toolCall = normalizeArray(assistantTurn.tool_calls)[0] || null;
    transcript.push({
      role: "tool",
      tool_call_id: asText(toolCall?.id) || `call_${index + 1}`,
      name:
        asText(toolCall?.function?.name) ||
        asText(step?.observation?.capability) ||
        "reviewed_tool",
      content: JSON.stringify(buildStepObservationPayload(step), null, 2),
    });
  });

  return transcript;
}

function looksLikeClarificationRequest(text = "") {
  const trimmed = asText(text);
  if (!trimmed) return false;
  if (trimmed.includes("?")) return true;
  return /^(what|which|who|where|when|why|how|please provide|can you|could you|do you want|soll ich|welche|welcher|welches|wer|wo|wann|warum|wie|bitte gib|bitte nenne|moechtest du|möchtest du|kannst du|brauchst du)/i.test(trimmed);
}

function buildNeedsInputQuestionsFromText(text = "") {
  const lines = asText(text)
    .split(/\n+/)
    .map((line) => asText(line))
    .filter(Boolean);
  const questions = [];
  for (const line of lines) {
    if (questions.length >= 6) break;
    questions.push({
      id: `question-${questions.length + 1}`,
      question: line,
      reason: "",
    });
  }
  if (!questions.length) {
    questions.push({
      id: "question-1",
      question: "Please provide the missing information.",
      reason: "",
    });
  }
  return questions;
}

function findFirstUrl(value) {
  if (typeof value === "string") {
    const text = asText(value);
    if (/^https?:\/\//i.test(text)) return text;
    return "";
  }
  if (Array.isArray(value)) {
    for (const entry of value) {
      const found = findFirstUrl(entry);
      if (found) return found;
    }
    return "";
  }
  if (value && typeof value === "object") {
    for (const entry of Object.values(value)) {
      const found = findFirstUrl(entry);
      if (found) return found;
    }
  }
  return "";
}

function collectUrlHosts(value, target = new Set()) {
  if (typeof value === "string") {
    const text = asText(value);
    if (/^https?:\/\//i.test(text)) {
      try {
        target.add(new URL(text).hostname.toLowerCase());
      } catch {}
    }
    return target;
  }
  if (Array.isArray(value)) {
    for (const entry of value) collectUrlHosts(entry, target);
    return target;
  }
  if (value && typeof value === "object") {
    for (const entry of Object.values(value)) collectUrlHosts(entry, target);
  }
  return target;
}

async function pTabsQuery(queryInfo) {
  return await new Promise((resolve, reject) => {
    chrome.tabs.query(queryInfo, (tabs) => {
      const error = chrome.runtime.lastError;
      if (error) reject(new Error(error.message));
      else resolve(Array.isArray(tabs) ? tabs : []);
    });
  });
}

async function pTabsUpdate(tabId, updateProps) {
  return await new Promise((resolve, reject) => {
    chrome.tabs.update(Number(tabId || 0), updateProps, (tab) => {
      const error = chrome.runtime.lastError;
      if (error) reject(new Error(error.message));
      else resolve(tab || null);
    });
  });
}

async function pExecuteScript(tabId, func, args = [], world = "MAIN") {
  return await new Promise((resolve, reject) => {
    chrome.scripting.executeScript({
      target: {
        tabId: Number(tabId || 0),
        allFrames: false,
      },
      world,
      func,
      args,
    }, (results) => {
      const error = chrome.runtime.lastError;
      if (error) reject(new Error(error.message));
      else resolve(Array.isArray(results) ? results : []);
    });
  });
}

async function readNewestArtifact(craftId, kind) {
  const craftSync = globalThis.SinepanelCraftSync;
  if (!craftSync?.listLocalArtifacts) return null;
  const list = await craftSync.listLocalArtifacts({
    craftId: asText(craftId),
    kind: asText(kind),
  });
  return Array.isArray(list) && list.length ? list[0] : null;
}

function shouldAllowEmbeddedCraftBundleRuntimeFallback(craft = null) {
  return craft?.sync?.readOnly === true || /^shared:/i.test(asText(craft?.id));
}

async function readCraftBundle(craft = null) {
  const craftId = asText(craft?.id);
  if (!craftId) {
    return craft?.bundle && typeof craft.bundle === "object"
      ? cloneJson(craft.bundle, null)
      : null;
  }
  const [
    trainingDataRecord,
    toolScriptsRecord,
    browserCapabilitiesRecord,
    weightsRecord,
    policyRecord,
  ] = await Promise.all([
    readNewestArtifact(craftId, TRAINING_DATA_ARTIFACT_KIND),
    readNewestArtifact(craftId, TOOL_SCRIPTS_ARTIFACT_KIND),
    readNewestArtifact(craftId, BROWSER_CAPABILITY_BUNDLE_ARTIFACT_KIND),
    readNewestArtifact(craftId, WEIGHTS_ARTIFACT_KIND),
    readNewestArtifact(craftId, POLICY_BUNDLE_ARTIFACT_KIND),
  ]);

  const hasLocalArtifacts = Boolean(
    trainingDataRecord || toolScriptsRecord || browserCapabilitiesRecord || weightsRecord || policyRecord,
  );
  const bundle = buildCapabilityBundle({
    craft,
    trainingDataRecord,
    toolScriptsRecord,
    browserCapabilitiesRecord,
    weightsRecord,
    policyRecord,
    preserveStoredBrowserCapabilities: true,
    toolScriptsOptions: STRICT_RUNTIME_TOOL_SCRIPTS_OPTIONS,
    browserCapabilityOptions: STRICT_RUNTIME_CAPABILITY_NORMALIZATION_OPTIONS,
  });
  if (hasLocalArtifacts) {
    return bundle;
  }
  return shouldAllowEmbeddedCraftBundleRuntimeFallback(craft) &&
    craft?.bundle && typeof craft.bundle === "object"
    ? cloneJson(craft.bundle, bundle)
    : bundle;
}

async function collectActiveTabContext() {
  const tabs = await pTabsQuery({
    active: true,
    currentWindow: true,
  });
  const tab = normalizeArray(tabs).find((entry) => /^https?:/i.test(asText(entry?.url))) || null;
  if (!tab) {
    return {
      tabId: 0,
      url: "",
      title: "",
      pageState: null,
    };
  }
  const pageState = await collectPageStateSafely(Number(tab.id || 0));
  return {
    tabId: Number(tab.id || 0),
    url: asText(tab.url),
    title: asText(tab.title),
    pageState,
  };
}

function isDirectActiveTextToolName(toolName = "") {
  const normalizedToolName = normalizeToolName(toolName);
  return normalizedToolName === "read_active_text_target" || normalizedToolName === "replace_active_text_target";
}

async function resolveDirectActiveTextTabContext(browserContext = null) {
  const preferredTabId = Number(browserContext?.tabId || 0);
  const preferredUrl = asText(browserContext?.url);
  if (preferredTabId > 0 && /^https?:/i.test(preferredUrl)) {
    return {
      tabId: preferredTabId,
      url: preferredUrl,
      title: asText(browserContext?.title),
      pageState: browserContext?.pageState || null,
    };
  }
  return await collectActiveTabContext();
}

async function executeDirectActiveTextCapability({
  capability = null,
  action = null,
  browserContext = null,
} = {}) {
  const toolName = normalizeToolName(capability?.toolName || capability?.id || capability?.name);
  const tabContext = await resolveDirectActiveTextTabContext(browserContext);
  const tabId = Number(tabContext?.tabId || 0);
  if (!tabId) {
    return {
      phase: "execute",
      pageState: null,
      finalUrl: "",
      ...normalizeCapabilityPhaseResult("execute", {
        ok: false,
        error: {
          code: "NO_ACTIVE_TEXT_TARGET",
          message: "Could not resolve an active HTTP(S) tab for the active-text runtime.",
        },
      }),
    };
  }

  await pTabsUpdate(tabId, { active: true }).catch(() => null);
  const inputArgs =
    action?.arguments && typeof action.arguments === "object"
      ? cloneJson(action.arguments, {})
      : {};
  const scriptResults = await pExecuteScript(
    tabId,
    async (currentToolName, rawInput) => {
      const asText = (value) => String(value == null ? "" : value);
      const clipboardTimeoutMs = 1_500;
      const activeTextFixtureSelector = '[data-sinepanel-active-text-fixture="true"]';
      const textInputTypes = new Set(["", "text", "search", "email", "url", "tel", "password", "number"]);
      const awaitWithTimeout = async (value, timeoutMs, fallbackValue) => {
        const waitMs = Math.max(50, Math.floor(Number(timeoutMs) || 0));
        if (!waitMs) return await value;
        let timeoutId = null;
        const timeoutPromise = new Promise((resolve) => {
          timeoutId = globalThis.setTimeout(() => resolve({ ok: false, value: fallbackValue }), waitMs);
        });
        try {
          const result = await Promise.race([
            Promise.resolve(value)
              .then((resolved) => ({ ok: true, value: resolved }))
              .catch(() => ({ ok: false, value: fallbackValue })),
            timeoutPromise,
          ]);
          return result && typeof result === "object" && Object.prototype.hasOwnProperty.call(result, "value")
            ? result.value
            : fallbackValue;
        } finally {
          if (timeoutId != null) globalThis.clearTimeout(timeoutId);
        }
      };
      const getDeepActiveElement = () => {
        let activeElement = document.activeElement || null;
        while (activeElement && activeElement.shadowRoot && activeElement.shadowRoot.activeElement) {
          activeElement = activeElement.shadowRoot.activeElement;
        }
        return activeElement || null;
      };
      const isVisibleElement = (element) => {
        if (!element || !(element instanceof Element)) return false;
        const rect = element.getBoundingClientRect();
        if (!(rect.width > 0 && rect.height > 0)) return false;
        const style = globalThis.getComputedStyle?.(element);
        return !!style && style.display !== "none" && style.visibility !== "hidden" && style.opacity !== "0";
      };
      const isTextControl = (element) => {
        if (!element) return false;
        const tagName = String(element.tagName || "").toLowerCase();
        if (tagName === "textarea") return true;
        if (tagName !== "input") return false;
        const type = String(element.type || "text").toLowerCase();
        return textInputTypes.has(type);
      };
      const isEditableElement = (element) => {
        if (!element) return false;
        if (isTextControl(element)) return true;
        return !!element.isContentEditable;
      };
      const resolveFallbackEditableElement = () => {
        const fixtureElement = document.querySelector(activeTextFixtureSelector);
        if (isEditableElement(fixtureElement) && isVisibleElement(fixtureElement)) {
          return fixtureElement;
        }
        const editableCandidates = Array.from(
          document.querySelectorAll("textarea,input,[contenteditable='true'],[contenteditable='plaintext-only']"),
        )
          .filter((element) => isEditableElement(element) && isVisibleElement(element))
          .filter((element) => !!asText("value" in element ? element.value : element.innerText || element.textContent))
          .slice(0, 8);
        return editableCandidates.length === 1 ? editableCandidates[0] : null;
      };
      const readTextControlSelection = (element) => {
        if (!isTextControl(element)) return "";
        const start = Number(element.selectionStart);
        const end = Number(element.selectionEnd);
        if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) return "";
        return asText(element.value).slice(start, end);
      };
      const readWindowSelectionText = () => {
        const selection = typeof window.getSelection === "function" ? window.getSelection() : null;
        if (!selection || selection.rangeCount < 1) return "";
        return asText(selection.toString());
      };
      const readFocusedEditableText = (element) => {
        if (!isEditableElement(element)) return "";
        if (isTextControl(element)) return asText(element.value);
        return asText(element.innerText || element.textContent);
      };
      const emitEditableEvents = (element) => {
        if (!element || typeof element.dispatchEvent !== "function") return;
        element.dispatchEvent(new Event("input", { bubbles: true }));
        element.dispatchEvent(new Event("change", { bubbles: true }));
      };
      const replaceTextControlSelection = (element, nextText) => {
        if (!isTextControl(element)) return false;
        const start = Number(element.selectionStart);
        const end = Number(element.selectionEnd);
        if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) return false;
        const value = asText(nextText);
        try {
          if (typeof element.focus === "function") element.focus({ preventScroll: true });
        } catch {}
        if (typeof element.setRangeText === "function") {
          element.setRangeText(value, start, end, "end");
        } else {
          const currentValue = asText(element.value);
          element.value = currentValue.slice(0, start) + value + currentValue.slice(end);
          if (typeof element.setSelectionRange === "function") {
            const cursor = start + value.length;
            element.setSelectionRange(cursor, cursor, "none");
          }
        }
        emitEditableEvents(element);
        return true;
      };
      const replaceFocusedEditableText = (element, nextText) => {
        if (!isEditableElement(element)) return false;
        const value = asText(nextText);
        try {
          if (typeof element.focus === "function") element.focus({ preventScroll: true });
        } catch {}
        if (isTextControl(element)) {
          element.value = value;
          if (typeof element.setSelectionRange === "function") {
            element.setSelectionRange(value.length, value.length, "none");
          }
        } else {
          element.textContent = value;
        }
        emitEditableEvents(element);
        return true;
      };
      const replaceWindowSelectionText = (nextText) => {
        const selection = typeof window.getSelection === "function" ? window.getSelection() : null;
        if (!selection || selection.rangeCount < 1) return false;
        const selectedText = asText(selection.toString());
        if (!selectedText) return false;
        const range = selection.getRangeAt(0);
        const value = asText(nextText);
        range.deleteContents();
        const node = document.createTextNode(value);
        range.insertNode(node);
        const nextRange = document.createRange();
        nextRange.setStartAfter(node);
        nextRange.collapse(true);
        selection.removeAllRanges();
        selection.addRange(nextRange);
        return true;
      };
      const readClipboardText = async () => {
        if (!globalThis.navigator?.clipboard?.readText) return null;
        try {
          return asText(
            await awaitWithTimeout(globalThis.navigator.clipboard.readText(), clipboardTimeoutMs, ""),
          );
        } catch {
          return null;
        }
      };
      const writeClipboardText = async (nextText) => {
        if (!globalThis.navigator?.clipboard?.writeText) return false;
        try {
          return !!(await awaitWithTimeout(
            globalThis.navigator.clipboard.writeText(asText(nextText)).then(() => true),
            clipboardTimeoutMs,
            false,
          ));
        } catch {
          return false;
        }
      };
      const buildOk = (targetType, text) => ({
        ok: true,
        data: {
          targetType,
          text: asText(text),
        },
      });
      const buildError = (code, message) => ({
        ok: false,
        error: {
          code: asText(code || "ACTIVE_TEXT_ERROR"),
          message: asText(message || "Active text operation failed."),
        },
      });

      const activeElement = getDeepActiveElement();
      const activeTarget = isEditableElement(activeElement) ? activeElement : resolveFallbackEditableElement();
      if (currentToolName === "read_active_text_target") {
        const focusedSelection = readTextControlSelection(activeTarget);
        if (focusedSelection) return buildOk("selection", focusedSelection);
        const windowSelection = readWindowSelectionText();
        if (windowSelection) return buildOk("selection", windowSelection);
        const focusedEditableText = readFocusedEditableText(activeTarget);
        if (focusedEditableText) return buildOk("focused_editable", focusedEditableText);
        const clipboardText = await readClipboardText();
        if (clipboardText) return buildOk("clipboard", clipboardText);
        return buildError(
          "NO_ACTIVE_TEXT_TARGET",
          "Could not resolve selected text, focused editable text, or clipboard text.",
        );
      }

      const nextText =
        typeof rawInput === "string"
          ? rawInput
          : rawInput && typeof rawInput === "object" && Object.prototype.hasOwnProperty.call(rawInput, "text")
            ? rawInput.text
            : "";
      if (!asText(nextText)) {
        return buildError("MISSING_TEXT", "replace_active_text_target requires a replacement text.");
      }
      if (replaceTextControlSelection(activeTarget, nextText)) return buildOk("selection", nextText);
      if (replaceWindowSelectionText(nextText)) return buildOk("selection", nextText);
      if (replaceFocusedEditableText(activeTarget, nextText)) return buildOk("focused_editable", nextText);
      if (await writeClipboardText(nextText)) return buildOk("clipboard", nextText);
      return buildError(
        "NO_ACTIVE_TEXT_TARGET",
        "Could not replace selected text, focused editable text, or clipboard text.",
      );
    },
    [toolName, inputArgs],
  );
  const rawResult = Array.isArray(scriptResults) ? scriptResults[0]?.result : null;
  const normalizedResult = normalizeActiveTextCapabilityResult(rawResult || {
    ok: false,
    error: {
      code: "ACTIVE_TEXT_EXECUTION_FAILED",
      message: "Active-text execution returned no result.",
    },
  });
  const pageState = await collectPageStateSafely(tabId);
  return {
    phase: "execute",
    pageState,
    finalUrl: asText(tabContext?.url),
    ...normalizeCapabilityPhaseResult("execute", normalizedResult),
  };
}

function buildRuntimeAllowedHosts(browserContext, action, capability) {
  const hosts = new Set();
  collectUrlHosts(browserContext?.url, hosts);
  collectUrlHosts(action?.arguments, hosts);
  collectUrlHosts(capability?.readsFrom, hosts);
  collectUrlHosts(capability?.writesTo, hosts);
  collectUrlHosts(capability?.resourceRefs, hosts);
  return [...hosts].filter(Boolean);
}

function buildRuntimeTargetUrl(browserContext, action, capability) {
  return (
    findFirstUrl(action?.arguments) ||
    // Reviewed domain workflows often declare their landing page via capability resource refs.
    findFirstUrl(capability?.resourceRefs) ||
    findFirstUrl(capability?.readsFrom) ||
    findFirstUrl(capability?.writesTo) ||
    asText(browserContext?.url)
  );
}

function buildCapabilityRuntimeToolsPrelude() {
  return [
    "const __getDeepActiveElement = () => {",
    "  let activeElement = document.activeElement || null;",
    "  while (activeElement && activeElement.shadowRoot && activeElement.shadowRoot.activeElement) {",
    "    activeElement = activeElement.shadowRoot.activeElement;",
    "  }",
    "  return activeElement || null;",
    "};",
    "const __textInputTypes = new Set(['', 'text', 'search', 'email', 'url', 'tel', 'password', 'number']);",
    "const __isTextControl = (element) => {",
    "  if (!element) return false;",
    "  const tagName = String(element.tagName || '').toLowerCase();",
    "  if (tagName === 'textarea') return true;",
    "  if (tagName !== 'input') return false;",
    "  const type = String(element.type || 'text').toLowerCase();",
    "  return __textInputTypes.has(type);",
    "};",
    "const __isEditableElement = (element) => {",
    "  if (!element) return false;",
    "  if (__isTextControl(element)) return true;",
    "  return !!element.isContentEditable;",
    "};",
    "const __asText = (value) => String(value == null ? '' : value);",
    "const __awaitWithTimeout = async (value, timeoutMs, fallbackValue) => {",
    "  const waitMs = Math.max(50, Math.floor(Number(timeoutMs) || 0));",
    "  if (!waitMs) return await value;",
    "  let timeoutId = null;",
    "  const timeoutPromise = new Promise((resolve) => {",
    "    timeoutId = globalThis.setTimeout(() => resolve({ ok: false, value: fallbackValue }), waitMs);",
    "  });",
    "  try {",
    "    const result = await Promise.race([",
    "      Promise.resolve(value).then((resolved) => ({ ok: true, value: resolved })).catch(() => ({ ok: false, value: fallbackValue })),",
    "      timeoutPromise,",
    "    ]);",
    "    return result && typeof result === 'object' && Object.prototype.hasOwnProperty.call(result, 'value')",
    "      ? result.value",
    "      : fallbackValue;",
    "  } finally {",
    "    if (timeoutId != null) globalThis.clearTimeout(timeoutId);",
    "  }",
    "};",
    "const __clipboardTimeoutMs = 1_500;",
    "const __emitEditableEvents = (element) => {",
    "  if (!element || typeof element.dispatchEvent !== 'function') return;",
    "  element.dispatchEvent(new Event('input', { bubbles: true }));",
    "  element.dispatchEvent(new Event('change', { bubbles: true }));",
    "};",
    "const __readTextControlSelection = (element) => {",
    "  if (!__isTextControl(element)) return '';",
    "  const start = Number(element.selectionStart);",
    "  const end = Number(element.selectionEnd);",
    "  if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) return '';",
    "  return __asText(element.value).slice(start, end);",
    "};",
    "const __readWindowSelectionText = () => {",
    "  const selection = typeof window.getSelection === 'function' ? window.getSelection() : null;",
    "  if (!selection || selection.rangeCount < 1) return '';",
    "  return __asText(selection.toString());",
    "};",
    "const __readFocusedEditableText = (element) => {",
    "  if (!__isEditableElement(element)) return '';",
    "  if (__isTextControl(element)) return __asText(element.value);",
    "  return __asText(element.innerText || element.textContent);",
    "};",
    "const __replaceTextControlSelection = (element, nextText) => {",
    "  if (!__isTextControl(element)) return false;",
    "  const start = Number(element.selectionStart);",
    "  const end = Number(element.selectionEnd);",
    "  if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) return false;",
    "  const value = __asText(nextText);",
    "  try {",
    "    if (typeof element.focus === 'function') element.focus({ preventScroll: true });",
    "  } catch {}",
    "  if (typeof element.setRangeText === 'function') {",
    "    element.setRangeText(value, start, end, 'end');",
    "  } else {",
    "    const currentValue = __asText(element.value);",
    "    element.value = currentValue.slice(0, start) + value + currentValue.slice(end);",
    "    if (typeof element.setSelectionRange === 'function') {",
    "      const cursor = start + value.length;",
    "      element.setSelectionRange(cursor, cursor, 'none');",
    "    }",
    "  }",
    "  __emitEditableEvents(element);",
    "  return true;",
    "};",
    "const __replaceFocusedEditableText = (element, nextText) => {",
    "  if (!__isEditableElement(element)) return false;",
    "  const value = __asText(nextText);",
    "  try {",
    "    if (typeof element.focus === 'function') element.focus({ preventScroll: true });",
    "  } catch {}",
    "  if (__isTextControl(element)) {",
    "    element.value = value;",
    "    if (typeof element.setSelectionRange === 'function') {",
    "      element.setSelectionRange(value.length, value.length, 'none');",
    "    }",
    "  } else {",
    "    element.textContent = value;",
    "  }",
    "  __emitEditableEvents(element);",
    "  return true;",
    "};",
    "const __replaceWindowSelectionText = (nextText) => {",
    "  const selection = typeof window.getSelection === 'function' ? window.getSelection() : null;",
    "  if (!selection || selection.rangeCount < 1) return false;",
    "  const selectedText = __asText(selection.toString());",
    "  if (!selectedText) return false;",
    "  const range = selection.getRangeAt(0);",
    "  const value = __asText(nextText);",
    "  range.deleteContents();",
    "  const node = document.createTextNode(value);",
    "  range.insertNode(node);",
    "  const nextRange = document.createRange();",
    "  nextRange.setStartAfter(node);",
    "  nextRange.collapse(true);",
    "  selection.removeAllRanges();",
    "  selection.addRange(nextRange);",
    "  const editableHost = node.parentElement?.closest?.('[contenteditable=\"true\"],[contenteditable=\"plaintext-only\"],textarea,input');",
    "  if (editableHost) __emitEditableEvents(editableHost);",
    "  return true;",
    "};",
    "const __readClipboardText = async () => {",
    "  if (!globalThis.navigator?.clipboard?.readText) return null;",
    "  try {",
    "    return __asText(await __awaitWithTimeout(globalThis.navigator.clipboard.readText(), __clipboardTimeoutMs, ''));",
    "  } catch {",
    "    return null;",
    "  }",
    "};",
    "const __writeClipboardText = async (nextText) => {",
    "  if (!globalThis.navigator?.clipboard?.writeText) return false;",
    "  try {",
    "    return !!(await __awaitWithTimeout(",
    "      globalThis.navigator.clipboard.writeText(__asText(nextText)).then(() => true),",
    "      __clipboardTimeoutMs,",
    "      false,",
    "    ));",
    "  } catch {",
    "    return false;",
    "  }",
    "};",
    "const __buildActiveTextOk = (targetType, text) => ({",
    "  ok: true,",
    "  data: {",
    "    targetType: __asText(targetType),",
    "    text: __asText(text),",
    "  },",
    "});",
    "const __buildActiveTextError = (code, message) => ({",
    "  ok: false,",
    "  error: {",
    "    code: __asText(code || 'ACTIVE_TEXT_ERROR'),",
    "    message: __asText(message || 'Active text operation failed.'),",
    "  },",
    "});",
    "const __readActiveTextTarget = async () => {",
    "  const activeElement = __getDeepActiveElement();",
    "  const focusedSelection = __readTextControlSelection(activeElement);",
    "  if (focusedSelection) return __buildActiveTextOk('selection', focusedSelection);",
    "  const windowSelection = __readWindowSelectionText();",
    "  if (windowSelection) return __buildActiveTextOk('selection', windowSelection);",
    "  const focusedEditableText = __readFocusedEditableText(activeElement);",
    "  if (focusedEditableText) return __buildActiveTextOk('focused_editable', focusedEditableText);",
    "  const clipboardText = await __readClipboardText();",
    "  if (clipboardText) return __buildActiveTextOk('clipboard', clipboardText);",
    "  return __buildActiveTextError('NO_ACTIVE_TEXT_TARGET', 'Could not resolve selected text, focused editable text, or clipboard text.');",
    "};",
    "const __replaceActiveTextTarget = async (input = {}) => {",
    "  const argsHasText = !!args && typeof args === 'object' && Object.prototype.hasOwnProperty.call(args, 'text');",
    "  const inputHasText = typeof input === 'string' || (!!input && typeof input === 'object' && Object.prototype.hasOwnProperty.call(input, 'text'));",
    "  if (!argsHasText && !inputHasText) {",
    "    return __buildActiveTextError('MISSING_TEXT', 'replace_active_text_target requires a replacement text.');",
    "  }",
    "  const nextText = typeof input === 'string'",
    "    ? input",
    "    : inputHasText",
    "      ? input.text",
    "      : args.text;",
    "  const activeElement = __getDeepActiveElement();",
    "  if (__replaceTextControlSelection(activeElement, nextText)) return __buildActiveTextOk('selection', nextText);",
    "  if (__replaceWindowSelectionText(nextText)) return __buildActiveTextOk('selection', nextText);",
    "  if (__replaceFocusedEditableText(activeElement, nextText)) return __buildActiveTextOk('focused_editable', nextText);",
    "  if (await __writeClipboardText(nextText)) return __buildActiveTextOk('clipboard', nextText);",
    "  return __buildActiveTextError('NO_ACTIVE_TEXT_TARGET', 'Could not replace selected text, focused editable text, or clipboard text.');",
    "};",
    "const __clipToolText = (value, max = 500) => {",
    "  const limit = Math.max(0, Math.floor(Number(max) || 0));",
    "  const text = __asText(value).replace(/\\s+/g, ' ').trim();",
    "  if (!limit || text.length <= limit) return text;",
    "  return `${text.slice(0, Math.max(0, limit - 3)).trim()}...`;",
    "};",
    "const __normalizeStringList = (value) => {",
    "  if (Array.isArray(value)) return value.map((entry) => __asText(entry)).filter(Boolean);",
    "  const text = __asText(value);",
    "  return text ? text.split(/[\\n,;]+/).map((entry) => __asText(entry)).filter(Boolean) : [];",
    "};",
    "const __resolveTextInput = (input = {}, fallbackArgs = args) => {",
    "  if (typeof input === 'string') return { ok: true, text: input };",
    "  if (input && typeof input === 'object' && Object.prototype.hasOwnProperty.call(input, 'text')) {",
    "    return { ok: true, text: input.text };",
    "  }",
    "  if (fallbackArgs && typeof fallbackArgs === 'object' && Object.prototype.hasOwnProperty.call(fallbackArgs, 'text')) {",
    "    return { ok: true, text: fallbackArgs.text };",
    "  }",
    "  return { ok: false, text: '' };",
    "};",
    "const __buildToolOk = (data = {}) => ({",
    "  ok: true,",
    "  data: data && typeof data === 'object' ? data : {},",
    "});",
    "const __buildToolError = (code, message) => ({",
    "  ok: false,",
    "  error: {",
    "    code: __asText(code || 'TOOL_ERROR'),",
    "    message: __asText(message || 'Tool execution failed.'),",
    "  },",
    "});",
    "const __readClipboardTextTool = async () => {",
    "  const text = await __readClipboardText();",
    "  if (text == null) return __buildToolError('CLIPBOARD_READ_FAILED', 'Could not read clipboard text.');",
    "  return __buildToolOk({ text: __asText(text) });",
    "};",
    "const __writeClipboardTextTool = async (input = {}) => {",
    "  const resolved = __resolveTextInput(input, args);",
    "  if (!resolved.ok) return __buildToolError('MISSING_TEXT', 'write_clipboard_text requires a text field.');",
    "  const nextText = __asText(resolved.text);",
    "  if (await __writeClipboardText(nextText)) return __buildToolOk({ text: nextText });",
    "  return __buildToolError('CLIPBOARD_WRITE_FAILED', 'Could not write clipboard text.');",
    "};",
    "const __buildMailtoUrl = ({ recipient = '', subject = '', body = '', cc = [], bcc = [], replyTo = '' } = {}) => {",
    "  const query = new URLSearchParams();",
    "  if (__asText(subject)) query.set('subject', __asText(subject));",
    "  if (__asText(body)) query.set('body', __asText(body));",
    "  if (Array.isArray(cc) && cc.length) query.set('cc', cc.map((entry) => __asText(entry)).filter(Boolean).join(','));",
    "  if (Array.isArray(bcc) && bcc.length) query.set('bcc', bcc.map((entry) => __asText(entry)).filter(Boolean).join(','));",
    "  if (__asText(replyTo)) query.set('replyTo', __asText(replyTo));",
    "  const queryText = query.toString();",
    "  return `mailto:${encodeURIComponent(__asText(recipient))}${queryText ? `?${queryText}` : ''}`;",
    "};",
    "const __buildEmailClipboardText = ({ recipient = '', subject = '', body = '', cc = [], bcc = [], replyTo = '' } = {}) => {",
    "  return [",
    "    __asText(recipient) ? `To: ${__asText(recipient)}` : '',",
    "    Array.isArray(cc) && cc.length ? `Cc: ${cc.join(', ')}` : '',",
    "    Array.isArray(bcc) && bcc.length ? `Bcc: ${bcc.join(', ')}` : '',",
    "    __asText(replyTo) ? `Reply-To: ${__asText(replyTo)}` : '',",
    "    __asText(subject) ? `Subject: ${__asText(subject)}` : '',",
    "    __asText(body),",
    "  ].filter(Boolean).join('\\n\\n');",
    "};",
    "const __composeEmailDraft = async (input = {}) => {",
    "  const params = input && typeof input === 'object' ? input : {};",
    "  const recipient = __asText(params.recipient || params.to);",
    "  const subject = __asText(params.subject);",
    "  const body = __asText(params.body);",
    "  if (!subject || !body) return __buildToolError('MISSING_EMAIL_FIELDS', 'compose_email requires subject and body.');",
    "  const cc = __normalizeStringList(params.cc);",
    "  const bcc = __normalizeStringList(params.bcc);",
    "  const replyTo = __asText(params.replyTo || params.reply_to);",
    "  return __buildToolOk({",
    "    recipient,",
    "    subject,",
    "    body,",
    "    cc,",
    "    bcc,",
    "    replyTo,",
    "    mailtoUrl: __buildMailtoUrl({ recipient, subject, body, cc, bcc, replyTo }),",
    "    clipboardText: __buildEmailClipboardText({ recipient, subject, body, cc, bcc, replyTo }),",
    "  });",
    "};",
    buildReviewedWebSearchRuntimePrelude(),
    "const __isVisibleElement = (element) => {",
    "  if (!element || typeof element.getBoundingClientRect !== 'function') return false;",
    "  const rect = element.getBoundingClientRect();",
    "  if (!rect || rect.width < 1 || rect.height < 1) return false;",
    "  const style = typeof globalThis.getComputedStyle === 'function' ? globalThis.getComputedStyle(element) : null;",
    "  if (style && (style.visibility === 'hidden' || style.display === 'none' || Number(style.opacity) === 0)) return false;",
    "  return true;",
    "};",
    "const __collectVisibleTextList = (selector, maxItems = 6, maxLen = 180) => {",
    "  try {",
    "    return Array.from(document.querySelectorAll(selector))",
    "      .filter((element) => __isVisibleElement(element))",
    "      .map((element) => __clipToolText(element.innerText || element.textContent || '', maxLen))",
    "      .filter(Boolean)",
    "      .slice(0, Math.max(1, Math.floor(Number(maxItems) || 0)));",
    "  } catch {",
    "    return [];",
    "  }",
    "};",
    "const __describeFocusedElement = () => {",
    "  const element = __getDeepActiveElement();",
    "  if (!element) return null;",
    "  return {",
    "    tagName: __asText(element.tagName).toLowerCase(),",
    "    inputType: __asText(element.type),",
    "    placeholder: __asText(element.placeholder),",
    "    name: __asText(element.name),",
    "    id: __asText(element.id),",
    "    ariaLabel: __asText(typeof element.getAttribute === 'function' ? element.getAttribute('aria-label') : ''),",
    "    valuePreview: __clipToolText(__readFocusedEditableText(element), 280),",
    "  };",
    "};",
    "const __captureBugReportContext = async (input = {}) => {",
    "  const params = input && typeof input === 'object' ? input : {};",
    "  const maxBodyChars = Math.max(200, Math.min(4_000, Math.floor(Number(params.maxBodyChars) || 1_400)));",
    "  const url = __asText(globalThis.location?.href || '');",
    "  const title = __asText(document?.title || url);",
    "  const summary = __clipToolText(params.summary || title || url, 160);",
    "  const selectionText = __clipToolText(__readWindowSelectionText(), 400);",
    "  const activeElement = __describeFocusedElement();",
    "  const headings = __collectVisibleTextList('h1, h2, h3', 6, 140);",
    "  const dialogs = __collectVisibleTextList('[role=\"dialog\"], dialog, [aria-modal=\"true\"]', 4, 220);",
    "  const visibleErrorTexts = __collectVisibleTextList('[role=\"alert\"], .error, .errors, .alert, [aria-invalid=\"true\"], [data-testid*=\"error\"]', 6, 220);",
    "  const bodyExcerpt = __clipToolText(document?.body?.innerText || document?.body?.textContent || '', maxBodyChars);",
    "  const stepsToReproduce = __normalizeStringList(params.stepsToReproduce).slice(0, 8);",
    "  const expectedBehavior = __clipToolText(params.expectedBehavior, 400);",
    "  const actualBehavior = __clipToolText(params.actualBehavior, 400);",
    "  const activeElementLine = activeElement",
    "    ? `Focused element: ${[activeElement.tagName, activeElement.inputType || '', activeElement.placeholder || activeElement.name || activeElement.id || '']",
    "        .filter(Boolean)",
    "        .join(' | ')}`",
    "    : '';",
    "  const emailSubject = __clipToolText(`Bug report: ${summary || title || url || 'Current page'}`, 160);",
    "  const emailBody = [",
    "    summary ? `Summary: ${summary}` : '',",
    "    url ? `URL: ${url}` : '',",
    "    title ? `Title: ${title}` : '',",
    "    selectionText ? `Selected text: ${selectionText}` : '',",
    "    activeElementLine,",
    "    stepsToReproduce.length ? `Steps to reproduce:\\n${stepsToReproduce.map((step, index) => `${index + 1}. ${step}`).join('\\n')}` : '',",
    "    expectedBehavior ? `Expected behavior: ${expectedBehavior}` : '',",
    "    actualBehavior ? `Actual behavior: ${actualBehavior}` : '',",
    "    visibleErrorTexts.length ? `Visible errors:\\n- ${visibleErrorTexts.join('\\n- ')}` : '',",
    "    dialogs.length ? `Open dialogs:\\n- ${dialogs.join('\\n- ')}` : '',",
    "    headings.length ? `Visible headings:\\n- ${headings.join('\\n- ')}` : '',",
    "    bodyExcerpt ? `Page excerpt:\\n${bodyExcerpt}` : '',",
    "  ].filter(Boolean).join('\\n\\n');",
    "  return __buildToolOk({",
    "    url,",
    "    title,",
    "    summary,",
    "    selectionText,",
    "    bodyExcerpt,",
    "    headings,",
    "    dialogs,",
    "    visibleErrorTexts,",
    "    stepsToReproduce,",
    "    expectedBehavior,",
    "    actualBehavior,",
    "    emailSubject,",
    "    emailBody,",
    "    ...(activeElement ? { activeElement } : {}),",
    "  });",
    "};",
    "const __reviewedToolRegistry = Object.create(null);",
    "const __builtinReviewedToolRegistry = Object.create(null);",
    "const __registerReviewedTool = (toolName, handler) => {",
    "  const key = __asText(toolName);",
    "  if (!key || typeof handler !== 'function') return handler;",
    "  __reviewedToolRegistry[key] = handler;",
    "  try {",
    "    globalThis[key] = handler;",
    "  } catch {}",
    "  return handler;",
    "};",
    "const __registerBuiltinReviewedTool = (toolName, handler) => {",
    "  const key = __asText(toolName);",
    "  if (!key || typeof handler !== 'function') return handler;",
    "  __builtinReviewedToolRegistry[key] = handler;",
    "  return __registerReviewedTool(key, handler);",
    "};",
    "const __resolveBuiltinReviewedTool = (toolName) => {",
    "  const key = __asText(toolName);",
    "  if (!key) return null;",
    "  return typeof __builtinReviewedToolRegistry[key] === 'function' ? __builtinReviewedToolRegistry[key] : null;",
    "};",
    "const __resolveReviewedTool = (toolName) => {",
    "  const key = __asText(toolName);",
    "  if (!key) return null;",
    "  if (typeof __reviewedToolRegistry[key] === 'function') return __reviewedToolRegistry[key];",
    "  if (typeof globalThis[key] === 'function') return globalThis[key];",
    "  return __resolveBuiltinReviewedTool(key);",
    "};",
    "const __normalizeToolInputPayload = (input = args) => {",
    "  if (typeof input === 'string') return { text: input };",
    "  if (input && typeof input === 'object' && !Array.isArray(input)) return input;",
    "  return {};",
    "};",
    "const __respondToReviewedTool = (value = null) => value;",
    "const __buildToolRuntime = (input = args) => {",
    "  const inputPayload = __normalizeToolInputPayload(input);",
    "  return Object.freeze({",
    "    ...inputPayload,",
    "    input: inputPayload,",
    "    args: inputPayload,",
    "    ctx,",
    "    page,",
    "    WAIT,",
    "    tools,",
    "    activeText: activeTextTools,",
    "    activeTextTools,",
    "    clipboard: clipboardTools,",
    "    clipboardTools,",
    "    prompt,",
    "    browserContext,",
    "    state,",
    "    observation,",
    "    modelOutput,",
    "    craft,",
    "    capability,",
    "    action,",
    "    webSearch: __webSearchTool,",
    "    web_search: __webSearchTool,",
    "    callReviewedTool: (toolName, nextInput = inputPayload) => __callReviewedTool(toolName, nextInput),",
    "    callBuiltin: (toolName, nextInput = inputPayload) => __callBuiltinReviewedTool(toolName, nextInput),",
    "    respond: (value = null) => __respondToReviewedTool(value),",
    "    ok: (data = {}) => __buildToolOk(data),",
    "    error: (code, message) => __buildToolError(code, message),",
    "  });",
    "};",
    "const __buildToolCallContext = (input = args, runtime = null) => {",
    "  const inputPayload = __normalizeToolInputPayload(input);",
    "  return Object.freeze({",
    "    ...ctx,",
    "    ...inputPayload,",
    "    input: inputPayload,",
    "    args: inputPayload,",
    "    runtime,",
    "    webSearch: __webSearchTool,",
    "    web_search: __webSearchTool,",
    "    callBuiltin: (toolName, nextInput = inputPayload) => __callBuiltinReviewedTool(toolName, nextInput),",
    "    respond: (value = null) => __respondToReviewedTool(value),",
    "    ok: (data = {}) => __buildToolOk(data),",
    "    error: (code, message) => __buildToolError(code, message),",
    "  });",
    "};",
    "const __invokeToolHandler = async (handler, input = args) => {",
    "  const runtime = __buildToolRuntime(input);",
    "  const callCtx = __buildToolCallContext(input, runtime);",
    "  return await handler(runtime, callCtx, runtime.args);",
    "};",
    "const __callBuiltinReviewedTool = async (toolName, input = args) => {",
    "  const handler = __resolveBuiltinReviewedTool(toolName);",
    "  if (typeof handler !== 'function') {",
    "    throw new ReferenceError(`${__asText(toolName) || 'builtin_reviewed_tool'} is not defined`);",
    "  }",
    "  return await __invokeToolHandler(handler, input);",
    "};",
    "const __callReviewedTool = async (toolName, input = args) => {",
    "  const handler = __resolveReviewedTool(toolName);",
    "  if (typeof handler !== 'function') {",
    "    throw new ReferenceError(`${__asText(toolName) || 'reviewed_tool'} is not defined`);",
    "  }",
    "  return await __invokeToolHandler(handler, input);",
    "};",
    "const read_active_text_target = __registerBuiltinReviewedTool('read_active_text_target', __readActiveTextTarget);",
    "const replace_active_text_target = __registerBuiltinReviewedTool('replace_active_text_target', __replaceActiveTextTarget);",
    "const readActiveTextTarget = __registerBuiltinReviewedTool('readActiveTextTarget', __readActiveTextTarget);",
    "const replaceActiveTextTarget = __registerBuiltinReviewedTool('replaceActiveTextTarget', __replaceActiveTextTarget);",
    "const read_clipboard_text = __registerBuiltinReviewedTool('read_clipboard_text', __readClipboardTextTool);",
    "const write_clipboard_text = __registerBuiltinReviewedTool('write_clipboard_text', __writeClipboardTextTool);",
    "const readClipboardText = __registerBuiltinReviewedTool('readClipboardText', __readClipboardTextTool);",
    "const writeClipboardText = __registerBuiltinReviewedTool('writeClipboardText', __writeClipboardTextTool);",
    "const compose_email = __registerBuiltinReviewedTool('compose_email', __composeEmailDraft);",
    "const composeEmail = __registerBuiltinReviewedTool('composeEmail', __composeEmailDraft);",
    "const web_search = __registerBuiltinReviewedTool('web_search', __webSearchTool);",
    "const webSearch = __registerBuiltinReviewedTool('webSearch', __webSearchTool);",
    "const capture_bug_report_context = __registerBuiltinReviewedTool('capture_bug_report_context', __captureBugReportContext);",
    "const captureBugReportContext = __registerBuiltinReviewedTool('captureBugReportContext', __captureBugReportContext);",
    "const activeTextTools = Object.freeze({",
    "  read: read_active_text_target,",
    "  replace: replace_active_text_target,",
    "});",
    "const clipboardTools = Object.freeze({",
    "  read: read_clipboard_text,",
    "  write: write_clipboard_text,",
    "});",
    "const tools = Object.freeze({",
    "  page,",
    "  WAIT,",
    "  activeText: activeTextTools,",
    "  clipboard: clipboardTools,",
    "  readActiveTextTarget: __readActiveTextTarget,",
    "  replaceActiveTextTarget: __replaceActiveTextTarget,",
    "  readClipboardText: __readClipboardTextTool,",
    "  writeClipboardText: __writeClipboardTextTool,",
    "  composeEmail: __composeEmailDraft,",
    "  webSearch: __webSearchTool,",
    "  captureBugReportContext: __captureBugReportContext,",
    "  read_active_text_target: __readActiveTextTarget,",
    "  replace_active_text_target: __replaceActiveTextTarget,",
    "  read_clipboard_text: __readClipboardTextTool,",
    "  write_clipboard_text: __writeClipboardTextTool,",
    "  compose_email: __composeEmailDraft,",
    "  web_search: __webSearchTool,",
    "  capture_bug_report_context: __captureBugReportContext,",
    "  callReviewedTool: __callReviewedTool,",
    "  callBuiltin: __callBuiltinReviewedTool,",
    "});",
    "const ctx = Object.freeze({",
    "  page,",
    "  WAIT,",
    "  tools,",
    "  activeText: activeTextTools,",
    "  activeTextTools,",
    "  clipboard: clipboardTools,",
    "  clipboardTools,",
    "  readActiveTextTarget: __readActiveTextTarget,",
    "  replaceActiveTextTarget: __replaceActiveTextTarget,",
    "  readClipboardText: __readClipboardTextTool,",
    "  writeClipboardText: __writeClipboardTextTool,",
    "  composeEmail: __composeEmailDraft,",
    "  webSearch: __webSearchTool,",
    "  captureBugReportContext: __captureBugReportContext,",
    "  read_active_text_target: __readActiveTextTarget,",
    "  replace_active_text_target: __replaceActiveTextTarget,",
    "  read_clipboard_text: __readClipboardTextTool,",
    "  write_clipboard_text: __writeClipboardTextTool,",
    "  compose_email: __composeEmailDraft,",
    "  web_search: __webSearchTool,",
    "  capture_bug_report_context: __captureBugReportContext,",
    "  callReviewedTool: __callReviewedTool,",
    "  callBuiltin: __callBuiltinReviewedTool,",
    "  args,",
    "  prompt,",
    "  browserContext,",
    "  state,",
    "  observation,",
    "  modelOutput,",
    "  craft,",
    "  capability,",
    "  action,",
    "});",
    "try { globalThis.ctx = ctx; } catch {}",
    "try { globalThis.tools = tools; } catch {}",
    "try { globalThis.activeTextTools = activeTextTools; } catch {}",
  ].join("\n");
}

function buildCapabilityPhaseScript({
  phase = "execute",
  source = "",
  bundle = null,
  craft = null,
  capability = null,
  action = null,
  prompt = "",
  browserContext = null,
  state = null,
  observation = null,
  modelOutput = null,
} = {}) {
  const runtimePayload = {
    phase: asText(phase),
    craft:
      craft && typeof craft === "object"
        ? {
            id: asText(craft.id),
            name: asText(craft.name),
            summary: asText(craft.summary),
            inputMode: asText(craft.inputMode),
          }
        : null,
    capability: cloneJson(capability, null),
    action: cloneJson(action, null),
    args:
      action?.arguments && typeof action.arguments === "object"
        ? cloneJson(action.arguments, {})
        : {},
    prompt: asText(prompt),
    browserContext: cloneJson(browserContext, null),
    state: cloneJson(state, {}),
    observation: cloneJson(observation, null),
    modelOutput: cloneJson(modelOutput, null),
  };
  const reviewedToolScriptsPrelude = buildReviewedToolScriptsPrelude(bundle?.toolScripts?.payload || null);
  return [
    `const __runtime = ${JSON.stringify(runtimePayload)};`,
    "const phase = __runtime.phase;",
    "const craft = __runtime.craft;",
    "const capability = __runtime.capability;",
    "const action = __runtime.action;",
    "const args = __runtime.args || {};",
    "const prompt = __runtime.prompt || '';",
    "const browserContext = __runtime.browserContext || null;",
    "const state = __runtime.state || {};",
    "const observation = __runtime.observation || null;",
    "const modelOutput = __runtime.modelOutput || null;",
    buildCapabilityRuntimeToolsPrelude(),
    // Reviewed capability scripts may depend on shared bundle helpers such as activeTextTools.
    reviewedToolScriptsPrelude,
    // Keep the deterministic active-text reviewed tools authoritative even when
    // stored tool scripts contain older or malformed handlers with the same ids.
    "__registerReviewedTool('read_active_text_target', __readActiveTextTarget);",
    "__registerReviewedTool('replace_active_text_target', __replaceActiveTextTarget);",
    "__registerReviewedTool('readActiveTextTarget', __readActiveTextTarget);",
    "__registerReviewedTool('replaceActiveTextTarget', __replaceActiveTextTarget);",
    String(source || "").trim(),
  ].filter((part) => asText(part)).join("\n\n");
}

function normalizeCapabilityPhaseResult(phase, rawResult) {
  const value = rawResult && typeof rawResult === "object" ? rawResult : rawResult ?? null;
  const source = value && typeof value === "object" ? value : null;
  const nextState =
    source?.state && typeof source.state === "object"
      ? cloneJson(source.state, {})
      : {};
  const output = Object.prototype.hasOwnProperty.call(source || {}, "output")
    ? cloneJson(source.output, source.output)
    : Object.prototype.hasOwnProperty.call(source || {}, "result")
      ? cloneJson(source.result, source.result)
      : cloneJson(value, value);
  return {
    phase: asText(phase),
    raw: cloneJson(value, value),
    state: nextState,
    output,
    summary: trimText(
      source?.summary ||
        source?.message ||
        `${phase} completed`,
      220,
    ),
  };
}

const CAPABILITY_AUXILIARY_TOP_LEVEL_FIELDS = new Set([
  "state",
  "summary",
  "browserPlan",
  "browser_plan",
]);

function stripCapabilityAuxiliaryFieldsForValidation(capability = null, value = null) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return cloneJson(value, value);
  }
  const schemaProperties =
    capability?.returnSchema?.properties &&
    typeof capability.returnSchema.properties === "object" &&
    !Array.isArray(capability.returnSchema.properties)
      ? capability.returnSchema.properties
      : {};
  const allowedTopLevelKeys = new Set(
    Object.keys(schemaProperties)
      .map((entry) => asText(entry))
      .filter(Boolean),
  );
  const looksLikeReviewedToolEnvelope =
    typeof value.ok === "boolean" ||
    Object.prototype.hasOwnProperty.call(value, "data") ||
    Object.prototype.hasOwnProperty.call(value, "error");
  if (!looksLikeReviewedToolEnvelope) {
    return cloneJson(value, value);
  }

  const normalizedValue = cloneJson(value, value);
  let changed = false;
  for (const key of CAPABILITY_AUXILIARY_TOP_LEVEL_FIELDS) {
    if (allowedTopLevelKeys.has(key)) continue;
    if (!Object.prototype.hasOwnProperty.call(normalizedValue || {}, key)) continue;
    delete normalizedValue[key];
    changed = true;
  }
  return changed ? normalizedValue : cloneJson(value, value);
}

function normalizeCapabilityOutputForRuntime(capability = null, value = null) {
  const normalizedValue = stripCapabilityAuxiliaryFieldsForValidation(capability, value);
  const toolName = normalizeToolName(capability?.toolName || capability?.id || capability?.name);
  if (
    toolName === "read_active_text_target" ||
    toolName === "replace_active_text_target"
  ) {
    return normalizeActiveTextCapabilityResult(normalizedValue);
  }
  return normalizedValue;
}

function normalizeCapabilityExecutionResult(phase, rawResult, capability = null) {
  return normalizeCapabilityPhaseResult(
    phase,
    normalizeCapabilityOutputForRuntime(capability, rawResult),
  );
}

function formatCapabilitySchemaPath(path = "$") {
  const text = asText(path) || "$";
  return text.startsWith("$") ? text : `$.${text}`;
}

function describeCapabilitySchemaValue(value) {
  if (value === null) return "null";
  if (Array.isArray(value)) return "array";
  if (typeof value === "number") {
    return Number.isInteger(value) ? "integer" : "number";
  }
  return typeof value;
}

function pruneCapabilitySchemaValue(schema = null, value = null) {
  const source = schema && typeof schema === "object" && !Array.isArray(schema) ? schema : null;
  if (!source || value == null) {
    return cloneJson(value, value);
  }
  if (Array.isArray(value)) {
    if (source.items && typeof source.items === "object") {
      return value.map((entry) => pruneCapabilitySchemaValue(source.items, entry));
    }
    return cloneJson(value, value);
  }
  if (typeof value !== "object") {
    return cloneJson(value, value);
  }
  const properties = source.properties && typeof source.properties === "object" && !Array.isArray(source.properties)
    ? source.properties
    : {};
  const pruned = {};
  for (const [key, entryValue] of Object.entries(value)) {
    if (source.additionalProperties === false && !Object.prototype.hasOwnProperty.call(properties, key)) {
      continue;
    }
    const propertySchema = Object.prototype.hasOwnProperty.call(properties, key) ? properties[key] : null;
    pruned[key] = pruneCapabilitySchemaValue(propertySchema, entryValue);
  }
  return pruned;
}

function formatCapabilitySchemaEnumValue(value) {
  if (typeof value === "string") return JSON.stringify(value);
  if (value === null) return "null";
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function validateCapabilitySchemaValue(schema = null, value = null, path = "$") {
  const source = schema && typeof schema === "object" && !Array.isArray(schema) ? schema : null;
  if (!source || !Object.keys(source).length) {
    return { ok: true };
  }

  if (Array.isArray(source.anyOf) && source.anyOf.length) {
    for (const candidate of source.anyOf) {
      const candidateResult = validateCapabilitySchemaValue(candidate, value, path);
      if (candidateResult.ok) return candidateResult;
    }
    return {
      ok: false,
      path,
      reason: "must match at least one allowed schema variant",
    };
  }

  if (Array.isArray(source.oneOf) && source.oneOf.length) {
    let matches = 0;
    for (const candidate of source.oneOf) {
      const candidateResult = validateCapabilitySchemaValue(candidate, value, path);
      if (candidateResult.ok) {
        matches += 1;
      }
    }
    if (matches === 1) {
      return { ok: true };
    }
    return {
      ok: false,
      path,
      reason: matches > 1
        ? "must match exactly one allowed schema variant"
        : "must match one allowed schema variant",
    };
  }

  if (Array.isArray(source.allOf) && source.allOf.length) {
    for (const candidate of source.allOf) {
      const candidateResult = validateCapabilitySchemaValue(candidate, value, path);
      if (!candidateResult.ok) return candidateResult;
    }
  }

  if (Object.prototype.hasOwnProperty.call(source, "const") && !Object.is(source.const, value)) {
    return {
      ok: false,
      path,
      reason: `must equal ${formatCapabilitySchemaEnumValue(source.const)}`,
    };
  }

  if (Array.isArray(source.enum) && source.enum.length) {
    const matchesEnum = source.enum.some((entry) => Object.is(entry, value));
    if (!matchesEnum) {
      return {
        ok: false,
        path,
        reason: `must be one of ${source.enum.map((entry) => formatCapabilitySchemaEnumValue(entry)).join(", ")}`,
      };
    }
  }

  const allowedTypes = Array.isArray(source.type)
    ? source.type.map((entry) => asText(entry).toLowerCase()).filter(Boolean)
    : asText(source.type)
      ? [asText(source.type).toLowerCase()]
      : [];
  if (allowedTypes.length) {
    const matchesType = allowedTypes.some((type) => {
      if (type === "null") return value === null;
      if (type === "array") return Array.isArray(value);
      if (type === "object") return Boolean(value) && typeof value === "object" && !Array.isArray(value);
      if (type === "integer") return typeof value === "number" && Number.isInteger(value);
      if (type === "number") return typeof value === "number" && Number.isFinite(value);
      return typeof value === type;
    });
    if (!matchesType) {
      return {
        ok: false,
        path,
        reason: `expected ${allowedTypes.join(" or ")} but received ${describeCapabilitySchemaValue(value)}`,
      };
    }
  }

  if (value == null) {
    return { ok: true };
  }

  if (Array.isArray(value)) {
    if (source.items && typeof source.items === "object") {
      for (let index = 0; index < value.length; index += 1) {
        const itemResult = validateCapabilitySchemaValue(source.items, value[index], `${path}[${index}]`);
        if (!itemResult.ok) return itemResult;
      }
    }
    return { ok: true };
  }

  if (typeof value !== "object") {
    return { ok: true };
  }

  const properties = source.properties && typeof source.properties === "object" && !Array.isArray(source.properties)
    ? source.properties
    : {};

  for (const requiredKey of normalizeArray(source.required)) {
    const propertyName = asText(requiredKey);
    if (!propertyName) continue;
    if (!Object.prototype.hasOwnProperty.call(value, propertyName)) {
      return {
        ok: false,
        path: formatCapabilitySchemaPath(`${path}.${propertyName}`),
        reason: "is required",
      };
    }
  }

  if (source.additionalProperties === false) {
    for (const key of Object.keys(value)) {
      if (Object.prototype.hasOwnProperty.call(properties, key)) continue;
      return {
        ok: false,
        path: formatCapabilitySchemaPath(`${path}.${key}`),
        reason: "is not allowed by the reviewed capability return schema",
      };
    }
  }

  for (const [propertyName, propertySchema] of Object.entries(properties)) {
    if (!Object.prototype.hasOwnProperty.call(value, propertyName)) continue;
    const propertyResult = validateCapabilitySchemaValue(
      propertySchema,
      value[propertyName],
      formatCapabilitySchemaPath(`${path}.${propertyName}`),
    );
    if (!propertyResult.ok) return propertyResult;
  }

  return { ok: true };
}

function createCapabilitySchemaValidationError(capability = null, value = null, validation = null) {
  const toolName = asText(capability?.toolName || capability?.id || capability?.name) || "reviewed_capability";
  const schemaPath = formatCapabilitySchemaPath(validation?.path || "$");
  const validationReason = asText(validation?.reason) || "returned data that does not match its return schema";
  let receivedPreview = "";
  try {
    receivedPreview = trimText(JSON.stringify(value ?? null), 320);
  } catch {
    receivedPreview = trimText(String(value ?? ""), 320);
  }
  const error = new Error(
    `The reviewed capability "${toolName}" returned schema-invalid data at ${schemaPath}: ${validationReason}.`,
  );
  error.name = "SchemaValidationError";
  error.detail = {
    reason: "reviewed_capability_schema_invalid",
    toolName,
    capabilityId: asText(capability?.id),
    schemaPath,
    validationReason,
    receivedPreview,
  };
  return error;
}

function validateCapabilityReturnSchema(capability = null, value = null) {
  const schema = capability?.returnSchema && typeof capability.returnSchema === "object"
    ? capability.returnSchema
    : null;
  const normalizedValue = normalizeCapabilityOutputForRuntime(capability, value);
  if (!schema || !Object.keys(schema).length) {
    return normalizedValue;
  }
  const sanitizedValue = pruneCapabilitySchemaValue(schema, normalizedValue);
  const validation = validateCapabilitySchemaValue(schema, sanitizedValue, "$");
  if (!validation.ok) {
    throw createCapabilitySchemaValidationError(capability, sanitizedValue, validation);
  }
  return sanitizedValue;
}

function getCapabilityReportedFailure(capability = null, value = null) {
  const source = value && typeof value === "object" && !Array.isArray(value)
    ? value
    : null;
  if (!source || source.ok !== false) {
    return null;
  }
  const errorSource = source.error && typeof source.error === "object" && !Array.isArray(source.error)
    ? source.error
    : null;
  const toolName = asText(capability?.toolName || capability?.id || capability?.name) || "reviewed_capability";
  const message = asText(errorSource?.message) || `The reviewed capability "${toolName}" could not complete.`;
  return {
    message,
    detail: {
      reason: "reviewed_capability_reported_failure",
      toolName,
      capabilityId: asText(capability?.id),
      code: asText(errorSource?.code) || "REVIEWED_CAPABILITY_REPORTED_FAILURE",
      finalOutput: cloneJson(source, source),
    },
  };
}

function buildCapabilityModelMessages({
  craft = null,
  capability = null,
  action = null,
  prompt = "",
  browserContext = null,
  preResult = null,
} = {}) {
  return [
    {
      role: "system",
      content: [
        `You are the inner model step for the reviewed browser capability "${asText(capability?.name) || "capability"}".`,
        asText(capability?.description) ? `Capability: ${asText(capability.description)}` : "",
        asText(capability?.skillRef) ? `Skill/policy ref: ${asText(capability.skillRef)}` : "",
        "Use the extracted browser data and the tool arguments to produce the payload needed for the next post-processing step.",
        "If the capability return schema implies JSON, return only valid JSON and nothing else.",
        "Otherwise return only the transformed content with no markdown fences and no explanations.",
      ].filter(Boolean).join("\n"),
    },
    {
      role: "user",
      content: JSON.stringify(
        {
          craft:
            craft && typeof craft === "object"
              ? {
                  id: asText(craft.id),
                  name: asText(craft.name),
                  summary: asText(craft.summary),
                }
              : null,
          capability:
            capability && typeof capability === "object"
              ? {
                  id: asText(capability.id),
                  name: asText(capability.name),
                  description: asText(capability.description),
                  parameterSchema: cloneJson(capability.parameterSchema, {}),
                  returnSchema: cloneJson(capability.returnSchema, {}),
                }
              : null,
          actionArguments:
            action?.arguments && typeof action.arguments === "object"
              ? cloneJson(action.arguments, {})
              : {},
          userPrompt: asText(prompt),
          browserContext: clipPromptPayload(browserContext, 3_000),
          extractedInput: clipPromptPayload(preResult?.output ?? preResult?.raw ?? null, 4_000),
        },
        null,
        2,
      ),
    },
  ];
}

async function runCapabilityModelTransform({
  craft = null,
  capability = null,
  action = null,
  prompt = "",
  browserContext = null,
  preResult = null,
  signal,
} = {}) {
  const returnSchema = capability?.returnSchema && typeof capability.returnSchema === "object"
    ? capability.returnSchema
    : null;
  const expectsJson = Boolean(
    returnSchema &&
      (
        returnSchema.type === "object" ||
        (returnSchema.properties && typeof returnSchema.properties === "object")
      ),
  );

  if (expectsJson) {
    const response = await llmChat(buildCraftModelRequest(craft, {
      localContext: buildCraftLocalContext(craft),
      messages: buildCapabilityModelMessages({
        craft,
        capability,
        action,
        prompt,
        browserContext,
        preResult,
      }),
      parameters: {
        temperature: 0,
        maxTokens: 700,
      },
      signal,
    }));
    const parsed = parseJsonLoose(response?.text || "");
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      throw new Error(`The reviewed capability "${asText(capability?.name) || "capability"}" expected JSON output.`);
    }
    return {
      ok: true,
      output: cloneJson(parsed, parsed),
      text: JSON.stringify(parsed, null, 2),
      usage: extractUsage(response),
      modelRef: asText(response?.resolved?.modelRef),
    };
  }

  const response = await llmChat(buildCraftModelRequest(craft, {
    localContext: buildCraftLocalContext(craft),
    messages: buildCapabilityModelMessages({
      craft,
      capability,
      action,
      prompt,
      browserContext,
      preResult,
    }),
    parameters: {
      temperature: 0.2,
      maxTokens: 700,
    },
    signal,
  }));
  const text = asText(response?.text);
  return {
    ok: true,
    output: text,
    text,
    usage: extractUsage(response),
    modelRef: asText(response?.resolved?.modelRef),
  };
}

async function runCapabilityBrowserPhase({
  phase = "execute",
  source = "",
  bundle = null,
  craft = null,
  capability = null,
  action = null,
  prompt = "",
  browserContext = null,
  state = null,
  observation = null,
  modelOutput = null,
  browserRuntime = null,
  signal,
} = {}) {
  const targetUrl = buildRuntimeTargetUrl(browserContext, action, capability);
  const baseUrl = targetUrl || asText(browserContext?.url) || "https://example.com/";
  const allowedHostSet = new Set(buildRuntimeAllowedHosts(browserContext, action, capability));
  // Keep the fallback browser target and the host allowlist self-consistent.
  collectUrlHosts(targetUrl, allowedHostSet);
  collectUrlHosts(baseUrl, allowedHostSet);
  const allowedHosts = [...allowedHostSet].filter(Boolean);
  const script = buildCapabilityPhaseScript({
    phase,
    source,
    bundle,
    craft,
    capability,
    action,
    prompt,
    browserContext,
    state,
    observation,
    modelOutput,
  });

  const result = await runRestrictedBrowserScript(browserRuntime || {}, {
    url: targetUrl,
    baseUrl,
    allowedHosts: allowedHosts.length ? allowedHosts : ["example.com"],
    active: true,
    timeoutMs: phase === "execute" ? 45_000 : 20_000,
    code: script,
    signal,
  });
  if (!result?.ok) {
    const error = new Error(asText(result?.error) || `${phase} script failed.`);
    error.detail = {
      phase: asText(phase),
      errorCode: asText(result?.data?.error_code || result?.errorCode),
      finalUrl: asText(result?.data?.url || ""),
      rawPreview: asText(result?.data?.raw_preview || ""),
      pageState:
        result?.data?.page_state && typeof result.data.page_state === "object"
          ? cloneJson(result.data.page_state, null)
          : null,
    };
    error.stack = asText(result?.errorStack) || error.stack;
    throw error;
  }
  return {
    phase: asText(phase),
    pageState: result?.data?.page_state || null,
    finalUrl: asText(result?.data?.url || ""),
    ...normalizeCapabilityExecutionResult(phase, result?.data?.raw, capability),
  };
}

async function executeReviewedCapability({
  craft = null,
  bundle = null,
  action = null,
  prompt = "",
  browserContext = null,
  signal,
} = {}) {
  const plan = compileRuntimeCapabilityExecutionPlan(action, bundle?.browserCapabilities?.payload || null);
  if (!plan?.ok) {
    return {
      ok: false,
      error: asText(plan?.error) || "Unknown reviewed capability.",
      capability: null,
    };
  }

  const browserRuntime = {};
  let state = {};
  let latestPageState = browserContext?.pageState || null;
  let latestUrl = asText(browserContext?.url);
  let modelUsage = {
    inputTokens: 0,
    outputTokens: 0,
    totalTokens: 0,
    costUsd: 0,
  };
  let modelRef = "";

  try {
    const preResult = asText(plan.invocation?.scripts?.pre)
      ? await runCapabilityBrowserPhase({
          phase: "pre",
          source: plan.invocation.scripts.pre,
          bundle,
          craft,
          capability: plan.capability,
          action: plan.action,
          prompt,
          browserContext,
          state,
          browserRuntime,
          signal,
        })
      : null;
    if (preResult?.state) {
      state = {
        ...state,
        ...preResult.state,
      };
    }
    if (preResult?.pageState) latestPageState = preResult.pageState;
    if (preResult?.finalUrl) latestUrl = preResult.finalUrl;

    const shouldRunLocalReviewedTool =
      !asText(plan.invocation?.scripts?.pre) &&
      !asText(plan.invocation?.scripts?.post) &&
      canExecuteReviewedToolLocally(bundle, plan.capability);
    const shouldRunDirectActiveTextCapability =
      !asText(plan.invocation?.scripts?.pre) &&
      !asText(plan.invocation?.scripts?.post) &&
      isDirectActiveTextToolName(plan.capability?.toolName || plan.capability?.id || plan.capability?.name);

    const executeResult = asText(plan.invocation?.scripts?.execute)
      ? shouldRunDirectActiveTextCapability
        ? await executeDirectActiveTextCapability({
            capability: plan.capability,
            action: plan.action,
            browserContext: {
              ...browserContext,
              url: latestUrl || browserContext?.url || "",
              pageState: latestPageState || browserContext?.pageState || null,
            },
          })
        : shouldRunLocalReviewedTool
        ? await executeReviewedToolLocally({
            bundle,
            craft,
            capability: plan.capability,
            action: plan.action,
            prompt,
            browserContext: {
              ...browserContext,
              url: latestUrl || browserContext?.url || "",
              pageState: latestPageState || browserContext?.pageState || null,
            },
            state,
            observation: preResult?.output ?? null,
            signal,
          })
        : await runCapabilityBrowserPhase({
            phase: "execute",
            source: plan.invocation.scripts.execute,
            bundle,
            craft,
            capability: plan.capability,
            action: plan.action,
            prompt,
            browserContext: {
              ...browserContext,
              url: latestUrl || browserContext?.url || "",
              pageState: latestPageState || browserContext?.pageState || null,
            },
            state,
            observation: preResult?.output ?? null,
            browserRuntime,
            signal,
          })
      : null;
    if (executeResult?.state) {
      state = {
        ...state,
        ...executeResult.state,
      };
    }
    if (executeResult?.pageState) latestPageState = executeResult.pageState;
    if (executeResult?.finalUrl) latestUrl = executeResult.finalUrl;

    const modelTransform = !executeResult
      ? await runCapabilityModelTransform({
          craft,
          capability: plan.capability,
          action: plan.action,
          prompt,
          browserContext: {
            ...browserContext,
            url: latestUrl || browserContext?.url || "",
            pageState: latestPageState || browserContext?.pageState || null,
          },
          preResult,
          signal,
        })
      : null;
    if (modelTransform?.usage) {
      modelUsage = addUsage(modelUsage, modelTransform.usage);
    }
    if (modelTransform?.modelRef) {
      modelRef = modelTransform.modelRef;
    }

    const postResult = asText(plan.invocation?.scripts?.post)
      ? await runCapabilityBrowserPhase({
          phase: "post",
          source: plan.invocation.scripts.post,
          bundle,
          craft,
          capability: plan.capability,
          action: plan.action,
          prompt,
          browserContext: {
            ...browserContext,
            url: latestUrl || browserContext?.url || "",
            pageState: latestPageState || browserContext?.pageState || null,
          },
          state,
          observation: executeResult?.output ?? preResult?.output ?? null,
          modelOutput: modelTransform?.output ?? null,
          browserRuntime,
          signal,
        })
      : null;
    if (postResult?.pageState) latestPageState = postResult.pageState;
    if (postResult?.finalUrl) latestUrl = postResult.finalUrl;

    const finalOutput = validateCapabilityReturnSchema(
      plan.capability,
      postResult?.output ??
        executeResult?.output ??
        modelTransform?.output ??
        preResult?.output ??
        null,
    );
    const reportedFailure = getCapabilityReportedFailure(plan.capability, finalOutput);
    if (reportedFailure) {
      return {
        ok: false,
        capability: {
          id: asText(plan.capability?.id),
          name: asText(plan.capability?.name),
          version: asText(plan.capability?.version),
        },
        preResult,
        executeResult,
        modelTransform,
        postResult,
        finalOutput,
        error: reportedFailure.message,
        errorDetail: reportedFailure.detail,
        errorStack: "",
        pageState: latestPageState,
        finalUrl: latestUrl,
        usage: modelUsage,
        modelRef,
      };
    }
    return {
      ok: true,
      capability: {
        id: asText(plan.capability?.id),
        name: asText(plan.capability?.name),
        version: asText(plan.capability?.version),
      },
      preResult,
      executeResult,
      modelTransform,
      postResult,
      finalOutput,
      pageState: latestPageState,
      finalUrl: latestUrl,
      usage: modelUsage,
      modelRef,
      summary: trimText(
        postResult?.summary ||
          executeResult?.summary ||
          (modelTransform?.text ? `Model output produced for ${asText(plan.capability?.name)}.` : "") ||
          preResult?.summary ||
          `${asText(plan.capability?.name) || "Capability"} executed.`,
        240,
      ),
    };
  } catch (error) {
    if (signal?.aborted) {
      throw signal.reason || error;
    }
    return {
      ok: false,
      capability: {
        id: asText(plan.capability?.id),
        name: asText(plan.capability?.name),
        version: asText(plan.capability?.version),
      },
      error: error instanceof Error ? error.message : String(error || "Capability execution failed."),
      errorDetail:
        error && typeof error === "object" && "detail" in error
          ? cloneJson(error.detail, null)
          : null,
      errorStack: asText(error?.stack || ""),
      pageState: latestPageState,
      finalUrl: latestUrl,
      usage: modelUsage,
      modelRef,
    };
  }
}

function formatFinishText(action, fallbackSummary = "") {
  const result = action?.result;
  if (typeof result === "string") return result;
  if (result && typeof result === "object") {
    if (asText(result.text)) return asText(result.text);
    if (asText(result.summary)) return asText(result.summary);
    try {
      return JSON.stringify(result, null, 2);
    } catch {}
  }
  return asText(action?.summary) || asText(fallbackSummary);
}

async function fallbackToDirectChat({ craft = null, prompt = "", signal } = {}) {
  if (!asText(prompt)) {
    return {
      status: "failed",
      text: "",
      error: "This craft has no reviewed browser capabilities yet and needs a prompt.",
      modelRef: "",
      usage: {
        inputTokens: 0,
        outputTokens: 0,
        totalTokens: 0,
        costUsd: 0,
      },
      steps: [],
    };
  }
  const response = await llmChat(buildCraftModelRequest(craft, {
    localContext: buildCraftLocalContext(craft),
    messages: [
      {
        role: "system",
        content: [
          `You are the runtime for the craft "${asText(craft?.name) || "craft"}".`,
          asText(craft?.summary) ? `Craft summary: ${asText(craft.summary)}` : "",
          "Answer directly and keep the output useful.",
        ].filter(Boolean).join("\n"),
      },
      {
        role: "user",
        content: asText(prompt),
      },
    ],
    parameters: {
      maxTokens: 700,
      temperature: 0,
    },
    signal,
  }));
  return {
    status: "done",
    text: asText(response?.text),
    error: "",
    modelRef: asText(response?.resolved?.modelRef),
    usage: extractUsage(response),
    steps: [],
  };
}

function buildDecisionMessages({
  craft = null,
  prompt = "",
  bundle = null,
  browserContext = null,
  steps = [],
  turn = 1,
  maxTurns = DEFAULT_MAX_RUNTIME_TURNS,
} = {}) {
  const capabilityPayload = bundle?.browserCapabilities?.payload || null;
  const policySpec = bundle?.policy?.payload?.policySpec && typeof bundle.policy.payload.policySpec === "object"
    ? bundle.policy.payload.policySpec
    : null;
  const transcript = JSON.stringify(
    {
      messages: buildRuntimeTranscriptMessages({
        craft,
        prompt,
        browserContext,
        steps,
        turn,
        maxTurns,
      }),
      tools: buildRuntimeCapabilityToolDefinitions(capabilityPayload),
    },
    null,
    2,
  );
  return [
    {
      role: "system",
      content: [
        `You are the runtime supervisor for the craft "${asText(craft?.name) || "craft"}".`,
        asText(craft?.summary) ? `Craft objective: ${asText(craft.summary)}` : "",
        buildRuntimeCapabilitySkillText(capabilityPayload),
        policySpec?.bundleSkill ? `Bundle policy: ${asText(policySpec.bundleSkill)}` : "",
        "Return exactly one assistant message as a JSON object.",
        "Prefer a reviewed browser capability whenever one can inspect, modify, or verify browser state needed for the task.",
        "Do not answer from prior knowledge when a reviewed capability can read or write the required browser state.",
        "Use assistant tool_calls only when a reviewed browser capability should run next.",
        "Use at most one tool call in the next assistant turn.",
        "If critical user intent is missing, ask the user directly in assistant content.",
        "If the task is done or blocked, return the final assistant response in content.",
        `This run may use at most ${Math.max(1, Number(maxTurns || DEFAULT_MAX_RUNTIME_TURNS))} turns.`,
      ].filter(Boolean).join("\n\n"),
    },
    {
      role: "user",
      content: [
        transcript,
        "Return the next assistant turn as one JSON object with role=\"assistant\".",
        "The object may include content and may include tool_calls.",
        "Do not return markdown fences or any text outside the JSON object.",
      ].filter(Boolean).join("\n\n"),
    },
  ];
}

function buildNativeDecisionPrompt({
  craft = null,
  prompt = "",
  bundle = null,
  browserContext = null,
  steps = [],
  turn = 1,
  maxTurns = DEFAULT_MAX_RUNTIME_TURNS,
} = {}) {
  const capabilityPayload = bundle?.browserCapabilities?.payload || null;
  const policySpec = bundle?.policy?.payload?.policySpec && typeof bundle.policy.payload.policySpec === "object"
    ? bundle.policy.payload.policySpec
    : null;
  const transcriptMessages = buildRuntimeTranscriptMessages({
    craft,
    prompt,
    browserContext,
    steps,
    turn,
    maxTurns,
  });
  const transcriptSystem = transcriptMessages[0]?.role === "system"
    ? asText(transcriptMessages[0].content)
    : "";
  const conversationMessages = transcriptMessages[0]?.role === "system"
    ? transcriptMessages.slice(1)
    : transcriptMessages.slice();
  const messages = [
    {
      role: "system",
      content: [
        `You are the runtime supervisor for the craft "${asText(craft?.name) || "craft"}".`,
        asText(craft?.summary) ? `Craft objective: ${asText(craft.summary)}` : "",
        transcriptSystem,
        buildRuntimeCapabilitySkillText(capabilityPayload, {
          outputSurface: "qwen_native_raw",
        }),
        policySpec?.bundleSkill ? `Bundle policy: ${asText(policySpec.bundleSkill)}` : "",
        "Prefer a reviewed browser capability whenever one can inspect, modify, or verify browser state needed for the task.",
        "Do not answer from prior knowledge when a reviewed capability can read or write the required browser state.",
        "Use at most one reviewed tool call in the next assistant turn.",
        "If critical user intent is missing, ask the user directly in plain assistant text before any tool call.",
        "If the task is done or blocked, return the final assistant response in plain text.",
        `This run may use at most ${Math.max(1, Number(maxTurns || DEFAULT_MAX_RUNTIME_TURNS))} turns.`,
      ].filter(Boolean).join("\n\n"),
    },
    ...conversationMessages,
  ];
  const tools = buildRuntimeCapabilityToolDefinitions(capabilityPayload);
  return {
    messages,
    tools,
    promptText: renderQwen35ChatTemplate(messages, {
      tools,
      addGenerationPrompt: true,
      enableThinking: false,
    }),
  };
}

async function decideNextAssistantTurn({
  craft = null,
  prompt = "",
  bundle = null,
  browserContext = null,
  steps = [],
  turn = 1,
  maxTurns = DEFAULT_MAX_RUNTIME_TURNS,
  signal,
} = {}) {
  const capabilityProfile = await resolveCraftCapabilityProfile(craft);
  const decisionMaxTokens = resolveReviewedDecisionMaxTokens(capabilityProfile);
  const capabilityNames = buildRuntimeCapabilityCatalog(bundle)
    .capabilities
    .map((entry) => asText(entry.toolName))
    .filter(Boolean);
  if (capabilityProfile.providerType === "local_qwen") {
    const nativePrompt = buildNativeDecisionPrompt({
      craft,
      prompt,
      bundle,
      browserContext,
      steps,
      turn,
      maxTurns,
    });
    const response = await llmChat(buildCraftModelRequest(craft, {
      localContext: buildCraftLocalContext(craft),
      messages: nativePrompt.messages,
      tools: nativePrompt.tools,
      promptText: nativePrompt.promptText,
      parameters: {
        temperature: 0,
        maxTokens: decisionMaxTokens,
      },
      signal,
    }));
    const assistantTurn = normalizePortableAssistantTurn(response.text, {
      allowedToolNames: capabilityNames,
      ...STRICT_RUNTIME_ASSISTANT_TURN_OPTIONS,
    });
    if (!assistantTurn) {
      throw new Error("Craft runtime did not receive a valid native Qwen assistant turn.");
    }
    return {
      assistantTurn,
      usage: extractUsage(response),
      modelRef: asText(response?.resolved?.modelRef),
    };
  }
  const response = await generateStructuredOutput(buildCraftModelRequest(craft, {
    localContext: buildCraftLocalContext(craft),
    schema: PORTABLE_ASSISTANT_TURN_SCHEMA,
    schemaName: "portable_assistant_turn",
    jsonSchema: buildPortableAssistantTurnJsonSchema({
      allowedToolNames: capabilityNames,
      maxToolCalls: 1,
    }),
    schemaDescription:
      "Return exactly one next assistant turn for the craft runtime. Prefer tool_calls whenever a reviewed capability can advance the task from browser state; use content only for clarification or final completion.",
    messages: buildDecisionMessages({
      craft,
      prompt,
      bundle,
      browserContext,
      steps,
      turn,
      maxTurns,
    }),
    parameters: {
      temperature: 0,
      maxTokens: decisionMaxTokens,
    },
    signal,
  }));

  const assistantTurn = normalizePortableAssistantTurn(response.object, {
    allowedToolNames: capabilityNames,
    ...STRICT_RUNTIME_ASSISTANT_TURN_OPTIONS,
  });
  if (!assistantTurn) {
    throw new Error("Craft runtime did not receive a valid assistant turn.");
  }

  return {
    assistantTurn,
    usage: extractUsage(response),
    modelRef: asText(response?.resolved?.modelRef),
  };
}

async function decideNextCanonicalActionFallback({
  craft = null,
  prompt = "",
  bundle = null,
  browserContext = null,
  steps = [],
  turn = 1,
  maxTurns = DEFAULT_MAX_RUNTIME_TURNS,
  signal,
} = {}) {
  const capabilityProfile = await resolveCraftCapabilityProfile(craft);
  const capabilityNames = buildRuntimeCapabilityCatalog(bundle)
    .capabilities
    .map((entry) => asText(entry.toolName))
    .filter(Boolean);
  const decisionMaxTokens = resolveReviewedDecisionMaxTokens(capabilityProfile);
  const canonicalDefaults = buildCanonicalActionDefaults(bundle);
  let response;
  try {
    response = await generateStructuredOutput(buildCraftModelRequest(craft, {
      localContext: buildCraftLocalContext(craft),
      schema: CANONICAL_TOOL_ACTION_SCHEMA,
      schemaName: "canonical_tool_action",
      jsonSchema: buildCanonicalToolActionJsonSchema({
        allowedToolNames: capabilityNames,
        includeAskUser: true,
        includeFinish: true,
      }),
      schemaDescription: "Compatibility fallback: decide exactly one next canonical action for the craft runtime.",
      messages: buildDecisionMessages({
        craft,
        prompt,
        bundle,
        browserContext,
        steps,
        turn,
        maxTurns,
      }),
      parameters: {
        temperature: 0,
        maxTokens: decisionMaxTokens,
      },
      signal,
    }));
  } catch (error) {
    const errorDetail = error && typeof error === "object" && "detail" in error ? error.detail || null : null;
    const candidateTexts = [
      errorDetail?.initialText,
      errorDetail?.repairedText,
    ].filter(Boolean);
    for (const candidateText of candidateTexts) {
      const parsedCandidate = parseJsonLoose(candidateText) || candidateText;
      const canonical = normalizeCanonicalToolAction(parsedCandidate, {
        ...canonicalDefaults,
        passThroughUnknown: false,
      });
      if (canonical) {
        return {
          action: canonical,
          usage: {
            inputTokens: 0,
            outputTokens: 0,
            totalTokens: 0,
            costUsd: 0,
          },
          modelRef: asText(errorDetail?.resolved?.modelRef),
        };
      }
      const assistantTurn = normalizePortableAssistantTurn(parsedCandidate, {
        allowedToolNames: capabilityNames,
        ...STRICT_RUNTIME_ASSISTANT_TURN_OPTIONS,
      });
      if (!assistantTurn) continue;
      const fallbackAction = convertPortableAssistantTurnToCanonicalAction(assistantTurn, canonicalDefaults);
      if (fallbackAction) {
        return {
          action: fallbackAction,
          usage: {
            inputTokens: 0,
            outputTokens: 0,
            totalTokens: 0,
            costUsd: 0,
          },
          modelRef: asText(errorDetail?.resolved?.modelRef),
        };
      }
    }
    throw error;
  }

  return {
    action: normalizeCanonicalToolAction(response.object, {
      ...canonicalDefaults,
      passThroughUnknown: false,
    }),
    usage: extractUsage(response),
    modelRef: asText(response?.resolved?.modelRef),
  };
}

export async function runCraftUse({
  craft = null,
  prompt = "",
  maxTurns = null,
  signal,
} = {}) {
  updateCraftUseDebug("run:start", {
    craftId: asText(craft?.id),
  });
  if (!craft || typeof craft !== "object") {
    throw new Error("Craft execution requires a craft object.");
  }

  const loadedBundle = await readCraftBundle(craft);
  const strictRuntimeBundle = resolveStrictRuntimeBundle(craft, loadedBundle);
  if (!strictRuntimeBundle.ok) {
    if (hasReviewedRuntimePackageSignals(craft, loadedBundle)) {
      return {
        status: "failed",
        text: "",
        error: strictRuntimeBundle.error,
        modelRef: "",
        usage: {
          inputTokens: 0,
          outputTokens: 0,
          totalTokens: 0,
          costUsd: 0,
        },
        steps: [],
      };
    }
    return await fallbackToDirectChat({ craft, prompt, signal });
  }
  const bundle = strictRuntimeBundle.bundle;
  const capabilityCatalog = buildRuntimeCapabilityCatalog(bundle);
  if (!capabilityCatalog.capabilities.length) {
    return await fallbackToDirectChat({ craft, prompt, signal });
  }
  const resolvedMaxTurns = resolveCraftRuntimeMaxTurns({
    craft,
    bundle,
    requestedMaxTurns: maxTurns,
  });
  const capabilityProfile = await resolveCraftCapabilityProfile(craft);
  const preferredInitialCapabilityToolName = getPreferredInitialCapabilityToolName(
    bundle,
    capabilityProfile,
  );
  const activeTextFlow = getInPlaceActiveTextFlow(bundle, craft);
  const shouldAllowDeterministicLocalFallback =
    capabilityProfile.providerType === "local_qwen" && !activeTextFlow.enabled;
  const deterministicLocalCapabilitySequence = shouldAllowDeterministicLocalFallback
    ? getDeterministicLocalCapabilitySequence(bundle)
    : [];

  updateCraftUseDebug("run:collect_initial_browser_context", {
    activeTextFlow: activeTextFlow.enabled,
  });
  const initialBrowserContext = deterministicLocalCapabilitySequence.length
    ? {
        tabId: 0,
        url: "",
        title: "",
        pageState: null,
      }
    : await collectActiveTabContext().catch(() => ({
        tabId: 0,
        url: "",
        title: "",
        pageState: null,
        }));
  updateCraftUseDebug("run:initial_browser_context_ready", {
    tabId: Number(initialBrowserContext?.tabId || 0),
    url: asText(initialBrowserContext?.url),
    activeTextFlow: activeTextFlow.enabled,
  });
  const steps = [];
  let browserContext = cloneJson(initialBrowserContext, initialBrowserContext);
  let usageTotals = {
    inputTokens: 0,
    outputTokens: 0,
    totalTokens: 0,
    costUsd: 0,
  };
  let lastModelRef = "";

  for (let turn = 1; turn <= resolvedMaxTurns; turn += 1) {
    updateCraftUseDebug("run:turn_start", {
      turn,
      stepCount: steps.length,
    });
    if (signal?.aborted) {
      throw signal.reason || new DOMException("Craft runtime aborted.", "AbortError");
    }
    let assistantTurn = null;
    let action = null;

    if (!steps.length && activeTextFlow.enabled) {
      action = buildForcedCapabilityAction(bundle, activeTextFlow.readToolName, {});
      assistantTurn = buildPortableAssistantTurnFromCanonicalAction(action, {
        callId: `call_${turn}`,
      });
    }

        if (!action && activeTextFlow.enabled && getLastStepToolName(steps) === activeTextFlow.readToolName) {
          const readPayload = extractActiveTextReadPayload(steps[steps.length - 1]);
          if (readPayload?.text) {
            const replacement = await waitForLocalReviewedToolResult(
              buildDirectActiveTextReplacement({
                craft,
                prompt,
                sourceText: readPayload.text,
                targetType: readPayload.targetType,
                signal,
              }),
              {
                signal,
                toolName: "active_text_replacement",
                timeoutMs: ACTIVE_TEXT_REPLACEMENT_TIMEOUT_MS,
              },
            );
            usageTotals = addUsage(usageTotals, replacement?.usage);
            if (replacement?.modelRef) lastModelRef = replacement.modelRef;
            action = buildForcedCapabilityAction(bundle, activeTextFlow.replaceToolName, {
              text: replacement?.text || "",
            });
        assistantTurn = buildPortableAssistantTurnFromCanonicalAction(action, {
          callId: `call_${turn}`,
        });
      }
    }

    if (!action && shouldAllowDeterministicLocalFallback) {
      const deterministicLocalAction = buildDeterministicLocalCapabilityAction(bundle, steps);
      if (deterministicLocalAction) {
        action = deterministicLocalAction;
        assistantTurn = buildPortableAssistantTurnFromCanonicalAction(action, {
          callId: `call_${turn}`,
        });
      }
    }

    if (!action && !steps.length && preferredInitialCapabilityToolName) {
      const forcedSingleCapabilityAction = buildForcedSingleCapabilityAction(bundle, preferredInitialCapabilityToolName);
      if (forcedSingleCapabilityAction) {
        action = forcedSingleCapabilityAction;
        assistantTurn = buildPortableAssistantTurnFromCanonicalAction(action, {
          callId: `call_${turn}`,
        });
      }
    }

    if (!action) {
      try {
        const decision = await decideNextAssistantTurn({
          craft,
          prompt,
          bundle,
          browserContext,
          steps,
          turn,
          maxTurns: resolvedMaxTurns,
          signal,
        });
        usageTotals = addUsage(usageTotals, decision.usage);
        if (decision.modelRef) lastModelRef = decision.modelRef;
        assistantTurn = decision.assistantTurn;
        action = convertPortableAssistantTurnToCanonicalAction(assistantTurn, {
          defaultBundleRef: asText(bundle?.browserCapabilities?.artifactId),
          defaultResourceRefs: buildBundleDefaultResourceRefs(bundle),
        });
        if (!action) {
          const deterministicLocalAction = shouldAllowDeterministicLocalFallback
            ? buildDeterministicLocalCapabilityAction(bundle, steps)
            : null;
          if (deterministicLocalAction) {
            action = deterministicLocalAction;
            assistantTurn = buildPortableAssistantTurnFromCanonicalAction(action, {
              callId: `call_${turn}`,
            });
          }
        }
        if (!action) {
          const content = asText(assistantTurn?.content);
          if (looksLikeClarificationRequest(content)) {
            const questions = buildNeedsInputQuestionsFromText(content);
            return {
              status: "needs_input",
              text: content,
              error: "",
              modelRef: lastModelRef,
              usage: usageTotals,
              questions,
              steps,
            };
          }
          if (!steps.length && preferredInitialCapabilityToolName) {
            action = buildForcedSingleCapabilityAction(bundle, preferredInitialCapabilityToolName);
            assistantTurn = buildPortableAssistantTurnFromCanonicalAction(action, {
              callId: `call_${turn}`,
            });
          } else {
            const forcedReplaceAction = coerceActiveTextReplyToReplaceAction(bundle, steps, content, activeTextFlow);
            if (forcedReplaceAction) {
              action = forcedReplaceAction;
              assistantTurn = buildPortableAssistantTurnFromCanonicalAction(action, {
                callId: `call_${turn}`,
              });
            } else {
              return {
                status: "done",
                text: content,
                error: "",
                modelRef: lastModelRef,
                usage: usageTotals,
                result: {
                  text: content,
                  summary: trimText(content, 240),
                },
                steps,
              };
            }
          }
        }
      } catch (assistantTurnError) {
        if (!shouldRetryDecisionViaCanonicalFallback(assistantTurnError)) {
          throw assistantTurnError;
        }
        const deterministicLocalAction = shouldAllowDeterministicLocalFallback
          ? buildDeterministicLocalCapabilityAction(bundle, steps)
          : null;
        if (deterministicLocalAction) {
          action = deterministicLocalAction;
        } else {
          const fallbackDecision = await decideNextCanonicalActionFallback({
            craft,
            prompt,
            bundle,
            browserContext,
            steps,
            turn,
            maxTurns: resolvedMaxTurns,
            signal,
          });
          usageTotals = addUsage(usageTotals, fallbackDecision.usage);
          if (fallbackDecision.modelRef) lastModelRef = fallbackDecision.modelRef;

          action = fallbackDecision.action;
          if (!action || typeof action !== "object") {
            throw new Error("Craft runtime did not receive a valid assistant turn or canonical action.");
          }

          if (action.type === "ask_user") {
            const questions = normalizeArray(action.questions)
              .map((entry, index) => ({
                id: asText(entry?.id) || `question-${index + 1}`,
                question: asText(entry?.question),
                reason: asText(entry?.reason),
              }))
              .filter((entry) => entry.question);
            return {
              status: "needs_input",
              text: questions.map((entry) => entry.question).join("\n"),
              error: "",
              modelRef: lastModelRef,
              usage: usageTotals,
              questions,
              steps,
            };
          }

          if (action.type === "finish" && !steps.length && preferredInitialCapabilityToolName) {
            action = buildForcedSingleCapabilityAction(bundle, preferredInitialCapabilityToolName);
          } else if (action.type === "finish") {
            const forcedReplaceAction = coerceActiveTextReplyToReplaceAction(
              bundle,
              steps,
              formatFinishText(action, ""),
              activeTextFlow,
            );
            if (forcedReplaceAction) {
              action = forcedReplaceAction;
            } else {
              return {
                status: "done",
                text: formatFinishText(action, "Craft execution finished."),
                error: "",
                modelRef: lastModelRef,
                usage: usageTotals,
                result: cloneJson(action.result, action.result),
                steps,
              };
            }
          }
        }

        assistantTurn = buildPortableAssistantTurnFromCanonicalAction(action, {
          callId: `call_${turn}`,
        });
      }
    }

    const executionPromise = executeReviewedCapability({
      craft,
      bundle,
      action,
      prompt,
      browserContext,
      signal,
    });
    const execution =
      activeTextFlow.enabled || isDirectActiveTextToolName(action?.tool_name)
        ? await waitForLocalReviewedToolResult(executionPromise, {
            signal,
            toolName: asText(action?.tool_name || "reviewed_capability"),
            timeoutMs: ACTIVE_TEXT_CAPABILITY_EXECUTION_TIMEOUT_MS,
          })
        : await executionPromise;
    updateCraftUseDebug("run:execution_complete", {
      turn,
      toolName: asText(action?.tool_name),
      ok: execution?.ok !== false,
      summary: trimText(execution?.summary, 160),
    });
    usageTotals = addUsage(usageTotals, execution.usage);
    if (execution.modelRef) lastModelRef = execution.modelRef;

    const observation = summarizeObservation(execution);
    steps.push({
      assistantTurn: cloneJson(assistantTurn, null),
      action: cloneJson(action, null),
      observation,
      execution: cloneJson(execution, null),
    });

    browserContext = {
      ...browserContext,
      url: asText(execution.finalUrl) || browserContext?.url || "",
      pageState: execution.pageState || browserContext?.pageState || null,
    };

    if (execution.ok === false) {
      const reportedBlocker =
        asText(execution?.errorDetail?.reason) === "reviewed_capability_reported_failure";
      return {
        status: reportedBlocker ? "blocked" : "failed",
        text: reportedBlocker ? asText(execution.error) : "",
        error: asText(execution.error) || "Capability execution failed.",
        errorDetail: execution.errorDetail && typeof execution.errorDetail === "object"
          ? cloneJson(execution.errorDetail, null)
          : null,
        errorStack: asText(execution.errorStack || ""),
        failingTool: asText(action?.tool_name || execution?.capability?.id || execution?.capability?.name),
        modelRef: lastModelRef,
        usage: usageTotals,
        steps,
      };
    }

    if (activeTextFlow.enabled && asText(action?.tool_name) === activeTextFlow.replaceToolName) {
      return {
        status: "done",
        text: asText(execution.summary) || "Active text updated.",
        error: "",
        modelRef: lastModelRef,
        usage: usageTotals,
        result: cloneJson(execution.finalOutput, execution.finalOutput),
        steps,
      };
    }

    if (hasCompletedDeterministicLocalCapabilitySequence(bundle, steps)) {
      return {
        status: "done",
        text: asText(execution.summary) || "Local reviewed tool workflow completed.",
        error: "",
        modelRef: lastModelRef,
        usage: usageTotals,
        result: cloneJson(execution.finalOutput, execution.finalOutput),
        steps,
      };
    }
  }

  const lastStep = steps[steps.length - 1] || null;
  return {
    status: "done",
    text: asText(lastStep?.observation?.summary) || "Craft execution reached the turn limit.",
    error: "",
    modelRef: lastModelRef,
    usage: usageTotals,
    result: cloneJson(lastStep?.execution?.finalOutput, lastStep?.execution?.finalOutput),
    steps,
  };
}
