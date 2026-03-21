import { llmChat } from "../bg/llm.js";
import {
  toolCollectPageState,
  toolNavigate,
  toolObserve,
  toolOpenUrl,
  toolRunBrowserNativeAction,
  toolRunPlaywrightCode,
} from "./browserAutomationTools.js";

export const DEFAULT_BROWSER_VISION_MODEL_REF = "custom_openai>Qwen/Qwen3.5-35B-A3B";
export const DEFAULT_BROWSER_VISION_IMAGE_DETAIL = "high";
export const ALLOWED_BROWSER_IMAGE_DETAILS = new Set(["auto", "low", "high"]);
const OPENAI_COMPATIBLE_VISION_PROVIDER_IDS = new Set([
  "openai",
  "custom_openai",
  "openrouter",
  "azure_openai",
]);
const LOCAL_QWEN_VISION_PROVIDER_ALIASES = new Set([
  "qwen",
  "local_qwen",
  "local-qwen",
  "local_qwen_vision",
  "local-qwen-vision",
]);

export const DEFAULT_EVENTUS_PORTAL_URL = "https://example.com/";
export const DEFAULT_EVENTUS_ALLOWED_HOSTS = Object.freeze(["example.com"]);
const RESTRICTED_BROWSER_TAB_READY_TIMEOUT_MS = 45_000;
const ATTACHABLE_TAB_STABLE_READY_MS = 1_200;

function asString(value) {
  return String(value == null ? "" : value);
}

function asInt(value, fallback = 0, min = Number.MIN_SAFE_INTEGER, max = Number.MAX_SAFE_INTEGER) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.max(min, Math.min(max, Math.floor(parsed)));
}

function safeChoice(value, allowed, fallback) {
  const raw = asString(value).trim().toLowerCase();
  return allowed.has(raw) ? raw : fallback;
}

function trimLine(value, max = 260) {
  const text = asString(value).replace(/\s+/g, " ").trim();
  if (text.length <= max) return text;
  return `${text.slice(0, Math.max(1, max - 1)).trimEnd()}...`;
}

function normalizeVisionProviderId(value, fallback = "openai") {
  const raw = asString(value).trim();
  const normalized = raw.toLowerCase();
  if (LOCAL_QWEN_VISION_PROVIDER_ALIASES.has(normalized)) {
    return "local_qwen";
  }
  return raw || asString(fallback).trim() || "openai";
}

function parseJsonLoose(raw) {
  const text = asString(raw).trim();
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch {
    const fenced = text.match(/```(?:json)?\s*([\s\S]*?)\s*```/i);
    if (fenced?.[1]) {
      try {
        return JSON.parse(fenced[1]);
      } catch {}
    }
    const objMatch = text.match(/\{[\s\S]*\}/);
    if (objMatch?.[0]) {
      try {
        return JSON.parse(objMatch[0]);
      } catch {}
    }
  }
  return null;
}

function normalizeBaseUrl(value) {
  return asString(value).trim().replace(/\/+$/, "");
}

function extractOpenAiCompatibleResponseText(payload) {
  const content = payload?.choices?.[0]?.message?.content;
  if (typeof content === "string") {
    const trimmed = content.trim();
    return trimmed || "";
  }
  if (Array.isArray(content)) {
    return content
      .map((item) => {
        if (typeof item?.text === "string") return item.text;
        if (typeof item?.content === "string") return item.content;
        return "";
      })
      .map((item) => asString(item).trim())
      .filter(Boolean)
      .join("\n")
      .trim();
  }
  return "";
}

async function callOpenAiCompatibleVisionInspector({
  providerId = "custom_openai",
  model = "",
  baseUrl = "",
  apiKey = "",
  prompt = "",
  screenshotDataUrl = "",
  imageDetail = DEFAULT_BROWSER_VISION_IMAGE_DETAIL,
  signal,
} = {}) {
  const normalizedBaseUrl = normalizeBaseUrl(baseUrl);
  if (!normalizedBaseUrl) {
    throw new Error(`Vision base URL is missing for provider ${providerId || "custom_openai"}.`);
  }
  const normalizedModel = asString(model).trim();
  if (!normalizedModel) {
    throw new Error(`Vision model is missing for provider ${providerId || "custom_openai"}.`);
  }
  const response = await fetch(`${normalizedBaseUrl}/chat/completions`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      ...(asString(apiKey).trim() ? { authorization: `Bearer ${asString(apiKey).trim()}` } : {}),
    },
    body: JSON.stringify({
      model: normalizedModel,
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: asString(prompt) },
            {
              type: "image_url",
              image_url: {
                url: screenshotDataUrl,
                detail: safeChoice(
                  imageDetail,
                  ALLOWED_BROWSER_IMAGE_DETAILS,
                  DEFAULT_BROWSER_VISION_IMAGE_DETAIL,
                ),
              },
            },
          ],
        },
      ],
      stream: false,
      store: false,
    }),
    signal,
  });

  let payload = null;
  try {
    payload = await response.json();
  } catch (error) {
    throw new Error(
      `Vision runtime ${providerId || "custom_openai"} returned invalid JSON: ${asString(error?.message || error)}`,
    );
  }

  if (!response.ok) {
    const detail =
      asString(payload?.error?.message) ||
      asString(payload?.error) ||
      `${response.status} ${response.statusText}`;
    throw new Error(`Vision runtime ${providerId || "custom_openai"} failed: ${detail}`);
  }

  const text = extractOpenAiCompatibleResponseText(payload);
  if (!text) {
    throw new Error(`Vision runtime ${providerId || "custom_openai"} returned no text.`);
  }
  return text;
}

function clipStructuredValue(value, maxChars = 12_000) {
  try {
    const json = JSON.stringify(value ?? null);
    if (json.length <= maxChars) {
      return {
        clipped: false,
        preview: json,
        value: value ?? null,
      };
    }
    return {
      clipped: true,
      preview: `${json.slice(0, Math.max(1, maxChars - 1)).trimEnd()}...`,
      value: null,
    };
  } catch {
    const fallback = trimLine(value, maxChars);
    return {
      clipped: true,
      preview: fallback,
      value: null,
    };
  }
}

function normalizeUrlHost(hostname = "") {
  return asString(hostname).trim().toLowerCase().replace(/\.+$/, "");
}

function isAllowedHost(hostname, allowedHosts = DEFAULT_EVENTUS_ALLOWED_HOSTS) {
  const normalizedHost = normalizeUrlHost(hostname);
  if (!normalizedHost) return false;
  const hosts = Array.isArray(allowedHosts) ? allowedHosts : [];
  return hosts.some((candidate) => {
    const normalizedCandidate = normalizeUrlHost(candidate);
    return (
      !!normalizedCandidate &&
      (normalizedHost === normalizedCandidate || normalizedHost.endsWith(`.${normalizedCandidate}`))
    );
  });
}

function sameUrlIgnoringHash(left, right) {
  try {
    const a = new URL(asString(left));
    const b = new URL(asString(right));
    a.hash = "";
    b.hash = "";
    return a.toString() === b.toString();
  } catch {
    return asString(left).trim() === asString(right).trim();
  }
}

export function resolveRestrictedBrowserUrl(rawUrl, {
  baseUrl = DEFAULT_EVENTUS_PORTAL_URL,
  allowedHosts = DEFAULT_EVENTUS_ALLOWED_HOSTS,
} = {}) {
  const base = new URL(asString(baseUrl).trim() || DEFAULT_EVENTUS_PORTAL_URL);
  const value = asString(rawUrl).trim();
  let resolved = null;

  try {
    if (!value) {
      resolved = new URL(base.toString());
    } else if (/^https?:\/\//i.test(value)) {
      resolved = new URL(value);
    } else if (value.startsWith("?") || value.startsWith("#")) {
      resolved = new URL(base.toString());
      if (value.startsWith("?")) {
        resolved.search = value;
      } else {
        resolved.hash = value;
      }
    } else {
      resolved = new URL(value.startsWith("/") ? value : `/${value}`, base);
    }
  } catch {
    throw new Error(`Ungueltige Browser-URL: ${trimLine(value || baseUrl, 180)}`);
  }

  if (!/^https?:$/i.test(resolved.protocol)) {
    throw new Error(`Nicht unterstuetztes URL-Schema: ${resolved.protocol}`);
  }
  if (!isAllowedHost(resolved.hostname, allowedHosts)) {
    throw new Error(`Browser-Ziel ausserhalb der erlaubten Hosts: ${resolved.hostname}`);
  }
  return resolved.toString();
}

function buildTabPatterns(allowedHosts = DEFAULT_EVENTUS_ALLOWED_HOSTS) {
  return (Array.isArray(allowedHosts) ? allowedHosts : [])
    .map((host) => normalizeUrlHost(host))
    .filter(Boolean)
    .flatMap((host) => [`https://${host}/*`, `https://*.${host}/*`, `http://${host}/*`, `http://*.${host}/*`]);
}

function isAttachableTab(tab) {
  const tabId = Number(tab?.id || 0);
  const url = asString(tab?.url || "");
  return Number.isFinite(tabId) && tabId > 0 && /^https?:/i.test(url);
}

function pTabsGet(tabId) {
  return new Promise((resolve, reject) => {
    chrome.tabs.get(Number(tabId), (tab) => {
      const err = chrome.runtime.lastError;
      if (err) reject(new Error(err.message));
      else resolve(tab);
    });
  });
}

function pTabsQuery(queryInfo) {
  return new Promise((resolve, reject) => {
    chrome.tabs.query(queryInfo, (tabs) => {
      const err = chrome.runtime.lastError;
      if (err) reject(new Error(err.message));
      else resolve(tabs || []);
    });
  });
}

function pTabsUpdate(tabId, updateProps) {
  return new Promise((resolve, reject) => {
    chrome.tabs.update(Number(tabId), updateProps, (tab) => {
      const err = chrome.runtime.lastError;
      if (err) reject(new Error(err.message));
      else resolve(tab);
    });
  });
}

async function waitForRestrictedTabReady(tabId, timeoutMs = RESTRICTED_BROWSER_TAB_READY_TIMEOUT_MS) {
  const numericTabId = Number(tabId || 0);
  if (!Number.isFinite(numericTabId) || numericTabId <= 0) return null;

  const deadline = Date.now() + Math.max(1_000, Math.floor(Number(timeoutMs) || RESTRICTED_BROWSER_TAB_READY_TIMEOUT_MS));
  let lastTab = await pTabsGet(numericTabId).catch(() => null);
  let lastAttachableUrl = "";
  let attachableSince = 0;
  while (Date.now() < deadline) {
    const tab = await pTabsGet(numericTabId).catch(() => null);
    if (tab) lastTab = tab;
    const status = asString(tab?.status || "").toLowerCase();
    if (isAttachableTab(tab)) {
      const attachableUrl = asString(tab?.url || "");
      if (attachableUrl !== lastAttachableUrl) {
        lastAttachableUrl = attachableUrl;
        attachableSince = Date.now();
      }
      if (!status || status === "complete" || Date.now() - attachableSince >= ATTACHABLE_TAB_STABLE_READY_MS) {
        await new Promise((resolve) => setTimeout(resolve, 80));
        return await pTabsGet(numericTabId).catch(() => tab);
      }
    } else {
      lastAttachableUrl = "";
      attachableSince = 0;
    }
    await new Promise((resolve) => setTimeout(resolve, 120));
  }
  return lastTab;
}

function pickBestRestrictedTab(tabs = [], allowedHosts = DEFAULT_EVENTUS_ALLOWED_HOSTS) {
  return (Array.isArray(tabs) ? tabs : [])
    .filter((tab) => isAttachableTab(tab))
    .filter((tab) => {
      try {
        return isAllowedHost(new URL(asString(tab.url)).hostname, allowedHosts);
      } catch {
        return false;
      }
    })
    .sort((left, right) => {
      const leftScore = Number(left?.active ? 10_000_000_000 : 0) + Number(left?.lastAccessed || 0);
      const rightScore = Number(right?.active ? 10_000_000_000 : 0) + Number(right?.lastAccessed || 0);
      return rightScore - leftScore;
    })[0] || null;
}

function normalizeInspectionTarget(target = {}) {
  const raw = target && typeof target === "object" ? target : {};
  const bboxCss = raw?.bbox_css && typeof raw.bbox_css === "object"
    ? {
        x: asInt(raw.bbox_css.x, 0),
        y: asInt(raw.bbox_css.y, 0),
        width: asInt(raw.bbox_css.width, 0),
        height: asInt(raw.bbox_css.height, 0),
      }
    : null;
  const bboxImg = raw?.bbox_img && typeof raw.bbox_img === "object"
    ? {
        x: asInt(raw.bbox_img.x, 0),
        y: asInt(raw.bbox_img.y, 0),
        width: asInt(raw.bbox_img.width, 0),
        height: asInt(raw.bbox_img.height, 0),
      }
    : null;
  return {
    selector_hint: trimLine(raw?.selector_hint || "", 180),
    text: trimLine(raw?.text || "", 220),
    role: trimLine(raw?.role || "", 80),
    action: safeChoice(raw?.action, new Set(["click", "type", "scroll", "wait", "ignore"]), "ignore"),
    reason: trimLine(raw?.reason || "", 220),
    confidence: Math.max(0, Math.min(1, Number(raw?.confidence || 0) || 0)),
    x_css: Number.isFinite(Number(raw?.x_css)) ? Math.round(Number(raw.x_css)) : null,
    y_css: Number.isFinite(Number(raw?.y_css)) ? Math.round(Number(raw.y_css)) : null,
    bbox_css: bboxCss,
    x_img: Number.isFinite(Number(raw?.x_img)) ? Math.round(Number(raw.x_img)) : null,
    y_img: Number.isFinite(Number(raw?.y_img)) ? Math.round(Number(raw.y_img)) : null,
    bbox_img: bboxImg,
  };
}

function buildVisionAgentRef(modelRef, {
  imageDetail = DEFAULT_BROWSER_VISION_IMAGE_DETAIL,
} = {}) {
  const fallbackRef = asString(DEFAULT_BROWSER_VISION_MODEL_REF).trim() || "openai>gpt-5.4";
  const sourceRef = asString(modelRef || fallbackRef).trim() || fallbackRef;
  const fallbackSplitIndex = fallbackRef.indexOf(">");
  const fallbackProviderId = normalizeVisionProviderId(
    fallbackSplitIndex >= 0 ? fallbackRef.slice(0, fallbackSplitIndex) : fallbackRef,
    "openai",
  );
  const fallbackModel =
    fallbackSplitIndex >= 0
      ? asString(fallbackRef.slice(fallbackSplitIndex + 1)).trim() || "gpt-5.4"
      : "gpt-5.4";
  const splitIndex = sourceRef.indexOf(">");
  const providerRaw = splitIndex >= 0 ? sourceRef.slice(0, splitIndex) : sourceRef;
  const modelRaw = splitIndex >= 0 ? sourceRef.slice(splitIndex + 1) : "";
  const providerId = normalizeVisionProviderId(providerRaw, fallbackProviderId);
  const model = asString(modelRaw).trim() || (providerId === "local_qwen" ? "" : fallbackModel);
  const params = {};
  const openAiCompatibleProviders = new Set(["openai", "custom_openai", "openrouter", "azure_openai"]);
  if (openAiCompatibleProviders.has(providerId.toLowerCase())) {
    params.store = false;
    params.textVerbosity = "low";
  }
  return {
    providerId,
    model,
    params,
    imageDetail: safeChoice(
      imageDetail,
      ALLOWED_BROWSER_IMAGE_DETAILS,
      DEFAULT_BROWSER_VISION_IMAGE_DETAIL,
    ),
  };
}

export async function askBrowserVisionInspector({
  visionModelRef = DEFAULT_BROWSER_VISION_MODEL_REF,
  visionBaseUrl = "",
  visionApiKey = "",
  imageDetail = DEFAULT_BROWSER_VISION_IMAGE_DETAIL,
  question,
  screenshotDataUrl,
  pageContext,
  imageMeta,
  signal,
} = {}) {
  const visionAgent = buildVisionAgentRef(visionModelRef, { imageDetail });
  const viewportW = Number(pageContext?.viewport?.w || 0);
  const viewportH = Number(pageContext?.viewport?.h || 0);
  const viewportDpr = Number(pageContext?.viewport?.dpr || 1);
  const imageW = Number(imageMeta?.width || 0);
  const imageH = Number(imageMeta?.height || 0);

  const prompt = [
    "You are a visual browser inspector.",
    "Analyze the currently VISIBLE viewport for a browser automation run.",
    "Reply ONLY as JSON with the following schema:",
    '{"answer":"string","completion_assessment":{"status":"not_started|in_progress|blocked|complete","reason":"string"},"observations":["string"],"ui_targets":[{"selector_hint":"string","text":"string","role":"string","action":"click|type|scroll|wait|ignore","reason":"string","confidence":0,"x_css":0,"y_css":0,"bbox_css":{"x":0,"y":0,"width":0,"height":0},"x_img":0,"y_img":0,"bbox_img":{"x":0,"y":0,"width":0,"height":0}}],"blockers":["string"],"qa_checks":[{"name":"string","status":"pass|fail|unknown","evidence":"string"}],"next_actions":["string"]}',
    "Coordinate rule: x_css/y_css are ALWAYS CSS pixels relative to the visible viewport.",
    "Optionally also provide x_img/y_img + bbox_img in screenshot pixels.",
    "Return only UI targets that are visible and realistically usable for the next step.",
    "If the task already appears fulfilled in the visible state, set completion_assessment.status='complete' and provide suitable qa_checks.",
    "Name overlay, modal, cookie, or login blockers explicitly in blockers.",
    "No Markdown. No prose outside JSON.",
    `Question: ${asString(question || "Describe the visible UI state and relevant blockers.")}`,
    `Page context: ${asString(pageContext?.title || "")} | ${asString(pageContext?.url || "")}`,
    `Viewport (CSS): width=${viewportW}, height=${viewportH}, dpr=${viewportDpr}`,
    `Screenshot (Pixel): width=${imageW}, height=${imageH}`,
  ].join("\n");

  const providerId = asString(visionAgent.providerId).trim().toLowerCase();
  let raw = "";
  if (OPENAI_COMPATIBLE_VISION_PROVIDER_IDS.has(providerId) && normalizeBaseUrl(visionBaseUrl)) {
    raw = await callOpenAiCompatibleVisionInspector({
      providerId,
      model: visionAgent.model,
      baseUrl: visionBaseUrl,
      apiKey: visionApiKey,
      prompt,
      screenshotDataUrl,
      imageDetail: visionAgent.imageDetail,
      signal,
    });
  } else {
    const response = await llmChat({
      slotId: "vision",
      modelRef: visionAgent.model
        ? `${visionAgent.providerId}>${visionAgent.model}`
        : visionAgent.providerId,
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: prompt },
            {
              type: "image",
              image: screenshotDataUrl,
              providerOptions: {
                openai: {
                  imageDetail: visionAgent.imageDetail,
                },
              },
            },
          ],
        },
      ],
      parameters: visionAgent.params,
      signal,
    });
    raw = asString(response?.text || "");
  }

  const parsed = parseJsonLoose(raw);
  if (!parsed || typeof parsed !== "object") {
    return {
      ok: false,
      error: "Vision output not valid JSON.",
      raw: trimLine(raw, 1000),
    };
  }

  return {
    ok: true,
    data: {
      answer: asString(parsed.answer || ""),
      completion_assessment: parsed?.completion_assessment && typeof parsed.completion_assessment === "object"
        ? {
            status: safeChoice(
              parsed.completion_assessment.status,
              new Set(["not_started", "in_progress", "blocked", "complete"]),
              "in_progress",
            ),
            reason: asString(parsed.completion_assessment.reason || ""),
          }
        : {
            status: "in_progress",
            reason: "",
          },
      observations: Array.isArray(parsed.observations) ? parsed.observations.map((item) => trimLine(item, 280)) : [],
      ui_targets: Array.isArray(parsed.ui_targets)
        ? parsed.ui_targets.map((item) => normalizeInspectionTarget(item)).slice(0, 12)
        : [],
      blockers: Array.isArray(parsed.blockers) ? parsed.blockers.map((item) => trimLine(item, 280)) : [],
      qa_checks: Array.isArray(parsed.qa_checks)
        ? parsed.qa_checks.slice(0, 12).map((check) => ({
            name: trimLine(check?.name || "", 120),
            status: safeChoice(check?.status, new Set(["pass", "fail", "unknown"]), "unknown"),
            evidence: trimLine(check?.evidence || "", 280),
          }))
        : [],
      next_actions: Array.isArray(parsed.next_actions) ? parsed.next_actions.map((item) => trimLine(item, 220)).slice(0, 12) : [],
    },
  };
}

export async function ensureRestrictedBrowserTab(runtime = {}, {
  url = "",
  baseUrl = DEFAULT_EVENTUS_PORTAL_URL,
  allowedHosts = DEFAULT_EVENTUS_ALLOWED_HOSTS,
  active = false,
} = {}) {
  const state = runtime && typeof runtime === "object" ? runtime : {};
  const requestedUrl = asString(url).trim();
  const targetUrl = resolveRestrictedBrowserUrl(requestedUrl || baseUrl, { baseUrl, allowedHosts });
  const hasExplicitTarget = !!requestedUrl;

  let tab = null;
  const currentTabId = Number(state.currentTabId || 0);
  if (Number.isFinite(currentTabId) && currentTabId > 0) {
    tab = await pTabsGet(currentTabId).catch(() => null);
    const currentAllowed = (() => {
      try {
        return isAllowedHost(new URL(asString(tab?.url || "")).hostname, allowedHosts);
      } catch {
        return false;
      }
    })();
    if (!isAttachableTab(tab) || !currentAllowed) {
      tab = null;
    }
  }

  if (!tab) {
    const patterns = buildTabPatterns(allowedHosts);
    const candidates = await pTabsQuery(patterns.length ? { url: patterns } : {}).catch(() => []);
    tab = pickBestRestrictedTab(candidates, allowedHosts);
  }

  let opened = false;
  let navigated = false;
  if (!tab) {
    const created = await toolOpenUrl(targetUrl, { active });
    tab = await pTabsGet(Number(created?.id || 0)).catch(() => created);
    opened = true;
  }

  const numericTabId = Number(tab?.id || 0);
  if (!Number.isFinite(numericTabId) || numericTabId <= 0) {
    throw new Error("Could not open a browser tab for the portal.");
  }

  const tabUrlOrPending = asString(tab?.url || tab?.pendingUrl || "");

  if (hasExplicitTarget && !sameUrlIgnoringHash(tabUrlOrPending, targetUrl)) {
    const nav = await toolNavigate(numericTabId, targetUrl, 0);
    if (!nav?.ok) {
      throw new Error(asString(nav?.error || "Navigation fehlgeschlagen."));
    }
    tab = await pTabsGet(numericTabId).catch(() => tab);
    navigated = true;
  } else if (active && !tab?.active) {
    tab = await pTabsUpdate(numericTabId, { active: true }).catch(() => tab);
  }

  tab = await waitForRestrictedTabReady(numericTabId).catch(() => null) || await pTabsGet(numericTabId).catch(() => tab) || tab;

  state.currentTabId = numericTabId;
  return {
    tabId: numericTabId,
    url: asString(tab?.url || targetUrl),
    title: asString(tab?.title || ""),
    active: !!tab?.active,
    opened,
    navigated,
    pageState: await toolCollectPageState(numericTabId).catch(() => null),
  };
}

export async function runRestrictedBrowserInspection(runtime = {}, {
  url = "",
  question = "",
  active = false,
  baseUrl = DEFAULT_EVENTUS_PORTAL_URL,
  allowedHosts = DEFAULT_EVENTUS_ALLOWED_HOSTS,
  visionModelRef = DEFAULT_BROWSER_VISION_MODEL_REF,
  visionBaseUrl = "",
  visionApiKey = "",
  imageDetail = DEFAULT_BROWSER_VISION_IMAGE_DETAIL,
  signal,
} = {}) {
  const tab = await ensureRestrictedBrowserTab(runtime, { url, active, baseUrl, allowedHosts });
  const observed = await toolObserve(tab.tabId).catch((error) => ({
    ok: false,
    error: asString(error?.message || error),
    error_stack: asString(error?.stack || ""),
  }));
  if (!observed?.ok) {
    return {
      ok: false,
      error: asString(observed?.error || "observe failed"),
      errorDetail: {
        backend: "vision_inspect",
        phase: "observe",
        finalUrl: tab.url,
        pageState: tab.pageState || null,
      },
      errorStack: asString(observed?.error_stack || ""),
      data: {
        tabId: tab.tabId,
        url: tab.url,
        title: tab.title,
        page_state: tab.pageState,
      },
    };
  }

  let vision;
  try {
    vision = await askBrowserVisionInspector({
      visionModelRef,
      visionBaseUrl,
      visionApiKey,
      imageDetail,
      question,
      screenshotDataUrl: observed.screenshot_data_url,
      pageContext: observed.page || {},
      imageMeta: observed.image || {},
      signal,
    });
  } catch (error) {
    return {
      ok: false,
      error: asString(error?.message || error || "vision failed"),
      errorDetail: {
        backend: "vision_inspect",
        phase: "vision",
        finalUrl: asString(observed?.page?.url || tab.url),
        pageState: observed.page || tab.pageState,
        image: observed.image || null,
        screenshotMode: asString(observed?.screenshot_mode || ""),
      },
      errorStack: asString(error?.stack || ""),
      data: {
        tabId: tab.tabId,
        url: asString(observed?.page?.url || tab.url),
        title: asString(observed?.page?.title || tab.title),
        page_state: observed.page || tab.pageState,
        image: observed.image || null,
        screenshot_mode: asString(observed?.screenshot_mode || ""),
      },
    };
  }

  if (!vision?.ok) {
    return {
      ok: false,
      error: asString(vision?.error || "vision failed"),
      errorDetail: {
        backend: "vision_inspect",
        phase: "vision",
        finalUrl: asString(observed?.page?.url || tab.url),
        pageState: observed.page || tab.pageState,
        image: observed.image || null,
        screenshotMode: asString(observed?.screenshot_mode || ""),
        visionRaw: trimLine(vision?.raw || "", 1200),
      },
      errorStack: asString(vision?.errorStack || vision?.error_stack || ""),
      data: {
        tabId: tab.tabId,
        url: asString(observed?.page?.url || tab.url),
        title: asString(observed?.page?.title || tab.title),
        page_state: observed.page || tab.pageState,
        image: observed.image || null,
        screenshot_mode: asString(observed?.screenshot_mode || ""),
        vision_raw: trimLine(vision?.raw || "", 1200),
      },
    };
  }

  return {
    ok: true,
    error: "",
    data: {
      tabId: tab.tabId,
      url: asString(observed?.page?.url || tab.url),
      title: asString(observed?.page?.title || tab.title),
      page_state: observed.page || tab.pageState,
      image: observed.image || null,
      screenshot_mode: asString(observed?.screenshot_mode || ""),
      answer: asString(vision?.data?.answer || ""),
      completion_assessment: vision?.data?.completion_assessment || null,
      observations: Array.isArray(vision?.data?.observations) ? vision.data.observations : [],
      blockers: Array.isArray(vision?.data?.blockers) ? vision.data.blockers : [],
      qa_checks: Array.isArray(vision?.data?.qa_checks) ? vision.data.qa_checks : [],
      next_actions: Array.isArray(vision?.data?.next_actions) ? vision.data.next_actions : [],
      ui_targets: Array.isArray(vision?.data?.ui_targets) ? vision.data.ui_targets : [],
    },
  };
}

export async function runRestrictedBrowserScript(runtime = {}, {
  url = "",
  code = "",
  timeoutMs = 45_000,
  active = false,
  baseUrl = DEFAULT_EVENTUS_PORTAL_URL,
  allowedHosts = DEFAULT_EVENTUS_ALLOWED_HOSTS,
} = {}) {
  const tab = await ensureRestrictedBrowserTab(runtime, { url, active, baseUrl, allowedHosts });
  const result = await toolRunPlaywrightCode(tab.tabId, {
    code: asString(code),
    timeoutMs: asInt(timeoutMs, 45_000, 500, 180_000),
  });

  const rawOutput = result?.result ?? result?.output ?? result ?? null;
  const clippedRaw = clipStructuredValue(rawOutput, 12_000);
  const pageState = result?.pageState && typeof result.pageState === "object"
    ? result.pageState
    : await toolCollectPageState(tab.tabId).catch(() => tab.pageState);

  return {
    ok: !!result?.ok,
    error: asString(result?.error || ""),
    errorStack: asString(result?.error_stack || ""),
    errorDetail: !result?.ok
      ? {
          backend: asString(result?.mode || "cdp_runtime_evaluate"),
          errorCode: asString(result?.error_code || ""),
          finalUrl: asString(result?.finalUrl || pageState?.url || tab.url),
          pageState: pageState || null,
          rawPreview: clippedRaw.preview,
          rawClipped: !!clippedRaw.clipped,
        }
      : null,
    data: {
      tabId: tab.tabId,
      url: asString(result?.finalUrl || pageState?.url || tab.url),
      title: asString(pageState?.title || tab.title),
      error_code: asString(result?.error_code || ""),
      raw: clippedRaw.value,
      raw_clipped: !!clippedRaw.clipped,
      raw_preview: clippedRaw.preview,
      page_state: pageState,
    },
  };
}

export async function runRestrictedBrowserAction(runtime = {}, {
  url = "",
  active = false,
  baseUrl = DEFAULT_EVENTUS_PORTAL_URL,
  allowedHosts = DEFAULT_EVENTUS_ALLOWED_HOSTS,
  action = "click",
  target = null,
  destination = null,
  deltaX = 0,
  deltaY = 0,
  textValue = "",
  keys = "",
  clear = false,
  waitMs = null,
  timeoutMs = 12_000,
  button = "left",
  steps = 20,
} = {}) {
  const tab = await ensureRestrictedBrowserTab(runtime, { url, active, baseUrl, allowedHosts });
  const result = await toolRunBrowserNativeAction(tab.tabId, {
    action,
    target,
    destination,
    deltaX,
    deltaY,
    textValue,
    keys,
    clear,
    waitMs,
    timeoutMs: asInt(timeoutMs, 12_000, 250, 60_000),
    button: asString(button || "left"),
    steps: asInt(steps, 20, 1, 60),
  });

  const rawResult = result?.result && typeof result.result === "object" ? { ...result.result } : {};
  const pageState = rawResult?.pageState && typeof rawResult.pageState === "object"
    ? rawResult.pageState
    : await toolCollectPageState(tab.tabId).catch(() => tab.pageState);
  delete rawResult.pageState;

  return {
    ok: !!result?.ok,
    error: asString(result?.error || ""),
    errorStack: asString(result?.error_stack || ""),
    errorDetail: !result?.ok
      ? {
          backend: asString(result?.mode || "crx_playwright_native"),
          action: asString(action || ""),
          errorCode: asString(result?.error_code || ""),
          finalUrl: asString(rawResult?.url || pageState?.url || tab.url),
          pageState: pageState || null,
        }
      : null,
    data: {
      tabId: tab.tabId,
      url: asString(rawResult?.url || pageState?.url || tab.url),
      title: asString(rawResult?.title || pageState?.title || tab.title),
      action: asString(action || ""),
      result: rawResult,
      page_state: pageState,
    },
  };
}
