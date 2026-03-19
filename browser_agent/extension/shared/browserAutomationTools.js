import { crx } from "../vendor/playwright-crx-client.mjs";

const PLAYWRIGHT_NAV_TIMEOUT_MS = 45_000;
const ATTACHABLE_TAB_STABLE_READY_MS = 1_200;
const CODE_TIMEOUT_MIN_MS = 500;
const CODE_TIMEOUT_MAX_MS = 180_000;
const CRX_PLAYWRIGHT_INIT_TIMEOUT_MS = 12_000;
const CRX_PLAYWRIGHT_ATTACH_TIMEOUT_MS = 15_000;
const CRX_NATIVE_ACTION_TIMEOUT_MS = 12_000;
const CRX_NATIVE_TYPE_DELAY_MS = 28;

let crxAppPromise = null;

function asString(v) {
  return String(v == null ? "" : v);
}

function asInt(v, fallback = 0) {
  const n = Number(v);
  if (!Number.isFinite(n)) return fallback;
  return Math.floor(n);
}

function clamp(n, min, max) {
  return Math.max(min, Math.min(max, n));
}

function isLikelyPlaywrightFunctionSource(source = "") {
  const text = asString(source).trim();
  if (!text) return false;
  if (/^(?:async\s+)?function\b/.test(text)) return true;
  return /^(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>/.test(text);
}

function buildPlaywrightUserCodeRunnerSource(sourceCode = "") {
  const source = asString(sourceCode).trim();
  if (!source) return "";
  if (!isLikelyPlaywrightFunctionSource(source)) return source;
  const serializedSource = JSON.stringify(source);
  return [
    `const __playwrightUserSource = ${serializedSource};`,
    'const __playwrightUserValue = eval("(" + __playwrightUserSource + ")");',
    "const __playwrightPageArg = Object.assign({}, page, { page });",
    'if (typeof __playwrightUserValue === "function") {',
    "  return await __playwrightUserValue.call(page, __playwrightPageArg, { page, runtime: { page } });",
    "}",
    "return await __playwrightUserValue;",
  ].join("\n");
}

function withTimeout(promise, timeoutMs, label) {
  const ms = Math.max(300, asInt(timeoutMs, 0));
  return Promise.race([
    promise,
    new Promise((_, reject) =>
      setTimeout(() => reject(new Error(`${label} timeout after ${ms}ms`)), ms),
    ),
  ]);
}

function isAttachableBrowserTab(tab) {
  const tabId = asInt(tab?.id, 0);
  const url = asString(tab?.url || "");
  return Number.isFinite(tabId) && tabId > 0 && /^https?:/i.test(url);
}

function pTabsCreate(url, updateProps = {}) {
  return new Promise((resolve, reject) => {
    chrome.tabs.create({ url, ...updateProps }, (tab) => {
      const err = chrome.runtime.lastError;
      if (err) reject(new Error(err.message));
      else resolve(tab);
    });
  });
}

function pTabsGet(tabId) {
  return new Promise((resolve, reject) => {
    chrome.tabs.get(tabId, (tab) => {
      const err = chrome.runtime.lastError;
      if (err) reject(new Error(err.message));
      else resolve(tab);
    });
  });
}

function pTabsUpdate(tabId, updateProps) {
  return new Promise((resolve, reject) => {
    chrome.tabs.update(tabId, updateProps, (tab) => {
      const err = chrome.runtime.lastError;
      if (err) reject(new Error(err.message));
      else resolve(tab);
    });
  });
}

function pExecuteScript({ tabId, world = "MAIN", func, args = [] }) {
  return new Promise((resolve, reject) => {
    chrome.scripting.executeScript(
      {
        target: { tabId, allFrames: false },
        world,
        func,
        args,
      },
      (results) => {
        const err = chrome.runtime.lastError;
        if (err) reject(new Error(err.message));
        else resolve(results || []);
      },
    );
  });
}

function pDebuggerAttach(tabId, version = "1.3") {
  return new Promise((resolve, reject) => {
    chrome.debugger.attach({ tabId: Number(tabId) }, version, () => {
      const err = chrome.runtime.lastError;
      if (err) reject(new Error(err.message));
      else resolve(true);
    });
  });
}

function pDebuggerDetach(tabId) {
  return new Promise((resolve, reject) => {
    chrome.debugger.detach({ tabId: Number(tabId) }, () => {
      const err = chrome.runtime.lastError;
      if (err) reject(new Error(err.message));
      else resolve(true);
    });
  });
}

function pDebuggerSendCommand(tabId, method, params = {}) {
  return new Promise((resolve, reject) => {
    chrome.debugger.sendCommand({ tabId: Number(tabId) }, method, params, (result) => {
      const err = chrome.runtime.lastError;
      if (err) reject(new Error(err.message));
      else resolve(result || {});
    });
  });
}

function ensureDebuggerApi() {
  if (!chrome?.debugger?.onEvent || typeof chrome.debugger.onEvent.addListener !== "function") {
    throw new Error("chrome.debugger API unavailable.");
  }
}

async function getCrxApp() {
  ensureDebuggerApi();
  if (crxAppPromise) return crxAppPromise;
  crxAppPromise = (async () => {
    try {
      try {
        await withTimeout(crx.start({}), CRX_PLAYWRIGHT_INIT_TIMEOUT_MS, "crx.start");
      } catch {
        // Bridge may already be active.
      }
      return await withTimeout(
        crx.get({ incognito: false }),
        CRX_PLAYWRIGHT_INIT_TIMEOUT_MS,
        "crx.get",
      );
    } catch (err) {
      crxAppPromise = null;
      throw err;
    }
  })();
  return crxAppPromise;
}

async function withCrxPage(tabId, fn, {
  attachTimeoutMs = CRX_PLAYWRIGHT_ATTACH_TIMEOUT_MS,
} = {}) {
  const numericTabId = asInt(tabId, 0);
  if (!Number.isFinite(numericTabId) || numericTabId <= 0) throw new Error("Invalid tab id.");
  const app = await getCrxApp();
  const page = await withTimeout(
    app.attach(numericTabId),
    attachTimeoutMs,
    `crx.attach(${numericTabId})`,
  );
  try {
    return await fn(page, app);
  } finally {
    try {
      await app.detach(numericTabId);
    } catch {}
  }
}

async function cdpCaptureScreenshotDataUrl(tabId) {
  const numericTabId = asInt(tabId, 0);
  if (!Number.isFinite(numericTabId) || numericTabId <= 0) throw new Error("Invalid tab id.");
  let attached = false;
  try {
    try {
      await pDebuggerAttach(numericTabId, "1.3");
      attached = true;
    } catch (attachErr) {
      const msg = asString(attachErr?.message || attachErr).toLowerCase();
      const alreadyAttached =
        msg.includes("already attached") ||
        msg.includes("another debugger is already attached") ||
        msg.includes("target is already attached");
      if (!alreadyAttached) throw attachErr;
      attached = false;
    }
    await pDebuggerSendCommand(numericTabId, "Page.enable", {});
    const shot = await pDebuggerSendCommand(numericTabId, "Page.captureScreenshot", {
      format: "png",
      fromSurface: true,
      // browser_inspect validates the visible viewport, not a stitched full-page image
      captureBeyondViewport: false,
      optimizeForSpeed: false,
    });
    const base64 = asString(shot?.data || "").trim();
    if (!base64) throw new Error("Page.captureScreenshot returned empty data.");
    return `data:image/png;base64,${base64}`;
  } finally {
    if (attached) {
      try { await pDebuggerDetach(numericTabId); } catch {}
    }
  }
}

async function waitForTabReady(tabId, timeoutMs = PLAYWRIGHT_NAV_TIMEOUT_MS) {
  const deadline = Date.now() + Math.max(1000, Math.floor(Number(timeoutMs) || PLAYWRIGHT_NAV_TIMEOUT_MS));
  let lastAttachableUrl = "";
  let attachableSince = 0;
  while (Date.now() < deadline) {
    const tab = await pTabsGet(Number(tabId));
    const status = asString(tab?.status || "").toLowerCase();
    if (isAttachableBrowserTab(tab)) {
      const attachableUrl = asString(tab?.url || "");
      if (attachableUrl !== lastAttachableUrl) {
        lastAttachableUrl = attachableUrl;
        attachableSince = Date.now();
      }
      if (!status || status === "complete") return tab;
      if (Date.now() - attachableSince >= ATTACHABLE_TAB_STABLE_READY_MS) return tab;
    } else {
      lastAttachableUrl = "";
      attachableSince = 0;
    }
    await new Promise((r) => setTimeout(r, 120));
  }
  throw new Error("Navigation timeout while waiting for tab ready state.");
}

function pngSizeFromDataUrl(dataUrl) {
  const base64 = asString(dataUrl).split(",")[1] || "";
  if (!base64) return { width: 0, height: 0, format: "png" };
  const bin = atob(base64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i += 1) bytes[i] = bin.charCodeAt(i);
  const w = (bytes[16] << 24) | (bytes[17] << 16) | (bytes[18] << 8) | bytes[19];
  const h = (bytes[20] << 24) | (bytes[21] << 16) | (bytes[22] << 8) | bytes[23];
  return { width: w >>> 0, height: h >>> 0, format: "png" };
}

function bytesToBase64(bytesLike) {
  const bytes = bytesLike instanceof Uint8Array
    ? bytesLike
    : bytesLike instanceof ArrayBuffer
      ? new Uint8Array(bytesLike)
      : new Uint8Array(bytesLike || []);
  if (!bytes.byteLength) return "";
  let binary = "";
  const chunkSize = 0x8000;
  for (let i = 0; i < bytes.length; i += chunkSize) {
    const chunk = bytes.subarray(i, Math.min(bytes.length, i + chunkSize));
    let chunkBinary = "";
    for (let j = 0; j < chunk.length; j += 1) {
      chunkBinary += String.fromCharCode(chunk[j] & 0xff);
    }
    binary += chunkBinary;
  }
  return btoa(binary);
}

async function crxCaptureScreenshotDataUrl(tabId) {
  const numericTabId = asInt(tabId, 0);
  if (!Number.isFinite(numericTabId) || numericTabId <= 0) throw new Error("Invalid tab id.");
  return await withCrxPage(numericTabId, async (page) => {
    const binary = await page.screenshot({
      type: "png",
      animations: "disabled",
      caret: "hide",
      scale: "css",
    });
    const base64 = bytesToBase64(binary);
    if (!base64) throw new Error("CRX screenshot returned empty data.");
    return `data:image/png;base64,${base64}`;
  });
}

function hasValue(v) {
  return v !== undefined && v !== null;
}

function normalizeNativeTarget(target = {}) {
  const raw = target && typeof target === "object" ? target : {};
  const frameSelectors = Array.isArray(raw.frameSelectors)
    ? raw.frameSelectors.map((v) => asString(v).trim()).filter(Boolean).slice(0, 6)
    : [];
  const x = Number(raw.x);
  const y = Number(raw.y);
  const imageWidth = Number(raw.imageWidth);
  const imageHeight = Number(raw.imageHeight);
  return {
    selector: asString(raw.selector).trim(),
    text: asString(raw.text).trim(),
    role: asString(raw.role).trim(),
    name: hasValue(raw.name) ? asString(raw.name).trim() : "",
    exact: !!raw.exact,
    frameSelectors,
    x: Number.isFinite(x) ? x : null,
    y: Number.isFinite(y) ? y : null,
    coordSpace: asString(raw.coordSpace || "viewport_css").trim().toLowerCase() === "image_px"
      ? "image_px"
      : "viewport_css",
    imageWidth: Number.isFinite(imageWidth) ? imageWidth : null,
    imageHeight: Number.isFinite(imageHeight) ? imageHeight : null,
  };
}

function nativeTargetHasLocator(target) {
  return !!(target?.selector || target?.text || target?.role);
}

function nativeTargetHasCoordinates(target) {
  return Number.isFinite(target?.x) && Number.isFinite(target?.y);
}

async function getCrxViewportSize(page) {
  try {
    const viewport = await page.viewportSize();
    const width = Number(viewport?.width || 0);
    const height = Number(viewport?.height || 0);
    if (width > 0 && height > 0) return { width, height };
  } catch {}
  const fallback = await page.evaluate(() => ({
    width: Number(window.innerWidth || 0),
    height: Number(window.innerHeight || 0),
  })).catch(() => ({ width: 0, height: 0 }));
  return {
    width: Math.max(0, asInt(fallback?.width, 0)),
    height: Math.max(0, asInt(fallback?.height, 0)),
  };
}

function resolveViewportCoordinates({
  x,
  y,
  coordSpace = "viewport_css",
  imageWidth = null,
  imageHeight = null,
  viewportWidth,
  viewportHeight,
  actionLabel = "browser action",
} = {}) {
  const inputX = Number(x);
  const inputY = Number(y);
  if (!Number.isFinite(inputX) || !Number.isFinite(inputY) || inputX < 0 || inputY < 0) {
    throw new Error(`${actionLabel}: x/y fehlen oder sind ungültig.`);
  }
  const width = Number(viewportWidth);
  const height = Number(viewportHeight);
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
    throw new Error(`${actionLabel}: viewport size unavailable.`);
  }

  const space = asString(coordSpace || "viewport_css").toLowerCase() === "image_px"
    ? "image_px"
    : "viewport_css";
  let resolvedX = inputX;
  let resolvedY = inputY;

  if (space === "image_px") {
    const imgW = Number(imageWidth);
    const imgH = Number(imageHeight);
    if (!Number.isFinite(imgW) || !Number.isFinite(imgH) || imgW <= 0 || imgH <= 0) {
      throw new Error(`${actionLabel}: imageWidth/imageHeight fehlen für coordSpace='image_px'.`);
    }
    resolvedX = (inputX * width) / imgW;
    resolvedY = (inputY * height) / imgH;
  }

  if (!Number.isFinite(resolvedX) || !Number.isFinite(resolvedY)) {
    throw new Error(`${actionLabel}: Koordinaten konnten nicht auf den Viewport aufgelöst werden.`);
  }

  const clampedX = clamp(Math.round(resolvedX), 0, Math.max(0, Math.round(width) - 1));
  const clampedY = clamp(Math.round(resolvedY), 0, Math.max(0, Math.round(height) - 1));
  return {
    input: {
      x: inputX,
      y: inputY,
      coordSpace: space,
      imageWidth: Number.isFinite(Number(imageWidth)) ? Number(imageWidth) : null,
      imageHeight: Number.isFinite(Number(imageHeight)) ? Number(imageHeight) : null,
    },
    resolved: {
      x: clampedX,
      y: clampedY,
      viewportWidth: Math.round(width),
      viewportHeight: Math.round(height),
    },
  };
}

function buildNativeLocatorFromContext(ctx, target) {
  if (!ctx || !target) return null;
  if (target.selector) return ctx.locator(target.selector).first();
  if (target.role) {
    const opts = {};
    if (target.name) opts.name = target.name;
    if (hasValue(target.exact)) opts.exact = !!target.exact;
    return ctx.getByRole(target.role, opts).first();
  }
  if (target.text) return ctx.getByText(target.text, { exact: !!target.exact }).first();
  return null;
}

async function probeNativeLocator(locator, timeoutMs) {
  if (!locator) return null;
  try {
    await locator.waitFor({
      state: "attached",
      timeout: clamp(asInt(timeoutMs, CRX_NATIVE_ACTION_TIMEOUT_MS), 250, CRX_NATIVE_ACTION_TIMEOUT_MS),
    });
    const box = await locator.boundingBox().catch(() => null);
    const visible = !!(box && Number(box.width) > 0 && Number(box.height) > 0);
    if (!visible) return null;
    return {
      locator,
      box,
      visible,
    };
  } catch {
    return null;
  }
}

function normalizeFrameInfo(frame) {
  if (!frame) return null;
  let url = "";
  let name = "";
  try { url = asString(frame.url?.() || ""); } catch {}
  try { name = asString(frame.name?.() || ""); } catch {}
  return {
    url,
    name,
  };
}

async function findNativeLocatorTarget(page, target, timeoutMs) {
  if (!nativeTargetHasLocator(target)) return null;

  if (Array.isArray(target.frameSelectors) && target.frameSelectors.length > 0) {
    let ctx = page;
    for (const selector of target.frameSelectors) {
      ctx = ctx.frameLocator(selector);
    }
    const locator = buildNativeLocatorFromContext(ctx, target);
    const probed = await probeNativeLocator(locator, timeoutMs);
    if (!probed) return null;
    return {
      ...probed,
      search: {
        strategy: target.selector ? "selector" : target.role ? "role" : "text",
        frameSelectors: target.frameSelectors,
        frame: null,
        autoFrameScan: false,
      },
    };
  }

  const direct = await probeNativeLocator(buildNativeLocatorFromContext(page, target), timeoutMs);
  if (direct) {
    return {
      ...direct,
      search: {
        strategy: target.selector ? "selector" : target.role ? "role" : "text",
        frameSelectors: [],
        frame: null,
        autoFrameScan: false,
      },
    };
  }

  const frames = Array.isArray(page.frames?.()) ? page.frames() : [];
  for (const frame of frames) {
    try {
      const locator = buildNativeLocatorFromContext(frame, target);
      const probed = await probeNativeLocator(locator, Math.min(timeoutMs, 2_000));
      if (!probed) continue;
      return {
        ...probed,
        search: {
          strategy: target.selector ? "selector" : target.role ? "role" : "text",
          frameSelectors: [],
          frame: normalizeFrameInfo(frame),
          autoFrameScan: true,
        },
      };
    } catch {}
  }
  return null;
}

async function resolveNativeActionTarget(page, target, {
  actionLabel = "browser action",
  timeoutMs = CRX_NATIVE_ACTION_TIMEOUT_MS,
} = {}) {
  const normalizedTarget = normalizeNativeTarget(target);
  const timeout = clamp(asInt(timeoutMs, CRX_NATIVE_ACTION_TIMEOUT_MS), 250, 60_000);

  const locatorHit = await findNativeLocatorTarget(page, normalizedTarget, timeout);
  if (locatorHit) {
    const box = locatorHit.box || {};
    const point = {
      x: Math.round(Number(box.x || 0) + (Number(box.width || 0) / 2)),
      y: Math.round(Number(box.y || 0) + (Number(box.height || 0) / 2)),
    };
    return {
      target: normalizedTarget,
      kind: "locator",
      locator: locatorHit.locator,
      search: locatorHit.search,
      bbox: locatorHit.box
        ? {
            x: Math.round(Number(locatorHit.box.x || 0)),
            y: Math.round(Number(locatorHit.box.y || 0)),
            width: Math.round(Number(locatorHit.box.width || 0)),
            height: Math.round(Number(locatorHit.box.height || 0)),
          }
        : null,
      point,
    };
  }

  if (nativeTargetHasCoordinates(normalizedTarget)) {
    const viewport = await getCrxViewportSize(page);
    const resolved = resolveViewportCoordinates({
      x: normalizedTarget.x,
      y: normalizedTarget.y,
      coordSpace: normalizedTarget.coordSpace,
      imageWidth: normalizedTarget.imageWidth,
      imageHeight: normalizedTarget.imageHeight,
      viewportWidth: viewport.width,
      viewportHeight: viewport.height,
      actionLabel,
    });
    return {
      target: normalizedTarget,
      kind: "coordinates",
      locator: null,
      search: {
        strategy: "coordinates",
        frameSelectors: normalizedTarget.frameSelectors,
        frame: null,
        autoFrameScan: false,
      },
      bbox: null,
      point: {
        x: Number(resolved.resolved.x),
        y: Number(resolved.resolved.y),
      },
      coordinateInput: resolved.input,
      coordinateResolved: resolved.resolved,
    };
  }

  throw new Error(
    `${actionLabel}: Ziel fehlt. Nutze target.selector, target.text, target.role/name oder target.x/target.y.`,
  );
}

async function maybeWaitAfterAction(page, action, waitMs) {
  if (action === "wait") return;
  const raw = Number(waitMs);
  const fallback = action === "move" ? 0 : (action === "wait" ? 0 : 300);
  const ms = clamp(Number.isFinite(raw) ? Math.floor(raw) : fallback, 0, 120_000);
  if (ms <= 0) return;
  await page.waitForTimeout(ms);
}

async function tryClearTypeTarget(page, locator) {
  if (!locator) return;
  await locator.fill("").catch(() => {});
  await page.keyboard.press("Meta+A").catch(async () => {
    await page.keyboard.press("Control+A").catch(() => {});
  });
  await page.keyboard.press("Backspace").catch(() => {});
}

export async function toolOpenUrl(url, { active = true } = {}) {
  const targetUrl = asString(url).trim();
  if (!targetUrl) throw new Error("toolOpenUrl: missing url");
  return await pTabsCreate(targetUrl, { active: !!active });
}

export async function toolNavigate(tabId, url, waitMs = 0) {
  const targetUrl = asString(url).trim();
  if (!targetUrl) return { ok: false, error: "missing url" };
  const numericTabId = asInt(tabId, 0);
  if (!Number.isFinite(numericTabId) || numericTabId <= 0) throw new Error("Invalid tab id.");
  await pTabsUpdate(numericTabId, { url: targetUrl });
  const tab = await waitForTabReady(numericTabId, PLAYWRIGHT_NAV_TIMEOUT_MS);
  const extraWait = clamp(asInt(waitMs, 0), 0, 120_000);
  if (extraWait > 0) await new Promise((r) => setTimeout(r, extraWait));
  return {
    ok: true,
    url: asString(tab?.url || targetUrl),
    status: null,
  };
}

export async function toolCollectPageState(tabId) {
  await pTabsGet(Number(tabId));
  const [ctxRes] = await pExecuteScript({
    tabId: Number(tabId),
    world: "MAIN",
    func: () => {
      const clean = (value, max = 180) => {
        const s = String(value == null ? "" : value).replace(/\s+/g, " ").trim();
        if (s.length <= max) return s;
        return `${s.slice(0, Math.max(1, max - 1)).trimEnd()}…`;
      };
      const round = (value) => {
        const n = Number(value);
        return Number.isFinite(n) ? Math.round(n) : 0;
      };
      const isVisible = (el) => {
        if (!el || !(el instanceof Element)) return false;
        const rect = el.getBoundingClientRect();
        if (!(rect.width > 0 && rect.height > 0)) return false;
        const style = getComputedStyle(el);
        return style.display !== "none" && style.visibility !== "hidden" && style.opacity !== "0";
      };
      const describeElement = (el) => {
        if (!el || !(el instanceof Element)) return null;
        const rect = el.getBoundingClientRect();
        const text = clean(
          el.innerText ||
          el.textContent ||
          el.getAttribute?.("aria-label") ||
          el.getAttribute?.("title") ||
          ("value" in el ? el.value : "") ||
          "",
          180,
        );
        return {
          tag: clean(el.tagName || "", 32).toLowerCase(),
          id: clean(el.id || "", 64),
          role: clean(el.getAttribute?.("role") || "", 40),
          name: clean(el.getAttribute?.("name") || "", 64),
          type: clean(el.getAttribute?.("type") || "", 40),
          placeholder: clean(el.getAttribute?.("placeholder") || "", 80),
          text,
          bbox: {
            x: round(rect.left),
            y: round(rect.top),
            width: round(rect.width),
            height: round(rect.height),
          },
        };
      };
      const unique = (elements) => {
        const seen = new Set();
        const out = [];
        for (const el of elements) {
          if (!el || seen.has(el)) continue;
          seen.add(el);
          out.push(el);
        }
        return out;
      };

      const scrollEl = document.scrollingElement || document.documentElement || document.body;
      const scrollX = round(window.scrollX || scrollEl?.scrollLeft || 0);
      const scrollY = round(window.scrollY || scrollEl?.scrollTop || 0);
      const maxScrollX = Math.max(0, round((scrollEl?.scrollWidth || 0) - (window.innerWidth || 0)));
      const maxScrollY = Math.max(0, round((scrollEl?.scrollHeight || 0) - (window.innerHeight || 0)));
      const progressY = maxScrollY > 0 ? Math.round((scrollY / maxScrollY) * 1000) / 10 : 0;

      const interactiveSelector = [
        "button",
        "a[href]",
        "input",
        "select",
        "textarea",
        "summary",
        "[role='button']",
        "[role='link']",
        "[role='tab']",
        "[role='menuitem']",
        "[role='checkbox']",
        "[role='radio']",
        "[role='option']",
        "[contenteditable='true']",
        "[tabindex]:not([tabindex='-1'])",
      ].join(",");

      const interactivePreview = unique(Array.from(document.querySelectorAll(interactiveSelector)))
        .filter((el) => isVisible(el))
        .slice(0, 14)
        .map((el) => describeElement(el))
        .filter(Boolean);

      const dialogs = Array.from(document.querySelectorAll("dialog,[role='dialog'],[aria-modal='true']"))
        .filter((el) => isVisible(el))
        .slice(0, 6)
        .map((el) => describeElement(el))
        .filter(Boolean);

      const headings = Array.from(document.querySelectorAll("h1,h2,h3"))
        .filter((el) => isVisible(el))
        .slice(0, 6)
        .map((el) => ({
          tag: clean(el.tagName || "", 16).toLowerCase(),
          text: clean(el.innerText || el.textContent || "", 160),
        }));

      const forms = Array.from(document.forms || [])
        .slice(0, 5)
        .map((form, index) => ({
          index,
          id: clean(form.id || "", 64),
          name: clean(form.getAttribute?.("name") || "", 64),
          method: clean(form.getAttribute?.("method") || "", 24),
          action: clean(form.getAttribute?.("action") || "", 180),
          fieldCount: form.querySelectorAll("input,select,textarea").length,
          submitCount: form.querySelectorAll("button[type='submit'],input[type='submit']").length,
        }));

      const focus = describeElement(document.activeElement);

      return {
        url: location.href,
        title: clean(document.title || "", 180),
        readyState: clean(document.readyState || "", 32),
        viewport: {
          w: round(window.innerWidth || 0),
          h: round(window.innerHeight || 0),
          dpr: Number.isFinite(Number(window.devicePixelRatio)) ? Number(window.devicePixelRatio) : 1,
        },
        scroll: {
          x: scrollX,
          y: scrollY,
          maxX: maxScrollX,
          maxY: maxScrollY,
          progressY,
          position: progressY <= 1 ? "top" : (progressY >= 99 ? "bottom" : "middle"),
        },
        focus,
        headings,
        forms,
        dialogs,
        interactivePreview,
        bodyTextPreview: clean(document.body?.innerText || "", 500),
        capturedAt: Date.now(),
      };
    },
  });
  return ctxRes?.result || {};
}

export async function toolObserve(tabId) {
  await waitForTabReady(Number(tabId), PLAYWRIGHT_NAV_TIMEOUT_MS).catch(() => null);
  await pTabsGet(Number(tabId));
  const pageState = await toolCollectPageState(Number(tabId));
  let screenshotDataUrl = "";
  let screenshotMode = "crx_playwright";
  try {
    screenshotDataUrl = await crxCaptureScreenshotDataUrl(Number(tabId));
  } catch {
    screenshotDataUrl = await cdpCaptureScreenshotDataUrl(Number(tabId));
    screenshotMode = "cdp_capture_screenshot";
  }
  const image = pngSizeFromDataUrl(screenshotDataUrl);
  return {
    ok: true,
    screenshot_id: `shot_${Date.now()}_${Math.floor(Math.random() * 1e6)}`,
    screenshot_data_url: screenshotDataUrl,
    screenshot_mode: screenshotMode,
    page: pageState || {},
    image,
  };
}

export async function toolRunBrowserNativeAction(tabId, {
  action = "click",
  target = null,
  destination = null,
  deltaX = 0,
  deltaY = 0,
  textValue = "",
  keys = "",
  clear = false,
  waitMs = null,
  timeoutMs = CRX_NATIVE_ACTION_TIMEOUT_MS,
  button = "left",
  steps = 20,
} = {}) {
  const numericTabId = asInt(tabId, 0);
  if (!Number.isFinite(numericTabId) || numericTabId <= 0) throw new Error("Invalid tab id.");

  const normalizedAction = asString(action).trim().toLowerCase();
  const actionTimeoutMs = clamp(asInt(timeoutMs, CRX_NATIVE_ACTION_TIMEOUT_MS), 250, 60_000);
  const normalizedButton = new Set(["left", "middle", "right"]).has(asString(button).trim().toLowerCase())
    ? asString(button).trim().toLowerCase()
    : "left";
  const moveSteps = clamp(asInt(steps, 20), 1, 60);

  try {
    return await withCrxPage(numericTabId, async (page) => {
      let resultData = {
        ok: true,
        tabId: numericTabId,
        action: normalizedAction,
        engine: "crx_playwright",
      };

      if (normalizedAction === "click" || normalizedAction === "double_click") {
        const resolvedTarget = await resolveNativeActionTarget(page, target, {
          actionLabel: normalizedAction,
          timeoutMs: actionTimeoutMs,
        });
        if (resolvedTarget.kind === "locator" && resolvedTarget.locator) {
          await resolvedTarget.locator.click({
            timeout: actionTimeoutMs,
            button: normalizedButton,
            clickCount: normalizedAction === "double_click" ? 2 : 1,
          });
        } else {
          await page.mouse.click(resolvedTarget.point.x, resolvedTarget.point.y, {
            button: normalizedButton,
            clickCount: normalizedAction === "double_click" ? 2 : 1,
          });
        }
        resultData = {
          ...resultData,
          target: {
            kind: resolvedTarget.kind,
            search: resolvedTarget.search,
            bbox: resolvedTarget.bbox,
            point: resolvedTarget.point,
            coordinateInput: resolvedTarget.coordinateInput || null,
            coordinateResolved: resolvedTarget.coordinateResolved || null,
          },
          clickCount: normalizedAction === "double_click" ? 2 : 1,
          button: normalizedButton,
        };
      } else if (normalizedAction === "move") {
        const resolvedTarget = await resolveNativeActionTarget(page, target, {
          actionLabel: "move",
          timeoutMs: actionTimeoutMs,
        });
        await page.mouse.move(resolvedTarget.point.x, resolvedTarget.point.y, { steps: moveSteps });
        resultData = {
          ...resultData,
          target: {
            kind: resolvedTarget.kind,
            search: resolvedTarget.search,
            bbox: resolvedTarget.bbox,
            point: resolvedTarget.point,
            coordinateInput: resolvedTarget.coordinateInput || null,
            coordinateResolved: resolvedTarget.coordinateResolved || null,
          },
          steps: moveSteps,
        };
      } else if (normalizedAction === "drag") {
        const sourceTarget = await resolveNativeActionTarget(page, target, {
          actionLabel: "drag source",
          timeoutMs: actionTimeoutMs,
        });
        const destinationTarget = await resolveNativeActionTarget(page, destination, {
          actionLabel: "drag destination",
          timeoutMs: actionTimeoutMs,
        });
        await page.mouse.move(sourceTarget.point.x, sourceTarget.point.y, {
          steps: Math.max(2, Math.floor(moveSteps / 2)),
        });
        await page.mouse.down({ button: normalizedButton });
        await page.mouse.move(destinationTarget.point.x, destinationTarget.point.y, { steps: moveSteps });
        await page.mouse.up({ button: normalizedButton });
        resultData = {
          ...resultData,
          sourceTarget: {
            kind: sourceTarget.kind,
            search: sourceTarget.search,
            bbox: sourceTarget.bbox,
            point: sourceTarget.point,
            coordinateInput: sourceTarget.coordinateInput || null,
            coordinateResolved: sourceTarget.coordinateResolved || null,
          },
          destinationTarget: {
            kind: destinationTarget.kind,
            search: destinationTarget.search,
            bbox: destinationTarget.bbox,
            point: destinationTarget.point,
            coordinateInput: destinationTarget.coordinateInput || null,
            coordinateResolved: destinationTarget.coordinateResolved || null,
          },
          button: normalizedButton,
          steps: moveSteps,
        };
      } else if (normalizedAction === "scroll") {
        let resolvedTarget = null;
        const normalizedScrollTarget = target && typeof target === "object" ? normalizeNativeTarget(target) : null;
        if (normalizedScrollTarget && (nativeTargetHasLocator(normalizedScrollTarget) || nativeTargetHasCoordinates(normalizedScrollTarget))) {
          resolvedTarget = await resolveNativeActionTarget(page, target, {
            actionLabel: "scroll anchor",
            timeoutMs: actionTimeoutMs,
          });
          await page.mouse.move(resolvedTarget.point.x, resolvedTarget.point.y, {
            steps: Math.max(2, Math.floor(moveSteps / 2)),
          });
        }
        const wheelDeltaX = Number(deltaX);
        const wheelDeltaY = Number(deltaY);
        if (!Number.isFinite(wheelDeltaX) || !Number.isFinite(wheelDeltaY) || (Math.abs(wheelDeltaX) < 0.001 && Math.abs(wheelDeltaY) < 0.001)) {
          throw new Error("scroll: deltaX/deltaY fehlen oder sind ungültig.");
        }
        await page.mouse.wheel(wheelDeltaX, wheelDeltaY);
        resultData = {
          ...resultData,
          target: resolvedTarget
            ? {
                kind: resolvedTarget.kind,
                search: resolvedTarget.search,
                bbox: resolvedTarget.bbox,
                point: resolvedTarget.point,
                coordinateInput: resolvedTarget.coordinateInput || null,
                coordinateResolved: resolvedTarget.coordinateResolved || null,
              }
            : null,
          deltaX: wheelDeltaX,
          deltaY: wheelDeltaY,
        };
      } else if (normalizedAction === "type") {
        const value = asString(textValue);
        let resolvedTarget = null;
        if (target && typeof target === "object" && (nativeTargetHasLocator(normalizeNativeTarget(target)) || nativeTargetHasCoordinates(normalizeNativeTarget(target)))) {
          resolvedTarget = await resolveNativeActionTarget(page, target, {
            actionLabel: "type",
            timeoutMs: actionTimeoutMs,
          });
          if (resolvedTarget.kind === "locator" && resolvedTarget.locator) {
            await resolvedTarget.locator.click({ timeout: actionTimeoutMs });
            if (clear) await tryClearTypeTarget(page, resolvedTarget.locator);
          } else {
            await page.mouse.click(resolvedTarget.point.x, resolvedTarget.point.y, { button: "left", clickCount: 1 });
            if (clear) await tryClearTypeTarget(page, null);
          }
        } else if (clear) {
          await tryClearTypeTarget(page, null);
        }
        await page.keyboard.type(value, {
          delay: CRX_NATIVE_TYPE_DELAY_MS,
        });
        resultData = {
          ...resultData,
          target: resolvedTarget
            ? {
                kind: resolvedTarget.kind,
                search: resolvedTarget.search,
                bbox: resolvedTarget.bbox,
                point: resolvedTarget.point,
                coordinateInput: resolvedTarget.coordinateInput || null,
                coordinateResolved: resolvedTarget.coordinateResolved || null,
              }
            : null,
          clear: !!clear,
          typedLength: value.length,
        };
      } else if (normalizedAction === "keypress") {
        const combo = asString(keys).trim();
        if (!combo) throw new Error("keypress: keys fehlen.");
        let resolvedTarget = null;
        if (target && typeof target === "object" && (nativeTargetHasLocator(normalizeNativeTarget(target)) || nativeTargetHasCoordinates(normalizeNativeTarget(target)))) {
          resolvedTarget = await resolveNativeActionTarget(page, target, {
            actionLabel: "keypress",
            timeoutMs: actionTimeoutMs,
          });
          if (resolvedTarget.kind === "locator" && resolvedTarget.locator) {
            await resolvedTarget.locator.click({ timeout: actionTimeoutMs });
          } else {
            await page.mouse.click(resolvedTarget.point.x, resolvedTarget.point.y, { button: "left", clickCount: 1 });
          }
        }
        await page.keyboard.press(combo, { delay: 20 });
        resultData = {
          ...resultData,
          target: resolvedTarget
            ? {
                kind: resolvedTarget.kind,
                search: resolvedTarget.search,
                bbox: resolvedTarget.bbox,
                point: resolvedTarget.point,
                coordinateInput: resolvedTarget.coordinateInput || null,
                coordinateResolved: resolvedTarget.coordinateResolved || null,
              }
            : null,
          keys: combo,
        };
      } else if (normalizedAction === "wait") {
        const waitForMs = clamp(asInt(waitMs, 0), 250, 10_000);
        if (waitForMs <= 0) throw new Error("wait: waitMs muss > 0 sein.");
        await page.waitForTimeout(waitForMs);
        resultData = {
          ...resultData,
          waitedMs: waitForMs,
        };
      } else {
        throw new Error(`Unsupported native browser action '${normalizedAction}'.`);
      }

      if (new Set(["click", "double_click", "drag", "keypress"]).has(normalizedAction)) {
        await page.waitForLoadState("domcontentloaded", {
          timeout: Math.min(actionTimeoutMs, 4_000),
        }).catch(() => {});
      }
      await maybeWaitAfterAction(page, normalizedAction, waitMs);
      const tab = await pTabsGet(numericTabId).catch(() => null);
      const pageState = await toolCollectPageState(numericTabId).catch(() => null);
      let finalUrl = "";
      let title = "";
      try { finalUrl = asString(page.url?.() || ""); } catch {}
      try { title = asString(await page.title()); } catch {}
      if (!finalUrl) finalUrl = asString(tab?.url || "");

      return {
        ok: true,
        mode: "crx_playwright_native",
        action: normalizedAction,
        result: {
          ...resultData,
          url: finalUrl,
          title,
          pageState: pageState || null,
        },
        finalUrl,
        pageState: pageState || null,
      };
    }, {
      attachTimeoutMs: Math.max(actionTimeoutMs, CRX_PLAYWRIGHT_ATTACH_TIMEOUT_MS),
    });
  } catch (err) {
    const msg = asString(err?.message || err);
    const pageState = await toolCollectPageState(numericTabId).catch(() => null);
    const tab = await pTabsGet(numericTabId).catch(() => null);
    return {
      ok: false,
      mode: "crx_playwright_native",
      action: normalizedAction,
      error: `CRX Playwright action failed: ${msg}`.slice(0, 800),
      error_code: /timeout/i.test(msg) ? "CRX_NATIVE_TIMEOUT" : "CRX_NATIVE_ACTION_ERROR",
      error_stack: asString(err?.stack || "").slice(0, 800),
      finalUrl: asString(tab?.url || ""),
      pageState: pageState || null,
    };
  }
}

export async function toolRunPlaywrightCode(tabId, { code = "", timeoutMs = 90_000 } = {}) {
  const sourceCode = asString(code).trim();
  if (!sourceCode) return { ok: false, error: "empty playwright code" };
  const executableSource = buildPlaywrightUserCodeRunnerSource(sourceCode);
  const cappedTimeout = clamp(asInt(timeoutMs, 90_000), CODE_TIMEOUT_MIN_MS, CODE_TIMEOUT_MAX_MS);
  const numericTabId = asInt(tabId, 0);
  if (!Number.isFinite(numericTabId) || numericTabId <= 0) throw new Error("Invalid tab id.");
  await waitForTabReady(numericTabId, PLAYWRIGHT_NAV_TIMEOUT_MS).catch(() => null);
  let attached = false;
  try {
    try {
      await pDebuggerAttach(numericTabId, "1.3");
      attached = true;
    } catch (attachErr) {
      const msg = asString(attachErr?.message || attachErr).toLowerCase();
      const alreadyAttached =
        msg.includes("already attached") ||
        msg.includes("another debugger is already attached") ||
        msg.includes("target is already attached");
      if (!alreadyAttached) throw attachErr;
      attached = false;
    }

    await pDebuggerSendCommand(numericTabId, "Runtime.enable", {});
    await pDebuggerSendCommand(numericTabId, "Page.enable", {});
    await pDebuggerSendCommand(numericTabId, "Page.setBypassCSP", { enabled: true });

    let contextId = null;
    const frameTreeResp = await pDebuggerSendCommand(numericTabId, "Page.getFrameTree", {});
    const topFrameId = asString(frameTreeResp?.frameTree?.frame?.id || "").trim();
    if (topFrameId) {
      const worldResp = await pDebuggerSendCommand(numericTabId, "Page.createIsolatedWorld", {
        frameId: topFrameId,
        worldName: "nwt_browser_automation_strict",
        grantUniveralAccess: false,
      });
      const parsedContextId = Number(worldResp?.executionContextId);
      if (Number.isFinite(parsedContextId) && parsedContextId > 0) contextId = parsedContextId;
    }

      const expression = `
(() => {
  const WAIT = (ms) => new Promise((resolve) => setTimeout(resolve, Math.max(0, Math.floor(Number(ms) || 0))));
  const cleanText = (v) => String(v == null ? "" : v).replace(/\\s+/g, " ").trim();
  const toArray = (x) => Array.isArray(x) ? x : Array.from(x || []);
  const clampNum = (n, min, max) => {
    const v = Number(n);
    if (!Number.isFinite(v)) return Number(min);
    return Math.min(max, Math.max(min, v));
  };
  const randomRange = (min, max) => min + (Math.random() * (max - min));
  const asPoint = (x, y) => ({
    x: clampNum(Number(x), 1, Math.max(1, (window.innerWidth || 1) - 1)),
    y: clampNum(Number(y), 1, Math.max(1, (window.innerHeight || 1) - 1)),
  });
  const vecSub = (a, b) => ({ x: a.x - b.x, y: a.y - b.y });
  const vecAdd = (a, b) => ({ x: a.x + b.x, y: a.y + b.y });
  const vecMult = (a, b) => ({ x: a.x * b, y: a.y * b });
  const vecMag = (a) => Math.sqrt((a.x * a.x) + (a.y * a.y));
  const vecUnit = (a) => {
    const len = vecMag(a);
    if (!Number.isFinite(len) || len <= 0) return { x: 0, y: 0 };
    return { x: a.x / len, y: a.y / len };
  };
  const vecPerp = (a) => ({ x: a.y, y: -1 * a.x });

  const buildGhostPath = (start, end) => {
    const from = asPoint(start.x, start.y);
    const to = asPoint(end.x, end.y);
    const dist = vecMag(vecSub(to, from));
    if (!Number.isFinite(dist) || dist < 2) return [from, to];

    const spread = clampNum(dist, 2, 200);
    const side = Math.random() > 0.5 ? 1 : -1;
    const randomOnLine = (a, b) => {
      const dir = vecSub(b, a);
      return vecAdd(a, vecMult(dir, Math.random()));
    };
    const anchor = () => {
      const mid = randomOnLine(from, to);
      const normal = vecMult(vecUnit(vecPerp(vecSub(mid, from))), spread * side);
      return randomOnLine(mid, vecAdd(mid, normal));
    };
    const a1 = anchor();
    const a2 = anchor();
    const width = 100;
    const fitts = Math.log2((Math.max(1, dist * 0.8) / width) + 1);
    const steps = clampNum(Math.ceil((Math.log2(fitts + 1) + (Math.random() * 18)) * 3), 18, 96);
    const out = [];
    const n = Math.max(2, Math.floor(steps));
    for (let i = 0; i < n; i += 1) {
      const t = i / (n - 1);
      const omt = 1 - t;
      const x = (omt ** 3) * from.x + (3 * (omt ** 2) * t * a1.x) + (3 * omt * (t ** 2) * a2.x) + ((t ** 3) * to.x);
      const y = (omt ** 3) * from.y + (3 * (omt ** 2) * t * a1.y) + (3 * omt * (t ** 2) * a2.y) + ((t ** 3) * to.y);
      out.push(asPoint(x, y));
    }
    out[0] = from;
    out[out.length - 1] = to;
    return out;
  };

  const getVirtualMouseState = () => {
    const stateKey = "__NWT_RUNTIME_VIRTUAL_MOUSE__";
    if (window[stateKey] && window[stateKey].node && document.contains(window[stateKey].node)) {
      return window[stateKey];
    }
    const styleId = "__nwt_runtime_virtual_mouse_style";
    const nodeId = "__nwt_runtime_virtual_mouse_node";
    if (!document.getElementById(styleId)) {
      const style = document.createElement("style");
      style.id = styleId;
      style.textContent = [
        "#" + nodeId + "{",
        "position:fixed;top:0;left:0;z-index:2147483647;width:20px;height:20px;",
        "margin:-10px 0 0 -10px;border-radius:999px;pointer-events:none;",
        "border:2px solid rgba(255,59,48,0.95);background:rgba(255,255,255,0.96);",
        "box-shadow:0 0 0 2px rgba(255,59,48,0.35),0 0 18px rgba(255,59,48,0.45);",
        "transition:transform .03s linear,background .08s ease,border-color .08s ease,opacity .08s ease;",
        "opacity:1;",
        "}",
        "#" + nodeId + ".is-down{background:rgba(255,59,48,0.96);border-color:rgba(255,255,255,1);}",
      ].join("");
      (document.head || document.documentElement).appendChild(style);
    }
    let node = document.getElementById(nodeId);
    if (!node) {
      node = document.createElement("div");
      node.id = nodeId;
      (document.body || document.documentElement).appendChild(node);
    }
    const initial = asPoint((window.innerWidth || 1) / 2, (window.innerHeight || 1) / 2);
    node.style.transform = "translate(" + Math.round(initial.x) + "px," + Math.round(initial.y) + "px)";
    const state = { x: initial.x, y: initial.y, node };
    window[stateKey] = state;
    return state;
  };

  const updateVirtualMouseVisual = (state, point, isDown = false) => {
    state.x = point.x;
    state.y = point.y;
    state.node.style.transform = "translate(" + Math.round(point.x) + "px," + Math.round(point.y) + "px)";
    state.node.classList.toggle("is-down", !!isDown);
  };

  const dispatchMouse = (target, type, init = {}) => {
    if (typeof PointerEvent === "function" && (type === "pointerdown" || type === "pointerup" || type === "pointermove")) {
      target.dispatchEvent(new PointerEvent(type, {
        bubbles: true,
        cancelable: true,
        view: window,
        pointerType: "mouse",
        isPrimary: true,
        ...init,
      }));
      return;
    }
    target.dispatchEvent(new MouseEvent(type, {
      bubbles: true,
      cancelable: true,
      view: window,
      ...init,
    }));
  };

  const moveVirtualMouseTo = async (x, y) => {
    const state = getVirtualMouseState();
    const to = asPoint(x, y);
    const from = asPoint(state.x, state.y);
    const points = buildGhostPath(from, to);
    const totalMs = clampNum(Math.round(vecMag(vecSub(to, from)) * 1.2), 120, 1_000);
    const perStepMs = points.length > 1 ? totalMs / (points.length - 1) : totalMs;
    for (let i = 0; i < points.length; i += 1) {
      const p = points[i];
      updateVirtualMouseVisual(state, p, false);
      const moveTarget = document.elementFromPoint(p.x, p.y) || document.body;
      dispatchMouse(moveTarget, "pointermove", { clientX: p.x, clientY: p.y, buttons: 0 });
      dispatchMouse(moveTarget, "mousemove", { clientX: p.x, clientY: p.y, buttons: 0 });
      if (i < points.length - 1) {
        await WAIT(clampNum(Math.round(perStepMs + randomRange(-2, 4)), 2, 26));
      }
    }
    return to;
  };

  const virtualMouseClickAt = async (x, y, { clickCount = 1 } = {}) => {
    const state = getVirtualMouseState();
    const point = await moveVirtualMouseTo(x, y);
    const target = document.elementFromPoint(point.x, point.y) || document.body;
    updateVirtualMouseVisual(state, point, true);
    dispatchMouse(target, "pointerdown", { clientX: point.x, clientY: point.y, button: 0, buttons: 1, detail: clickCount });
    dispatchMouse(target, "mousedown", { clientX: point.x, clientY: point.y, button: 0, buttons: 1, detail: clickCount });
    await WAIT(clampNum(Math.round(randomRange(24, 70)), 18, 90));
    updateVirtualMouseVisual(state, point, false);
    dispatchMouse(target, "pointerup", { clientX: point.x, clientY: point.y, button: 0, buttons: 0, detail: clickCount });
    dispatchMouse(target, "mouseup", { clientX: point.x, clientY: point.y, button: 0, buttons: 0, detail: clickCount });
    if (typeof target.click === "function") {
      const safeCount = Math.max(1, Math.min(3, Math.floor(Number(clickCount) || 1)));
      for (let i = 0; i < safeCount; i += 1) target.click();
      if (safeCount > 1) {
        dispatchMouse(target, "dblclick", { clientX: point.x, clientY: point.y, button: 0, buttons: 0, detail: safeCount });
      }
    } else {
      dispatchMouse(target, "click", { clientX: point.x, clientY: point.y, button: 0, buttons: 0, detail: clickCount });
      if (clickCount > 1) {
        dispatchMouse(target, "dblclick", { clientX: point.x, clientY: point.y, button: 0, buttons: 0, detail: clickCount });
      }
    }
    return { x: point.x, y: point.y, clickCount };
  };

  const virtualMouseWheel = async (deltaX = 0, deltaY = 0) => {
    const dx = Number(deltaX);
    const dy = Number(deltaY);
    if (!Number.isFinite(dx) || !Number.isFinite(dy) || (Math.abs(dx) < 0.001 && Math.abs(dy) < 0.001)) return;
    const state = getVirtualMouseState();
    const anchor = asPoint(state.x, state.y);
    updateVirtualMouseVisual(state, anchor, false);
    const longest = Math.max(Math.abs(dx), Math.abs(dy));
    const steps = clampNum(Math.ceil(longest / 140), 1, 24);
    const stepX = dx / steps;
    const stepY = dy / steps;
    const scrollingEl = document.scrollingElement || document.documentElement || document.body;
    for (let i = 0; i < steps; i += 1) {
      const target = document.elementFromPoint(anchor.x, anchor.y) || scrollingEl || document.body;
      target.dispatchEvent(new WheelEvent("wheel", {
        bubbles: true,
        cancelable: true,
        clientX: anchor.x,
        clientY: anchor.y,
        deltaX: stepX,
        deltaY: stepY,
      }));
      try {
        if (scrollingEl && typeof scrollingEl.scrollBy === "function") {
          scrollingEl.scrollBy(stepX, stepY);
        } else {
          window.scrollBy(stepX, stepY);
        }
      } catch {}
      if (i < steps - 1) await WAIT(clampNum(Math.round(randomRange(8, 22)), 6, 28));
    }
  };

  const isValueElement = (el) => !!el && ("value" in el) && (typeof el.value === "string");
  const isContentEditableElement = (el) => !!el && !!el.isContentEditable;

  const getEditableText = (el) => {
    if (!el) return "";
    if (isValueElement(el)) return String(el.value || "");
    return String(el.textContent || "");
  };

  const setEditableText = (el, next) => {
    const value = String(next ?? "");
    if (!el) return;
    if (isValueElement(el)) el.value = value;
    else el.textContent = value;
  };

  const emitInputEvent = (el, data, inputType) => {
    try {
      if (typeof InputEvent === "function") {
        el.dispatchEvent(new InputEvent("input", {
          bubbles: true,
          cancelable: false,
          data: data == null ? null : String(data),
          inputType: String(inputType || "insertText"),
        }));
        return;
      }
    } catch {}
    el.dispatchEvent(new Event("input", { bubbles: true }));
  };

  const emitKeyboardEvent = (el, type, key, opts = {}) => {
    const payload = {
      key: String(key || ""),
      bubbles: true,
      cancelable: true,
      shiftKey: !!opts.shiftKey,
      ctrlKey: !!opts.ctrlKey,
      metaKey: !!opts.metaKey,
      altKey: !!opts.altKey,
      repeat: !!opts.repeat,
    };
    try {
      el.dispatchEvent(new KeyboardEvent(type, payload));
    } catch {
      el.dispatchEvent(new Event(type, { bubbles: true, cancelable: true }));
    }
  };

  const keyFromChar = (ch) => {
    if (ch === "\\n") return "Enter";
    if (ch === "\\t") return "Tab";
    if (ch === "\\b") return "Backspace";
    return String(ch || "");
  };

  const shouldHumanType = (el) => {
    if (!el) return false;
    const tag = String(el.tagName || "").toLowerCase();
    if (tag === "textarea") return true;
    if (isContentEditableElement(el)) return true;
    if (tag !== "input") return isValueElement(el);
    const t = String(el.type || "text").toLowerCase();
    return t === "" || t === "text" || t === "search" || t === "email" || t === "url" || t === "tel" || t === "password" || t === "number";
  };

  const caretToEnd = (el) => {
    if (!el || !isValueElement(el)) return;
    try {
      const len = String(el.value || "").length;
      if (typeof el.setSelectionRange === "function") el.setSelectionRange(len, len);
    } catch {}
  };

  const humanDelayForChar = (ch) => {
    const c = String(ch || "");
    let base = randomRange(26, 94);
    if (/[.,;:!?]/.test(c)) base += randomRange(28, 92);
    if (/\s/.test(c)) base += randomRange(6, 24);
    return clampNum(Math.round(base), 16, 190);
  };

  const ensureFocusedForTyping = async (el) => {
    if (!el) return;
    const active = document.activeElement;
    if (active === el) return;
    try {
      const r = el.getBoundingClientRect();
      if (r && Number.isFinite(r.width) && Number.isFinite(r.height) && r.width > 0 && r.height > 0) {
        const x = r.left + Math.min(Math.max(r.width / 2, 1), Math.max(1, r.width - 1));
        const y = r.top + Math.min(Math.max(r.height / 2, 1), Math.max(1, r.height - 1));
        await virtualMouseClickAt(x, y, { clickCount: 1 });
      } else if (typeof el.focus === "function") {
        el.focus({ preventScroll: true });
      }
    } catch {
      try {
        if (typeof el.focus === "function") el.focus({ preventScroll: true });
      } catch {}
    }
    try {
      if (typeof el.focus === "function") el.focus({ preventScroll: true });
    } catch {}
    caretToEnd(el);
  };

  const humanBackspaceClear = async (el) => {
    let current = getEditableText(el);
    if (!current) return;
    if (current.length > 160) {
      setEditableText(el, "");
      emitInputEvent(el, null, "deleteContentBackward");
      return;
    }
    while (current.length > 0) {
      emitKeyboardEvent(el, "keydown", "Backspace");
      current = current.slice(0, -1);
      setEditableText(el, current);
      emitInputEvent(el, null, "deleteContentBackward");
      emitKeyboardEvent(el, "keyup", "Backspace");
      await WAIT(clampNum(Math.round(randomRange(14, 52)), 8, 78));
    }
  };

  const humanTypeText = async (el, text, { append = true } = {}) => {
    const value = String(text ?? "");
    if (!append) setEditableText(el, "");
    if (!value) return;
    for (const ch of value) {
      const key = keyFromChar(ch);
      const shiftNeeded = key.length === 1 && key.toUpperCase() === key && key.toLowerCase() !== key;
      emitKeyboardEvent(el, "keydown", key, { shiftKey: shiftNeeded });
      let next = getEditableText(el);
      if (ch === "\\b") next = next.slice(0, -1);
      else next += ch;
      setEditableText(el, next);
      emitInputEvent(el, ch === "\\b" ? null : ch, ch === "\\b" ? "deleteContentBackward" : "insertText");
      emitKeyboardEvent(el, "keyup", key, { shiftKey: shiftNeeded });
      await WAIT(humanDelayForChar(ch));
      if (Math.random() < 0.03) await WAIT(clampNum(Math.round(randomRange(70, 170)), 50, 220));
    }
  };

  const makeLocator = (resolver) => ({
    first() {
      return makeLocator(() => {
        const arr = toArray(resolver());
        return arr.length ? [arr[0]] : [];
      });
    },
    nth(n) {
      return makeLocator(() => {
        const arr = toArray(resolver());
        const idx = Math.max(0, Math.floor(Number(n) || 0));
        return arr[idx] ? [arr[idx]] : [];
      });
    },
    async count() {
      return toArray(resolver()).length;
    },
    async click() {
      const el = toArray(resolver())[0];
      if (!el) throw new Error("locator click target not found");
      const r = el.getBoundingClientRect();
      if (!(r && Number.isFinite(r.width) && Number.isFinite(r.height) && r.width > 0 && r.height > 0)) {
        throw new Error("locator click target has no visible bounding box");
      }
      const x = r.left + Math.min(Math.max(r.width / 2, 1), Math.max(1, r.width - 1));
      const y = r.top + Math.min(Math.max(r.height / 2, 1), Math.max(1, r.height - 1));
      await virtualMouseClickAt(x, y, { clickCount: 1 });
    },
    async fill(value) {
      const el = toArray(resolver())[0];
      if (!el) throw new Error("locator fill target not found");
      const v = String(value ?? "");
      if (!shouldHumanType(el)) {
        setEditableText(el, v);
        el.dispatchEvent(new Event("input", { bubbles: true }));
        el.dispatchEvent(new Event("change", { bubbles: true }));
        return;
      }
      await ensureFocusedForTyping(el);
      await humanBackspaceClear(el);
      await humanTypeText(el, v, { append: true });
      el.dispatchEvent(new Event("change", { bubbles: true }));
    },
    async type(value) {
      const el = toArray(resolver())[0];
      if (!el) throw new Error("locator type target not found");
      const add = String(value ?? "");
      if (!add) return;
      if (!shouldHumanType(el)) {
        setEditableText(el, getEditableText(el) + add);
        el.dispatchEvent(new Event("input", { bubbles: true }));
        return;
      }
      await ensureFocusedForTyping(el);
      await humanTypeText(el, add, { append: true });
    },
    async selectOption(value) {
      const el = toArray(resolver())[0];
      if (!el) throw new Error("locator selectOption target not found");
      const tag = String(el.tagName || "").toLowerCase();
      if (tag !== "select") throw new Error("locator selectOption target is not <select>");
      const pick = Array.isArray(value) ? value[0] : value;
      const next = typeof pick === "object" && pick !== null && "value" in pick ? String(pick.value ?? "") : String(pick ?? "");
      el.value = next;
      el.dispatchEvent(new Event("input", { bubbles: true }));
      el.dispatchEvent(new Event("change", { bubbles: true }));
    },
    async press(key) {
      const el = toArray(resolver())[0] || document.activeElement || document.body;
      const k = String(key || "Enter");
      el.dispatchEvent(new KeyboardEvent("keydown", { key: k, bubbles: true }));
      el.dispatchEvent(new KeyboardEvent("keyup", { key: k, bubbles: true }));
    },
    async isVisible() {
      const el = toArray(resolver())[0];
      if (!el) return false;
      const st = getComputedStyle(el);
      const r = el.getBoundingClientRect();
      return st.display !== "none" && st.visibility !== "hidden" && r.width > 0 && r.height > 0;
    },
    async innerText() {
      const el = toArray(resolver())[0];
      return cleanText(el?.innerText || el?.textContent || "");
    },
    async textContent() {
      const el = toArray(resolver())[0];
      return String(el?.textContent || "");
    },
    async inputValue() {
      const el = toArray(resolver())[0];
      return String(el && "value" in el ? el.value : "");
    },
    async evaluate(fn) {
      const el = toArray(resolver())[0];
      if (!el) throw new Error("locator evaluate target not found");
      if (typeof fn !== "function") throw new Error("locator evaluate expects function");
      return fn(el);
    },
    async boundingBox() {
      const el = toArray(resolver())[0];
      if (!el) return null;
      const r = el.getBoundingClientRect();
      return { x: r.x, y: r.y, width: r.width, height: r.height };
    },
  });

  const byText = (text, exact = false) => {
    const needle = cleanText(text).toLowerCase();
    const all = Array.from(document.querySelectorAll("a,button,input,textarea,label,div,span,p,li,h1,h2,h3,h4,h5,h6"));
    return all.filter((el) => {
      const v = cleanText(el.innerText || el.textContent || el.getAttribute?.("aria-label") || "");
      if (!v) return false;
      const low = v.toLowerCase();
      return exact ? low === needle : low.includes(needle);
    });
  };

  const byRole = (role, name, exact = false) => {
    const r = String(role || "").toLowerCase();
    const reName = name instanceof RegExp ? name : null;
    let selector = "*";
    if (r === "button") selector = "button,[role='button'],input[type='button'],input[type='submit']";
    else if (r === "link") selector = "a,[role='link']";
    else if (r === "textbox") selector = "input:not([type]),input[type='text'],input[type='search'],input[type='email'],textarea,[role='textbox']";
    const arr = Array.from(document.querySelectorAll(selector));
    if (!name) return arr;
    return arr.filter((el) => {
      const v = cleanText(el.innerText || el.textContent || el.value || el.getAttribute?.("aria-label") || "");
      if (!v) return false;
      if (reName) return reName.test(v);
      const needle = cleanText(name).toLowerCase();
      const low = v.toLowerCase();
      return exact ? low === needle : low.includes(needle);
    });
  };

  const page = {
    url() { return location.href; },
    async title() { return document.title || ""; },
    async setContent(html) {
      const markup = String(html ?? "");
      document.open();
      document.write(markup);
      document.close();
      await WAIT(0);
      return true;
    },
    async focus(selector) {
      const el = document.querySelector(String(selector || ""));
      if (!el) throw new Error("page.focus target not found");
      if (typeof el.focus === "function") {
        el.focus({ preventScroll: true });
      }
      return true;
    },
    async waitForTimeout(ms) { await WAIT(ms); },
    async waitForLoadState() {
      const t0 = Date.now();
      while (Date.now() - t0 < 30_000) {
        if (document.readyState === "complete" || document.readyState === "interactive") return true;
        await WAIT(50);
      }
      throw new Error("waitForLoadState timeout");
    },
    locator(selector) { return makeLocator(() => document.querySelectorAll(String(selector || ""))); },
    getByText(text, opts = {}) { return makeLocator(() => byText(text, !!opts?.exact)); },
    getByRole(role, opts = {}) { return makeLocator(() => byRole(role, opts?.name, !!opts?.exact)); },
    $$eval(selector, fn, ...args) {
      if (typeof fn !== "function") throw new Error("$$eval expects function");
      const els = Array.from(document.querySelectorAll(String(selector || "")));
      return fn(els, ...args);
    },
    $eval(selector, fn, ...args) {
      if (typeof fn !== "function") throw new Error("$eval expects function");
      const el = document.querySelector(String(selector || ""));
      if (!el) throw new Error("$eval target not found");
      return fn(el, ...args);
    },
    evaluate(fn, ...args) {
      if (typeof fn !== "function") throw new Error("page.evaluate expects function");
      return fn(...args);
    },
    frames() {
      return [];
    },
    frameLocator(selector) {
      throw new Error(
        "page.frameLocator is not supported in this runner. Use execute_browser_control(action='click_text') for iframe targets.",
      );
    },
    keyboard: {
      async press(key) {
        const el = document.activeElement || document.body;
        const k = String(key || "Enter");
        el.dispatchEvent(new KeyboardEvent("keydown", { key: k, bubbles: true }));
        el.dispatchEvent(new KeyboardEvent("keyup", { key: k, bubbles: true }));
      },
    },
    mouse: {
      async move(x, y) {
        await moveVirtualMouseTo(Number(x), Number(y));
      },
      async click(x, y, options = {}) {
        const countRaw = Number(options?.clickCount ?? options?.count ?? 1);
        const clickCount = Math.max(1, Math.min(3, Math.floor(Number.isFinite(countRaw) ? countRaw : 1)));
        await virtualMouseClickAt(Number(x), Number(y), { clickCount });
      },
      async wheel(deltaX = 0, deltaY = 0) {
        await virtualMouseWheel(Number(deltaX), Number(deltaY));
      },
    },
  };

  return (async () => {
${executableSource}
  })();
})();
`.trim();

    const out = await pDebuggerSendCommand(numericTabId, "Runtime.evaluate", {
      expression,
      ...(contextId ? { contextId } : {}),
      includeCommandLineAPI: true,
      awaitPromise: true,
      returnByValue: true,
      userGesture: true,
      allowUnsafeEvalBlockedByCSP: true,
      timeout: cappedTimeout,
      replMode: false,
    });

    if (out?.exceptionDetails) {
      const txt = asString(out?.exceptionDetails?.text || "Runtime.evaluate exception");
      const desc = asString(out?.result?.description || "");
      throw new Error(desc ? `${txt}: ${desc}` : txt);
    }

    const tab = await pTabsGet(numericTabId).catch(() => null);
    const pageState = await toolCollectPageState(numericTabId).catch(() => null);
    return {
      ok: true,
      mode: "cdp_runtime_evaluate",
      result: out?.result?.value ?? null,
      output: out?.result?.value ?? null,
      finalUrl: asString(tab?.url || ""),
      pageState: pageState || null,
    };
  } catch (err) {
    const msg = asString(err?.message || err);
    const tab = await pTabsGet(numericTabId).catch(() => null);
    const pageState = await toolCollectPageState(numericTabId).catch(() => null);
    return {
      ok: false,
      mode: "cdp_runtime_evaluate",
      error: `CDP evaluate failed: ${msg}`.slice(0, 800),
      error_code: /timeout/i.test(msg) ? "INJECTION_TIMEOUT" : "INJECTION_RUNTIME_ERROR",
      error_stack: asString(err?.stack || "").slice(0, 800),
      finalUrl: asString(tab?.url || ""),
      pageState: pageState || null,
    };
  } finally {
    try { await pDebuggerSendCommand(numericTabId, "Page.setBypassCSP", { enabled: false }); } catch {}
    if (attached) {
      try { await pDebuggerDetach(numericTabId); } catch {}
    }
  }
}
