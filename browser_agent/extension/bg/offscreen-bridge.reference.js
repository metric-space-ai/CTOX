const OFFSCREEN_URL = "bg/ml-inference.html";
const OFFSCREEN_MESSAGE_TIMEOUTS_MS = Object.freeze({
  OFFSCREEN_LOCAL_QWEN_CHAT: 25_000,
  OFFSCREEN_LOCAL_QWEN_FORWARD_BENCHMARK: 30_000,
  OFFSCREEN_LOCAL_QWEN_TRAINING_START: 60_000,
  OFFSCREEN_LOCAL_QWEN_TRAINING_STATUS: 15_000,
  OFFSCREEN_EMBED_TEXT: 15_000,
  OFFSCREEN_FLORENCE2_RUN: 30_000,
  OFFSCREEN_REMOTE_COMPUTE_BOOTSTRAP: 15_000,
});

let offscreenInitPromise = null;

function asText(value) {
  return String(value == null ? "" : value).trim();
}

function getOffscreenMessageTimeoutMs(type) {
  const configured = Number(OFFSCREEN_MESSAGE_TIMEOUTS_MS[asText(type)] || 0);
  if (Number.isFinite(configured) && configured > 0) {
    return configured;
  }
  return 20_000;
}

export async function ensureOffscreenDocument() {
  if (!chrome.offscreen) {
    console.warn("[ml-bridge] chrome.offscreen is not available");
    return;
  }

  if (offscreenInitPromise) {
    return offscreenInitPromise;
  }

  offscreenInitPromise = (async () => {
    try {
      if (chrome.offscreen.hasDocument) {
        const has = await chrome.offscreen.hasDocument();
        if (has) return;
      }

      await chrome.offscreen.createDocument({
        url: OFFSCREEN_URL,
        reasons: ["WORKERS"],
        justification: "Run transformers.js and ORT-WASM outside the MV3 service worker lifecycle",
      });
    } catch (error) {
      const msg = String(error?.message || error || "");
      if (msg.includes("Only a single offscreen document may be created")) {
        return;
      }
      throw error;
    }
  })();

  try {
    await offscreenInitPromise;
  } finally {
    offscreenInitPromise = null;
  }
}

export async function sendMessageToOffscreen(type, payload = {}) {
  await ensureOffscreenDocument();

  return await new Promise((resolve) => {
    const timeoutMs = getOffscreenMessageTimeoutMs(type);
    let settled = false;
    const finish = (value) => {
      if (settled) return;
      settled = true;
      if (timeoutId) {
        globalThis.clearTimeout(timeoutId);
      }
      resolve(value);
    };
    const timeoutId = globalThis.setTimeout(() => {
      finish({
        ok: false,
        error: `offscreen request timed out (${asText(type)})`,
        errorDetail: {
          bridge: "offscreen",
          messageType: asText(type),
          reason: "offscreen_timeout",
          timeoutMs,
        },
      });
    }, timeoutMs);
    chrome.runtime.sendMessage({ type, ...payload }, (response) => {
      const err = chrome.runtime.lastError;
      if (err) {
        finish({
          ok: false,
          error: err.message || String(err),
          errorDetail: {
            bridge: "offscreen",
            messageType: String(type || ""),
            reason: "chrome_runtime_last_error",
          },
          errorStack: String(err?.stack || ""),
        });
        return;
      }
      finish(response || {
        ok: false,
        error: `no response from offscreen (${type})`,
        errorDetail: {
          bridge: "offscreen",
          messageType: String(type || ""),
          reason: "missing_offscreen_response",
        },
      });
    });
  });
}

export function registerEmbeddingRoute() {
  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message?.type !== "embed-text") {
      return;
    }

    (async () => {
      try {
        const response = await sendMessageToOffscreen("OFFSCREEN_EMBED_TEXT", {
          text: message.text || "",
        });
        sendResponse(response);
      } catch (error) {
        sendResponse({
          ok: false,
          error: String(error?.message || error),
          errorDetail:
            error && typeof error === "object" && "detail" in error
              ? error.detail || null
              : null,
          errorStack: String(error?.stack || ""),
        });
      }
    })();

    return true;
  });
}
