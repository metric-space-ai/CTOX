const DEFAULT_BROWSER_AGENT_BRIDGE_BASE_URL = "http://127.0.0.1:8765";
const DEFAULT_BROWSER_AGENT_BRIDGE_POLL_MS = 2_000;

function asString(value) {
  return String(value == null ? "" : value);
}

function asInt(value, fallback = 0, min = Number.MIN_SAFE_INTEGER, max = Number.MAX_SAFE_INTEGER) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.max(min, Math.min(max, Math.floor(parsed)));
}

async function readJsonSafe(response) {
  const text = await response.text();
  if (!text.trim()) return {};
  try {
    return JSON.parse(text);
  } catch {
    return {
      ok: false,
      error: `Invalid JSON from browser bridge: ${text.slice(0, 300)}`,
    };
  }
}

const BRIDGE_STATE = {
  workerId: `ext-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`,
  baseUrl: DEFAULT_BROWSER_AGENT_BRIDGE_BASE_URL,
  pollMs: DEFAULT_BROWSER_AGENT_BRIDGE_POLL_MS,
  enabled: false,
  timerId: 0,
  inFlight: false,
  lastError: "",
  lastPollAt: "",
  lastJobId: "",
};

async function bridgeFetch(pathname, init = {}) {
  const baseUrl = asString(BRIDGE_STATE.baseUrl || DEFAULT_BROWSER_AGENT_BRIDGE_BASE_URL).replace(/\/$/, "");
  const response = await fetch(`${baseUrl}${pathname}`, {
    ...init,
    headers: {
      "content-type": "application/json",
      ...(init.headers && typeof init.headers === "object" ? init.headers : {}),
    },
  });
  const payload = await readJsonSafe(response);
  if (!response.ok) {
    const message =
      payload?.error ||
      payload?.message ||
      `Browser bridge HTTP ${response.status}`;
    throw new Error(asString(message).trim() || `Browser bridge HTTP ${response.status}`);
  }
  return payload;
}

async function completeJob(jobId, result) {
  await bridgeFetch(`/api/browser-agent/jobs/${encodeURIComponent(jobId)}/complete`, {
    method: "POST",
    body: JSON.stringify({
      workerId: BRIDGE_STATE.workerId,
      result,
      worker: {
        extensionId: chrome.runtime?.id || "",
        extensionVersion: chrome.runtime?.getManifest?.().version || "",
      },
    }),
  });
}

async function pollOnce(handler) {
  if (BRIDGE_STATE.inFlight || typeof handler !== "function") return;
  BRIDGE_STATE.inFlight = true;
  BRIDGE_STATE.lastPollAt = new Date().toISOString();
  let leasedJobId = "";
  try {
    const poll = await bridgeFetch(
      `/api/browser-agent/worker/poll?workerId=${encodeURIComponent(BRIDGE_STATE.workerId)}`,
      { method: "GET" },
    );
    const job = poll?.job && typeof poll.job === "object" ? poll.job : null;
    if (!job?.jobId || !job?.request) {
      BRIDGE_STATE.lastError = "";
      return;
    }
    leasedJobId = asString(job.jobId).trim();
    BRIDGE_STATE.lastJobId = leasedJobId;
    const result = await handler(job.request);
    await completeJob(leasedJobId, result);
    BRIDGE_STATE.lastError = "";
  } catch (error) {
    BRIDGE_STATE.lastError = error instanceof Error ? error.message : String(error || "Unknown bridge error");
    if (leasedJobId) {
      try {
        await completeJob(leasedJobId, {
          ok: false,
          error: BRIDGE_STATE.lastError,
          summary: BRIDGE_STATE.lastError,
        });
      } catch {}
    }
  } finally {
    BRIDGE_STATE.inFlight = false;
  }
}

export function getBrowserAgentBridgeState() {
  return {
    workerId: BRIDGE_STATE.workerId,
    baseUrl: BRIDGE_STATE.baseUrl,
    pollMs: BRIDGE_STATE.pollMs,
    enabled: BRIDGE_STATE.enabled,
    inFlight: BRIDGE_STATE.inFlight,
    lastError: BRIDGE_STATE.lastError,
    lastPollAt: BRIDGE_STATE.lastPollAt,
    lastJobId: BRIDGE_STATE.lastJobId,
  };
}

export function stopBrowserAgentBridgeLoop() {
  BRIDGE_STATE.enabled = false;
  if (BRIDGE_STATE.timerId) {
    clearInterval(BRIDGE_STATE.timerId);
    BRIDGE_STATE.timerId = 0;
  }
}

export function startBrowserAgentBridgeLoop({
  handler,
  baseUrl = DEFAULT_BROWSER_AGENT_BRIDGE_BASE_URL,
  pollMs = DEFAULT_BROWSER_AGENT_BRIDGE_POLL_MS,
} = {}) {
  BRIDGE_STATE.baseUrl = asString(baseUrl || DEFAULT_BROWSER_AGENT_BRIDGE_BASE_URL).trim() || DEFAULT_BROWSER_AGENT_BRIDGE_BASE_URL;
  BRIDGE_STATE.pollMs = asInt(pollMs, DEFAULT_BROWSER_AGENT_BRIDGE_POLL_MS, 500, 60_000);
  BRIDGE_STATE.enabled = true;

  if (!BRIDGE_STATE.timerId) {
    BRIDGE_STATE.timerId = setInterval(() => {
      void pollOnce(handler);
    }, BRIDGE_STATE.pollMs);
  }
  void pollOnce(handler);
  return getBrowserAgentBridgeState();
}
