import { normalizePortableAssistantTurn } from "../shared/qwen-agent-adapter.mjs";

const TOOL_SMOKE_PAGE_NAME = "browser-tool-smoke.html";
const TOOL_SMOKE_PAGE_URL = globalThis.chrome?.runtime?.getURL?.(TOOL_SMOKE_PAGE_NAME) || TOOL_SMOKE_PAGE_NAME;
const TOOL_SMOKE_READY_EXPR = "(() => Boolean(globalThis.__toolSmoke && typeof globalThis.__toolSmoke.getSnapshot === 'function'))()";
const TOOL_SMOKE_RESET_EXPR = "(() => globalThis.__toolSmoke.resetState())()";
const TOOL_SMOKE_SNAPSHOT_EXPR = "(() => globalThis.__toolSmoke.getSnapshot())()";
const TOOL_SMOKE_ACTION_EXPR = "(() => globalThis.__toolSmoke.openRelationMap())()";
const TOOL_SMOKE_CODE_EXPR = `
(() => {
  const button = document.querySelector("[data-testid='hero-action']");
  if (!button) {
    return {
      ok: false,
      error: "hero action missing",
    };
  }
  button.click();
  const titles = Array.from(document.querySelectorAll("[data-testid='article-card-title']"))
    .map((node) => String(node.textContent || "").trim())
    .filter(Boolean)
    .slice(0, 5);
  const relationPanel = document.querySelector("[data-testid='relation-panel']");
  const statusPill = document.querySelector("[data-testid='status-pill']");
  return {
    ok: titles.length === 5,
    clicked: true,
    articleCount: titles.length,
    articleTitles: titles,
    status: String(statusPill?.textContent || "").trim(),
    relationText: String(relationPanel?.textContent || "").trim(),
  };
})()
`;

let rememberedToolSmokeTabId = 0;

function asText(value) {
  return String(value == null ? "" : value).trim();
}

function trimText(value, max = 240) {
  const text = asText(value).replace(/\s+/g, " ").trim();
  if (text.length <= max) return text;
  return `${text.slice(0, Math.max(1, max - 1)).trimEnd()}...`;
}

function normalizeAnswerText(value) {
  return asText(value)
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function parseJsonObject(value) {
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

function nowIso() {
  return new Date().toISOString();
}

function sleep(ms) {
  return new Promise((resolve) => globalThis.setTimeout(resolve, ms));
}

function createDiagnosis({ failureKind = "", summary = "", nextAction = "", keyFacts = [] } = {}) {
  return {
    failureKind: asText(failureKind),
    summary: asText(summary),
    nextAction: asText(nextAction),
    keyFacts: Array.isArray(keyFacts)
      ? keyFacts
          .map((entry) => ({
            label: asText(entry?.label),
            value: asText(entry?.value),
          }))
          .filter((entry) => entry.label && entry.value)
      : [],
  };
}

function createToolSmokeReport(type, request = {}) {
  return {
    reportVersion: 1,
    type,
    timestamp: nowIso(),
    request,
    result: null,
    error: "",
    diagnosis: null,
    timingsMs: {},
  };
}

function pTabsCreate(url, updateProps = {}) {
  return new Promise((resolve, reject) => {
    chrome.tabs.create({ url, ...updateProps }, (tab) => {
      const error = chrome.runtime.lastError;
      if (error) reject(new Error(error.message));
      else resolve(tab);
    });
  });
}

function pTabsGet(tabId) {
  return new Promise((resolve, reject) => {
    chrome.tabs.get(Number(tabId), (tab) => {
      const error = chrome.runtime.lastError;
      if (error) reject(new Error(error.message));
      else resolve(tab);
    });
  });
}

function pTabsUpdate(tabId, updateProps) {
  return new Promise((resolve, reject) => {
    chrome.tabs.update(Number(tabId), updateProps, (tab) => {
      const error = chrome.runtime.lastError;
      if (error) reject(new Error(error.message));
      else resolve(tab);
    });
  });
}

function pCaptureVisibleTab(windowId, options = { format: "png" }) {
  return new Promise((resolve, reject) => {
    chrome.tabs.captureVisibleTab(Number(windowId), options, (dataUrl) => {
      const error = chrome.runtime.lastError;
      if (error) reject(new Error(error.message));
      else resolve(dataUrl || "");
    });
  });
}

function pDebuggerAttach(tabId, version = "1.3") {
  return new Promise((resolve, reject) => {
    chrome.debugger.attach({ tabId: Number(tabId) }, version, () => {
      const error = chrome.runtime.lastError;
      if (error) reject(new Error(error.message));
      else resolve(true);
    });
  });
}

function pDebuggerDetach(tabId) {
  return new Promise((resolve, reject) => {
    chrome.debugger.detach({ tabId: Number(tabId) }, () => {
      const error = chrome.runtime.lastError;
      if (error) reject(new Error(error.message));
      else resolve(true);
    });
  });
}

function pDebuggerSendCommand(tabId, method, params = {}) {
  return new Promise((resolve, reject) => {
    chrome.debugger.sendCommand({ tabId: Number(tabId) }, method, params, (result) => {
      const error = chrome.runtime.lastError;
      if (error) reject(new Error(error.message));
      else resolve(result || {});
    });
  });
}

async function withAttachedDebugger(tabId, handler) {
  await pDebuggerAttach(tabId, "1.3");
  try {
    await pDebuggerSendCommand(tabId, "Runtime.enable", {});
    await pDebuggerSendCommand(tabId, "Page.enable", {});
    return await handler();
  } finally {
    try {
      await pDebuggerDetach(tabId);
    } catch {}
  }
}

async function evaluateInTab(tabId, expression) {
  return await withAttachedDebugger(tabId, async () => {
    const response = await pDebuggerSendCommand(tabId, "Runtime.evaluate", {
      expression,
      awaitPromise: true,
      returnByValue: true,
      userGesture: true,
    });
    if (response?.exceptionDetails) {
      const description =
        response.result?.description ||
        response.exceptionDetails?.text ||
        "Runtime.evaluate failed.";
      throw new Error(trimText(description, 320));
    }
    return response?.result?.value ?? null;
  });
}

async function waitForToolSmokePageReady(tabId, timeoutMs = 12_000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const tab = await pTabsGet(tabId).catch(() => null);
    if (tab && (!tab.status || tab.status === "complete")) {
      try {
        const ready = await evaluateInTab(tabId, TOOL_SMOKE_READY_EXPR);
        if (ready) return tab;
      } catch {}
    }
    await sleep(140);
  }
  throw new Error("Browser tool smoke page did not finish loading.");
}

async function ensureToolSmokeTab({ active = true } = {}) {
  let tab = null;
  if (rememberedToolSmokeTabId > 0) {
    tab = await pTabsGet(rememberedToolSmokeTabId).catch(() => null);
    if (!tab || !asText(tab.url).startsWith(TOOL_SMOKE_PAGE_URL)) {
      tab = null;
    }
  }

  if (!tab) {
    tab = await pTabsCreate(TOOL_SMOKE_PAGE_URL, { active });
    rememberedToolSmokeTabId = Number(tab?.id || 0);
  } else if (active && !tab.active) {
    tab = await pTabsUpdate(tab.id, { active: true });
  }

  const readyTab = await waitForToolSmokePageReady(Number(tab?.id || 0));
  rememberedToolSmokeTabId = Number(readyTab?.id || 0);
  await evaluateInTab(readyTab.id, TOOL_SMOKE_RESET_EXPR).catch(() => null);
  return readyTab;
}

async function captureToolSmokeScreenshot(tab) {
  const activeTab = tab?.active ? tab : await pTabsUpdate(tab.id, { active: true }).catch(() => tab);
  await sleep(160);
  const image = await pCaptureVisibleTab(activeTab.windowId, { format: "png" });
  if (!image) {
    throw new Error("captureVisibleTab returned empty image data.");
  }
  return image;
}

async function getToolSmokeSnapshot(tabId) {
  return (await evaluateInTab(tabId, TOOL_SMOKE_SNAPSHOT_EXPR)) || null;
}

async function triggerToolSmokeAction(tabId) {
  return (await evaluateInTab(tabId, TOOL_SMOKE_ACTION_EXPR)) || null;
}

async function runToolSmokeCode(tabId) {
  return (await evaluateInTab(tabId, TOOL_SMOKE_CODE_EXPR)) || null;
}

export async function runBrowserToolTabSmoke() {
  const startedAt = Date.now();
  const report = createToolSmokeReport("browser_tool_tabs_smoke", {
    pageUrl: TOOL_SMOKE_PAGE_URL,
  });

  try {
    const openStartedAt = Date.now();
    const firstTab = await ensureToolSmokeTab({ active: false });
    report.timingsMs.openTab = Date.now() - openStartedAt;

    const firstSnapshot = await getToolSmokeSnapshot(firstTab.id);
    const reuseStartedAt = Date.now();
    const reusedTab = await ensureToolSmokeTab({ active: true });
    report.timingsMs.reuseTab = Date.now() - reuseStartedAt;
    const reusedSnapshot = await getToolSmokeSnapshot(reusedTab.id);

    report.result = {
      firstTabId: Number(firstTab?.id || 0),
      reusedTabId: Number(reusedTab?.id || 0),
      reusedSameTab: Number(firstTab?.id || 0) === Number(reusedTab?.id || 0),
      status: asText(reusedSnapshot?.status),
      articleCount: Number(reusedSnapshot?.articleCount || 0),
      pageUrl: asText(reusedTab?.url),
      active: reusedTab?.active === true,
      firstSnapshot,
    };
    report.ok = Boolean(
      report.result.reusedSameTab &&
        report.result.active &&
        report.result.articleCount === 5 &&
        report.result.status === "READY",
    );
    if (!report.ok) {
      report.diagnosis = createDiagnosis({
        failureKind: "tab_management",
        summary: "The tool smoke page did not stay reusable as one managed browser tab.",
        nextAction: "Check the new tabs/debugger permissions and rerun the tab smoke test.",
        keyFacts: [
          { label: "sameTab", value: String(report.result.reusedSameTab) },
          { label: "articleCount", value: String(report.result.articleCount) },
          { label: "status", value: report.result.status || "unknown" },
        ],
      });
    }
  } catch (error) {
    report.ok = false;
    report.error = error instanceof Error ? error.message : String(error || "Unknown tab smoke error");
    report.diagnosis = createDiagnosis({
      failureKind: "tab_runtime",
      summary: trimText(report.error || "The tab-management smoke test failed.", 180),
      nextAction: "Reload the extension once. If it still fails, inspect the copied report and the new browser-tool permissions.",
    });
  }

  report.totalDurationMs = Date.now() - startedAt;
  return report;
}

export async function runBrowserToolVisualSmoke({
  describeResolvedModel,
  llmChat,
} = {}) {
  const startedAt = Date.now();
  const report = createToolSmokeReport("browser_tool_visual_smoke", {
    pageUrl: TOOL_SMOKE_PAGE_URL,
    slotId: "vision",
    expectedButtonLabel: "RELATION MAP",
  });

  try {
    const tab = await ensureToolSmokeTab({ active: true });
    const beforeSnapshot = await getToolSmokeSnapshot(tab.id);
    report.timingsMs.prepareTab = Date.now() - startedAt;

    const screenshotStartedAt = Date.now();
    const screenshotDataUrl = await captureToolSmokeScreenshot(tab);
    report.timingsMs.captureScreenshot = Date.now() - screenshotStartedAt;

    const resolveStartedAt = Date.now();
    const resolved = await describeResolvedModel({ slotId: "vision" });
    report.timingsMs.resolveVision = Date.now() - resolveStartedAt;

    const visionStartedAt = Date.now();
    const response = await llmChat({
      slotId: "vision",
      providerId: resolved?.providerId,
      modelName: resolved?.modelName,
      messages: [
        {
          role: "system",
          content: "Read the browser screenshot and reply with exactly the large orange button label. No punctuation.",
        },
        {
          role: "user",
          content: [
            {
              type: "text",
              text: "Reply with exactly the text on the large orange button.",
            },
            {
              type: "image",
              image: screenshotDataUrl,
              providerOptions: {
                openai: {
                  imageDetail: "high",
                },
              },
            },
          ],
        },
      ],
      parameters: {
        maxTokens: 16,
        temperature: 0,
        reasoningMode: "no_think",
      },
    });
    report.timingsMs.visionCall = Date.now() - visionStartedAt;

    const answer = asText(response?.text);
    const normalizedAnswer = normalizeAnswerText(answer);
    const matchedLabel = normalizedAnswer === "relation map";
    const callSucceeded = Boolean(response && typeof response === "object");
    let afterSnapshot = beforeSnapshot;
    if (matchedLabel) {
      await triggerToolSmokeAction(tab.id);
      afterSnapshot = await getToolSmokeSnapshot(tab.id);
    }

    report.result = {
      resolved: resolved || null,
      beforeSnapshot,
      afterSnapshot,
      answer,
      normalizedAnswer,
    };
    report.error = "";
    report.ok = Boolean(
      callSucceeded &&
        matchedLabel &&
        asText(afterSnapshot?.status) === "RELATION MAP OPEN" &&
        afterSnapshot?.relationOpen === true,
    );

    if (!report.ok) {
      report.diagnosis = callSucceeded
        ? createDiagnosis({
            failureKind: "vision_quality",
            summary: "The vision path did not read the tool-smoke button label reliably enough to continue.",
            nextAction: "Use a stronger vision slot, then rerun the visual browser-tool smoke test.",
            keyFacts: [
              { label: "answer", value: normalizedAnswer || "<empty>" },
              { label: "model", value: asText(resolved?.modelRef || resolved?.modelName) || "unknown" },
            ],
          })
        : createDiagnosis({
            failureKind: "vision_runtime",
            summary: trimText(report.error || "The visual browser-tool smoke test failed.", 180),
            nextAction: "Check the configured vision slot and rerun the test.",
            keyFacts: [
              { label: "model", value: asText(resolved?.modelRef || resolved?.modelName) || "unknown" },
            ],
          });
    }
  } catch (error) {
    report.ok = false;
    report.error = error instanceof Error ? error.message : String(error || "Unknown visual tool smoke error");
    report.diagnosis = createDiagnosis({
      failureKind: "visual_runtime",
      summary: trimText(report.error || "The visual browser-tool smoke test failed.", 180),
      nextAction: "Reload the extension once. If it still fails, inspect the copied report and the configured vision slot.",
    });
  }

  report.totalDurationMs = Date.now() - startedAt;
  return report;
}

export async function runBrowserToolSelfUseSmoke({
  describeResolvedModel,
  llmChat,
  providerId = "",
  modelName = "",
} = {}) {
  const startedAt = Date.now();
  const report = createToolSmokeReport("browser_tool_self_use_smoke", {
    pageUrl: TOOL_SMOKE_PAGE_URL,
    slotId: "vision",
    providerId: asText(providerId),
    modelName: asText(modelName),
    expectedTool: "open_relation_map",
  });

  try {
    const tab = await ensureToolSmokeTab({ active: true });
    const beforeSnapshot = await getToolSmokeSnapshot(tab.id);
    report.timingsMs.prepareTab = Date.now() - startedAt;

    const screenshotStartedAt = Date.now();
    const screenshotDataUrl = await captureToolSmokeScreenshot(tab);
    report.timingsMs.captureScreenshot = Date.now() - screenshotStartedAt;

    const resolveStartedAt = Date.now();
    const resolved = await describeResolvedModel({
      slotId: "vision",
      providerId: asText(providerId),
      modelName: asText(modelName),
    });
    report.timingsMs.resolveVision = Date.now() - resolveStartedAt;

    const providerType = asText(resolved?.provider?.type).toLowerCase();
    if (providerType !== "local_qwen") {
      throw new Error("The self-use smoke test requires a local_qwen vision model.");
    }

    const tools = [
      {
        type: "function",
        function: {
          name: "open_relation_map",
          description: "Open the visible RELATION MAP button when that button is clearly visible in the screenshot.",
          parameters: {
            type: "object",
            additionalProperties: false,
            properties: {},
          },
        },
      },
    ];

    const selfUseStartedAt = Date.now();
    const response = await llmChat({
      slotId: "vision",
      providerId: resolved?.providerId,
      modelName: resolved?.modelName,
      tools,
      messages: [
        {
          role: "system",
          content: [
            "You are a browser-safe visual agent.",
            "Return exactly one next assistant turn.",
            "If the large orange RELATION MAP button is visible, call open_relation_map with an empty object.",
            "If that button is not visible, reply with plain assistant text that the button is unavailable.",
            "Do not describe the page outside the tool decision.",
          ].join("\n"),
        },
        {
          role: "user",
          content: [
            {
              type: "text",
              text: "Inspect the screenshot and choose the next assistant action. Use open_relation_map if the RELATION MAP button is visible.",
            },
            {
              type: "image",
              image: screenshotDataUrl,
              providerOptions: {
                openai: {
                  imageDetail: "high",
                },
              },
            },
          ],
        },
      ],
      parameters: {
        maxTokens: 96,
        temperature: 0,
        reasoningMode: "no_think",
      },
    });
    report.timingsMs.selfUseCall = Date.now() - selfUseStartedAt;

    const responseText = asText(response?.text);
    const assistantTurn = normalizePortableAssistantTurn(responseText, {
      allowedToolNames: ["open_relation_map"],
    });
    const toolCall = Array.isArray(assistantTurn?.tool_calls) ? assistantTurn.tool_calls[0] || null : null;
    const toolName = asText(toolCall?.function?.name);
    const toolArguments = parseJsonObject(toolCall?.function?.arguments);
    const usedExpectedTool = toolName === "open_relation_map";
    const usedEmptyArgs = usedExpectedTool && Object.keys(toolArguments).length === 0;

    let afterSnapshot = beforeSnapshot;
    if (usedExpectedTool) {
      await triggerToolSmokeAction(tab.id);
      afterSnapshot = await getToolSmokeSnapshot(tab.id);
    }

    report.result = {
      resolved: resolved || null,
      beforeSnapshot,
      afterSnapshot,
      responseText,
      assistantTurn: assistantTurn || null,
      toolName,
      toolArguments,
    };
    report.error = "";
    report.ok = Boolean(
      usedExpectedTool &&
        usedEmptyArgs &&
        asText(afterSnapshot?.status) === "RELATION MAP OPEN" &&
        afterSnapshot?.relationOpen === true,
    );

    if (!report.ok) {
      report.diagnosis = usedExpectedTool
        ? createDiagnosis({
            failureKind: "tool_execution",
            summary: "The self-use smoke emitted the expected tool call, but the relation map did not open cleanly.",
            nextAction: "Inspect the copied report for the post-click snapshot and rerun the self-use smoke test.",
            keyFacts: [
              { label: "tool", value: toolName || "<empty>" },
              { label: "status", value: asText(afterSnapshot?.status) || "unknown" },
            ],
          })
        : createDiagnosis({
            failureKind: "tool_routing",
            summary: "The local multimodal Qwen run did not emit the expected reviewed tool call from the screenshot.",
            nextAction: "Use a stronger local vision model or simplify the visual prompt, then rerun the self-use smoke test.",
            keyFacts: [
              { label: "tool", value: toolName || "<none>" },
              { label: "reply", value: trimText(responseText || "<empty>", 140) },
              { label: "model", value: asText(resolved?.modelRef || resolved?.modelName) || "unknown" },
            ],
          });
    }
  } catch (error) {
    report.ok = false;
    report.error = error instanceof Error ? error.message : String(error || "Unknown self-use smoke error");
    report.diagnosis = createDiagnosis({
      failureKind: "self_use_runtime",
      summary: trimText(report.error || "The local self-use smoke test failed.", 180),
      nextAction: "Reload the extension once. If it still fails, inspect the copied report and the configured local vision slot.",
    });
  }

  report.totalDurationMs = Date.now() - startedAt;
  return report;
}

export async function runBrowserToolCodeSmoke() {
  const startedAt = Date.now();
  const report = createToolSmokeReport("browser_tool_code_smoke", {
    pageUrl: TOOL_SMOKE_PAGE_URL,
    path: "freeform_code",
  });

  try {
    const tab = await ensureToolSmokeTab({ active: true });
    const beforeSnapshot = await getToolSmokeSnapshot(tab.id);
    const codeStartedAt = Date.now();
    const codeResult = await runToolSmokeCode(tab.id);
    report.timingsMs.executeCode = Date.now() - codeStartedAt;
    const afterSnapshot = await getToolSmokeSnapshot(tab.id);

    report.result = {
      beforeSnapshot,
      afterSnapshot,
      codeResult,
    };
    const relationText = normalizeAnswerText(codeResult?.relationText);
    report.ok = Boolean(
      codeResult?.ok &&
        codeResult?.clicked === true &&
        Number(codeResult?.articleCount || 0) === 5 &&
        Array.isArray(codeResult?.articleTitles) &&
        codeResult.articleTitles.length === 5 &&
        asText(afterSnapshot?.status) === "RELATION MAP OPEN" &&
        relationText.includes("infrastructure") &&
        relationText.includes("regulation"),
    );
    if (!report.ok) {
      report.diagnosis = createDiagnosis({
        failureKind: "code_path",
        summary: "The freeform browser-code path did not complete the tool smoke workflow.",
        nextAction: "Inspect the copied report for the returned code payload and verify the debugger-backed code path.",
        keyFacts: [
          { label: "status", value: asText(afterSnapshot?.status) || "unknown" },
          { label: "articleCount", value: String(Number(codeResult?.articleCount || 0)) },
        ],
      });
    }
  } catch (error) {
    report.ok = false;
    report.error = error instanceof Error ? error.message : String(error || "Unknown code tool smoke error");
    report.diagnosis = createDiagnosis({
      failureKind: "code_runtime",
      summary: trimText(report.error || "The browser-code smoke test failed.", 180),
      nextAction: "Reload the extension once. If it still fails, inspect the copied report and the debugger-backed code path.",
    });
  }

  report.totalDurationMs = Date.now() - startedAt;
  return report;
}
