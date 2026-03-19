import { llmChat } from "./llm.js";
import { FEW_SHOT_PLAYWRIGHT_CODE_EXAMPLES } from "../shared/browserAutomationFewShots.js";
import {
  DEFAULT_BROWSER_VISION_MODEL_REF,
  runRestrictedBrowserAction,
  runRestrictedBrowserInspection,
  runRestrictedBrowserScript,
} from "../shared/browserAutomationRuntime.js";

const DEFAULT_MAX_TURNS = 10;
const DEFAULT_TOOL_TIMEOUT_MS = 45_000;
const ALLOWED_TOOL_NAMES = new Set(["inspect", "action", "script", "finish"]);

function asString(value) {
  return String(value == null ? "" : value);
}

function asArray(value) {
  return Array.isArray(value) ? value : [];
}

function asInt(value, fallback = 0, min = Number.MIN_SAFE_INTEGER, max = Number.MAX_SAFE_INTEGER) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.max(min, Math.min(max, Math.floor(parsed)));
}

function trimLine(value, max = 320) {
  const text = asString(value).replace(/\s+/g, " ").trim();
  if (text.length <= max) return text;
  return `${text.slice(0, Math.max(1, max - 1)).trimEnd()}...`;
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
    return null;
  }
}

function normalizeHost(host) {
  return asString(host).trim().toLowerCase().replace(/^\.+|\.+$/g, "");
}

function deriveAllowedHosts({ allowedHosts = [], startUrl = "", baseUrl = "" } = {}) {
  const out = [];
  const seen = new Set();
  for (const item of asArray(allowedHosts)) {
    const normalized = normalizeHost(item);
    if (!normalized || seen.has(normalized)) continue;
    seen.add(normalized);
    out.push(normalized);
  }
  for (const urlText of [startUrl, baseUrl]) {
    try {
      const normalized = normalizeHost(new URL(asString(urlText).trim()).hostname);
      if (!normalized || seen.has(normalized)) continue;
      seen.add(normalized);
      out.push(normalized);
    } catch {}
  }
  return out.length ? out : ["example.com"];
}

function clipStructured(value, maxChars = 8_000) {
  try {
    const json = JSON.stringify(value ?? null);
    if (json.length <= maxChars) {
      return {
        clipped: false,
        text: json,
        value: value ?? null,
      };
    }
    return {
      clipped: true,
      text: `${json.slice(0, Math.max(1, maxChars - 1)).trimEnd()}...`,
      value: null,
    };
  } catch {
    const fallback = trimLine(value, maxChars);
    return {
      clipped: true,
      text: fallback,
      value: null,
    };
  }
}

function normalizeSeedRecord(raw, index = 0) {
  const source = raw && typeof raw === "object" ? raw : {};
  const title = trimLine(
    source.title || source.headline || source.name || source.subject || source.text || "",
    220,
  );
  const url = asString(source.url || source.link || "").trim() || null;
  const summary = trimLine(source.summary || source.abstract || source.snippet || "", 500);
  const bodyText = asString(
    source.body_text || source.bodyText || source.content || source.description || summary || "",
  ).trim();
  const categories =
    source.categories && typeof source.categories === "object" ? { ...source.categories } : {};
  const attributes =
    source.attributes && typeof source.attributes === "object"
      ? { ...source.attributes }
      : { ...source };
  delete attributes.title;
  delete attributes.url;
  delete attributes.body_text;
  delete attributes.bodyText;
  delete attributes.content;
  delete attributes.description;
  delete attributes.summary;
  delete attributes.abstract;
  delete attributes.snippet;
  delete attributes.categories;
  delete attributes.attributes;
  return {
    seed_id: asString(source.seed_id || source.id || `browser-seed-${index + 1}`).trim(),
    title,
    body_text: bodyText,
    url,
    summary,
    attributes,
    categories,
  };
}

function harvestRecords(value) {
  if (!value || typeof value !== "object") return [];
  if (Array.isArray(value.records)) return value.records;
  if (Array.isArray(value.items)) return value.items;
  if (value.evidence && typeof value.evidence === "object") {
    if (Array.isArray(value.evidence.records)) return value.evidence.records;
    if (Array.isArray(value.evidence.items)) return value.evidence.items;
  }
  return [];
}

function buildCollectionObjective(request) {
  const runtimeConfig = request?.runtime_config && typeof request.runtime_config === "object"
    ? request.runtime_config
    : {};
  const taskSpec = request?.task_spec && typeof request.task_spec === "object"
    ? request.task_spec
    : {};
  const recipePayload = request?.recipe_payload && typeof request.recipe_payload === "object"
    ? request.recipe_payload
    : {};
  const explicit =
    runtimeConfig.prompt ||
    runtimeConfig.collectionPrompt ||
    runtimeConfig.instructions ||
    recipePayload.runtimeConfig?.prompt ||
    "";
  if (asString(explicit).trim()) return asString(explicit).trim();
  const lines = [
    asString(taskSpec.task_name || "").trim(),
    asString(taskSpec.task_goal || "").trim(),
    asString(taskSpec.output_schema || "").trim()
      ? `Output schema hint: ${asString(taskSpec.output_schema).trim()}`
      : "",
  ].filter(Boolean);
  return lines.join("\n").trim() || "Collect grounded browser seed records for the requested task.";
}

function buildPlannerMessages({
  objective,
  request,
  state,
  startUrl,
  baseUrl,
  allowedHosts,
}) {
  const fewShots = FEW_SHOT_PLAYWRIGHT_CODE_EXAMPLES.slice(0, 2).map((shot) => ({
    title: shot.title,
    intent: shot.intent,
    code: shot.code,
  }));
  const payload = {
    task: {
      kind: request?.kind || "browser_collection",
      objective,
      start_url: startUrl,
      base_url: baseUrl,
      allowed_hosts: allowedHosts,
      expected_finish_shape: {
        summary: "string",
        records: [
          {
            title: "string",
            url: "string",
            summary: "string",
            body_text: "string",
            attributes: {},
            categories: {},
          },
        ],
        evidence: ["string"],
      },
    },
    state,
    few_shot_playwright_examples: fewShots,
  };

  return [
    {
      role: "system",
      content: [
        "You are the browser-automation sub-agent for the training factory.",
        "Return exactly one JSON object and nothing else.",
        "Available tool_name values: inspect, action, script, finish.",
        "Use inspect first to visually understand the current page before clicking or writing Playwright code.",
        "Use action for tab-visible interactions like click, type, scroll, keypress, wait or drag.",
        "Use script when you need reliable DOM extraction, article reading, pagination or multi-step automation with Playwright.",
        "Do not invent facts or records. Only finish when you have grounded evidence.",
        "For browser_collection you should usually finish with records containing title, url, summary and body_text.",
        "Keep tool_args minimal and deterministic.",
        'JSON schema: {"thought":"string","tool_name":"inspect|action|script|finish","tool_args":{},"final":{"summary":"string","records":[],"evidence":["string"]}}',
      ].join("\n"),
    },
    {
      role: "user",
      content: JSON.stringify(payload, null, 2),
    },
  ];
}

async function callPlanner({
  objective,
  request,
  state,
  startUrl,
  baseUrl,
  allowedHosts,
  plannerModelRef = "",
  plannerReasoningEffort = "medium",
}) {
  const response = await llmChat({
    slotId: "agent",
    modelRef: plannerModelRef,
    reasoningEffort: asString(plannerReasoningEffort).trim() || "medium",
    parameters: {
      temperature: 0.15,
      maxTokens: 1400,
    },
    messages: buildPlannerMessages({
      objective,
      request,
      state,
      startUrl,
      baseUrl,
      allowedHosts,
    }),
  });
  const text = asString(response?.text || "");
  let parsed = parseJsonLoose(text);
  if (!parsed || typeof parsed !== "object") {
    const repair = await llmChat({
      slotId: "agent",
      modelRef: plannerModelRef,
      reasoningEffort: "minimal",
      parameters: {
        temperature: 0,
        maxTokens: 900,
      },
      messages: [
        {
          role: "system",
          content: [
            "Repair invalid planner output into valid JSON.",
            "Return exactly one JSON object with keys thought, tool_name, tool_args and optional final.",
            "Do not add markdown or explanations.",
          ].join("\n"),
        },
        {
          role: "user",
          content: JSON.stringify({ candidate_text: text }, null, 2),
        },
      ],
    });
    parsed = parseJsonLoose(repair?.text || "");
  }
  if (!parsed || typeof parsed !== "object") {
    throw new Error("Browser planner returned invalid JSON.");
  }
  const toolName = asString(parsed.tool_name || "").trim().toLowerCase();
  if (!ALLOWED_TOOL_NAMES.has(toolName)) {
    throw new Error(`Browser planner requested unsupported tool: ${toolName || "unknown"}`);
  }
  return {
    thought: trimLine(parsed.thought || "", 280),
    toolName,
    toolArgs: parsed.tool_args && typeof parsed.tool_args === "object" ? parsed.tool_args : {},
    final: parsed.final && typeof parsed.final === "object" ? parsed.final : {},
    rawText: text,
  };
}

function summarizeToolResult(toolName, result) {
  if (!result || typeof result !== "object") {
    return `${toolName}: no result`;
  }
  if (result.ok === false) {
    return `${toolName}: ${trimLine(result.error || "failed", 220)}`;
  }
  if (toolName === "inspect") {
    return trimLine(
      [
        `inspect ok`,
        asString(result?.data?.title || ""),
        asString(result?.data?.answer || ""),
      ].filter(Boolean).join(" | "),
      260,
    );
  }
  if (toolName === "script") {
    return trimLine(
      [
        `script ok`,
        asString(result?.data?.title || ""),
        asString(result?.data?.raw_preview || ""),
      ].filter(Boolean).join(" | "),
      260,
    );
  }
  return trimLine(`${toolName} ok ${asString(result?.data?.title || result?.data?.url || "")}`, 260);
}

function buildTraceEntry(turn, planner, toolResult) {
  return {
    turn,
    thought: planner.thought,
    tool_name: planner.toolName,
    tool_args: planner.toolArgs,
    tool_summary: summarizeToolResult(planner.toolName, toolResult),
    ok: toolResult?.ok !== false,
  };
}

async function executePlannerTool({
  runtime,
  planner,
  objective,
  startUrl,
  baseUrl,
  allowedHosts,
  runtimeConfig,
}) {
  const common = {
    url: asString(planner.toolArgs?.url || startUrl || "").trim(),
    active: planner.toolArgs?.active === true,
    baseUrl,
    allowedHosts,
  };

  if (planner.toolName === "inspect") {
    return await runRestrictedBrowserInspection(runtime, {
      ...common,
      question: asString(planner.toolArgs?.question || objective).trim() || objective,
      visionModelRef: asString(runtimeConfig?.visionModelRef || DEFAULT_BROWSER_VISION_MODEL_REF).trim(),
      visionBaseUrl: asString(
        runtimeConfig?.visionBaseUrl ||
        runtimeConfig?.preferredVisionBaseUrl ||
        runtimeConfig?.localBaseUrl ||
        "",
      ).trim(),
      visionApiKey: asString(
        runtimeConfig?.visionApiKey ||
        runtimeConfig?.preferredVisionApiKey ||
        runtimeConfig?.localApiKey ||
        "",
      ).trim(),
      imageDetail: asString(runtimeConfig?.imageDetail || "high").trim() || "high",
    });
  }

  if (planner.toolName === "action") {
    return await runRestrictedBrowserAction(runtime, {
      ...common,
      action: planner.toolArgs?.action,
      target: planner.toolArgs?.target,
      destination: planner.toolArgs?.destination,
      deltaX: planner.toolArgs?.deltaX,
      deltaY: planner.toolArgs?.deltaY,
      textValue: planner.toolArgs?.textValue,
      keys: planner.toolArgs?.keys,
      clear: planner.toolArgs?.clear,
      waitMs: planner.toolArgs?.waitMs,
      timeoutMs: planner.toolArgs?.timeoutMs,
      button: planner.toolArgs?.button,
      steps: planner.toolArgs?.steps,
    });
  }

  if (planner.toolName === "script") {
    return await runRestrictedBrowserScript(runtime, {
      ...common,
      code: asString(planner.toolArgs?.code || "").trim(),
      timeoutMs: asInt(
        planner.toolArgs?.timeoutMs || runtimeConfig?.codeTimeoutMs || DEFAULT_TOOL_TIMEOUT_MS,
        DEFAULT_TOOL_TIMEOUT_MS,
        500,
        180_000,
      ),
    });
  }

  return {
    ok: false,
    error: `Unsupported tool execution: ${planner.toolName}`,
    data: null,
  };
}

export async function runBrowserCollectionSubagent(request = {}) {
  const runtimeConfig = request?.runtime_config && typeof request.runtime_config === "object"
    ? request.runtime_config
    : {};
  const startUrl = asString(
    runtimeConfig.startUrl ||
    runtimeConfig.url ||
    runtimeConfig.targetUrl ||
    runtimeConfig.baseUrl ||
    request?.task_spec?.source_url ||
    "",
  ).trim();
  const baseUrl = asString(runtimeConfig.baseUrl || startUrl || "https://example.com/").trim() || "https://example.com/";
  const allowedHosts = deriveAllowedHosts({
    allowedHosts: runtimeConfig.allowedHosts,
    startUrl,
    baseUrl,
  });
  const objective = buildCollectionObjective(request);
  const maxTurns = asInt(runtimeConfig.maxTurns || runtimeConfig.max_steps, DEFAULT_MAX_TURNS, 1, 20);
  const plannerModelRef = asString(runtimeConfig.plannerModelRef || runtimeConfig.modelRef || "").trim();
  const plannerReasoningEffort = asString(runtimeConfig.plannerReasoningEffort || "medium").trim() || "medium";
  const runtime = {};
  const trace = [];
  let lastToolResult = null;

  for (let turn = 1; turn <= maxTurns; turn += 1) {
    const clippedLast = clipStructured(lastToolResult, 8_000);
    const planner = await callPlanner({
      objective,
      request,
      state: {
        turn,
        max_turns: maxTurns,
        last_tool_result: clippedLast.text,
        trace,
      },
      startUrl,
      baseUrl,
      allowedHosts,
      plannerModelRef,
      plannerReasoningEffort,
    });

    if (planner.toolName === "finish") {
      const rawRecords = asArray(planner.final?.records).length
        ? planner.final.records
        : harvestRecords(lastToolResult?.data?.raw || {});
      const records = rawRecords.map((item, index) => normalizeSeedRecord(item, index));
      return {
        ok: true,
        summary: trimLine(
          planner.final?.summary ||
            `Browser sub-agent finished after ${turn} turns with ${records.length} records.`,
          500,
        ),
        records,
        evidence: asArray(planner.final?.evidence).map((item) => trimLine(item, 320)).slice(0, 16),
        trace,
      };
    }

    lastToolResult = await executePlannerTool({
      runtime,
      planner,
      objective,
      startUrl,
      baseUrl,
      allowedHosts,
      runtimeConfig,
    });
    trace.push(buildTraceEntry(turn, planner, lastToolResult));
  }

  const fallbackRecords = harvestRecords(lastToolResult?.data?.raw || {}).map((item, index) =>
    normalizeSeedRecord(item, index),
  );
  return {
    ok: false,
    error: `Browser sub-agent reached maxTurns=${maxTurns} without finish.`,
    summary: `Browser sub-agent reached maxTurns=${maxTurns} without finish.`,
    records: fallbackRecords,
    evidence: trace.map((entry) => entry.tool_summary).slice(-8),
    trace,
  };
}

export async function runBrowserActionCapabilityTest(request = {}) {
  const runtimeConfig = request?.runtime_config && typeof request.runtime_config === "object"
    ? request.runtime_config
    : {};
  const recipePayload = request?.recipe_payload && typeof request.recipe_payload === "object"
    ? request.recipe_payload
    : {};
  const startUrl = asString(
    runtimeConfig.startUrl ||
    runtimeConfig.url ||
    runtimeConfig.targetUrl ||
    runtimeConfig.baseUrl ||
    "",
  ).trim();
  const baseUrl = asString(runtimeConfig.baseUrl || startUrl || "https://example.com/").trim() || "https://example.com/";
  const allowedHosts = deriveAllowedHosts({
    allowedHosts: runtimeConfig.allowedHosts,
    startUrl,
    baseUrl,
  });
  const code = asString(
    request?.code ||
    runtimeConfig.playwrightCode ||
    runtimeConfig.code ||
    recipePayload.runtimeConfig?.playwrightCode ||
    recipePayload.runtimeConfig?.code ||
    "",
  ).trim();
  const runtime = {};
  const tests = [
    {
      name: "bridge_request_present",
      ok: true,
      detail: "Browser action request reached extension worker.",
    },
    {
      name: "start_url_present",
      ok: !!startUrl || !!baseUrl,
      detail: "Capability test needs a target URL or base URL.",
    },
    {
      name: "playwright_code_present",
      ok: !!code,
      detail: "Capability test needs Playwright code.",
    },
  ];

  let result = null;
  if (code) {
    result = await runRestrictedBrowserScript(runtime, {
      url: startUrl || baseUrl,
      baseUrl,
      allowedHosts,
      code,
      timeoutMs: asInt(runtimeConfig.codeTimeoutMs || request?.timeout_ms, DEFAULT_TOOL_TIMEOUT_MS, 500, 180_000),
    });
    tests.push({
      name: "playwright_execution_ok",
      ok: !!result?.ok,
      detail: trimLine(result?.error || result?.data?.raw_preview || "Playwright execution finished.", 320),
    });
    tests.push({
      name: "playwright_result_has_payload",
      ok: !!asString(result?.data?.raw_preview || "").trim() || result?.data?.raw != null,
      detail: "Capability should return structured evidence or payload.",
    });
  }

  const passed = tests.filter((item) => item.ok).length;
  return {
    ok: tests.every((item) => item.ok),
    summary: `Browser action capability test: ${passed}/${tests.length} checks passed.`,
    tests,
    data: result?.data || null,
  };
}

export async function handleBrowserAgentBridgeRequest(request = {}) {
  const kind = asString(request?.kind || "").trim().toLowerCase();
  if (kind === "browser_collection") {
    return await runBrowserCollectionSubagent(request);
  }
  if (kind === "browser_action_test") {
    return await runBrowserActionCapabilityTest(request);
  }
  return {
    ok: false,
    error: `Unsupported browser bridge kind: ${kind || "unknown"}`,
    summary: `Unsupported browser bridge kind: ${kind || "unknown"}`,
  };
}
