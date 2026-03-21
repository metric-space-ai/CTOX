import "../shared/craft-storage.js";

import {
  Agent,
  Runner,
  RunState,
  aisdk,
  generateText,
  tool,
  webSearchTool,
  z,
} from "../vendor/agent-bundle.mjs";
import {
  getLanguageModelForSlot,
  llmChat,
  planTrainingSampleOps,
  resolveProviderAndModel,
} from "./llm.js";
import {
  runRestrictedBrowserAction,
  runRestrictedBrowserInspection,
  runRestrictedBrowserScript,
} from "../shared/browserAutomationRuntime.js";
import {
  formatCraftingAgentToolLabels,
  normalizeCraftingAgentToolingPayload,
} from "../shared/crafting-agent-tooling.mjs";
import {
  createAgentReportedCraftMaturity,
  createEmptyCraftMaturity,
  gateCraftMaturityForCapability,
  normalizeCraftMaturity,
} from "../shared/craft-maturity.mjs";
import {
  CRAFTING_AGENT_DESCRIPTION_CONTRACT,
  CRAFTING_AGENT_FINAL_OUTPUT_CONTRACT,
  CRAFTING_AGENT_MATURITY_CONTRACT,
  CRAFTING_AGENT_NAME_CONTRACT,
  CRAFTING_AGENT_PROVENANCE_CONTRACT,
  CRAFTING_AGENT_QUESTION_CONTRACT,
  CRAFTING_AGENT_REPORT_CONTRACT,
  CRAFTING_AGENT_TRAINING_DRAFT_CONTRACT,
  CRAFTING_AGENT_USE_SURFACE_CONTRACT,
} from "../shared/crafting-agent-contracts.mjs";
import {
  listAgentRunStates,
  upsertAgentRunState,
} from "../shared/agent-run-store.mjs";
import {
  extractStructuredOutputObject,
  repairStructuredOutputText,
} from "../shared/structured-output-repair.mjs";
import {
  BROWSER_CAPABILITY_BUNDLE_ARTIFACT_KIND,
  buildCapabilityBundle,
  getBrowserCapabilityBundleArtifactId,
  getToolScriptsArtifactId,
  normalizeToolScriptsPayload,
  POLICY_BUNDLE_ARTIFACT_KIND,
  TOOL_SCRIPTS_ARTIFACT_KIND,
  TRAINING_DATA_ARTIFACT_KIND,
  WEIGHTS_ARTIFACT_KIND,
} from "../shared/capability-bundle.mjs";
import {
  BROWSER_CAPABILITY_RUNTIME_PACKAGE_VERSION,
  compilePublishedBrowserCapabilityBundlePayload,
  normalizeBrowserCapabilityBundlePayload,
} from "../shared/browser-capability-bundle.mjs";
import {
  inspectPortableTrainingTrace,
} from "../shared/local_qwen_training_contract.mjs";
import {
  getPreferredLocalQwenVanillaModelName,
  getLocalQwenModelRepoUrl,
  getLocalQwenRuntimePlan,
  getSupportedLocalQwenModels,
} from "../shared/local_qwen_runtime.mjs";
import { readUiPreferences } from "../shared/ui-preferences.mjs";
import { runCraftUse } from "./craft-use-runner.js";
import { sendMessageToOffscreen } from "./offscreen-bridge.reference.js";

const craftSync = globalThis.SinepanelCraftSync;
const craftStore = globalThis.SinepanelCraftStorage;

const RUNS = new Map();
const DEFAULT_MAX_TURNS = 120;
const DEFAULT_LOCAL_BROWSER_TRAINING_MODEL = "unsloth/Qwen3.5-0.8B (Vanilla)";
const DEFAULT_SEGMENT_MAX_TURNS = 12;
const MAX_COMPACTION_PASSES = Math.ceil(DEFAULT_MAX_TURNS / DEFAULT_SEGMENT_MAX_TURNS);
const MAX_COMPACT_MEMORIES = 6;
const MAX_SERIALIZED_STATE_CHARS = 72_000;
const MAX_TOOL_TRACE = 10;
const MAX_EXPERIMENT_LEDGER = 48;
const MAX_LOGS = 120;
const DEFAULT_SUPERVISOR_TOOL_TIMEOUT_MS = 90_000;
const SUPERVISOR_TOOL_TIMEOUTS_MS = Object.freeze({
  inspect_training_data: 12_000,
  inspect_bundle_snapshot: 12_000,
  workspace_code_dive: 25_000,
  web_search: 30_000,
  browser_inspect: 60_000,
  browser_action: 60_000,
  browser_tabs: 20_000,
  playwright_ctx: 60_000,
  save_tool_scripts: 20_000,
  save_browser_capabilities: 20_000,
  request_codex_repair: 20_000,
  get_codex_repair_status: 30_000,
  run_agentic_smoke: 90_000,
  run_capability_eval: 120_000,
  generate_training_batch: 180_000,
  start_training_run: 45_000,
  get_training_run_status: 20_000,
  record_experiment_decision: 12_000,
  draft_training_changes: 60_000,
});
const DDG_SEARCH_ENDPOINT = "https://html.duckduckgo.com/html/";
const SEARCH_RESULT_LIMIT = 5;
const DEFAULT_LOCAL_TRAINING_BRIDGE_BASE_URL = "http://127.0.0.1:8765";
const ASK_USER_TOOL_NAME = "ask_user_clarification";
const FINAL_OUTPUT_STATUSES = new Set(["done", "blocked", "continue"]);
const RUN_PERSIST_DEBOUNCE_MS = 80;
const MAX_TERMINAL_RUNS_IN_MEMORY = 24;
const TERMINAL_RUN_RETENTION_MS = 20 * 60 * 1000;
const DEV_MODE_PREFERENCE_CACHE_MS = 1500;
const DEV_OBSERVABILITY_FLUSH_MS = 80;
const DEV_OBSERVABILITY_MAX_BUFFER = 240;
const DEV_OBSERVABILITY_LOG_TAIL = 10;
const DEV_OBSERVABILITY_TOOL_TRACE_TAIL = 8;
const AUTO_DATASET_GROWTH_TARGET_MATURITY_PERCENT = 90;
const AUTO_DATASET_GROWTH_MIN_READY_SAMPLES = 8;
const AUTO_DATASET_GROWTH_MIN_RUNNABLE_SAMPLES = 8;
const AUTO_DATASET_GROWTH_MAX_BATCH_ATTEMPTS = 2;
const AUTO_DATASET_GROWTH_BATCH_SIZE = 8;
const STRICT_RUNTIME_TOOL_SCRIPTS_OPTIONS = Object.freeze({
  inferPlaceholderScripts: false,
  allowToolFallback: false,
});
const STRICT_RUNTIME_BROWSER_CAPABILITY_OPTIONS = Object.freeze({
  allowToolNameInference: false,
  allowSyntheticCapabilities: false,
  allowFallbackExecuteScript: false,
  allowBundleFallback: false,
});
const PROVIDER_WEB_SEARCH_TYPES = new Set(["openai", "azure_openai", "anthropic"]);
const BROWSER_TOOL_ACTIONS = new Set([
  "browser_inspect",
  "browser_action",
  "browser_tabs",
  "playwright_ctx",
]);
const PASSIVE_DIAGNOSIS_TOOL_ACTIONS = new Set([
  "inspect_training_data",
  "inspect_bundle_snapshot",
  "get_training_run_status",
  "workspace_code_dive",
]);
const CODEX_REPAIR_TOOL_ACTIONS = new Set([
  "request_codex_repair",
  "get_codex_repair_status",
]);
const NON_WORKSPACE_DIAGNOSIS_TOOL_ACTIONS = new Set([
  ...PASSIVE_DIAGNOSIS_TOOL_ACTIONS,
  ...CODEX_REPAIR_TOOL_ACTIONS,
]);
const devModePreferenceCache = {
  value: false,
  loadedAt: 0,
  promise: null,
};
const devObservabilityQueue = [];
let devObservabilityFlushTimer = 0;
let devObservabilityFlushPromise = null;
const DEFAULT_BROWSER_TOOL_BASE_URL = "https://example.com/";
const ACTIVE_TEXT_SMOKE_FIXTURE_ELEMENT_ID = "sinepanel-active-text-fixture";
const ACTIVE_TEXT_SMOKE_FIXTURE_TEXT = "Ths is a smple txt with speling erors and bad grammer.";
const WORKSPACE_REPO_ROOT = "fuck-api-train-local-ai";
const WORKSPACE_CODE_DIVE_MAX_FILES = 6;
const WORKSPACE_CODE_DIVE_MAX_MATCHES_PER_FILE = 3;
const WORKSPACE_CODE_DIVE_SNIPPET_RADIUS = 2;
const MAX_CODEX_REPAIR_ATTEMPTS_PER_FINGERPRINT = 1;
const AUTO_CODEX_REPAIR_STATUS_WAIT_MS = 10_000;
const AUTO_CODEX_REPAIR_POLL_INTERVAL_MS = 2_000;
const MAX_CODEX_REPAIR_STATUS_WAIT_MS = 8_000;
const WORKSPACE_CODE_DIVE_STOP_WORDS = new Set([
  "the",
  "der",
  "die",
  "das",
  "dem",
  "den",
  "and",
  "und",
  "oder",
  "mit",
  "with",
  "from",
  "fuer",
  "fur",
  "that",
  "this",
  "there",
  "have",
  "will",
  "dann",
  "weil",
  "eine",
  "einen",
  "einer",
  "eines",
  "einem",
  "eines",
  "when",
  "after",
  "before",
  "into",
  "unter",
  "uber",
  "ueber",
  "nicht",
  "noch",
  "auch",
  "then",
  "kann",
  "koennte",
  "soll",
  "sollte",
  "wird",
  "wurde",
  "werden",
  "wieder",
  "wenn",
  "direkt",
  "lokale",
  "lokalen",
  "local",
  "try",
  "running",
  "while",
  "per",
  "ich",
  "muss",
  "beim",
  "ersten",
  "derzeit",
  "zum",
  "von",
  "bei",
  "ein",
  "aus",
  "aber",
  "zwar",
  "wie",
  "not",
  "fehlgeschlagen",
  "please",
  "again",
  "error",
  "failed",
  "tool",
  "runtime",
]);
const WORKSPACE_CODE_DIVE_FILE_MANIFEST = Object.freeze([
  Object.freeze({
    runtimePath: "manifest.json",
    repoPath: `${WORKSPACE_REPO_ROOT}/manifest.json`,
    subsystem: "extension_shell",
    tags: ["mv3", "permissions", "sidepanel", "service_worker"],
    description: "Chrome extension manifest, permissions, background worker, and sidepanel entrypoint.",
  }),
  Object.freeze({
    runtimePath: "sidepanel.js",
    repoPath: `${WORKSPACE_REPO_ROOT}/sidepanel.js`,
    subsystem: "sidepanel_ui",
    tags: ["sidepanel", "debug_json", "agent_run_ui", "clipboard"],
    description: "Sidepanel state, agent progress UI, and debug JSON export.",
  }),
  Object.freeze({
    runtimePath: "bg/service_worker.js",
    repoPath: `${WORKSPACE_REPO_ROOT}/bg/service_worker.js`,
    subsystem: "background_bridge",
    tags: ["service_worker", "runtime_messages", "background"],
    description: "Background message router that dispatches agent, model, and smoke-test requests.",
  }),
  Object.freeze({
    runtimePath: "bg/crafting-agent-runner.js",
    repoPath: `${WORKSPACE_REPO_ROOT}/bg/crafting-agent-runner.js`,
    subsystem: "crafting_agent",
    tags: ["crafting_agent", "runner", "tools", "smoke", "diagnosis"],
    description: "Crafting-agent supervisor, tool definitions, run lifecycle, and blocked-state handling.",
  }),
  Object.freeze({
    runtimePath: "bg/craft-use-runner.js",
    repoPath: `${WORKSPACE_REPO_ROOT}/bg/craft-use-runner.js`,
    subsystem: "capability_runtime",
    tags: ["craft_use", "reviewed_capabilities", "active_text", "tool_runtime"],
    description: "Runtime that executes reviewed capabilities and active-text tool flows.",
  }),
  Object.freeze({
    runtimePath: "bg/llm.js",
    repoPath: `${WORKSPACE_REPO_ROOT}/bg/llm.js`,
    subsystem: "model_bridge",
    tags: ["llm", "local_qwen", "provider_resolution", "offscreen_bridge"],
    description: "Model resolution and local_qwen handoff into the offscreen runtime.",
  }),
  Object.freeze({
    runtimePath: "bg/ml-inference.js",
    repoPath: `${WORKSPACE_REPO_ROOT}/bg/ml-inference.js`,
    subsystem: "local_runtime",
    tags: ["local_qwen", "onnx", "webgpu", "offscreen", "training"],
    description: "Offscreen local Qwen WebGPU/ONNX runtime, diagnostics, and training execution.",
  }),
  Object.freeze({
    runtimePath: "shared/capability-bundle.mjs",
    repoPath: `${WORKSPACE_REPO_ROOT}/shared/capability-bundle.mjs`,
    subsystem: "bundle_storage",
    tags: ["bundle", "tool_scripts", "browser_capabilities", "policy"],
    description: "Capability bundle assembly and artifact IDs for tool scripts, capabilities, weights, and policy.",
  }),
  Object.freeze({
    runtimePath: "shared/browser-capability-bundle.mjs",
    repoPath: `${WORKSPACE_REPO_ROOT}/shared/browser-capability-bundle.mjs`,
    subsystem: "capability_contracts",
    tags: ["browser_capabilities", "active_text", "tool_contract"],
    description: "Reviewed browser capability normalization and active-text capability defaults.",
  }),
  Object.freeze({
    runtimePath: "shared/browserAutomationRuntime.js",
    repoPath: `${WORKSPACE_REPO_ROOT}/shared/browserAutomationRuntime.js`,
    subsystem: "browser_runtime",
    tags: ["browser_runtime", "script_eval", "cdp", "playwright_code"],
    description: "Restricted browser runtime wrapper that executes reviewed browser scripts and returns execution errors.",
  }),
  Object.freeze({
    runtimePath: "shared/browserAutomationTools.js",
    repoPath: `${WORKSPACE_REPO_ROOT}/shared/browserAutomationTools.js`,
    subsystem: "browser_runtime",
    tags: ["browser_runtime", "cdp", "execute_script", "injection"],
    description: "Low-level browser automation and CDP runtime evaluation used for reviewed capability scripts.",
  }),
  Object.freeze({
    runtimePath: "shared/crafting-agent-contracts.mjs",
    repoPath: `${WORKSPACE_REPO_ROOT}/shared/crafting-agent-contracts.mjs`,
    subsystem: "agent_contracts",
    tags: ["contracts", "structured_output", "agent_report"],
    description: "Structured output contracts for crafting-agent final reports and training drafts.",
  }),
  Object.freeze({
    runtimePath: "shared/crafting-agent-tooling.mjs",
    repoPath: `${WORKSPACE_REPO_ROOT}/shared/crafting-agent-tooling.mjs`,
    subsystem: "agent_tooling",
    tags: ["tooling", "supervisor_tools", "tool_labels"],
    description: "Fixed supervisor tool labels and sidepanel tooling metadata.",
  }),
  Object.freeze({
    runtimePath: "shared/local_qwen_runtime.mjs",
    repoPath: `${WORKSPACE_REPO_ROOT}/shared/local_qwen_runtime.mjs`,
    subsystem: "local_runtime",
    tags: ["local_qwen", "runtime_plan", "onnx", "webgpu"],
    description: "Local Qwen runtime plans, supported ONNX aliases, and message normalization.",
  }),
  Object.freeze({
    runtimePath: "README.md",
    repoPath: `${WORKSPACE_REPO_ROOT}/README.md`,
    subsystem: "project_docs",
    tags: ["docs", "architecture", "overview"],
    description: "High-level project overview for the Chrome extension architecture.",
  }),
]);
const WORKSPACE_CODE_DIVE_RUNTIME_PATHS = new Set(
  WORKSPACE_CODE_DIVE_FILE_MANIFEST.map((entry) => entry.runtimePath),
);
const CRAFTING_AGENT_TOOL_LOG_METADATA = Object.freeze({
  inspect_training_data: Object.freeze({
    title: "Inspect training data",
    startDetail: "Checking the current sample set and split coverage.",
    stageId: "inspect",
  }),
  inspect_bundle_snapshot: Object.freeze({
    title: "Inspect current bundle",
    startDetail: "Checking saved scripts, capabilities, and bundle artifacts.",
    stageId: "inspect",
  }),
  workspace_code_dive: Object.freeze({
    title: "Inspect workspace code",
    startDetail: "Scanning packaged extension source files for the failing runtime path.",
    stageId: "diagnose",
  }),
  web_search: Object.freeze({
    title: "Gather external references",
    startDetail: "Searching for grounded references before editing the craft.",
    stageId: "research",
  }),
  browser_tabs: Object.freeze({
    title: "Prepare browser context",
    startDetail: "Opening or switching browser tabs for the next step.",
    stageId: "research",
  }),
  browser_inspect: Object.freeze({
    title: "Inspect browser state",
    startDetail: "Reading the visible browser state before taking action.",
    stageId: "research",
  }),
  browser_action: Object.freeze({
    title: "Interact with the browser",
    startDetail: "Performing a direct browser step in the active tab.",
    stageId: "research",
  }),
  playwright_ctx: Object.freeze({
    title: "Run scripted browser step",
    startDetail: "Executing a deterministic in-tab browser script.",
    stageId: "research",
  }),
  save_tool_scripts: Object.freeze({
    title: "Save reviewed tool scripts",
    startDetail: "Persisting deterministic runtime scripts for the craft.",
    stageId: "tools",
  }),
  save_browser_capabilities: Object.freeze({
    title: "Save reviewed browser functions",
    startDetail: "Saving the reviewed browser functions bundle for the craft runtime.",
    stageId: "tools",
  }),
  request_codex_repair: Object.freeze({
    title: "Start Codex repair",
    startDetail: "Submitting the local workspace defect to the Codex repair bridge.",
    stageId: "repair",
  }),
  get_codex_repair_status: Object.freeze({
    title: "Track Codex repair",
    startDetail: "Polling the local Codex repair bridge for job progress.",
    stageId: "repair",
  }),
  run_agentic_smoke: Object.freeze({
    title: "Run live smoke test",
    startDetail: "Testing the reviewed capability path through the real runtime.",
    stageId: "validate",
  }),
  run_capability_eval: Object.freeze({
    title: "Run capability evaluation",
    startDetail: "Checking the reviewed capability bundle against manual test cases.",
    stageId: "validate",
  }),
  generate_training_batch: Object.freeze({
    title: "Grow training dataset",
    startDetail: "Generating grounded training rows from the local batch bridge.",
    stageId: "dataset",
  }),
  draft_training_changes: Object.freeze({
    title: "Draft training examples",
    startDetail: "Preparing concrete training data edits for this craft.",
    stageId: "dataset",
  }),
  start_training_run: Object.freeze({
    title: "Start model training",
    startDetail: "Launching the local training runtime for this craft.",
    stageId: "train",
  }),
  get_training_run_status: Object.freeze({
    title: "Track model training",
    startDetail: "Polling the live training run for fresh progress.",
    stageId: "train",
  }),
  record_experiment_decision: Object.freeze({
    title: "Record experiment decision",
    startDetail: "Logging whether the latest challenger should be kept, discarded, or parked.",
    stageId: "validate",
  }),
  update_craft_maturity: Object.freeze({
    title: "Record maturity milestone",
    startDetail: "Updating the explicit maturity state shown in the UI.",
    stageId: "train",
  }),
  ask_user_clarification: Object.freeze({
    title: "Wait for clarification",
    startDetail: "Collecting a missing answer before the run can continue.",
    stageId: "validate",
  }),
});
const CLARIFICATION_BLOCKING_HINT_RE =
  /\b(block(?:ed|ing)?|cannot continue|can't continue|unable to continue|required(?:\s+before|\s+to)? continue|must know|missing critical|critical detail|decisive|determin(?:e|es|ing)|ambig(?:uous|uity)|incompatible|wrong (?:contract|schema|target|site|behavior)|approval|permission|credential|api key|auth|site-specific|domain-specific|current-tab|dom state)\b/i;
const CLARIFICATION_CONVENIENCE_HINT_RE =
  /\b(open|visit|load|navigate|mark|highlight|select|focus|retry|resume|continue|verify|check|probe)\b[\s\S]{0,80}\b(page|website|site|tab|text|selection|field|input|dom|browser)\b|\bactive http\(s\) tab\b|\blive dom\b|\bbrowser automation\b|\bbrowser probe\b/i;
const CLARIFICATION_DECISION_HINT_RE =
  /\b(which|choose|pick|should|confirm)\b[\s\S]{0,60}\b(schema|format|field|target|site|domain|behavior|output|contract|tool)\b/i;

const runPersistTimers = new Map();
let runsHydrated = false;
let runsHydrationPromise = null;
const workspaceSourceTextCache = new Map();
const craftArtifactWriteBarriers = new Map();

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

async function waitForCraftArtifactWrites(craftId = "") {
  const key = asText(craftId) || "__global__";
  const barrier = craftArtifactWriteBarriers.get(key);
  if (!barrier || typeof barrier.then !== "function") return;
  await Promise.resolve(barrier).catch(() => {});
}

async function withCraftArtifactWriteBarrier(craftId = "", operation = null) {
  if (typeof operation !== "function") {
    throw new TypeError("Artifact write barrier requires an operation.");
  }
  const key = asText(craftId) || "__global__";
  const previous = craftArtifactWriteBarriers.get(key) || Promise.resolve();
  let releaseCurrent = () => {};
  const current = new Promise((resolve) => {
    releaseCurrent = resolve;
  });
  craftArtifactWriteBarriers.set(key, current);
  await Promise.resolve(previous).catch(() => {});
  try {
    return await operation();
  } finally {
    releaseCurrent();
    if (craftArtifactWriteBarriers.get(key) === current) {
      craftArtifactWriteBarriers.delete(key);
    }
  }
}

function trimText(value, max = 280) {
  const text = asText(value).replace(/\s+/g, " ").trim();
  if (text.length <= max) return text;
  return `${text.slice(0, Math.max(1, max - 1)).trimEnd()}...`;
}

function updateSmokeDebug(stage = "", detail = null) {
  try {
    globalThis.__sinepanelSmokeDebug = {
      stage: asText(stage),
      detail: detail && typeof detail === "object" ? cloneJson(detail, {}) : detail ?? null,
      updatedAt: new Date().toISOString(),
    };
  } catch {}
}

function uniqueTextList(values = [], limit = 48) {
  const out = [];
  const seen = new Set();
  for (const value of Array.isArray(values) ? values : [values]) {
    const text = asText(value);
    if (!text || seen.has(text)) continue;
    seen.add(text);
    out.push(text);
    if (out.length >= limit) break;
  }
  return out;
}

const EXPERIMENT_MODE_VALUES = new Set(["baseline", "champion", "challenger"]);
const EXPERIMENT_MUTATION_SCOPE_VALUES = new Set([
  "none",
  "tool_scripts",
  "browser_capabilities",
  "training_rows",
  "training_config",
  "bundle_policy",
  "multi_artifact",
]);
const EXPERIMENT_DECISION_POLICY_VALUES = new Set([
  "keep_if_better",
  "keep_if_equal_or_better",
  "park_unless_clear_win",
  "manual_review",
]);
const EXPERIMENT_DECISION_VALUES = new Set(["pending", "keep", "discard", "park"]);

function normalizeExperimentEnumValue(value, allowedValues) {
  const normalized = asText(value).toLowerCase();
  return allowedValues.has(normalized) ? normalized : "";
}

function normalizeExperimentMetadata(rawExperiment, {
  requireCandidateId = false,
  requireDecision = false,
} = {}) {
  const source = rawExperiment && typeof rawExperiment === "object" ? rawExperiment : null;
  if (!source) return null;
  const candidateId = asText(source.candidateId || source.candidate_id);
  const mode = normalizeExperimentEnumValue(source.mode, EXPERIMENT_MODE_VALUES);
  const compareAgainst = asText(source.compareAgainst || source.compare_against);
  const hypothesis = trimText(source.hypothesis, 600);
  const expectedSignal = trimText(source.expectedSignal || source.expected_signal, 400);
  const mutationScope = normalizeExperimentEnumValue(
    source.mutationScope || source.mutation_scope,
    EXPERIMENT_MUTATION_SCOPE_VALUES,
  );
  const decisionPolicy = normalizeExperimentEnumValue(
    source.decisionPolicy || source.decision_policy,
    EXPERIMENT_DECISION_POLICY_VALUES,
  );
  const suiteId = asText(source.suiteId || source.suite_id);
  const evalSetId = asText(source.evalSetId || source.eval_set_id);
  const decision = normalizeExperimentEnumValue(source.decision, EXPERIMENT_DECISION_VALUES);
  const rationale = trimText(source.rationale, 600);
  const tags = uniqueTextList(source.tags, 12);
  const metrics =
    source.metrics && typeof source.metrics === "object"
      ? cloneJson(source.metrics, {})
      : null;
  const hasContent = Boolean(
    candidateId ||
      mode ||
      compareAgainst ||
      hypothesis ||
      expectedSignal ||
      mutationScope ||
      decisionPolicy ||
      suiteId ||
      evalSetId ||
      decision ||
      rationale ||
      tags.length ||
      metrics,
  );
  if (!hasContent) return null;
  if (requireCandidateId && !candidateId) return null;
  if (requireDecision && !decision) return null;
  return {
    candidateId,
    mode,
    compareAgainst,
    hypothesis,
    expectedSignal,
    mutationScope,
    decisionPolicy,
    suiteId,
    evalSetId,
    decision,
    rationale,
    tags,
    metrics,
  };
}

function buildExperimentLabel(experiment = null) {
  const normalized = normalizeExperimentMetadata(experiment);
  if (!normalized) return "";
  return [
    normalized.candidateId,
    normalized.mode,
    normalized.mutationScope,
  ].filter(Boolean).join(" · ");
}

function normalizeExperimentLedgerEntry(rawEntry, index = 0) {
  const experiment = normalizeExperimentMetadata(rawEntry);
  const action = asText(rawEntry?.action);
  const outcome = asText(rawEntry?.outcome).toLowerCase();
  const summary = trimText(rawEntry?.summary, 240);
  const recordedAt = asText(rawEntry?.recordedAt || rawEntry?.recorded_at);
  if (!experiment && !action && !summary) return null;
  return {
    id: asText(rawEntry?.id) || buildStableId("experiment", `${buildExperimentLabel(experiment)}|${action}|${summary}`, index),
    ...(experiment || {
      candidateId: "",
      mode: "",
      compareAgainst: "",
      hypothesis: "",
      expectedSignal: "",
      mutationScope: "",
      decisionPolicy: "",
      suiteId: "",
      evalSetId: "",
      decision: "",
      rationale: "",
      tags: [],
      metrics: null,
    }),
    action,
    outcome,
    summary,
    recordedAt: recordedAt || new Date().toISOString(),
  };
}

function normalizeExperimentLedger(rawEntries) {
  return (Array.isArray(rawEntries) ? rawEntries : [])
    .map((entry, index) => normalizeExperimentLedgerEntry(entry, index))
    .filter(Boolean)
    .slice(-MAX_EXPERIMENT_LEDGER);
}

function appendRunExperimentLedger(run, entry = null) {
  const normalizedEntry = normalizeExperimentLedgerEntry(entry, run?.experimentLedger?.length || 0);
  if (!normalizedEntry) return null;
  run.experimentLedger = normalizeExperimentLedger([
    ...(Array.isArray(run.experimentLedger) ? run.experimentLedger : []),
    normalizedEntry,
  ]);
  if (normalizedEntry.candidateId) {
    run.latestExperimentCandidateId = normalizedEntry.candidateId;
  }
  run.updatedAt = new Date().toISOString();
  scheduleRunPersistence(run);
  return normalizedEntry;
}

function buildPublishedBrowserCapabilitiesPayload(payload, craft = null, toolScriptsPayload = null, publicationContext = {}) {
  const normalizedToolScriptsPayload = normalizeToolScriptsPayload(
    toolScriptsPayload,
    craft,
    STRICT_RUNTIME_TOOL_SCRIPTS_OPTIONS,
  );
  const compiled = compilePublishedBrowserCapabilityBundlePayload(payload, {
    craft,
    toolScriptsPayload: normalizedToolScriptsPayload,
    publishedAt: publicationContext.publishedAt,
    publishedBy: asText(publicationContext.publishedBy) || "crafting_agent",
  });
  if (!compiled?.ok) {
    throw new Error(asText(compiled?.error) || "Reviewed browser capability package could not be compiled.");
  }
  return compiled.payload;
}

function mergeRecordsById(existing = [], incoming = []) {
  const merged = new Map();
  for (const entry of Array.isArray(existing) ? existing : []) {
    if (!entry || typeof entry !== "object") continue;
    const id = asText(entry.id || entry.name);
    if (!id) continue;
    merged.set(id, cloneJson(entry, {}));
  }
  for (const entry of Array.isArray(incoming) ? incoming : []) {
    if (!entry || typeof entry !== "object") continue;
    const id = asText(entry.id || entry.name);
    if (!id) continue;
    merged.set(id, cloneJson(entry, {}));
  }
  return [...merged.values()];
}

function normalizeCraftNameCandidate(value, max = 80) {
  const text = asText(value)
    .replace(/\s+/g, " ")
    .replace(/^["'`]+|["'`]+$/g, "")
    .replace(/[.?!,:;]+$/g, "")
    .trim();
  if (!text) return "";
  if (text.length <= max) return text;
  return text.slice(0, max).trimEnd();
}

function normalizeFinalRunStatus(value) {
  const status = asText(value).toLowerCase();
  return FINAL_OUTPUT_STATUSES.has(status) ? status : "";
}

function hasUnsupportedPathNote(value) {
  const text = asText(value).toLowerCase();
  if (!text) return false;
  return /(keine freigegebenen werkzeuge|kein passender ablauf|nicht aufgenommen|vorerst ausgelassen|not supported|unsupported|missing tool|cannot be covered|nicht abgedeckt)/.test(
    text,
  );
}

function finalOutputHasLimitationNote(finalOutput) {
  if (!finalOutput || typeof finalOutput !== "object") return false;
  const texts = [
    finalOutput.summary,
    finalOutput.responseText,
    finalOutput.report?.currentState,
    finalOutput.report?.nextAction,
    ...(Array.isArray(finalOutput.provenance)
      ? finalOutput.provenance.flatMap((entry) => [entry?.title, entry?.detail])
      : []),
  ];
  return texts.some((entry) => hasUnsupportedPathNote(entry));
}

function describeToolActionForUser(action) {
  const normalized = asText(action);
  if (normalized === "run_agentic_smoke") return "the smoke test";
  if (normalized === "run_capability_eval") return "the mini eval suite";
  if (normalized === "generate_training_batch") return "training-batch generation";
  if (normalized === "save_tool_scripts") return "saving the tool scripts";
  if (normalized === "save_browser_capabilities") return "saving the browser capabilities";
  if (normalized === "record_experiment_decision") return "recording the experiment decision";
  if (normalized === "request_codex_repair") return "starting the local Codex repair run";
  if (normalized === "get_codex_repair_status") return "checking the local Codex repair run";
  if (normalized === "browser_action") return "a browser step";
  if (normalized === "browser_inspect") return "browser inspection";
  if (normalized === "playwright_ctx") return "the browser script";
  return normalized ? `the ${normalized} tool` : "a required tool";
}

function detectDevelopmentToolingBlocker(run, finalOutput) {
  const traces = Array.isArray(run?.toolTrace) ? run.toolTrace.slice().reverse() : [];
  for (const trace of traces) {
    const action = asText(trace?.action);
    const summary = asText(trace?.summary).toLowerCase();
    const ok = trace?.ok !== false;
    if (!action) continue;
    if (action === ASK_USER_TOOL_NAME) continue;
    if (PASSIVE_DIAGNOSIS_TOOL_ACTIONS.has(action)) {
      continue;
    }
    if (
      action === "run_agentic_smoke" &&
      /keine reviewed capability verwendet|did not use.*reviewed capability|without reviewed capability/.test(summary)
    ) {
      return {
        currentState:
          "The run stops here because the smoke test has not used the saved function through the intended path yet.",
        nextAction:
          "Inspect the capability selection and tool contract, fix the path, then rerun the same test.",
      };
    }
    if (action === "run_capability_eval" && !ok) {
      return {
        currentState:
          "The run stops here because the saved function does not pass the mini eval suite yet.",
        nextAction:
          "Inspect the function choice and tool contract, fix the failure, then rerun the same eval cases.",
      };
    }
    if (!ok) {
      return {
        currentState: `The run stops here because ${describeToolActionForUser(action)} failed in the current pass.`,
        nextAction:
          "Fix the affected tool or tool contract first, then start a new validation run.",
      };
    }
  }
  if (finalOutputHasLimitationNote(finalOutput)) {
    return {
      currentState:
        "The run stops here because a required tool path is still missing or not approved yet.",
      nextAction:
        "Extend or repair the tool path first and validate it again before adding more data or training.",
    };
  }
  return null;
}

function detectRecoverableValidationIteration(run, finalOutput) {
  if (!run || typeof run !== "object") return null;
  const traces = Array.isArray(run?.toolTrace) ? run.toolTrace.slice().reverse() : [];
  for (const trace of traces) {
    const action = asText(trace?.action);
    const summary = asText(trace?.summary).toLowerCase();
    const ok = trace?.ok !== false;
    if (!action) continue;
    if (action === ASK_USER_TOOL_NAME) continue;
    if (PASSIVE_DIAGNOSIS_TOOL_ACTIONS.has(action)) {
      continue;
    }
    if (
      action === "run_agentic_smoke" &&
      (
        !ok ||
        /keine reviewed capability verwendet|did not use.*reviewed capability|without reviewed capability/.test(summary)
      )
    ) {
      return {
        currentState:
          "The last smoke test failed. This is still normal iteration work on the local script and capability path.",
        nextAction:
          "Check the saved bundle and source first, separate artifact mistakes from real runtime faults, then revise the script or flow and test again in the same run.",
      };
    }
    if (action === "run_capability_eval" && !ok) {
      return {
        currentState:
          "The last mini eval suite failed. This is still normal iteration work on the local script and capability path.",
        nextAction:
          "Check the saved bundle and source first, separate artifact mistakes from real runtime faults, then revise the script or flow and rerun the same cases in the same run.",
      };
    }
    return null;
  }
  if (finalOutputHasLimitationNote(finalOutput)) {
    return null;
  }
  return null;
}

function normalizeNameComparison(value) {
  return asText(value)
    .replace(/\s+/g, " ")
    .trim()
    .toLowerCase();
}

function buildLegacyAbilityNameFromDescription(description) {
  const text = String(description || "")
    .replace(/\s+/g, " ")
    .trim();
  if (!text) return "";
  const firstSentence = text.split(/[.!?]/)[0]?.trim() || text;
  const compact = firstSentence.slice(0, 48).trim();
  if (!compact) return "";
  return compact.charAt(0).toUpperCase() + compact.slice(1);
}

function isPlaceholderCraftName(name) {
  const value = asText(name);
  return /^new capability(?: \d+)?$/i.test(value) || /^craft \d+$/i.test(value);
}

function shouldDeriveCraftNameSuggestion(run) {
  const nameSource = asText(run?.craft?.nameSource).toLowerCase();
  if (nameSource === "agent" || nameSource === "user") return false;
  const currentName = asText(run?.craft?.name);
  if (!currentName) return true;
  if (nameSource === "placeholder" || isPlaceholderCraftName(currentName)) return true;
  const legacyName = buildLegacyAbilityNameFromDescription(
    run?.craft?.summary || run?.craft?.agentPrompt || run?.brief || "",
  );
  return Boolean(legacyName) && normalizeNameComparison(currentName) === normalizeNameComparison(legacyName);
}

function formatClock(value = new Date()) {
  try {
    return new Intl.DateTimeFormat("de-DE", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    }).format(value);
  } catch {
    const iso = value instanceof Date ? value.toISOString() : new Date().toISOString();
    return iso.slice(11, 19);
  }
}

function humanizeToolName(toolName = "") {
  const normalized = asText(toolName)
    .replace(/[_-]+/g, " ")
    .trim();
  if (!normalized) return "Agent step";
  return normalized.replace(/\b[a-z]/g, (match) => match.toUpperCase());
}

function getCraftingAgentToolLogMetadata(toolName = "") {
  const key = asText(toolName);
  const meta = CRAFTING_AGENT_TOOL_LOG_METADATA[key] || null;
  return {
    toolName: key,
    title: meta?.title || humanizeToolName(key),
    startDetail: meta?.startDetail || "Working on this step.",
    stageId: meta?.stageId || "",
  };
}

function normalizeLogStatus(value = "") {
  const status = asText(value).toLowerCase();
  return ["running", "done", "warn", "error"].includes(status) ? status : "";
}

function serializeErrorDetail(value) {
  if (value == null) return null;
  if (typeof value === "string") {
    const text = trimText(value, 2_000);
    return text ? { message: text } : null;
  }
  if (value && typeof value === "object") {
    return cloneJson(value, { message: trimText(String(value), 2_000) });
  }
  const text = trimText(String(value), 2_000);
  return text ? { message: text } : null;
}

function resolveRunErrorDetail(run, fallback = null) {
  return serializeErrorDetail(
    fallback ??
      run?.errorDetail ??
      run?.lastToolFailure?.errorDetail ??
      null,
  );
}

function buildErrorInfo(error) {
  const message = error instanceof Error ? error.message : String(error || "Unknown error");
  return {
    message: asText(message) || "Unknown error",
    stack: trimText(error instanceof Error ? error.stack : "", 4_000),
    detail: serializeErrorDetail(error && typeof error === "object" && "detail" in error ? error.detail : null),
  };
}

function cloneStructuredErrorDetail(value) {
  return value && typeof value === "object"
    ? cloneJson(value, null)
    : serializeErrorDetail(value);
}

function collectStructuredErrorTexts(value, bucket = []) {
  if (value == null) return bucket;
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    const text = asText(value);
    if (text) bucket.push(text);
    return bucket;
  }
  if (!value || typeof value !== "object") return bucket;
  bucket.push(
    asText(value.reason),
    asText(value.message),
    asText(value.numericCode),
    asText(value?.error?.message),
    asText(value?.error?.props?.rawMessage),
  );
  if (Array.isArray(value.failures)) {
    value.failures.slice(0, 4).forEach((entry) => collectStructuredErrorTexts(entry, bucket));
  }
  if (value.runtime && typeof value.runtime === "object") {
    collectStructuredErrorTexts(value.runtime, bucket);
  }
  return bucket;
}

function describeLocalQwenRuntimeFailure(errorText = "", errorDetail = null) {
  const haystack = [
    asText(errorText),
    ...collectStructuredErrorTexts(errorDetail),
  ].join(" ").toLowerCase();
  if (/local_qwen_vision_pixel_budget_exceeded|safe browser screenshot budget|numeric runtime code 9468408|memory access out of bounds|out of bounds/.test(haystack)) {
    return "The local Qwen vision path is processing browser screenshots with a pixel budget that is too large for the WebGPU plan.";
  }
  if (/local_qwen|qwen|onnx|webgpu|ortrun|offscreen/.test(haystack)) {
    return "The local Qwen offscreen path is failing during WebGPU/ONNX execution.";
  }
  return asText(errorText);
}

function summarizeLocalQwenTrainingStartFailure(errorText = "", errorDetail = null) {
  const detail = errorDetail && typeof errorDetail === "object" ? errorDetail : null;
  if (asText(detail?.reason) !== "local_qwen_browser_training_manifest_missing") {
    return null;
  }
  const runtimeModelId = asText(detail?.runtimeModelId);
  const expectedRelativePath = asText(detail?.expectedRelativePath);
  const availableRuntimeModelIds = Array.isArray(detail?.availableRuntimeModelIds)
    ? detail.availableRuntimeModelIds.map((value) => asText(value)).filter(Boolean)
    : [];
  return {
    summary: "Local browser LoRA training is not available for this Qwen package.",
    currentState: expectedRelativePath
      ? `${expectedRelativePath} is missing from the workspace.`
      : runtimeModelId
        ? `The training manifest for ${runtimeModelId} is missing from the workspace.`
        : "The packaged training manifest is missing from the workspace.",
    nextAction: availableRuntimeModelIds.length
      ? `Package the training manifest and ONNX training artifacts for ${runtimeModelId}, or use a packaged model such as ${availableRuntimeModelIds[0]}.`
      : "Package the training manifest and matching ONNX training artifacts for this model.",
    error: asText(errorText),
  };
}

function readStructuredErrorStack(value) {
  const stack = asText(
    value?.errorStack ||
      value?.error_stack ||
      value?.stack ||
      value?.errorDetail?.stack ||
      value?.errorDetail?.errorStack ||
      "",
  );
  return trimText(stack, 4_000);
}

function buildSyntheticToolFailureDetail(action, result = {}) {
  const data = result?.data && typeof result.data === "object" ? result.data : {};
  const snapshot = {
    toolAction: asText(action),
    summary: trimText(asText(result?.summary || result?.error || ""), 260),
    error: trimText(asText(result?.error || ""), 600),
    underlyingFailingTool: asText(data?.underlyingFailingTool || data?.failingTool || ""),
    backend: asText(data?.backend || ""),
    operation: asText(data?.operation || data?.action || ""),
    errorCode: asText(data?.error_code || data?.errorCode || ""),
    url: trimText(asText(data?.url || data?.finalUrl || data?.tab?.url || ""), 400),
    title: trimText(asText(data?.title || data?.tab?.title || ""), 240),
    rawPreview: trimText(asText(data?.raw_preview || data?.rawPreview || ""), 600),
  };
  if (data?.page_state && typeof data.page_state === "object") {
    snapshot.pageState = cloneJson(data.page_state, null);
  } else if (data?.pageState && typeof data.pageState === "object") {
    snapshot.pageState = cloneJson(data.pageState, null);
  }
  return serializeErrorDetail(snapshot);
}

function createLog(level, message, meta = {}) {
  const next = {
    level: asText(level) || "info",
    message: String(message || ""),
    time: formatClock(new Date()),
  };
  const kind = asText(meta?.kind).toLowerCase();
  if (kind) next.kind = kind;
  const title = asText(meta?.title);
  if (title) next.title = title;
  const detail = asText(meta?.detail);
  if (detail) next.detail = detail;
  const toolName = asText(meta?.toolName);
  if (toolName) next.toolName = toolName;
  const stageId = asText(meta?.stageId);
  if (stageId) next.stageId = stageId;
  const status = normalizeLogStatus(meta?.status);
  if (status) next.status = status;
  const data = meta?.data && typeof meta.data === "object" ? cloneJson(meta.data, null) : null;
  if (data) next.data = data;
  return next;
}

function buildStableId(prefix, text, index = 0) {
  const normalized = String(text || "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 24);
  return `${prefix}-${index + 1}-${normalized || "entry"}`;
}

function normalizeQuestion(rawQuestion, index = 0) {
  const source = rawQuestion && typeof rawQuestion === "object" ? rawQuestion : {};
  const question = asText(source.question || source.text);
  if (!question) return null;
  return {
    id: asText(source.id) || buildStableId("question", question, index),
    question,
    reason: asText(source.reason),
    answer: asText(source.answer),
  };
}

function normalizeQuestions(rawQuestions) {
  return (Array.isArray(rawQuestions) ? rawQuestions : [])
    .map((entry, index) => normalizeQuestion(entry, index))
    .filter(Boolean)
    .slice(0, 6);
}

function normalizeProvenanceEntry(rawEntry, index = 0) {
  const source = rawEntry && typeof rawEntry === "object" ? rawEntry : {};
  const title = asText(source.title || source.label);
  const detail = asText(source.detail || source.reason || source.message);
  if (!title && !detail) return null;
  const kind = asText(source.kind).toLowerCase();
  const operationType = asText(source.operationType || source.operation_type).toLowerCase();
  return {
    id: asText(source.id) || buildStableId("prov", title || detail, index),
    title: title || "Signal",
    detail,
    kind: ["match", "constraint", "operation", "sample"].includes(kind) ? kind : "match",
    sampleId: asText(source.sampleId || source.sample_id),
    operationType: ["add", "update", "delete"].includes(operationType) ? operationType : "",
  };
}

function normalizeProvenance(rawEntries) {
  return (Array.isArray(rawEntries) ? rawEntries : [])
    .map((entry, index) => normalizeProvenanceEntry(entry, index))
    .filter(Boolean)
    .slice(0, 18);
}

function normalizeReport(rawReport, fallback = {}) {
  const report = rawReport && typeof rawReport === "object" ? rawReport : {};
  const fallbackReport = fallback && typeof fallback === "object" ? fallback : {};
  const matchingSignals = Array.from(
    new Set(
      (Array.isArray(report.matchingSignals) ? report.matchingSignals : [])
        .map((entry) => asText(entry))
        .filter(Boolean),
    ),
  ).slice(0, 8);
  return {
    objective: asText(report.objective || fallbackReport.objective),
    currentState: asText(report.currentState || fallbackReport.currentState),
    nextAction: asText(report.nextAction || fallbackReport.nextAction),
    matchingSignals,
  };
}

function normalizeMaturity(rawMaturity, fallback = null) {
  const source = rawMaturity && typeof rawMaturity === "object" ? rawMaturity : null;
  const fallbackValue =
    fallback && typeof fallback === "object" ? fallback : createEmptyCraftMaturity();
  if (asText(source?.kind) !== "agent_reported") {
    return normalizeCraftMaturity(null, fallbackValue);
  }
  return normalizeCraftMaturity(
    createAgentReportedCraftMaturity({
      percent: source?.percent,
      phase: source?.phase,
      rationale: source?.rationale,
      updatedAt: source?.updatedAt,
    }),
    fallbackValue,
  );
}

function nullIfEmptyText(value) {
  const text = asText(value);
  return text ? text : null;
}

function normalizeStructuredNameSuggestionContract(rawSuggestion) {
  const source =
    rawSuggestion && typeof rawSuggestion === "object"
      ? rawSuggestion
      : typeof rawSuggestion === "string"
        ? { name: rawSuggestion }
        : {};
  const name = normalizeCraftNameCandidate(
    source?.name || source?.suggestedName || source?.suggested_name || source?.title,
  );
  if (!name || isPlaceholderCraftName(name)) return null;
  return CRAFTING_AGENT_NAME_CONTRACT.parse({
    name,
    reason: nullIfEmptyText(source?.reason || source?.rationale || source?.why),
  });
}

function normalizeStructuredOfficialDescriptionContract(rawDescription) {
  const source =
    rawDescription && typeof rawDescription === "object"
      ? rawDescription
      : typeof rawDescription === "string"
        ? { text: rawDescription }
        : {};
  const text = trimText(
    asText(source?.text || source?.description || source?.summary || source?.body),
    600,
  );
  if (!text) return null;
  return CRAFTING_AGENT_DESCRIPTION_CONTRACT.parse({
    text,
    reason: nullIfEmptyText(source?.reason || source?.rationale || source?.why),
  });
}

function normalizeStructuredReportContract(rawReport, fallback = {}) {
  return CRAFTING_AGENT_REPORT_CONTRACT.parse({
    objective: nullIfEmptyText(rawReport?.objective ?? fallback?.objective),
    currentState: nullIfEmptyText(rawReport?.currentState ?? fallback?.currentState),
    nextAction: nullIfEmptyText(rawReport?.nextAction ?? fallback?.nextAction),
    matchingSignals: Array.from(
      new Set(
        (Array.isArray(rawReport?.matchingSignals)
          ? rawReport.matchingSignals
          : Array.isArray(fallback?.matchingSignals)
            ? fallback.matchingSignals
            : []
        )
          .map((entry) => asText(entry))
          .filter(Boolean),
      ),
    ).slice(0, 8),
  });
}

function normalizeStructuredMaturityContract(rawMaturity) {
  if (!rawMaturity || typeof rawMaturity !== "object") return null;
  return CRAFTING_AGENT_MATURITY_CONTRACT.parse({
    kind: "agent_reported",
    percent: Number(rawMaturity?.percent ?? 0),
    phase: asText(rawMaturity?.phase) === "capability_readiness"
      ? "capability_readiness"
      : "crafting_progress",
    rationale: nullIfEmptyText(rawMaturity?.rationale),
  });
}

function normalizeStructuredQuestionContract(rawQuestion, index = 0) {
  const source = rawQuestion && typeof rawQuestion === "object" ? rawQuestion : {};
  const question = asText(source.question || source.text);
  if (!question) return null;
  return CRAFTING_AGENT_QUESTION_CONTRACT.parse({
    id: nullIfEmptyText(source.id) || buildStableId("question", question, index),
    question,
    reason: nullIfEmptyText(source.reason),
    answer: nullIfEmptyText(source.answer),
  });
}

function normalizeStructuredQuestionContracts(rawQuestions) {
  return (Array.isArray(rawQuestions) ? rawQuestions : [])
    .map((entry, index) => normalizeStructuredQuestionContract(entry, index))
    .filter(Boolean)
    .slice(0, 6);
}

function normalizeStructuredProvenanceContract(rawEntry, index = 0) {
  const source = rawEntry && typeof rawEntry === "object" ? rawEntry : {};
  const title = asText(source.title || source.label);
  const detail = asText(source.detail || source.reason || source.message);
  if (!title && !detail) return null;
  const kind = asText(source.kind).toLowerCase();
  const operationType = asText(source.operationType || source.operation_type).toLowerCase();
  return CRAFTING_AGENT_PROVENANCE_CONTRACT.parse({
    id: nullIfEmptyText(source.id) || buildStableId("prov", title || detail, index),
    title: title || "Signal",
    detail: nullIfEmptyText(detail),
    kind: ["match", "constraint", "operation", "sample"].includes(kind) ? kind : "match",
    sampleId: nullIfEmptyText(source.sampleId || source.sample_id),
    operationType: ["add", "update", "delete"].includes(operationType) ? operationType : null,
  });
}

function normalizeStructuredProvenanceContracts(rawEntries, limit = 18) {
  return (Array.isArray(rawEntries) ? rawEntries : [])
    .map((entry, index) => normalizeStructuredProvenanceContract(entry, index))
    .filter(Boolean)
    .slice(0, limit);
}

function normalizeStructuredUseSurfaceContract(rawUseSurface) {
  const source = rawUseSurface && typeof rawUseSurface === "object" ? rawUseSurface : {};
  const inputMode = asText(source.inputMode).toLowerCase();
  return CRAFTING_AGENT_USE_SURFACE_CONTRACT.parse({
    inputMode: ["free_text", "mixed", "selection", "current_tab", "context_only"].includes(inputMode)
      ? inputMode
      : null,
    inputHint: nullIfEmptyText(source.inputHint),
    inputPlaceholder: nullIfEmptyText(source.inputPlaceholder),
    actionLabel: nullIfEmptyText(source.actionLabel),
    inputExamples: Array.from(
      new Set(
        (Array.isArray(source.inputExamples) ? source.inputExamples : [])
          .map((entry) => asText(entry))
          .filter(Boolean),
      ),
    ).slice(0, 6),
  });
}

function getTrainingDataArtifactId(craftId) {
  return `training-samples:${asText(craftId)}`;
}

function getWeightsArtifactId(craftId) {
  return `capability-weights:${asText(craftId)}`;
}

function createEmptyCapabilityEvidence(craftId = "") {
  return {
    artifactId: getWeightsArtifactId(craftId),
    hasTrainedCapability: false,
    status: "base_model_only",
    checkedAt: "",
    completedAt: "",
  };
}

function normalizeCapabilityEvidence(rawEvidence = null, craftId = "") {
  const source = rawEvidence && typeof rawEvidence === "object" ? rawEvidence : {};
  return {
    artifactId: getWeightsArtifactId(craftId),
    hasTrainedCapability: source.hasTrainedCapability === true,
    status: asText(source.status) || (source.hasTrainedCapability === true ? "trained_adapter" : "base_model_only"),
    checkedAt: asText(source.checkedAt),
    completedAt: asText(source.completedAt),
  };
}

function hasTrainedCapabilityArtifact(record = null) {
  const payload = record?.payload && typeof record.payload === "object" ? record.payload : {};
  const meta = record?.meta && typeof record.meta === "object" ? record.meta : {};
  const status = asText(payload.status || meta.status).toLowerCase();
  const hasAdapter =
    meta.hasAdapter === true ||
    (Array.isArray(payload?.adapter?.modules) &&
      payload.adapter.modules.some(
        (entry) => asText(entry?.modulePath) && asText(entry?.loraADataUrl) && asText(entry?.loraBDataUrl),
      ));
  return status === "trained_adapter" && Boolean(hasAdapter);
}

function normalizeTrainingSampleStatus(value) {
  const candidate = asText(value).toLowerCase();
  return ["draft", "review", "ready", "blocked"].includes(candidate) ? candidate : "draft";
}

function normalizeTrainingSampleSplit(value) {
  const candidate = asText(value).toLowerCase();
  return ["train", "validation", "test"].includes(candidate) ? candidate : "train";
}

function stringifyTrainingJson(value) {
  if (typeof value === "string") return value;
  if (value == null) return "{}";
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return "{}";
  }
}

function normalizeTrainingMessages(value) {
  return Array.isArray(value) ? cloneJson(value, []) : [];
}

function normalizeTrainingTools(value) {
  return Array.isArray(value) ? cloneJson(value, []) : [];
}

function normalizeTrainingTargetTurnIndex(value) {
  if (!Number.isInteger(value)) return null;
  return value >= 0 ? value : null;
}

function hasStructuredTrainingTrace(sample) {
  return Array.isArray(sample?.messages) && sample.messages.length > 0 && Number.isInteger(sample?.targetTurnIndex);
}

function extractTrainingMessageText(message) {
  const content = message?.content;
  if (typeof content === "string") return String(content).trim();
  if (!Array.isArray(content)) return "";
  return content
    .map((part) => {
      if (typeof part === "string") return String(part).trim();
      if (part?.type === "text") return String(part.text || "").trim();
      return "";
    })
    .filter(Boolean)
    .join(" ")
    .trim();
}

function summarizeTrainingTraceMessage(message) {
  const role = String(message?.role || "message").trim().toLowerCase();
  const text = trimText(extractTrainingMessageText(message), 160);
  if (text) return `${role}: ${text}`;
  const toolCallNames = (Array.isArray(message?.tool_calls) ? message.tool_calls : [])
    .map((toolCall) => String(toolCall?.function?.name || toolCall?.name || "").trim())
    .filter(Boolean);
  if (toolCallNames.length) return `${role}: [tool_call ${toolCallNames.join(", ")}]`;
  if (role === "tool") {
    const toolName = String(message?.name || "").trim();
    return toolName ? `tool: ${toolName}` : "tool";
  }
  return role || "message";
}

function summarizeStructuredTrainingPrompt(sample, limit = 3) {
  const messages = Array.isArray(sample?.messages) ? sample.messages : [];
  const targetTurnIndex = Number.isInteger(sample?.targetTurnIndex) ? sample.targetTurnIndex : messages.length;
  const promptMessages = messages.slice(0, Math.max(0, targetTurnIndex));
  const lines = promptMessages
    .map((message) => summarizeTrainingTraceMessage(message))
    .filter(Boolean)
    .slice(0, limit);
  return lines.join("\n").trim() || "Stored as multi-turn transcript.";
}

function summarizeStructuredTrainingTarget(sample) {
  if (!hasStructuredTrainingTrace(sample)) return "";
  const targetMessage = sample.messages[sample.targetTurnIndex];
  return summarizeTrainingTraceMessage(targetMessage) || "assistant";
}

function renderStructuredTrainingPromptSummary(sample, limit = 3) {
  const messages = Array.isArray(sample?.messages) ? sample.messages : [];
  const targetTurnIndex = Number.isInteger(sample?.targetTurnIndex) ? sample.targetTurnIndex : messages.length;
  return messages
    .slice(0, Math.max(0, targetTurnIndex))
    .map((message) => summarizeTrainingTraceMessage(message))
    .filter(Boolean)
    .slice(0, limit)
    .join("\n")
    .trim() || "Stored as multi-turn transcript.";
}

function renderStructuredTrainingTargetSummary(sample) {
  if (!hasStructuredTrainingTrace(sample)) return "";
  const targetMessage = sample.messages[sample.targetTurnIndex];
  return summarizeTrainingTraceMessage(targetMessage) || "assistant";
}

function hasStructuredTrainingPayload(sample) {
  return (
    (Array.isArray(sample?.messages) && sample.messages.length > 0) ||
    Number.isInteger(sample?.targetTurnIndex)
  );
}

function describeStructuredTrainingTrace(sample, inspection = null) {
  const messageCount = Array.isArray(inspection?.normalizedMessages)
    ? inspection.normalizedMessages.length
    : Array.isArray(sample?.messages)
      ? sample.messages.length
      : 0;
  const targetTurnIndex = Number.isInteger(inspection?.normalizedTargetTurnIndex)
    ? inspection.normalizedTargetTurnIndex
    : Number.isInteger(sample?.targetTurnIndex)
      ? sample.targetTurnIndex
      : null;
  const details = [];
  if (messageCount > 0) details.push(`${messageCount} messages`);
  if (Number.isInteger(targetTurnIndex)) details.push(`target turn ${targetTurnIndex + 1}`);
  if (inspection?.hasToolResponses) details.push("tool responses included");
  return details.length ? `Multi-turn transcript · ${details.join(" · ")}.` : "Multi-turn transcript.";
}

function ensurePersistableStructuredTrainingTrace(messages = [], targetTurnIndex = null) {
  const inspection = inspectPortableTrainingTrace(messages, targetTurnIndex);
  if (inspection.ok) return inspection;
  throw new Error(
    [
      "Multi-turn training rows must include a valid assistant target turn and matching tool responses for every earlier assistant tool call before that supervised turn.",
      inspection.reason,
    ].filter(Boolean).join(" "),
  );
}

function ensureNativeQwenTrainingSample(sample) {
  if (hasStructuredTrainingPayload(sample)) {
    return ensurePersistableStructuredTrainingTrace(sample?.messages, sample?.targetTurnIndex);
  }
  const hasLegacyPayload = asText(sample?.promptText) || asText(sample?.expectedJsonText);
  if (hasLegacyPayload) {
    throw new Error(
      "Local Qwen training rows must use messages + tools + targetTurnIndex. Legacy promptText/expectedJson rows are no longer supported.",
    );
  }
  return null;
}

function validateTrainingSample(sample) {
  if (hasStructuredTrainingPayload(sample)) {
    const inspection = inspectPortableTrainingTrace(sample?.messages, sample?.targetTurnIndex);
    return {
      runnable: inspection.ok,
      invalidJson: false,
      invalidTrace: !inspection.ok,
      traceState: inspection,
      detail: inspection.ok ? describeStructuredTrainingTrace(sample, inspection) : inspection.reason,
    };
  }
  const hasLegacyPayload = asText(sample?.promptText) || asText(sample?.expectedJsonText);
  return {
    runnable: false,
    invalidJson: false,
    invalidTrace: !!hasLegacyPayload,
    traceState: null,
    detail: hasLegacyPayload
      ? "Local Qwen training rows must use messages + tools + targetTurnIndex."
      : "Add messages, tools, and a supervised assistant target turn for a native Qwen training row.",
  };
}

function createTrainingSampleDraft(index = 0, overrides = {}) {
  const now = new Date().toISOString();
  return {
    id: `sample-${Date.now()}-${Math.random().toString(16).slice(2, 8)}-${index + 1}`,
    promptText: "",
    expectedJsonText: "",
    messages: [],
    tools: [],
    targetTurnIndex: null,
    split: "train",
    status: "draft",
    source: "agent",
    createdAt: now,
    updatedAt: now,
    ...overrides,
  };
}

function resolveTrainingSampleTargetText(sample) {
  if (typeof sample?.expectedJsonText === "string") return sample.expectedJsonText;
  if (typeof sample?.expected_json_text === "string") return sample.expected_json_text;
  if (Object.prototype.hasOwnProperty.call(sample || {}, "expected_json")) {
    return stringifyTrainingJson(sample.expected_json);
  }
  if (Object.prototype.hasOwnProperty.call(sample || {}, "target")) {
    return stringifyTrainingJson(sample.target);
  }
  if (Object.prototype.hasOwnProperty.call(sample || {}, "output")) {
    return stringifyTrainingJson(sample.output);
  }
  return "{}";
}

function normalizeTrainingSample(sample, index = 0) {
  const fallback = createTrainingSampleDraft(index);
  const messages = normalizeTrainingMessages(sample?.messages);
  const targetTurnIndex = normalizeTrainingTargetTurnIndex(sample?.targetTurnIndex ?? sample?.target_turn_index);
  const structuredPayload = messages.length > 0 || Number.isInteger(targetTurnIndex);
  return {
    id: asText(sample?.id || fallback.id),
    promptText: String(sample?.promptText ?? sample?.prompt_text ?? sample?.prompt ?? ""),
    expectedJsonText: structuredPayload ? "" : String(resolveTrainingSampleTargetText(sample)),
    messages,
    tools: normalizeTrainingTools(sample?.tools ?? sample?.available_tools),
    targetTurnIndex,
    split: normalizeTrainingSampleSplit(sample?.split),
    status: normalizeTrainingSampleStatus(sample?.status),
    source: String(sample?.source ?? sample?.sourceRef ?? fallback.source),
    createdAt: String(sample?.createdAt || fallback.createdAt),
    updatedAt: String(sample?.updatedAt || sample?.createdAt || fallback.updatedAt),
  };
}

function parseTrainingSampleJson(text) {
  const source = asText(text);
  if (!source) return { ok: false, value: null };
  try {
    return {
      ok: true,
      value: JSON.parse(source),
    };
  } catch {
    return {
      ok: false,
      value: null,
    };
  }
}

function serializeTrainingSamples(samples = []) {
  return (Array.isArray(samples) ? samples : []).map((sample, index) => {
    const normalized = normalizeTrainingSample(sample, index);
    return {
      id: normalized.id,
      prompt_text: normalized.promptText,
      expected_json_text: "",
      expected_json: null,
      messages: normalized.messages,
      tools: normalized.tools,
      target_turn_index: normalized.targetTurnIndex,
      ...(normalized.messages.length && Number.isInteger(normalized.targetTurnIndex) && normalized.tools.length
        ? { output_mode: "multiturn_tool_agent" }
        : {}),
      split: normalized.split,
      status: normalized.status,
      source: normalized.source,
      createdAt: normalized.createdAt,
      updatedAt: normalized.updatedAt,
    };
  });
}

function summarizeTrainingSamples(samples = [], limit = 16) {
  return (Array.isArray(samples) ? samples : [])
    .slice(0, limit)
    .map((sample) => {
      const structured = hasStructuredTrainingTrace(sample);
      return {
        id: asText(sample.id),
        mode: "multiturn",
        split: asText(sample.split || "train"),
        status: asText(sample.status || "draft"),
        source: asText(sample.source),
        promptText: structured
          ? trimText(renderStructuredTrainingPromptSummary(sample), 280)
          : trimText(sample.promptText, 280),
        targetTurnSummary: trimText(renderStructuredTrainingTargetSummary(sample), 280),
        targetTurnIndex: structured && Number.isInteger(sample.targetTurnIndex) ? sample.targetTurnIndex : null,
        messageCount: structured && Array.isArray(sample.messages) ? sample.messages.length : 0,
        messages: structured && Array.isArray(sample.messages) ? sample.messages : [],
        tools: structured && Array.isArray(sample.tools) ? sample.tools : [],
        updatedAt: asText(sample.updatedAt || sample.createdAt),
      };
    });
}

function getTrainingDataMeta(samples = []) {
  const normalizedSamples = Array.isArray(samples) ? samples : [];
  const totalSamples = normalizedSamples.length;
  const readySamples = normalizedSamples.filter((sample) => {
    return sample.status === "ready" && validateTrainingSample(sample).runnable;
  }).length;
  const runnableSamples = normalizedSamples.filter((sample) => {
    return sample.status !== "blocked" && validateTrainingSample(sample).runnable;
  }).length;
  const invalidJsonSamples = normalizedSamples.filter((sample) => {
    return sample.status !== "blocked" && validateTrainingSample(sample).invalidJson;
  }).length;
  const invalidTraceSamples = normalizedSamples.filter((sample) => {
    return sample.status !== "blocked" && validateTrainingSample(sample).invalidTrace;
  }).length;
  return {
    totalSamples,
    readySamples,
    runnableSamples,
    invalidJsonSamples,
    invalidTraceSamples,
    invalidSamples: invalidJsonSamples + invalidTraceSamples,
  };
}

function hasStructuredSerializedTrainingTrace(sample) {
  return inspectPortableTrainingTrace(sample?.messages, sample?.target_turn_index).ok;
}

function pickTrainingDatasetRows(samples = []) {
  const serialized = serializeTrainingSamples(samples);
  const validRows = serialized.filter((sample) => {
    return sample.status !== "blocked" && hasStructuredSerializedTrainingTrace(sample);
  });
  if (!validRows.length) {
    return {
      train: [],
      validation: [],
      test: [],
    };
  }

  if (validRows.length === 1) {
    return {
      train: validRows.slice(),
      validation: validRows.slice(),
      test: validRows.slice(),
    };
  }

  const explicitTrain = validRows.filter((sample) => sample.split === "train");
  const explicitValidation = validRows.filter((sample) => sample.split === "validation");
  const explicitTest = validRows.filter((sample) => sample.split === "test");
  const baseTrainRows = explicitTrain.length
    ? explicitTrain
    : validRows.filter((sample) => sample.split !== "validation" && sample.split !== "test");
  const trainPool = (baseTrainRows.length ? baseTrainRows : validRows).slice();

  const takeFromTrainPool = (desiredCount = 0) => {
    const maxRemovable = Math.max(0, trainPool.length - 1);
    const count = Math.max(0, Math.min(Number(desiredCount || 0), maxRemovable));
    if (!count) return [];
    return trainPool.splice(trainPool.length - count, count);
  };

  const fallbackValidationSize = Math.max(
    1,
    Math.min(Math.floor(validRows.length * 0.15), Math.max(1, trainPool.length - 1)),
  );
  const fallbackTestSize = Math.max(
    1,
    Math.min(Math.floor(validRows.length * 0.2), Math.max(1, trainPool.length - 1)),
  );

  let validation = explicitValidation.slice();
  let test = explicitTest.slice();
  if (!validation.length && !test.length) {
    test = takeFromTrainPool(fallbackTestSize);
    validation = takeFromTrainPool(fallbackValidationSize);
  } else {
    if (!validation.length) validation = takeFromTrainPool(fallbackValidationSize);
    if (!test.length) test = takeFromTrainPool(fallbackTestSize);
  }

  const train = trainPool.length ? trainPool : validRows.slice(0, 1);
  if (!validation.length) validation = test.length ? test.slice(0, 1) : train.slice(-1);
  if (!test.length) test = validation.length ? validation.slice(0, 1) : train.slice(-1);

  return {
    train,
    validation,
    test,
  };
}

async function buildTrainingDatasetPayloadForCraft(craftId, fallbackSamples = []) {
  const resolvedCraftId = asText(craftId);
  const samples = await readTrainingSamplesForCraft(resolvedCraftId, fallbackSamples);
  const dataset = pickTrainingDatasetRows(samples);
  if (!dataset.train.length || !dataset.validation.length || !dataset.test.length) {
    return null;
  }
  return {
    train: dataset.train,
    validation: dataset.validation,
    test: dataset.test,
    meta: {
      source: "crafting_agent_run_samples",
      craftId: resolvedCraftId,
      artifactId: resolvedCraftId ? getTrainingDataArtifactId(resolvedCraftId) : "",
      sampleCount: dataset.train.length + dataset.validation.length + dataset.test.length,
      splitCounts: {
        train: dataset.train.length,
        validation: dataset.validation.length,
        test: dataset.test.length,
      },
    },
  };
}

async function readTrainingSamplesForCraft(craftId, fallbackSamples = []) {
  const artifactId = getTrainingDataArtifactId(craftId);
  const record = craftSync?.readLocalArtifact ? await craftSync.readLocalArtifact(artifactId, null) : null;
  if (record?.payload?.samples && Array.isArray(record.payload.samples)) {
    return record.payload.samples.map((sample, index) => normalizeTrainingSample(sample, index));
  }
  return (Array.isArray(fallbackSamples) ? fallbackSamples : []).map((sample, index) =>
    normalizeTrainingSample(sample, index),
  );
}

async function refreshRunCapabilityEvidence(run) {
  const craftId = asText(run?.craftId);
  if (!craftId) {
    run.capabilityEvidence = createEmptyCapabilityEvidence("");
    return run.capabilityEvidence;
  }
  const artifactId = getWeightsArtifactId(craftId);
  const record = craftSync?.readLocalArtifact ? await craftSync.readLocalArtifact(artifactId, null) : null;
  const payload = record?.payload && typeof record.payload === "object" ? record.payload : {};
  run.capabilityEvidence = normalizeCapabilityEvidence(
    {
      hasTrainedCapability: hasTrainedCapabilityArtifact(record),
      status: payload.status,
      checkedAt: new Date().toISOString(),
      completedAt: payload?.run?.completedAt,
    },
    craftId,
  );
  return run.capabilityEvidence;
}

async function syncCraftTrainingSummary(craftId, samples) {
  if (!craftStore?.readLocalCrafts || !craftStore?.writeCrafts) return;
  const localCrafts = await craftStore.readLocalCrafts();
  const targetIndex = localCrafts.findIndex((craft) => asText(craft?.id) === asText(craftId));
  if (targetIndex < 0) return;
  const nextCrafts = localCrafts.slice();
  const meta = getTrainingDataMeta(samples);
  nextCrafts[targetIndex] = {
    ...nextCrafts[targetIndex],
    seedRows: meta.totalSamples,
    datasetRows: meta.runnableSamples,
    updatedAt: new Date().toISOString(),
  };
  await craftStore.writeCrafts(nextCrafts);
}

async function writeTrainingSamplesForCraft(craftId, samples) {
  const normalizedSamples = (Array.isArray(samples) ? samples : []).map((sample, index) =>
    normalizeTrainingSample(sample, index),
  );
  for (const sample of normalizedSamples) {
    ensureNativeQwenTrainingSample(sample);
  }
  const serializedSamples = serializeTrainingSamples(normalizedSamples);
  if (craftSync?.putLocalArtifact) {
    await craftSync.putLocalArtifact({
      id: getTrainingDataArtifactId(craftId),
      craftId: asText(craftId),
      kind: "training_samples",
      payload: {
        samples: serializedSamples,
      },
      meta: {
        sampleCount: serializedSamples.length,
        updatedAt: Date.now(),
      },
    });
  }
  await syncCraftTrainingSummary(craftId, normalizedSamples);
  return normalizedSamples;
}

function summarizeOperationCounts(operations = []) {
  const counts = { add: 0, update: 0, delete: 0 };
  for (const operation of Array.isArray(operations) ? operations : []) {
    const type = asText(operation?.type).toLowerCase();
    if (Object.prototype.hasOwnProperty.call(counts, type)) {
      counts[type] += 1;
    }
  }
  return Object.entries(counts)
    .filter(([, count]) => count > 0)
    .map(([type, count]) => {
      if (type === "add") return `${count} hinzugefuegt`;
      if (type === "update") return `${count} aktualisiert`;
      if (type === "delete") return `${count} entfernt`;
      return `${count} ${type}`;
    })
    .join(" · ");
}

function getOperationCounts(operations = []) {
  const counts = { add: 0, update: 0, delete: 0 };
  for (const operation of Array.isArray(operations) ? operations : []) {
    const type = asText(operation?.type).toLowerCase();
    if (Object.prototype.hasOwnProperty.call(counts, type)) {
      counts[type] += 1;
    }
  }
  return counts;
}

function getSingleOperationType(operations = []) {
  const activeTypes = Object.entries(getOperationCounts(operations))
    .filter(([, count]) => count > 0)
    .map(([type]) => type);
  return activeTypes.length === 1 ? activeTypes[0] : "";
}

function formatGermanCount(count, singular, plural) {
  return `${count} ${count === 1 ? singular : plural}`;
}

function describeOperationMilestone(operations = []) {
  const counts = getOperationCounts(operations);
  if (counts.add && !counts.update && !counts.delete) {
    return `${formatGermanCount(counts.add, "training sample", "training samples")} added`;
  }
  if (counts.update && !counts.add && !counts.delete) {
    return `${formatGermanCount(counts.update, "training sample", "training samples")} updated`;
  }
  if (counts.delete && !counts.add && !counts.update) {
    return `${formatGermanCount(counts.delete, "training sample", "training samples")} removed`;
  }
  const summary = summarizeOperationCounts(operations);
  return summary ? `Training data revised (${summary})` : "Training data revised";
}

function describeOperationMilestoneDetail(operations = []) {
  const counts = getOperationCounts(operations);
  if (counts.add && !counts.update && !counts.delete) {
    return "The agent created new training samples for this capability.";
  }
  if (counts.update && !counts.add && !counts.delete) {
    return "The agent improved existing training samples for this capability.";
  }
  if (counts.delete && !counts.add && !counts.update) {
    return "The agent removed unsuitable training samples from the local dataset.";
  }
  return "The agent revised the local training data for this capability.";
}

function summarizeProvenanceLinks(items = [], limit = 3) {
  return (Array.isArray(items) ? items : [])
    .slice(0, limit)
    .map((entry) => {
      const title = trimText(asText(entry?.title || entry?.url || entry?.summary), 120);
      const url = trimText(asText(entry?.url), 160);
      return [title, url].filter(Boolean).join(" · ");
    })
    .filter(Boolean)
    .join(" | ");
}

function buildResearchMilestoneProvenance(query, results = []) {
  if (!Array.isArray(results) || !results.length) return [];
  return normalizeProvenance([
    {
      title: `Research scoped: ${trimText(asText(query) || "current craft", 96)}`,
      detail: summarizeProvenanceLinks(results, 3) || `${results.length} candidate sources found.`,
      kind: "match",
    },
  ]);
}

function buildBrowserMilestoneProvenance(activeTab, records = [], trace = []) {
  const recordSummary = summarizeProvenanceLinks(
    (Array.isArray(records) ? records : []).map((entry) => ({
      title: asText(entry?.title) || "Browser signal",
      url: asText(entry?.url || entry?.summary || entry?.body_text),
    })),
    3,
  );
  const traceSummary = (Array.isArray(trace) ? trace : [])
    .slice(-2)
    .map((entry) => trimText(asText(entry?.tool_summary), 140))
    .filter(Boolean)
    .join(" | ");
  if (!recordSummary && !traceSummary) return [];
  return normalizeProvenance([
    {
      title: `Browser grounding captured${asText(activeTab?.hostname) ? ` on ${activeTab.hostname}` : ""}`,
      detail: [recordSummary, traceSummary].filter(Boolean).join(" | "),
      kind: "sample",
    },
  ]);
}

function buildOperationMilestoneProvenance(operations = [], appliedMessages = []) {
  if (!Array.isArray(operations) || !operations.length) return [];
  return normalizeProvenance([
    {
      title: describeOperationMilestone(operations),
      detail: describeOperationMilestoneDetail(operations),
      kind: "operation",
      operationType: getSingleOperationType(operations),
    },
  ]);
}

function applyTrainingSampleOperation(samples, operation) {
  const nextSamples = (Array.isArray(samples) ? samples : []).map((sample, index) =>
    normalizeTrainingSample(sample, index),
  );
  const type = asText(operation?.type).toLowerCase();
  const fields = operation?.fields && typeof operation.fields === "object" ? operation.fields : {};
  const sampleId = asText(operation?.sampleId);
  const reason = asText(operation?.reason);
  const now = new Date().toISOString();

  if (type === "add") {
    const promptText = asText(fields.promptText);
    const messages = normalizeTrainingMessages(fields.messages);
    const targetTurnIndex = normalizeTrainingTargetTurnIndex(fields.targetTurnIndex);
    const hasStructuredPayload = messages.length > 0 || Number.isInteger(targetTurnIndex);
    if (!hasStructuredPayload) {
      throw new Error("Add operation needs a native Qwen multi-turn messages + targetTurnIndex payload.");
    }
    ensurePersistableStructuredTrainingTrace(messages, targetTurnIndex);
    const sample = createTrainingSampleDraft(nextSamples.length, {
      promptText,
      expectedJsonText: "",
      messages,
      tools: normalizeTrainingTools(fields.tools),
      targetTurnIndex,
      split: normalizeTrainingSampleSplit(fields.split),
      status: normalizeTrainingSampleStatus(fields.status || "review"),
      source: asText(fields.source) || "agent",
      createdAt: now,
      updatedAt: now,
    });
    nextSamples.unshift(sample);
    return {
      samples: nextSamples,
      message: `Added sample ${sample.id}${reason ? ` · ${reason}` : ""}`,
    };
  }

  const sample = nextSamples.find((entry) => entry.id === sampleId);
  if (!sample) {
    throw new Error(`Referenced sample not found: ${sampleId || "missing id"}`);
  }

  if (type === "delete") {
    return {
      samples: nextSamples.filter((entry) => entry.id !== sample.id),
      message: `Deleted sample ${sample.id}${reason ? ` · ${reason}` : ""}`,
    };
  }

  if (type === "update") {
    if (Object.prototype.hasOwnProperty.call(fields, "promptText")) {
      sample.promptText = String(fields.promptText || "");
    }
    if (Object.prototype.hasOwnProperty.call(fields, "messages")) {
      sample.messages = normalizeTrainingMessages(fields.messages);
    }
    if (Object.prototype.hasOwnProperty.call(fields, "tools")) {
      sample.tools = normalizeTrainingTools(fields.tools);
    }
    if (Object.prototype.hasOwnProperty.call(fields, "targetTurnIndex")) {
      sample.targetTurnIndex = normalizeTrainingTargetTurnIndex(fields.targetTurnIndex);
    }
    if (Object.prototype.hasOwnProperty.call(fields, "split")) {
      sample.split = normalizeTrainingSampleSplit(fields.split);
    }
    if (Object.prototype.hasOwnProperty.call(fields, "status")) {
      sample.status = normalizeTrainingSampleStatus(fields.status);
    }
    if (Object.prototype.hasOwnProperty.call(fields, "source")) {
      sample.source = String(fields.source || "");
    }
    ensureNativeQwenTrainingSample(sample);
    sample.updatedAt = now;
    return {
      samples: nextSamples,
      message: `Updated sample ${sample.id}${reason ? ` · ${reason}` : ""}`,
    };
  }

  throw new Error(`Unsupported training sample operation: ${type || "unknown"}`);
}

function decodeHtmlEntities(value) {
  return String(value || "")
    .replace(/&amp;/g, "&")
    .replace(/&quot;/g, "\"")
    .replace(/&#39;/g, "'")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&#x2F;/gi, "/")
    .replace(/&#x3D;/gi, "=")
    .replace(/&#x26;/gi, "&");
}

function stripHtml(value) {
  return decodeHtmlEntities(String(value || "").replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim());
}

function unwrapDuckDuckGoUrl(rawUrl) {
  const value = asText(rawUrl);
  if (!value) return "";
  try {
    const parsed = new URL(value, DDG_SEARCH_ENDPOINT);
    const direct = parsed.searchParams.get("uddg");
    return direct ? decodeURIComponent(direct) : parsed.toString();
  } catch {
    return value;
  }
}

function extractDuckDuckGoResults(html, limit = SEARCH_RESULT_LIMIT) {
  const source = String(html || "");
  const blockPattern = /<div[^>]*class="[^"]*result[^"]*"[^>]*>([\s\S]*?)<\/div>\s*<\/div>/gi;
  const blocks = [];
  for (const match of source.matchAll(blockPattern)) {
    if (!match?.[1]) continue;
    blocks.push(match[1]);
    if (blocks.length >= limit * 3) break;
  }

  const results = [];
  for (const block of blocks) {
    const anchorMatch = block.match(/<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>([\s\S]*?)<\/a>/i);
    if (!anchorMatch) continue;
    const snippetMatch = block.match(/<a[^>]*class="[^"]*result__snippet[^"]*"[^>]*>([\s\S]*?)<\/a>|<div[^>]*class="[^"]*result__snippet[^"]*"[^>]*>([\s\S]*?)<\/div>/i);
    const url = unwrapDuckDuckGoUrl(anchorMatch[1]);
    const title = stripHtml(anchorMatch[2]);
    const snippet = stripHtml(snippetMatch?.[1] || snippetMatch?.[2] || "");
    if (!url || !title) continue;
    results.push({
      title: trimText(title, 180),
      url,
      snippet: trimText(snippet, 280),
    });
    if (results.length >= limit) break;
  }
  return results;
}

function parseJsonLoose(raw) {
  const text = asText(raw);
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

function buildToolTraceEntry(turn, planner, toolResult) {
  return {
    turn,
    thought: trimText(planner?.thought, 280),
    action: asText(planner?.action),
    args: cloneJson(planner?.args, {}),
    summary: trimText(toolResult?.summary || toolResult?.error || "", 260),
    ok: toolResult?.ok !== false,
  };
}

function buildToolExecutionFailureResult(action, error) {
  const errorInfo = buildErrorInfo(error);
  const message = errorInfo.message || "Tool execution failed.";
  return {
    ok: false,
    summary: `An error occurred while running the tool. Please try again. Error: ${message}`,
    error: message,
    errorDetail: errorInfo.detail,
    errorStack: errorInfo.stack,
    report: {
      currentState: `${describeToolActionForUser(action)} failed: ${trimText(message, 220)}`,
      nextAction:
        action === "workspace_code_dive"
          ? "Use the already visible error and runtime data, or repeat the diagnosis later."
          : "Inspect the affected tool path first or run a source-code diagnosis next.",
      matchingSignals: [trimText(message, 180)].filter(Boolean),
    },
    provenance: [],
  };
}

function createEmptyCodexRepairState() {
  return {
    pendingJobId: "",
    pendingFingerprint: "",
    history: {},
  };
}

function createEmptyReloadResumeState() {
  return {
    pending: false,
    reason: "",
    requestedAt: "",
  };
}

function normalizeReloadResumeState(value = null) {
  const source = value && typeof value === "object" ? value : {};
  return {
    pending: source.pending === true,
    reason: asText(source.reason),
    requestedAt: asText(source.requestedAt),
  };
}

function normalizeCodexRepairHistoryEntry(entry = {}) {
  const source = entry && typeof entry === "object" ? entry : {};
  return {
    attempts: Math.max(0, Math.min(8, Number(source.attempts || 0) || 0)),
    lastJobId: asText(source.lastJobId),
    lastStatus: asText(source.lastStatus),
    lastSummary: trimText(asText(source.lastSummary), 300),
    updatedAt: asText(source.updatedAt),
  };
}

function ensureCodexRepairState(run) {
  const source = run?.codexRepair && typeof run.codexRepair === "object"
    ? run.codexRepair
    : createEmptyCodexRepairState();
  const rawHistory =
    source.history && typeof source.history === "object" && !Array.isArray(source.history)
      ? source.history
      : {};
  const history = {};
  for (const [fingerprint, entry] of Object.entries(rawHistory)) {
    const key = asText(fingerprint);
    if (!key) continue;
    history[key] = normalizeCodexRepairHistoryEntry(entry);
  }
  run.codexRepair = {
    pendingJobId: asText(source.pendingJobId),
    pendingFingerprint: asText(source.pendingFingerprint),
    history,
  };
  return run.codexRepair;
}

function buildCodexRepairFingerprint(run) {
  const codingHandoff =
    run?.workspaceCodeDive?.codingHandoff && typeof run.workspaceCodeDive.codingHandoff === "object"
      ? run.workspaceCodeDive.codingHandoff
      : {};
  const patchPaths = Array.isArray(codingHandoff?.patchTargets)
    ? codingHandoff.patchTargets
      .map((entry) => asText(entry?.path))
      .filter(Boolean)
      .slice(0, 4)
    : [];
  return [
    asText(run?.craftId),
    asText(codingHandoff?.failingTool || run?.lastToolFailure?.action),
    trimText(asText(codingHandoff?.errorText || run?.lastToolFailure?.error), 220),
    ...patchPaths,
  ]
    .filter(Boolean)
    .join(" | ")
    .toLowerCase();
}

function getCodexRepairAttemptCount(run, fingerprint = "") {
  const state = ensureCodexRepairState(run);
  const key = asText(fingerprint);
  if (!key) return 0;
  return Math.max(0, Number(state.history?.[key]?.attempts || 0) || 0);
}

function updateCodexRepairHistory(run, fingerprint = "", patch = {}) {
  const key = asText(fingerprint);
  if (!key) return createEmptyCodexRepairState();
  const state = ensureCodexRepairState(run);
  const previous = normalizeCodexRepairHistoryEntry(state.history?.[key]);
  const nextAttempts = patch.incrementAttempt
    ? previous.attempts + 1
    : Math.max(0, Number(patch.attempts ?? previous.attempts) || 0);
  state.history[key] = normalizeCodexRepairHistoryEntry({
    ...previous,
    ...patch,
    attempts: nextAttempts,
    updatedAt: asText(patch.updatedAt) || new Date().toISOString(),
  });
  return state;
}

function setPendingCodexRepair(run, {
  jobId = "",
  fingerprint = "",
} = {}) {
  const state = ensureCodexRepairState(run);
  state.pendingJobId = asText(jobId);
  state.pendingFingerprint = asText(fingerprint);
  return state;
}

function clearPendingCodexRepair(run) {
  const state = ensureCodexRepairState(run);
  state.pendingJobId = "";
  state.pendingFingerprint = "";
  return state;
}

function parseTimestampMs(value) {
  const parsed = Date.parse(asText(value));
  return Number.isFinite(parsed) ? parsed : 0;
}

function getRunLastActivityMs(run) {
  return (
    parseTimestampMs(run?.updatedAt) ||
    parseTimestampMs(run?.completedAt) ||
    parseTimestampMs(run?.startedAt)
  );
}

function isRunTerminalStatus(status = "") {
  return ["done", "blocked", "failed", "stopped"].includes(asText(status).toLowerCase());
}

function isRunResumable(run) {
  if (!run || typeof run !== "object") return false;
  if (normalizeReloadResumeState(run.reloadResume).pending) {
    return true;
  }
  return Boolean(asText(run.serializedState) && asText(run.pendingApprovalCallId));
}

function compactRunForTerminalRetention(run) {
  if (!run || typeof run !== "object") return run;
  if (!isRunTerminalStatus(run.status) || isRunResumable(run)) return run;
  run.craft = {
    id: asText(run?.craftId),
    name: asText(run?.craft?.name),
  };
  run.samples = [];
  run.resolved = null;
  run.parameters = {};
  run.activeTabContext = null;
  run.browserRuntime = { currentTabId: Number(run?.browserRuntime?.currentTabId || 0) || 0 };
  run.researchNotes = [];
  run.browserNotes = [];
  run.compactedMemory = [];
  run.serializedState = "";
  run.pendingApprovalCallId = "";
  run.pendingUserAnswersByCallId = {};
  run.previousQuestions = normalizeQuestions(run.previousQuestions)
    .filter((entry) => asText(entry.answer))
    .map((entry) => ({
      id: entry.id,
      question: entry.question,
      reason: entry.reason,
      answer: entry.answer,
    }));
  run.latestSuggestedQuestions = [];
  run.executionPromise = null;
  run.abortController = null;
  run.stopRequested = false;
  return run;
}

function pruneRetainedRuns() {
  const now = Date.now();
  const terminalRuns = [];
  for (const [jobId, run] of RUNS.entries()) {
    if (!run || typeof run !== "object") {
      RUNS.delete(jobId);
      continue;
    }
    if (!isRunTerminalStatus(run.status) || isRunResumable(run)) continue;
    compactRunForTerminalRetention(run);
    const lastActivityMs = getRunLastActivityMs(run);
    if (lastActivityMs && now - lastActivityMs > TERMINAL_RUN_RETENTION_MS) {
      RUNS.delete(jobId);
      continue;
    }
    terminalRuns.push([jobId, lastActivityMs]);
  }
  if (terminalRuns.length <= MAX_TERMINAL_RUNS_IN_MEMORY) return;
  terminalRuns
    .sort((left, right) => (right[1] || 0) - (left[1] || 0))
    .slice(MAX_TERMINAL_RUNS_IN_MEMORY)
    .forEach(([jobId]) => {
      RUNS.delete(jobId);
    });
}

function createRunSnapshot(run) {
  return cloneJson(
    {
      jobId: run.jobId,
      craftId: run.craftId,
      status: run.status,
      phase: run.phase,
      mode: run.mode,
      modelRef: run.modelRef,
      responseText: run.responseText,
      finalStatus: normalizeFinalRunStatus(run.finalStatus),
      tokens: run.tokens,
      costUsd: run.costUsd,
      error: run.error,
      errorDetail: resolveRunErrorDetail(run),
      report: run.report,
      questions: run.questions,
      provenance: run.provenance,
      suggestedName: run.suggestedName,
      officialDescription: run.officialDescription,
      activeTab: run.activeTab,
      logs: run.logs,
      toolTrace: run.toolTrace,
      lastToolFailure: run.lastToolFailure && typeof run.lastToolFailure === "object"
        ? cloneJson(run.lastToolFailure, null)
        : null,
      workspaceCodeDive: run.workspaceCodeDive && typeof run.workspaceCodeDive === "object"
        ? cloneJson(run.workspaceCodeDive, null)
        : null,
      experimentLedger: normalizeExperimentLedger(run.experimentLedger),
      latestExperimentCandidateId: asText(run.latestExperimentCandidateId),
      operations: run.operations,
      useSurface: run.useSurface,
      latestTrainingJobId: run.latestTrainingJobId,
      latestCodexRepairJobId: run.latestCodexRepairJobId,
      codexRepair: ensureCodexRepairState(run),
      reloadResume: normalizeReloadResumeState(run.reloadResume),
      maturity: normalizeMaturity(run.maturity, createEmptyCraftMaturity()),
      turnsUsed: run.turnsUsed,
      maxTurns: run.maxTurns,
      updatedAt: run.updatedAt,
      startedAt: run.startedAt,
      completedAt: run.completedAt,
    },
    null,
  );
}

function createPersistableRunRuntime(run) {
  if (!run || typeof run !== "object") return null;
  const runtime = {
    ...createRunSnapshot(run),
    brief: asText(run.brief),
    previousQuestions: normalizeQuestions(run.previousQuestions),
  };
  if (!isRunResumable(run)) {
    return cloneJson(runtime, null);
  }
  return cloneJson(
    {
      ...runtime,
      craft: cloneJson(run.craft, {}),
      samples: Array.isArray(run.samples) ? cloneJson(run.samples, []) : [],
      agentTooling: normalizeCraftingAgentToolingPayload(run.agentTooling),
      resolved: run.resolved ? cloneJson(run.resolved, {}) : null,
      requestedSlotId: asText(run.requestedSlotId),
      requestedModelRef: asText(run.requestedModelRef),
      requestedProviderId: asText(run.requestedProviderId),
      requestedModelName: asText(run.requestedModelName),
      parameters: run.parameters && typeof run.parameters === "object" ? cloneJson(run.parameters, {}) : {},
      reasoningEffort: asText(run.reasoningEffort),
      activeTabContext:
        run.activeTabContext && typeof run.activeTabContext === "object"
          ? cloneJson(run.activeTabContext, null)
          : null,
      browserRuntime:
        run.browserRuntime && typeof run.browserRuntime === "object"
          ? cloneJson(run.browserRuntime, {})
          : {},
      researchNotes: Array.isArray(run.researchNotes) ? run.researchNotes.slice(-8) : [],
      browserNotes: Array.isArray(run.browserNotes) ? run.browserNotes.slice(-8) : [],
      compactedMemory: Array.isArray(run.compactedMemory) ? cloneJson(run.compactedMemory, []) : [],
      compactionCount: Number(run.compactionCount || 0) || 0,
      turnOffset: Number(run.turnOffset || 0) || 0,
      segmentCount: Number(run.segmentCount || 0) || 0,
      segmentMaxTurns: Number(run.segmentMaxTurns || DEFAULT_SEGMENT_MAX_TURNS) || DEFAULT_SEGMENT_MAX_TURNS,
      initialized: run.initialized === true,
      serializedState: asText(run.serializedState),
      pendingApprovalCallId: asText(run.pendingApprovalCallId),
      pendingUserAnswersByCallId:
        run.pendingUserAnswersByCallId && typeof run.pendingUserAnswersByCallId === "object"
          ? cloneJson(run.pendingUserAnswersByCallId, {})
          : {},
      latestSuggestedQuestions: normalizeQuestions(run.latestSuggestedQuestions),
      accountedUsage: normalizeUsageTotals(run.accountedUsage),
      liveUsage: normalizeUsageTotals(run.liveUsage),
      capabilityEvidence: normalizeCapabilityEvidence(run.capabilityEvidence, run.craftId),
    },
    null,
  );
}

async function persistRunStateNow(run) {
  const craftId = asText(run?.craftId);
  if (!craftId) return;
  try {
    await upsertAgentRunState({
      craftId,
      snapshot: createRunSnapshot(run),
      runtime: createPersistableRunRuntime(run),
      meta: {
        jobId: asText(run?.jobId),
        status: asText(run?.status),
      },
    });
    emitDevObservabilityEvent(run, "run_snapshot_persisted", {
      run: buildRunObservabilitySnapshot(run),
    });
  } catch (error) {
    console.warn("[crafting-agent-runner] failed to persist run", error);
  }
}

function scheduleRunPersistence(run) {
  const craftId = asText(run?.craftId);
  if (!craftId) return;
  globalThis.clearTimeout(runPersistTimers.get(craftId) || 0);
  const timer = globalThis.setTimeout(() => {
    runPersistTimers.delete(craftId);
    void persistRunStateNow(run);
  }, RUN_PERSIST_DEBOUNCE_MS);
  runPersistTimers.set(craftId, timer);
}

function restorePersistedRun(runtime) {
  const run = cloneJson(runtime, null);
  if (!run || typeof run !== "object") return null;
  if (!asText(run.jobId) || !asText(run.craftId)) return null;
  run.executionPromise = null;
  run.abortController = null;
  run.stopRequested = false;
  run.stopReason = "";
  run.logs = Array.isArray(run.logs) ? run.logs : [];
  run.questions = normalizeQuestions(run.questions);
  run.previousQuestions = normalizeQuestions(run.previousQuestions);
  run.provenance = normalizeProvenance(run.provenance);
  run.suggestedName = normalizeCraftNameCandidate(run.suggestedName);
  run.officialDescription =
    normalizeStructuredOfficialDescriptionContract(run.officialDescription)?.text || null;
  run.report = normalizeReport(run.report, {
    objective: asText(run?.brief),
    currentState: asText(run?.responseText),
  });
  run.toolTrace = Array.isArray(run.toolTrace) ? run.toolTrace : [];
  run.experimentLedger = normalizeExperimentLedger(run.experimentLedger);
  run.latestExperimentCandidateId = asText(run.latestExperimentCandidateId);
  run.researchNotes = Array.isArray(run.researchNotes) ? run.researchNotes : [];
  run.browserNotes = Array.isArray(run.browserNotes) ? run.browserNotes : [];
  run.browserRuntime =
    run.browserRuntime && typeof run.browserRuntime === "object"
      ? cloneJson(run.browserRuntime, {})
      : {};
  ensureRunBrowserRuntime(run);
  run.compactedMemory = Array.isArray(run.compactedMemory) ? run.compactedMemory : [];
  run.parameters = run.parameters && typeof run.parameters === "object" ? cloneJson(run.parameters, {}) : {};
  run.agentTooling = normalizeCraftingAgentToolingPayload(run.agentTooling);
  run.accountedUsage = normalizeUsageTotals(run.accountedUsage);
  run.liveUsage = normalizeUsageTotals(run.liveUsage);
  run.finalStatus = normalizeFinalRunStatus(run.finalStatus);
  run.reloadResume = normalizeReloadResumeState(run.reloadResume);
  run.maturity = normalizeMaturity(run.maturity, createEmptyCraftMaturity());
  run.capabilityEvidence = normalizeCapabilityEvidence(run.capabilityEvidence, run.craftId);
  run.errorDetail = resolveRunErrorDetail(run);
  run.lastToolFailure =
    run.lastToolFailure && typeof run.lastToolFailure === "object"
      ? cloneJson(run.lastToolFailure, null)
      : null;
  run.workspaceCodeDive =
    run.workspaceCodeDive && typeof run.workspaceCodeDive === "object"
      ? cloneJson(run.workspaceCodeDive, null)
      : null;
  run.completedAt = asText(run.completedAt);
  return run;
}

function recoverPersistedRun(run) {
  const status = asText(run?.status);
  const reloadResume = normalizeReloadResumeState(run?.reloadResume);
  if (reloadResume.pending) {
    run.reloadResume = reloadResume;
    return run;
  }
  if (!["starting", "running", "resuming"].includes(status)) return run;
  run.status = "failed";
  run.phase = "failed";
  run.error = asText(run.error) || "The background agent was interrupted by a restart.";
  run.finalStatus = "";
  run.completedAt = new Date().toISOString();
  run.report = normalizeReport(run.report, {
    currentState: run.error,
    nextAction: "Restart the agent run to continue from a fresh runtime.",
  });
  run.logs = [...(Array.isArray(run.logs) ? run.logs : []), createLog("error", run.error)].slice(-MAX_LOGS);
  run.maturity = normalizeMaturity(run.maturity, createEmptyCraftMaturity());
  run.capabilityEvidence = normalizeCapabilityEvidence(run.capabilityEvidence, run.craftId);
  return run;
}

async function ensureRunsHydrated() {
  if (runsHydrated) return;
  if (runsHydrationPromise) return runsHydrationPromise;

  runsHydrationPromise = (async () => {
    const records = await listAgentRunStates().catch(() => []);
    for (const record of records) {
      const restored = restorePersistedRun(record?.runtime);
      if (!restored) continue;
      const recovered = recoverPersistedRun(restored);
      compactRunForTerminalRetention(recovered);
      RUNS.set(recovered.jobId, recovered);
    }
    pruneRetainedRuns();
    runsHydrated = true;
    runsHydrationPromise = null;
  })().catch((error) => {
    runsHydrationPromise = null;
    throw error;
  });

  return runsHydrationPromise;
}

async function readPersistedRunByJobId(jobId = "") {
  const normalizedJobId = asText(jobId);
  if (!normalizedJobId) return null;
  const records = await listAgentRunStates().catch(() => []);
  const record = records.find((entry) => {
    const runtimeJobId = asText(entry?.runtime?.jobId);
    const snapshotJobId = asText(entry?.snapshot?.jobId);
    return runtimeJobId === normalizedJobId || snapshotJobId === normalizedJobId;
  });
  if (!record) return null;
  const restored = restorePersistedRun(record?.runtime || record?.snapshot);
  if (!restored) return null;
  const recovered = recoverPersistedRun(restored);
  compactRunForTerminalRetention(recovered);
  return recovered;
}

function pushRunLog(run, level, message, meta = {}) {
  const nextLog = createLog(level, message, meta);
  run.logs.push(nextLog);
  if (run.logs.length > MAX_LOGS) {
    run.logs = run.logs.slice(-MAX_LOGS);
  }
  run.updatedAt = new Date().toISOString();
  scheduleRunPersistence(run);
  emitDevObservabilityEvent(run, "run_log", {
    log: summarizeRunLogEntry(nextLog),
  });
}

function finalizeRunRetention(run) {
  compactRunForTerminalRetention(run);
  scheduleRunPersistence(run);
  pruneRetainedRuns();
}

function pushRunProvenance(run, entries) {
  const merged = normalizeProvenance([...(run.provenance || []), ...(Array.isArray(entries) ? entries : [])]);
  run.provenance = merged;
  run.updatedAt = new Date().toISOString();
  scheduleRunPersistence(run);
}

function setRunReport(run, report, fallback = {}) {
  run.report = normalizeReport(report, fallback);
  run.updatedAt = new Date().toISOString();
  scheduleRunPersistence(run);
}

async function resolveRunMaturity(run, maturity, fallback = null) {
  await refreshRunCapabilityEvidence(run);
  return gateCraftMaturityForCapability(
    normalizeMaturity(maturity, fallback || run?.maturity || createEmptyCraftMaturity()),
    {
      hasTrainedCapability: run.capabilityEvidence?.hasTrainedCapability === true,
    },
  );
}

async function setRunMaturity(run, maturity, fallback = null) {
  run.maturity = await resolveRunMaturity(run, maturity, fallback);
  if (run.maturity?.isExplicit) {
    run.maturity.updatedAt = new Date().toISOString();
  }
  run.updatedAt = new Date().toISOString();
  scheduleRunPersistence(run);
}

function buildDefaultReport(run, fallback = {}) {
  const sampleCount = Array.isArray(run.samples) ? run.samples.length : 0;
  return normalizeReport(run.report, {
    objective: trimText(run.brief, 180),
    currentState: sampleCount
      ? `${sampleCount} local training rows are available as context.`
      : "No local training rows exist for this craft yet.",
    nextAction: fallback.nextAction || "Wait for the next agent step.",
    matchingSignals: run.report?.matchingSignals || [],
  });
}

function mergeQuestionsWithAnswers(nextQuestions, previousQuestions = []) {
  const answerById = new Map(
    normalizeQuestions(previousQuestions).map((entry) => [entry.id, asText(entry.answer)]),
  );
  return normalizeQuestions(nextQuestions).map((entry) => ({
    ...entry,
    answer: answerById.get(entry.id) || entry.answer || "",
  }));
}

function normalizeBrowserTabContext(rawTab = {}, fallbackUrl = "") {
  const source = rawTab && typeof rawTab === "object" ? rawTab : {};
  const tabId = Number(source.id || source.tabId || source.tab_id || 0);
  const title = asText(source.title || source.pageState?.title || source.page_state?.title);
  const urlText = asText(
    source.url ||
      source.pendingUrl ||
      source.pending_url ||
      source.finalUrl ||
      source.final_url ||
      source.pageState?.url ||
      source.page_state?.url ||
      fallbackUrl,
  );
  if (!urlText || !/^https?:/i.test(urlText)) return null;
  try {
    const parsed = new URL(urlText);
    return {
      tabId: Number.isFinite(tabId) && tabId > 0 ? tabId : null,
      title,
      url: parsed.toString(),
      hostname: parsed.hostname,
      active: source.active === true,
      windowId:
        Number.isFinite(Number(source.windowId)) && Number(source.windowId) > 0
          ? Number(source.windowId)
          : null,
    };
  } catch {
    return null;
  }
}

function ensureRunBrowserRuntime(run) {
  const runtime =
    run?.browserRuntime && typeof run.browserRuntime === "object"
      ? run.browserRuntime
      : {};
  const currentTabId = Number(runtime.currentTabId || 0);
  runtime.currentTabId = Number.isFinite(currentTabId) && currentTabId > 0 ? currentTabId : 0;
  run.browserRuntime = runtime;
  return runtime;
}

function syncRunBrowserContext(run, context, { clearOnNull = false, fallbackUrl = "" } = {}) {
  const normalized = normalizeBrowserTabContext(context, fallbackUrl);
  const runtime = ensureRunBrowserRuntime(run);
  if (!normalized) {
    if (clearOnNull) {
      run.activeTabContext = null;
      runtime.currentTabId = 0;
    }
    return null;
  }
  run.activeTabContext = normalized;
  runtime.currentTabId = Number(normalized.tabId || 0) || 0;
  return normalized;
}

function pTabsQuery(queryInfo = {}) {
  if (!globalThis.chrome?.tabs?.query) {
    return Promise.reject(new Error("chrome.tabs.query unavailable."));
  }
  return new Promise((resolve, reject) => {
    globalThis.chrome.tabs.query(queryInfo, (tabs) => {
      const err = globalThis.chrome?.runtime?.lastError;
      if (err) reject(new Error(err.message));
      else resolve(Array.isArray(tabs) ? tabs : []);
    });
  });
}

function pTabsGet(tabId) {
  if (!globalThis.chrome?.tabs?.get) {
    return Promise.reject(new Error("chrome.tabs.get unavailable."));
  }
  return new Promise((resolve, reject) => {
    globalThis.chrome.tabs.get(Number(tabId), (tab) => {
      const err = globalThis.chrome?.runtime?.lastError;
      if (err) reject(new Error(err.message));
      else resolve(tab || null);
    });
  });
}

function pTabsCreate(createProperties = {}) {
  if (!globalThis.chrome?.tabs?.create) {
    return Promise.reject(new Error("chrome.tabs.create unavailable."));
  }
  return new Promise((resolve, reject) => {
    globalThis.chrome.tabs.create(createProperties, (tab) => {
      const err = globalThis.chrome?.runtime?.lastError;
      if (err) reject(new Error(err.message));
      else resolve(tab || null);
    });
  });
}

function pTabsUpdate(tabId, updateProperties = {}) {
  if (!globalThis.chrome?.tabs?.update) {
    return Promise.reject(new Error("chrome.tabs.update unavailable."));
  }
  return new Promise((resolve, reject) => {
    globalThis.chrome.tabs.update(Number(tabId), updateProperties, (tab) => {
      const err = globalThis.chrome?.runtime?.lastError;
      if (err) reject(new Error(err.message));
      else resolve(tab || null);
    });
  });
}

function pTabsRemove(tabIds) {
  if (!globalThis.chrome?.tabs?.remove) {
    return Promise.reject(new Error("chrome.tabs.remove unavailable."));
  }
  const ids = (Array.isArray(tabIds) ? tabIds : [tabIds])
    .map((value) => Number(value || 0))
    .filter((value) => Number.isFinite(value) && value > 0);
  if (!ids.length) {
    return Promise.reject(new Error("No valid tab ids were provided."));
  }
  return new Promise((resolve, reject) => {
    globalThis.chrome.tabs.remove(ids.length === 1 ? ids[0] : ids, () => {
      const err = globalThis.chrome?.runtime?.lastError;
      if (err) reject(new Error(err.message));
      else resolve(true);
    });
  });
}

function pExecuteScript(tabId, func, args = [], { world = "MAIN" } = {}) {
  if (!globalThis.chrome?.scripting?.executeScript) {
    return Promise.reject(new Error("chrome.scripting.executeScript unavailable."));
  }
  return new Promise((resolve, reject) => {
    globalThis.chrome.scripting.executeScript(
      {
        target: { tabId: Number(tabId), allFrames: false },
        world,
        func,
        args,
      },
      (results) => {
        const err = globalThis.chrome?.runtime?.lastError;
        if (err) reject(new Error(err.message));
        else resolve(Array.isArray(results) ? results : []);
      },
    );
  });
}

async function waitForBrowserTabContext(run, {
  preferredTabId = 0,
  fallbackUrl = "",
  allowActiveFallback = false,
  timeoutMs = 2_500,
  pollMs = 125,
} = {}) {
  const targetTabId = Number(preferredTabId || 0);
  const deadline = Date.now() + Math.max(250, Number(timeoutMs || 0) || 0);
  const sleepMs = Math.max(50, Math.min(250, Number(pollMs || 0) || 125));
  let fallbackContext = null;

  while (Date.now() <= deadline) {
    if (Number.isFinite(targetTabId) && targetTabId > 0) {
      const direct = await getTabContextById(targetTabId).catch(() => null);
      if (direct) return syncRunBrowserContext(run, direct, { clearOnNull: false });
      const rawTab = await pTabsGet(targetTabId).catch(() => null);
      fallbackContext = normalizeBrowserTabContext(rawTab, fallbackUrl) || fallbackContext;
    }

    if (allowActiveFallback) {
      const active = await queryActiveTabContext();
      if (active) return syncRunBrowserContext(run, active, { clearOnNull: false });
    }

    if (Date.now() >= deadline) break;
    await new Promise((resolve) => setTimeout(resolve, sleepMs));
  }

  if (fallbackContext) {
    return syncRunBrowserContext(run, fallbackContext, {
      clearOnNull: false,
      fallbackUrl,
    });
  }

  return null;
}

function collectBrowserHosts(values = []) {
  const hosts = [];
  const seen = new Set();
  for (const value of Array.isArray(values) ? values : [values]) {
    const text = asText(value);
    if (!text) continue;
    let candidate = text;
    try {
      candidate = new URL(text).hostname;
    } catch {}
    const normalized = candidate.toLowerCase().replace(/^\.+|\.+$/g, "");
    if (!normalized || seen.has(normalized)) continue;
    seen.add(normalized);
    hosts.push(normalized);
  }
  return hosts;
}

function deriveBrowserAllowedHosts(run, args = {}, tabContext = null) {
  const explicitHosts = Array.isArray(args?.allowedHosts) ? args.allowedHosts : [];
  return collectBrowserHosts([
    ...explicitHosts,
    asText(args?.url),
    asText(tabContext?.url),
    asText(tabContext?.hostname),
    asText(run?.activeTabContext?.url),
    asText(run?.activeTabContext?.hostname),
  ]);
}

function deriveBrowserBaseUrl(args = {}, tabContext = null) {
  return asText(args?.url) || asText(tabContext?.url) || DEFAULT_BROWSER_TOOL_BASE_URL;
}

function summarizeBrowserTabs(tabs = [], limit = 12) {
  return (Array.isArray(tabs) ? tabs : [])
    .map((tab) => normalizeBrowserTabContext(tab))
    .filter(Boolean)
    .sort((left, right) => {
      const leftScore = Number(left?.active ? 1_000_000_000 : 0);
      const rightScore = Number(right?.active ? 1_000_000_000 : 0);
      return rightScore - leftScore;
    })
    .slice(0, limit);
}

async function readNewestCraftArtifact(craftId, kind) {
  if (!craftSync?.listLocalArtifacts) return null;
  const records = await craftSync.listLocalArtifacts({
    craftId: asText(craftId),
    kind: asText(kind),
  });
  return Array.isArray(records) && records.length ? records[0] : null;
}

async function readCraftBundleSnapshot(craft = null) {
  const craftId = asText(craft?.id);
  if (!craftId) {
    return {
      craftId: "",
      trainingDataRecord: null,
      toolScriptsRecord: null,
      browserCapabilitiesRecord: null,
      weightsRecord: null,
      policyRecord: null,
      bundle: buildCapabilityBundle({ craft }),
    };
  }

  const [
    trainingDataRecord,
    toolScriptsRecord,
    browserCapabilitiesRecord,
    weightsRecord,
    policyRecord,
  ] = await Promise.all([
    readNewestCraftArtifact(craftId, TRAINING_DATA_ARTIFACT_KIND),
    readNewestCraftArtifact(craftId, TOOL_SCRIPTS_ARTIFACT_KIND),
    readNewestCraftArtifact(craftId, BROWSER_CAPABILITY_BUNDLE_ARTIFACT_KIND),
    readNewestCraftArtifact(craftId, WEIGHTS_ARTIFACT_KIND),
    readNewestCraftArtifact(craftId, POLICY_BUNDLE_ARTIFACT_KIND),
  ]);

  return {
    craftId,
    trainingDataRecord,
    toolScriptsRecord,
    browserCapabilitiesRecord,
    weightsRecord,
    policyRecord,
    bundle: buildCapabilityBundle({
      craft,
      trainingDataRecord,
      toolScriptsRecord,
      browserCapabilitiesRecord,
      weightsRecord,
      policyRecord,
      preserveStoredBrowserCapabilities: true,
      toolScriptsOptions: STRICT_RUNTIME_TOOL_SCRIPTS_OPTIONS,
      browserCapabilityOptions: STRICT_RUNTIME_BROWSER_CAPABILITY_OPTIONS,
    }),
  };
}

function buildArtifactRecord(id, craftId, kind, payload, meta = {}) {
  return {
    id,
    craftId: asText(craftId),
    kind: asText(kind),
    payload: cloneJson(payload, {}),
    meta: cloneJson(meta, {}),
  };
}

function createBundleFromSnapshot(craft, snapshot, bundleOverride = {}) {
  const craftId = asText(craft?.id);
  const trainingDataRecord = bundleOverride?.trainingData && typeof bundleOverride.trainingData === "object"
    ? buildArtifactRecord(
        snapshot?.bundle?.trainingData?.artifactId || `training-samples:${craftId}`,
        craftId,
        TRAINING_DATA_ARTIFACT_KIND,
        bundleOverride.trainingData,
        {
          ...(snapshot?.trainingDataRecord?.meta || {}),
          updatedAt: Date.now(),
          smokeOverride: true,
        },
      )
    : snapshot?.trainingDataRecord || null;
  const toolScriptsRecord = bundleOverride?.toolScripts && typeof bundleOverride.toolScripts === "object"
    ? buildArtifactRecord(
        getToolScriptsArtifactId(craftId),
        craftId,
        TOOL_SCRIPTS_ARTIFACT_KIND,
        normalizeToolScriptsPayload(bundleOverride.toolScripts, craft, STRICT_RUNTIME_TOOL_SCRIPTS_OPTIONS),
        {
          scriptCount: Array.isArray(bundleOverride.toolScripts?.scripts) ? bundleOverride.toolScripts.scripts.length : 0,
          updatedAt: Date.now(),
          smokeOverride: true,
        },
      )
    : snapshot?.toolScriptsRecord || null;
  const browserCapabilitiesRecord = bundleOverride?.browserCapabilities && typeof bundleOverride.browserCapabilities === "object"
    ? buildArtifactRecord(
        getBrowserCapabilityBundleArtifactId(craftId),
        craftId,
        BROWSER_CAPABILITY_BUNDLE_ARTIFACT_KIND,
        buildPublishedBrowserCapabilitiesPayload(
          bundleOverride.browserCapabilities,
          craft,
          toolScriptsRecord?.payload || null,
          {
            publishedBy: "crafting_agent_smoke_override",
          },
        ),
        {
          capabilityCount: Array.isArray(bundleOverride.browserCapabilities?.capabilities)
            ? bundleOverride.browserCapabilities.capabilities.length
            : 0,
          updatedAt: Date.now(),
          smokeOverride: true,
        },
      )
    : snapshot?.browserCapabilitiesRecord || null;
  const weightsRecord = bundleOverride?.weights && typeof bundleOverride.weights === "object"
    ? buildArtifactRecord(
        snapshot?.bundle?.weights?.artifactId || `capability-weights:${craftId}`,
        craftId,
        WEIGHTS_ARTIFACT_KIND,
        bundleOverride.weights,
        {
          ...(snapshot?.weightsRecord?.meta || {}),
          updatedAt: Date.now(),
          smokeOverride: true,
        },
      )
    : snapshot?.weightsRecord || null;
  const policyRecord = bundleOverride?.policy && typeof bundleOverride.policy === "object"
    ? buildArtifactRecord(
        snapshot?.bundle?.policy?.artifactId || `policy-bundle:${craftId}`,
        craftId,
        POLICY_BUNDLE_ARTIFACT_KIND,
        bundleOverride.policy,
        {
          ...(snapshot?.policyRecord?.meta || {}),
          updatedAt: Date.now(),
          smokeOverride: true,
        },
      )
    : snapshot?.policyRecord || null;

  return buildCapabilityBundle({
    craft,
    trainingDataRecord,
    toolScriptsRecord,
    browserCapabilitiesRecord,
    weightsRecord,
    policyRecord,
    preserveStoredBrowserCapabilities: true,
    toolScriptsOptions: STRICT_RUNTIME_TOOL_SCRIPTS_OPTIONS,
    browserCapabilityOptions: STRICT_RUNTIME_BROWSER_CAPABILITY_OPTIONS,
  });
}

function buildWorkspaceRepoPath(runtimePath = "") {
  const normalized = asText(runtimePath).replace(/^\/+/, "");
  return normalized ? `${WORKSPACE_REPO_ROOT}/${normalized}` : WORKSPACE_REPO_ROOT;
}

function normalizeWorkspaceSearchText(value = "") {
  return asText(value)
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/ß/g, "ss")
    .toLowerCase();
}

function normalizeWorkspaceRuntimePathCandidate(value = "") {
  const normalized = normalizeWorkspaceSearchText(value)
    .replace(/^chrome-extension:\/\/[^/]+\//, "")
    .replace(/^.*?fuck-api-train-local-ai\//, "")
    .replace(/^\/+/, "")
    .replace(/[),.;]+$/g, "");
  return WORKSPACE_CODE_DIVE_RUNTIME_PATHS.has(normalized) ? normalized : "";
}

async function readPackagedWorkspaceSourceText(runtimePath = "") {
  const normalizedPath = asText(runtimePath).replace(/^\/+/, "");
  if (!normalizedPath) {
    throw new Error("workspace_code_dive requires a packaged runtime path.");
  }
  if (!workspaceSourceTextCache.has(normalizedPath)) {
    const runtimeUrl = globalThis.chrome?.runtime?.getURL
      ? globalThis.chrome.runtime.getURL(normalizedPath)
      : normalizedPath;
    workspaceSourceTextCache.set(
      normalizedPath,
      fetch(runtimeUrl)
        .then(async (response) => {
          if (!response.ok) {
            throw new Error(`Failed to read packaged source ${normalizedPath}: ${response.status}`);
          }
          return await response.text();
        })
        .catch((error) => {
          workspaceSourceTextCache.delete(normalizedPath);
          throw error;
        }),
    );
  }
  return await workspaceSourceTextCache.get(normalizedPath);
}

function extractWorkspaceCodeDiveFocusReferences(...values) {
  const refs = [];
  const seen = new Set();
  for (const value of values.flat()) {
    const text = asText(value);
    if (!text) continue;
    const pattern =
      /(?:fuck-api-train-local-ai\/)?((?:manifest\.json|sidepanel\.js|(?:bg|shared)\/[a-z0-9_.-]+\.(?:js|mjs|json)))(?::(\d+))?(?::(\d+))?/gi;
    let match;
    while ((match = pattern.exec(text))) {
      const runtimePath = normalizeWorkspaceRuntimePathCandidate(match[1] || "");
      if (!runtimePath) continue;
      const line = Number(match[2] || 0);
      const column = Number(match[3] || 0);
      const key = `${runtimePath}:${line || 0}:${column || 0}`;
      if (seen.has(key)) continue;
      seen.add(key);
      refs.push({
        runtimePath,
        line: line > 0 ? line : 0,
        column: column > 0 ? column : 0,
      });
    }
  }
  return refs;
}

function collectWorkspaceCodeDiveErrorTexts(run, args = {}) {
  const texts = [
    args.errorText,
    run?.lastToolFailure?.error,
    run?.lastToolFailure?.errorStack,
  ];
  const failures = Array.isArray(run?.lastToolFailure?.errorDetail?.failures)
    ? run.lastToolFailure.errorDetail.failures
    : [];
  for (const failure of failures) {
    texts.push(failure?.error?.message, failure?.error?.stack);
  }
  return texts.filter(Boolean);
}

function buildWorkspaceCodeDiveFocusReferences(args = {}, run = null) {
  return extractWorkspaceCodeDiveFocusReferences(
    Array.isArray(args.focusFiles) ? args.focusFiles : [],
    collectWorkspaceCodeDiveErrorTexts(run, args),
  ).slice(0, 10);
}

function formatWorkspaceFocusReference(ref = {}) {
  const runtimePath = asText(ref?.runtimePath);
  if (!runtimePath) return "";
  const line = Number(ref?.line || 0);
  return line > 0 ? `${runtimePath}:${line}` : runtimePath;
}

function buildWorkspaceCodeDiveTerms(args = {}) {
  const rawTexts = [
    args.problem,
    args.errorText,
    args.failingTool,
    ...(Array.isArray(args.focusTags) ? args.focusTags : []),
    ...(Array.isArray(args.focusFiles) ? args.focusFiles : []),
  ];
  const tokens = [];
  const seen = new Set();
  for (const rawText of rawTexts) {
    const parts = normalizeWorkspaceSearchText(rawText)
      .split(/[^a-z0-9_./-]+/g)
      .map((entry) => entry.replace(/^[./-]+|[./-]+$/g, ""))
      .filter(Boolean);
    for (const part of parts) {
      const normalized = part.replace(/[_./-]+/g, " ").trim();
      const candidates = [part, ...normalized.split(/\s+/g)];
      for (const candidate of candidates) {
        const token = asText(candidate).toLowerCase();
        if (
          !token ||
          token.length < 3 ||
          token.length > 40 ||
          !/[a-z]/.test(token) ||
          /^\d+$/.test(token) ||
          ["js", "mjs", "json"].includes(token) ||
          WORKSPACE_CODE_DIVE_STOP_WORDS.has(token) ||
          seen.has(token)
        ) {
          continue;
        }
        seen.add(token);
        tokens.push(token);
        if (tokens.length >= 24) {
          return tokens;
        }
      }
    }
  }
  return tokens;
}

function findNearestWorkspaceSymbol(lines = [], lineIndex = 0) {
  const start = Math.max(0, Number(lineIndex || 0));
  const end = Math.max(0, Number(lineIndex || 0) - 16);
  for (let cursor = start; cursor >= end; cursor -= 1) {
    const line = String(lines[cursor] || "");
    const functionMatch =
      /^\s*(?:export\s+)?function\s+([A-Za-z0-9_]+)/.exec(line) ||
      /^\s*(?:export\s+)?async function\s+([A-Za-z0-9_]+)/.exec(line) ||
      /^\s*class\s+([A-Za-z0-9_]+)/.exec(line);
    if (functionMatch?.[1]) return functionMatch[1];
  }
  for (let cursor = start; cursor >= end; cursor -= 1) {
    const line = String(lines[cursor] || "");
    const variableMatch =
      /^\s*(?:const|let|var)\s+([A-Za-z0-9_]+)\s*=\s*(?:async\s*)?\(/.exec(line) ||
      /^\s*(?:const|let|var)\s+([A-Za-z0-9_]+)\s*=/.exec(line);
    if (variableMatch?.[1]) return variableMatch[1];
  }
  return "";
}

function buildWorkspaceMatchExcerpt(lines = [], matchLineIndex = 0, radius = WORKSPACE_CODE_DIVE_SNIPPET_RADIUS) {
  const start = Math.max(0, Number(matchLineIndex || 0) - radius);
  const end = Math.min(lines.length, Number(matchLineIndex || 0) + radius + 1);
  return lines
    .slice(start, end)
    .map((line, index) => `${start + index + 1}: ${String(line || "").slice(0, 220)}`)
    .join("\n");
}

function collectWorkspaceCodeDiveMatches(sourceText = "", terms = [], focusReferences = []) {
  const text = String(sourceText || "");
  const lines = text.split("\n");
  const matches = [];
  const seenLines = new Set();
  for (const ref of Array.isArray(focusReferences) ? focusReferences : []) {
    const lineIndex = Math.max(0, Number(ref?.line || 0) - 1);
    if (lineIndex >= lines.length || seenLines.has(lineIndex)) continue;
    const line = String(lines[lineIndex] || "");
    const haystack = normalizeWorkspaceSearchText(line);
    matches.push({
      line: lineIndex + 1,
      symbol: findNearestWorkspaceSymbol(lines, lineIndex),
      matchedTerms: uniqueTextList(
        [
          "stack",
          ...terms.filter((term) => haystack.includes(term)),
        ],
        6,
      ),
      excerpt: buildWorkspaceMatchExcerpt(lines, lineIndex),
    });
    seenLines.add(lineIndex);
    if (matches.length >= WORKSPACE_CODE_DIVE_MAX_MATCHES_PER_FILE) {
      return matches;
    }
  }
  for (const [index, rawLine] of lines.entries()) {
    if (seenLines.has(index)) continue;
    const line = String(rawLine || "");
    const haystack = normalizeWorkspaceSearchText(line);
    const matchedTerms = terms.filter((term) => haystack.includes(term)).slice(0, 6);
    if (!matchedTerms.length) continue;
    matches.push({
      line: index + 1,
      symbol: findNearestWorkspaceSymbol(lines, index),
      matchedTerms,
      excerpt: buildWorkspaceMatchExcerpt(lines, index),
    });
    if (matches.length >= WORKSPACE_CODE_DIVE_MAX_MATCHES_PER_FILE) break;
  }
  return matches;
}

function buildWorkspaceCodeDiveHeuristics(args = {}, focusReferences = []) {
  const failingTool = asText(args.failingTool).toLowerCase();
  const haystack = [
    args.problem,
    args.errorText,
    args.failingTool,
    ...(Array.isArray(args.focusTags) ? args.focusTags : []),
  ].map((entry) => asText(entry).toLowerCase()).join(" ");
  const preferredRuntimePaths = [];
  const preferredSubsystems = [];
  const suggestedExecutionPath = [];
  const focusRuntimePaths = uniqueTextList(
    (Array.isArray(focusReferences) ? focusReferences : []).map((entry) => entry?.runtimePath),
    8,
  );

  preferredRuntimePaths.push(...focusRuntimePaths);

  if (/referenceerror[^.\n]*is not defined|is not defined|runtime injection|execute binding|tool script mapping|aufrufbare funktion|injizierte laufzeit|tool script entrypoint/.test(haystack)) {
    preferredRuntimePaths.push(
      "bg/crafting-agent-runner.js",
      "bg/craft-use-runner.js",
      "shared/capability-bundle.mjs",
      "shared/browserAutomationRuntime.js",
      "shared/browserAutomationTools.js",
    );
    preferredSubsystems.push("capability_runtime", "bundle_storage", "browser_runtime", "crafting_agent");
    suggestedExecutionPath.push(
      `${WORKSPACE_REPO_ROOT}/bg/crafting-agent-runner.js`,
      `${WORKSPACE_REPO_ROOT}/shared/capability-bundle.mjs`,
      `${WORKSPACE_REPO_ROOT}/bg/craft-use-runner.js`,
      `${WORKSPACE_REPO_ROOT}/shared/browserAutomationRuntime.js`,
      `${WORKSPACE_REPO_ROOT}/shared/browserAutomationTools.js`,
    );
  }

  if (/modeltransform|model transform|pre\/execute\/post null|empty scripts|entrypoint|tool script wiring|capability scripts|reviewed runtime path/.test(haystack)) {
    preferredRuntimePaths.push(
      "bg/crafting-agent-runner.js",
      "bg/craft-use-runner.js",
      "shared/capability-bundle.mjs",
      "shared/browser-capability-bundle.mjs",
      "shared/browserAutomationRuntime.js",
      "shared/browserAutomationTools.js",
    );
    preferredSubsystems.push("capability_runtime", "capability_contracts", "bundle_storage", "browser_runtime");
    suggestedExecutionPath.push(
      `${WORKSPACE_REPO_ROOT}/bg/crafting-agent-runner.js`,
      `${WORKSPACE_REPO_ROOT}/shared/capability-bundle.mjs`,
      `${WORKSPACE_REPO_ROOT}/shared/browser-capability-bundle.mjs`,
      `${WORKSPACE_REPO_ROOT}/bg/craft-use-runner.js`,
      `${WORKSPACE_REPO_ROOT}/shared/browserAutomationRuntime.js`,
      `${WORKSPACE_REPO_ROOT}/shared/browserAutomationTools.js`,
    );
  }

  if (/local_qwen|qwen|onnx|webgpu|ortrun|offscreen/.test(haystack)) {
    preferredRuntimePaths.push(
      "bg/crafting-agent-runner.js",
      "bg/llm.js",
      "bg/ml-inference.js",
      "shared/local_qwen_runtime.mjs",
      "bg/service_worker.js",
    );
    preferredSubsystems.push("local_runtime", "model_bridge", "crafting_agent");
    suggestedExecutionPath.push(
      `${WORKSPACE_REPO_ROOT}/sidepanel.js`,
      `${WORKSPACE_REPO_ROOT}/bg/service_worker.js`,
      `${WORKSPACE_REPO_ROOT}/bg/crafting-agent-runner.js`,
      `${WORKSPACE_REPO_ROOT}/bg/llm.js`,
      `${WORKSPACE_REPO_ROOT}/bg/ml-inference.js`,
      `${WORKSPACE_REPO_ROOT}/shared/local_qwen_runtime.mjs`,
    );
  }

  if (/module is not defined|module\.exports|exports is not defined|commonjs|script format/.test(haystack)) {
    preferredRuntimePaths.push(
      "bg/crafting-agent-runner.js",
      "bg/craft-use-runner.js",
      "shared/browserAutomationRuntime.js",
      "shared/browserAutomationTools.js",
      "shared/capability-bundle.mjs",
      "shared/browser-capability-bundle.mjs",
    );
    preferredSubsystems.push("capability_runtime", "browser_runtime", "bundle_storage", "capability_contracts");
    if (!suggestedExecutionPath.length) {
      suggestedExecutionPath.push(
        `${WORKSPACE_REPO_ROOT}/bg/crafting-agent-runner.js`,
        `${WORKSPACE_REPO_ROOT}/shared/capability-bundle.mjs`,
        `${WORKSPACE_REPO_ROOT}/shared/browser-capability-bundle.mjs`,
        `${WORKSPACE_REPO_ROOT}/bg/craft-use-runner.js`,
        `${WORKSPACE_REPO_ROOT}/shared/browserAutomationRuntime.js`,
        `${WORKSPACE_REPO_ROOT}/shared/browserAutomationTools.js`,
      );
    }
  }

  if (failingTool === "run_agentic_smoke" || /reviewed capability|capability path|active text|clipboard|focused editable/.test(haystack)) {
    preferredRuntimePaths.push(
      "bg/crafting-agent-runner.js",
      "bg/craft-use-runner.js",
      "shared/browser-capability-bundle.mjs",
      "shared/capability-bundle.mjs",
    );
    preferredSubsystems.push("capability_runtime", "capability_contracts", "bundle_storage");
    if (!suggestedExecutionPath.length) {
      suggestedExecutionPath.push(
        `${WORKSPACE_REPO_ROOT}/sidepanel.js`,
        `${WORKSPACE_REPO_ROOT}/bg/service_worker.js`,
        `${WORKSPACE_REPO_ROOT}/bg/crafting-agent-runner.js`,
        `${WORKSPACE_REPO_ROOT}/bg/craft-use-runner.js`,
        `${WORKSPACE_REPO_ROOT}/shared/browser-capability-bundle.mjs`,
      );
    }
  }

  if (/debug json|clipboard|copy debug/.test(haystack)) {
    preferredRuntimePaths.push("sidepanel.js");
    preferredSubsystems.push("sidepanel_ui");
  }

  return {
    preferredRuntimePaths: uniqueTextList(preferredRuntimePaths, 12),
    preferredSubsystems: uniqueTextList(preferredSubsystems, 8),
    suggestedExecutionPath: uniqueTextList(suggestedExecutionPath, 8),
    focusRuntimePaths,
  };
}

function inferWorkspaceValidationTargets(selectedFiles = [], heuristics = {}) {
  const repoPaths = new Set((Array.isArray(selectedFiles) ? selectedFiles : []).map((entry) => asText(entry?.path)));
  const tests = new Set();
  if (
    [...repoPaths].some((path) =>
      /bg\/ml-inference\.js|bg\/llm\.js|shared\/local_qwen_runtime\.mjs/.test(path),
    )
  ) {
    tests.add("tests/local_qwen_error_regressions.test.mjs");
    tests.add("tests/local_qwen_runtime.test.mjs");
  }
  if (
    [...repoPaths].some((path) =>
      /bg\/craft-use-runner\.js|shared\/browser-capability-bundle\.mjs|shared\/capability-bundle\.mjs|shared\/browserAutomationRuntime\.js|shared\/browserAutomationTools\.js/.test(path),
    )
  ) {
    tests.add("tests/craft_use_runtime_regressions.test.mjs");
    tests.add("tests/qwen_agent_adapter.test.mjs");
  }
  if (
    [...repoPaths].some((path) =>
      /sidepanel\.js|bg\/crafting-agent-runner\.js/.test(path),
    ) || Array.isArray(heuristics?.suggestedExecutionPath) && heuristics.suggestedExecutionPath.length
  ) {
    tests.add("tests/crafting_agent_ui_regressions.test.mjs");
  }
  return [...tests].slice(0, 8);
}

function scoreWorkspaceCodeDiveFile(entry = {}, terms = [], heuristics = {}) {
  const haystack = [
    entry.repoPath,
    entry.subsystem,
    entry.description,
    ...(Array.isArray(entry.tags) ? entry.tags : []),
  ].map((value) => asText(value).toLowerCase()).join(" ");
  let score = 0;
  const matchedTags = [];
  for (const term of terms) {
    if (!term) continue;
    if (haystack.includes(term)) {
      score += 8;
      matchedTags.push(term);
    }
  }
  if ((heuristics?.preferredRuntimePaths || []).includes(entry.runtimePath)) {
    score += 36;
  }
  if ((heuristics?.focusRuntimePaths || []).includes(entry.runtimePath)) {
    score += 42;
  }
  if ((heuristics?.preferredSubsystems || []).includes(entry.subsystem)) {
    score += 18;
  }
  return {
    score,
    matchedTags: uniqueTextList(matchedTags, 8),
  };
}

function buildWorkspaceRootCauseHypotheses(args = {}, selectedFiles = [], heuristics = {}) {
  const haystack = [args.problem, args.errorText, args.failingTool]
    .map((entry) => asText(entry).toLowerCase())
    .join(" ");
  const hypotheses = [];
  if (/referenceerror[^.\n]*is not defined|is not defined|runtime injection|execute binding|tool script mapping|aufrufbare funktion|injizierte laufzeit|tool script entrypoint/.test(haystack)) {
    hypotheses.push({
      summary: "The saved tool scripts are not being bound into the injected browser runtime under the expected function name or entrypoint. That leaves read_active_text_target undefined in the execute script.",
      confidence: "high",
      evidence: [
        `${WORKSPACE_REPO_ROOT}/bg/craft-use-runner.js`,
        `${WORKSPACE_REPO_ROOT}/shared/capability-bundle.mjs`,
        `${WORKSPACE_REPO_ROOT}/shared/browserAutomationRuntime.js`,
        `${WORKSPACE_REPO_ROOT}/shared/browserAutomationTools.js`,
      ],
    });
  }
  if (/modeltransform|model transform|pre\/execute\/post null|empty scripts|entrypoint|tool script wiring|capability scripts|reviewed runtime path/.test(haystack)) {
    hypotheses.push({
      summary: "The reviewed capability path is wired, but the saved tool scripts are not being materialized into real pre/execute/post runtime sections. The path falls back to modelTransform instead of the JavaScript tool scripts.",
      confidence: "high",
      evidence: [
        `${WORKSPACE_REPO_ROOT}/bg/craft-use-runner.js`,
        `${WORKSPACE_REPO_ROOT}/shared/browser-capability-bundle.mjs`,
        `${WORKSPACE_REPO_ROOT}/shared/capability-bundle.mjs`,
      ],
    });
  }
  if (/module is not defined|module\.exports|exports is not defined|commonjs|script format/.test(haystack)) {
    hypotheses.push({
      summary: "The reviewed tool-script path mixes a CommonJS export pattern into direct browser script execution, where module does not exist.",
      confidence: "high",
      evidence: [
        `${WORKSPACE_REPO_ROOT}/bg/craft-use-runner.js`,
        `${WORKSPACE_REPO_ROOT}/shared/browserAutomationRuntime.js`,
        `${WORKSPACE_REPO_ROOT}/shared/browserAutomationTools.js`,
      ],
    });
  }
  if (/runtime\.callbuiltin is not a function|runtime\.respond is not a function|callbuiltin is not a function|respond is not a function/.test(haystack)) {
    hypotheses.push({
      summary: "The injected reviewed tool runtime is not passing a compatible runtime adapter with callBuiltin or respond into the saved tool script.",
      confidence: "high",
      evidence: [
        `${WORKSPACE_REPO_ROOT}/bg/craft-use-runner.js`,
        `${WORKSPACE_REPO_ROOT}/sidepanel.js`,
        `${WORKSPACE_REPO_ROOT}/shared/reviewed-tool-script-runtime.mjs`,
      ],
    });
  }
  if (/memory access out of bounds|out of bounds/.test(haystack) && /browser_inspect|vision|local_qwen|qwen/.test(haystack)) {
    hypotheses.push({
      summary: "The local Qwen vision path is processing browser screenshots with a pixel budget that is too large for the WebGPU plan.",
      confidence: "high",
      evidence: [
        `${WORKSPACE_REPO_ROOT}/bg/ml-inference.js`,
        `${WORKSPACE_REPO_ROOT}/shared/local_qwen_runtime.mjs`,
        `${WORKSPACE_REPO_ROOT}/shared/browserAutomationRuntime.js`,
      ],
    });
  }
  if (/local_qwen|qwen|onnx|webgpu|ortrun|offscreen/.test(haystack)) {
    hypotheses.push({
      summary: "The local Qwen offscreen path is failing during WebGPU/ONNX execution.",
      confidence: "high",
      evidence: [
        `${WORKSPACE_REPO_ROOT}/bg/ml-inference.js`,
        `${WORKSPACE_REPO_ROOT}/bg/llm.js`,
        `${WORKSPACE_REPO_ROOT}/shared/local_qwen_runtime.mjs`,
      ],
    });
    hypotheses.push({
      summary: "The local Qwen request is being routed to the offscreen runtime through the wrong model or runtime path.",
      confidence: "medium",
      evidence: [
        `${WORKSPACE_REPO_ROOT}/bg/llm.js`,
        `${WORKSPACE_REPO_ROOT}/shared/local_qwen_runtime.mjs`,
      ],
    });
  }
  if (/run_agentic_smoke|reviewed capability|active text|clipboard|focused editable/.test(haystack)) {
    hypotheses.push({
      summary: "The reviewed capability path or its active-text contract is saved, but it is not being executed reliably in the smoke run.",
      confidence: hypotheses.length ? "medium" : "high",
      evidence: [
        `${WORKSPACE_REPO_ROOT}/bg/crafting-agent-runner.js`,
        `${WORKSPACE_REPO_ROOT}/bg/craft-use-runner.js`,
        `${WORKSPACE_REPO_ROOT}/shared/browser-capability-bundle.mjs`,
      ],
    });
  }
  if (!hypotheses.length) {
    hypotheses.push({
      summary: "The blocker is probably in the extension's local runner or model path, not in the craft UI itself.",
      confidence: "medium",
      evidence: uniqueTextList(
        (Array.isArray(selectedFiles) ? selectedFiles : []).map((entry) => entry?.path),
        4,
      ),
    });
  }
  return {
    hypotheses: hypotheses.slice(0, 4),
    bestHypothesis: hypotheses[0] || {
      summary: "The extension's local runtime path should be narrowed down technically first.",
      confidence: "medium",
      evidence: heuristics?.suggestedExecutionPath || [],
    },
  };
}

function buildWorkspacePatchTargets(selectedFiles = [], bestHypothesis = null) {
  return (Array.isArray(selectedFiles) ? selectedFiles : [])
    .filter((entry) => asText(entry?.path) && asText(entry?.path) !== `${WORKSPACE_REPO_ROOT}/README.md`)
    .slice(0, 4)
    .map((entry, index) => ({
      path: asText(entry.path),
      symbols: uniqueTextList(
        (Array.isArray(entry.snippets) ? entry.snippets : []).map((snippet) => snippet?.symbol),
        6,
      ),
      why:
        asText(entry?.rationale) ||
        (
          index === 0 && bestHypothesis?.summary
            ? bestHypothesis.summary
            : `Relevant to ${asText(entry?.subsystem || "the affected runtime path")}.`
        ),
      confidence: index === 0 ? "high" : index < 3 ? "medium" : "low",
    }));
}

function buildWorkspaceCodeDivePrompt(args = {}, selectedFiles = [], executionPath = [], patchTargets = [], validationTargets = []) {
  const lines = [
    "Patch this Chrome extension workspace to resolve the blocked crafting-agent runtime path.",
    `Primary problem: ${asText(args.problem) || "Blocked crafting-agent run with a local workspace defect."}`,
    asText(args.failingTool) ? `Failing tool: ${asText(args.failingTool)}` : "",
    asText(args.errorText) ? `Observed error: ${asText(args.errorText)}` : "",
    executionPath.length ? `Start with this execution path: ${executionPath.join(" -> ")}` : "",
    patchTargets.length
      ? `Inspect these patch targets first: ${patchTargets.map((entry) => entry.path).join(", ")}`
      : "",
    validationTargets.length
      ? `Validate the fix with: ${validationTargets.join(", ")}`
      : "",
    selectedFiles.length
      ? `Relevant source files already narrowed down: ${selectedFiles.map((entry) => entry.path).join(", ")}`
      : "",
    "Keep the fix inside this workspace. Do not paper over the broken reviewed path with a workaround.",
    "Preserve the reviewed capability path, keep user-facing German copy concise, and rerun the same smoke or eval after patching.",
  ];
  return lines.filter(Boolean).join("\n");
}

async function executeWorkspaceCodeDiveTool(run, args = {}) {
  const terms = buildWorkspaceCodeDiveTerms(args);
  const focusReferences = buildWorkspaceCodeDiveFocusReferences(args, run);
  const heuristics = buildWorkspaceCodeDiveHeuristics(args, focusReferences);
  const requestedMaxFiles = Math.max(1, Math.min(
    WORKSPACE_CODE_DIVE_MAX_FILES,
    Number(args.maxFiles || WORKSPACE_CODE_DIVE_MAX_FILES) || WORKSPACE_CODE_DIVE_MAX_FILES,
  ));
  const records = await Promise.all(
    WORKSPACE_CODE_DIVE_FILE_MANIFEST.map(async (entry) => {
      try {
        const text = await readPackagedWorkspaceSourceText(entry.runtimePath);
        const fileFocusReferences = focusReferences.filter((ref) => ref.runtimePath === entry.runtimePath);
        const contentMatches = collectWorkspaceCodeDiveMatches(text, terms, fileFocusReferences);
        const scoreMeta = scoreWorkspaceCodeDiveFile(entry, terms, heuristics);
        const contentScore = contentMatches.reduce((total, match) => total + match.matchedTerms.length * 4, 0);
        return {
          ...entry,
          score: scoreMeta.score + contentScore + fileFocusReferences.length * 18,
          matchedTerms: uniqueTextList(
            [
              ...scoreMeta.matchedTags,
              ...contentMatches.flatMap((match) => match.matchedTerms),
            ],
            12,
          ),
          snippets: contentMatches,
          focusReferences: fileFocusReferences,
          lineCount: text.split("\n").length,
        };
      } catch (error) {
        return {
          ...entry,
          score: (heuristics?.preferredRuntimePaths || []).includes(entry.runtimePath) ? 4 : 0,
          matchedTerms: [],
          snippets: [],
          focusReferences: [],
          lineCount: 0,
          readError: error instanceof Error ? error.message : String(error || "Unable to read source."),
        };
      }
    }),
  );

  const selectedFiles = records
    .filter((entry) => entry.score > 0 || entry.snippets.length > 0 || (heuristics?.preferredRuntimePaths || []).includes(entry.runtimePath))
    .sort((left, right) => Number(right.score || 0) - Number(left.score || 0))
    .slice(0, requestedMaxFiles)
    .map((entry) => ({
      path: asText(entry.repoPath || buildWorkspaceRepoPath(entry.runtimePath)),
      runtimePath: asText(entry.runtimePath),
      subsystem: asText(entry.subsystem),
      tags: Array.isArray(entry.tags) ? entry.tags : [],
      rationale:
        entry.focusReferences.length
          ? `Stack references: ${entry.focusReferences.map((ref) => formatWorkspaceFocusReference(ref)).join(", ")}`
          : entry.matchedTerms.length
            ? `Matched terms: ${entry.matchedTerms.join(", ")}`
          : (heuristics?.preferredRuntimePaths || []).includes(entry.runtimePath)
            ? "Selected from the most likely failing runtime path."
            : entry.readError
              ? `Source read failed: ${entry.readError}`
              : "Selected as part of the relevant workspace area.",
      snippets: (Array.isArray(entry.snippets) ? entry.snippets : []).map((snippet) => ({
        line: Number(snippet?.line || 0),
        symbol: asText(snippet?.symbol),
        matchedTerms: Array.isArray(snippet?.matchedTerms) ? snippet.matchedTerms : [],
        excerpt: trimText(snippet?.excerpt, 1_600),
      })),
    }));
  const executionPath = heuristics.suggestedExecutionPath.length
    ? heuristics.suggestedExecutionPath
    : uniqueTextList(selectedFiles.map((entry) => entry.path), 6);
  const hypothesisPack = buildWorkspaceRootCauseHypotheses(args, selectedFiles, heuristics);
  const validationTargets = inferWorkspaceValidationTargets(selectedFiles, heuristics);
  const patchTargets = buildWorkspacePatchTargets(selectedFiles, hypothesisPack.bestHypothesis);
  const codingPrompt = buildWorkspaceCodeDivePrompt(args, selectedFiles, executionPath, patchTargets, validationTargets);
  const suspectedSubsystems = uniqueTextList(
    selectedFiles.map((entry) => entry.subsystem).filter(Boolean),
    8,
  );
  const codingHandoff = {
    format: "patch_handoff_v1",
    repoKind: "chrome_extension_mv3_sidepanel",
    repoPathHint: WORKSPACE_REPO_ROOT,
    objective: asText(run?.brief || args.problem),
    problem: asText(args.problem),
    failingTool: asText(args.failingTool),
    errorText: asText(args.errorText),
    executionPath,
    patchTargets,
    validationTargets,
    prompt: codingPrompt,
    logExcerpt: (Array.isArray(run?.logs) ? run.logs : []).slice(-8),
  };

  return {
    ok: Boolean(selectedFiles.length),
    summary: selectedFiles.length
      ? `Source-code diagnosis narrowed down: ${patchTargets.slice(0, 3).map((entry) => entry.path.split("/").slice(-2).join("/")).join(", ")}`
      : "Source-code diagnosis could not narrow down relevant files.",
    data: {
      query: {
        problem: asText(args.problem),
        failingTool: asText(args.failingTool),
        errorText: asText(args.errorText),
        terms,
        focusFiles: uniqueTextList(focusReferences.map((ref) => ref.runtimePath), 8),
      },
      suspectedSubsystems,
      executionPath,
      files: selectedFiles,
      rootCauseHypotheses: hypothesisPack.hypotheses,
      bestHypothesis: hypothesisPack.bestHypothesis,
      patchTargets,
      validationTargets,
      codingHandoff,
    },
    report: {
      currentState: selectedFiles.length
        ? "The blocker looks like a concrete defect in the local extension code path. The key files have been narrowed down for a patch."
        : "The blocker cannot yet be narrowed down cleanly to specific extension files.",
      nextAction: selectedFiles.length
        ? "Hand the patch hints to a coding agent and then rerun the same smoke or eval path."
        : "Add more error context or repeat the run with stronger technical signals before you start a coding agent.",
      matchingSignals: [
        ...suspectedSubsystems,
        ...patchTargets.slice(0, 2).map((entry) => entry.path),
      ].filter(Boolean).slice(0, 4),
    },
    provenance: selectedFiles.length
      ? [
        {
            title: "Source-code diagnosis narrowed down",
            detail: trimText(
              patchTargets
                .slice(0, 3)
                .map((entry) => entry.path)
                .join(" | "),
              220,
            ),
            kind: "match",
          },
        ]
      : [],
  };
}

function buildBrowserToolBootstrapResult(run, args = {}, toolLabel = "Browser tool") {
  const bootstrapHints = buildCraftBootstrapHints(run, args);
  if (!bootstrapHints?.genericSelectionBootstrap) return null;
  return {
    ok: true,
    summary:
      `${toolLabel} skipped because no active browser tab or target URL was found. Continue with generic active-text bootstrap rows instead of blocking on live DOM evidence.`,
    data: {
      activeTab: null,
      fallback: "generic_selection_bootstrap",
      bootstrapHints,
    },
    report: {
      currentState:
        "No active HTTP(S) tab or target URL is available. For generic selection, focus, or clipboard flows, the agent can still draft seed rows.",
      nextAction:
        "Use draft_training_changes now and derive portable multi-turn training rows from the generic active-text tool contract.",
      matchingSignals: [
        "Portable contract: read_active_text_target -> replace_active_text_target",
        "Selection/focused-field/clipboard flow without extra free-text input",
      ],
    },
    provenance: [
      {
        title: `${toolLabel} skipped`,
        detail:
          "No active HTTP(S) tab or target URL was available, so the run should continue from the reviewed generic active-text contract instead of interrupting the user.",
        kind: "constraint",
      },
    ],
  };
}

function buildMissingBrowserTargetResult(toolLabel = "Browser tool") {
  return {
    ok: false,
    summary: `${toolLabel} could not start because no active browser tab or target URL was found.`,
    error: "No active HTTP(S) tab or target URL is available.",
    errorDetail: {
      toolLabel: asText(toolLabel),
      reason: "missing_browser_target",
    },
  };
}

async function getTabContextById(tabId) {
  const tab = await pTabsGet(Number(tabId));
  return normalizeBrowserTabContext(tab);
}

async function queryActiveTabContext() {
  const tabs = await pTabsQuery({ active: true, lastFocusedWindow: true }).catch(() => []);
  return normalizeBrowserTabContext(Array.isArray(tabs) ? tabs[0] : null);
}

async function refreshRunActiveTabContext(run, preferredTabId = 0) {
  const runtime = ensureRunBrowserRuntime(run);
  const candidateIds = [preferredTabId, runtime.currentTabId]
    .map((value) => Number(value || 0))
    .filter((value, index, list) => Number.isFinite(value) && value > 0 && list.indexOf(value) === index);
  for (const tabId of candidateIds) {
    const context = await getTabContextById(tabId).catch(() => null);
    if (context) {
      return syncRunBrowserContext(run, context);
    }
  }
  const active = await queryActiveTabContext();
  return syncRunBrowserContext(run, active, { clearOnNull: true });
}

async function resolveBrowserToolSession(run, args = {}, {
  toolLabel = "Browser tool",
  allowBootstrapFallback = true,
} = {}) {
  const runtime = ensureRunBrowserRuntime(run);
  const explicitTabId = Number(args?.tabId || 0);
  if (Number.isFinite(explicitTabId) && explicitTabId > 0) {
    runtime.currentTabId = explicitTabId;
  }
  const tabContext = await refreshRunActiveTabContext(run, explicitTabId || runtime.currentTabId);
  const requestedUrl = asText(args?.url);
  if (!requestedUrl && !tabContext?.url) {
    if (allowBootstrapFallback) {
      const bootstrapResult = buildBrowserToolBootstrapResult(run, args, toolLabel);
      if (bootstrapResult) {
        return { kind: "skip", result: bootstrapResult };
      }
    }
    return { kind: "error", result: buildMissingBrowserTargetResult(toolLabel) };
  }
  return {
    kind: "session",
    runtime,
    tabContext,
    requestedUrl,
    baseUrl: deriveBrowserBaseUrl(args, tabContext),
    allowedHosts: deriveBrowserAllowedHosts(run, args, tabContext),
  };
}

async function syncRunBrowserContextFromResult(run, result) {
  const data = result?.data && typeof result.data === "object" ? result.data : {};
  const preferredTabId = Number(data.tabId || data.tab_id || 0);
  if (Number.isFinite(preferredTabId) && preferredTabId > 0) {
    const refreshed = await refreshRunActiveTabContext(run, preferredTabId);
    if (refreshed) return refreshed;
  }
  return syncRunBrowserContext(run, data, { clearOnNull: false });
}

function createEmptyUsageTotals() {
  return {
    requests: 0,
    inputTokens: 0,
    outputTokens: 0,
    totalTokens: 0,
  };
}

function normalizeUsageTotals(rawUsage) {
  const usage = rawUsage && typeof rawUsage === "object" ? rawUsage : {};
  return {
    requests: Math.max(0, Number(usage.requests || 0) || 0),
    inputTokens: Math.max(0, Number(usage.inputTokens || usage.input_tokens || 0) || 0),
    outputTokens: Math.max(0, Number(usage.outputTokens || usage.output_tokens || 0) || 0),
    totalTokens: Math.max(
      0,
      Number(
        usage.totalTokens ||
          usage.total_tokens ||
          (Number(usage.inputTokens || usage.input_tokens || 0) || 0) +
            (Number(usage.outputTokens || usage.output_tokens || 0) || 0),
      ) || 0,
    ),
  };
}

function safeReadRunStateJson(value) {
  try {
    if (!value) return null;
    if (typeof value === "string") {
      const parsed = JSON.parse(value);
      return parsed && typeof parsed === "object" ? parsed : null;
    }
    if (typeof value?.toJSON === "function") {
      const parsed = value.toJSON();
      return parsed && typeof parsed === "object" ? parsed : null;
    }
  } catch {}
  return null;
}

function safeSerializeRunState(state) {
  try {
    if (!state || typeof state.toString !== "function") return "";
    return String(state.toString() || "");
  } catch {
    return "";
  }
}

function readRunStateCurrentTurn(state) {
  const stateJson = safeReadRunStateJson(state);
  return Math.max(0, Number(stateJson?.currentTurn || 0) || 0);
}

function readRunStateUsageTotals(state) {
  const stateJson = safeReadRunStateJson(state);
  return normalizeUsageTotals(stateJson?.context?.usage);
}

function flattenPromptContentParts(parts) {
  const items = Array.isArray(parts) ? parts : [];
  return items
    .map((part) => {
      if (!part || typeof part !== "object") return "";
      if (part.type === "text") return String(part.text || "");
      if (part.type === "reasoning") return String(part.text || "");
      if (part.type === "tool-call") {
        return `[tool_call ${asText(part.toolName)} ${JSON.stringify(part.input ?? {}, null, 2)}]`;
      }
      if (part.type === "tool-result") {
        const output = part.output && typeof part.output === "object" ? part.output : {};
        const textValue =
          output.type === "text"
            ? asText(output.value)
            : output.type === "content"
              ? JSON.stringify(output.value ?? null)
              : JSON.stringify(output);
        return `[tool_result ${asText(part.toolName)} ${textValue}]`;
      }
      if (part.type === "file") {
        return `[file ${asText(part.mediaType)} ${trimText(part.data, 200)}]`;
      }
      return JSON.stringify(part);
    })
    .filter(Boolean)
    .join("\n");
}

function formatLocalPromptMessages(promptMessages) {
  return (Array.isArray(promptMessages) ? promptMessages : [])
    .map((message, index) => {
      const role = asText(message?.role || "user") || "user";
      const content =
        typeof message?.content === "string"
          ? message.content
          : flattenPromptContentParts(message?.content);
      return {
        step: index + 1,
        role,
        content: trimText(content, 4_000),
      };
    })
    .filter((entry) => entry.content);
}

function buildLocalToolCatalog(tools = []) {
  return (Array.isArray(tools) ? tools : []).map((entry) => ({
    name: asText(entry?.name),
    description: asText(entry?.description),
    inputSchema: entry?.inputSchema && typeof entry.inputSchema === "object" ? entry.inputSchema : {},
  }));
}

function buildLocalResponseFormatDescriptor(responseFormat) {
  if (!responseFormat || typeof responseFormat !== "object") {
    return { type: "text" };
  }
  const type = asText(responseFormat.type).toLowerCase();
  if (type === "json" && responseFormat.schema && typeof responseFormat.schema === "object") {
    return {
      type: "json",
      name: asText(responseFormat.name) || "output",
      schema: cloneJson(responseFormat.schema, {}),
    };
  }
  return { type: "text" };
}

function parseLocalModelDecision(rawText, { toolNames = [], expectsStructuredFinal = false } = {}) {
  const text = asText(rawText);
  const parsed = parseJsonLoose(text);
  if (!parsed || typeof parsed !== "object") {
    return expectsStructuredFinal ? null : { kind: "final", text };
  }
  const kind = asText(parsed.kind || parsed.type || parsed.mode).toLowerCase();
  const toolName = asText(parsed.toolName || parsed.tool_name || parsed.name);
  const args = parsed.arguments && typeof parsed.arguments === "object" ? parsed.arguments : {};
  if ((kind === "tool" || kind === "tool_call") && toolNames.includes(toolName)) {
    return { kind: "tool", toolName, arguments: args };
  }
  if (toolName && toolNames.includes(toolName)) {
    return { kind: "tool", toolName, arguments: args };
  }
  if (expectsStructuredFinal) {
    const finalOutput =
      extractStructuredOutputObject(parsed) ||
      (kind && !["final", "done", "text"].includes(kind)
        ? null
        : extractStructuredOutputObject(text));
    if (finalOutput && typeof finalOutput === "object" && !Array.isArray(finalOutput)) {
      return {
        kind: "final",
        text: JSON.stringify(finalOutput),
      };
    }
    const nestedObject = parseJsonLoose(
      typeof parsed.text === "string"
        ? parsed.text
        : typeof parsed.final === "string"
          ? parsed.final
          : text,
    );
    if (nestedObject && typeof nestedObject === "object" && !Array.isArray(nestedObject)) {
      return {
        kind: "final",
        text: JSON.stringify(nestedObject),
      };
    }
    return null;
  }
  if (kind === "final" || kind === "done" || kind === "text") {
    const finalText =
      typeof parsed.text === "string"
        ? parsed.text
        : typeof parsed.final === "string"
          ? parsed.final
          : parsed.text && typeof parsed.text === "object"
            ? JSON.stringify(parsed.text, null, 2)
            : parsed.final && typeof parsed.final === "object"
              ? JSON.stringify(parsed.final, null, 2)
              : text;
    return { kind: "final", text: finalText };
  }
  return { kind: "final", text };
}

async function repairLocalModelDecision({
  run,
  candidateText,
  transcript,
  tools,
  responseFormat,
}) {
  const expectsStructuredFinal = responseFormat?.type === "json";
  const response = await llmChat({
    slotId: run.resolved?.slotId || "agent",
    modelRef: run.modelRef,
    parameters: {
      ...(run.parameters && typeof run.parameters === "object" ? run.parameters : {}),
      temperature: 0,
      maxTokens: 800,
    },
    reasoningEffort: run.reasoningEffort || "minimal",
    messages: [
      {
        role: "system",
        content: [
          "Repair invalid output for a local Agents SDK adapter.",
          "Return exactly one JSON object and nothing else.",
          expectsStructuredFinal
            ? 'Allowed shapes: {"kind":"tool","toolName":"<tool>","arguments":{}} or {"kind":"final","output":{...}}'
            : 'Allowed shapes: {"kind":"tool","toolName":"<tool>","arguments":{}} or {"kind":"final","text":"<assistant text>"}',
          "Never add markdown or prose.",
        ].join("\n"),
      },
      {
        role: "user",
        content: JSON.stringify(
          {
            candidateText: String(candidateText || ""),
            transcript,
            tools,
            responseFormat,
          },
          null,
          2,
        ),
      },
    ],
  });
  return parseLocalModelDecision(response?.text || "", {
    toolNames: tools.map((entry) => entry.name),
    expectsStructuredFinal,
  });
}

function createLocalQwenAiSdkLanguageModel(run) {
  return {
    provider: "local_qwen",
    modelId: asText(run.resolved?.modelName) || "local_qwen",
    async doGenerate(request) {
      const transcript = formatLocalPromptMessages(request?.prompt);
      const tools = buildLocalToolCatalog(request?.tools);
      const responseFormat = buildLocalResponseFormatDescriptor(request?.responseFormat);
      const expectsStructuredFinal = responseFormat.type === "json";
      const response = await llmChat({
        slotId: run.resolved?.slotId || "agent",
        modelRef: run.modelRef,
        parameters: {
          ...(run.parameters && typeof run.parameters === "object" ? run.parameters : {}),
          temperature: 0.1,
          maxTokens: Math.max(
            320,
            Math.min(
              1_200,
              Number(
                run.parameters?.maxTokens ||
                  run.parameters?.max_new_tokens ||
                  900,
              ) || 900,
            ),
          ),
        },
        reasoningEffort: run.reasoningEffort || "medium",
        messages: [
          {
            role: "system",
            content: [
              "You are a local tool-calling adapter for an Agents SDK runner.",
              "Read the transcript and available tools carefully.",
              "Return exactly one JSON object and nothing else.",
              'If a tool is needed, respond with {"kind":"tool","toolName":"<tool>","arguments":{}}.',
              expectsStructuredFinal
                ? 'If no tool is needed, respond with {"kind":"final","output":{...}} where output matches the provided JSON schema.'
                : 'If no tool is needed, respond with {"kind":"final","text":"<assistant final text>"}.',
              "Do not call unknown tools.",
              "Do not add markdown fences or commentary outside JSON.",
              expectsStructuredFinal
                ? "When you choose final, the output object must satisfy the provided schema exactly."
                : "When you choose final, the text must be the assistant's final reply exactly as intended.",
            ].join("\n"),
          },
          {
            role: "user",
            content: JSON.stringify(
              {
                transcript,
                availableTools: tools,
                responseFormat,
              },
              null,
              2,
            ),
          },
        ],
      });
      const parsed =
        parseLocalModelDecision(response?.text || "", {
          toolNames: tools.map((entry) => entry.name),
          expectsStructuredFinal,
        }) ||
        (await repairLocalModelDecision({
          run,
          candidateText: response?.text || "",
          transcript,
          tools,
          responseFormat,
        })) || {
          kind: "final",
          text: expectsStructuredFinal ? "{}" : String(response?.text || ""),
        };
      const responseId = `local-qwen-${Date.now()}-${Math.random().toString(16).slice(2, 8)}`;
      const usage = normalizeUsageTotals(response?.usage);
      if (parsed.kind === "tool") {
        return {
          content: [
            {
              type: "tool-call",
              toolCallId: `call_${Math.random().toString(16).slice(2, 10)}`,
              toolName: parsed.toolName,
              input: parsed.arguments || {},
            },
          ],
          usage,
          response: {
            id: responseId,
          },
          providerMetadata: {
            localRuntime: response?.runtime || null,
          },
        };
      }
      return {
        content: [
          {
            type: "text",
            text: expectsStructuredFinal
              ? repairStructuredOutputText(
                  String(parsed.text || response?.text || ""),
                  responseFormat.schema,
                )
              : String(parsed.text || response?.text || ""),
          },
        ],
        usage,
        response: {
          id: responseId,
        },
        providerMetadata: {
          localRuntime: response?.runtime || null,
        },
      };
    },
    async doStream(request) {
      const generated = await this.doGenerate(request);
      async function* buildStream() {
        yield {
          type: "response-metadata",
          id: generated?.response?.id || `local-qwen-${Date.now()}`,
        };
        for (const entry of Array.isArray(generated?.content) ? generated.content : []) {
          if (entry?.type === "tool-call") {
            yield {
              type: "tool-call",
              toolCallId: entry.toolCallId,
              toolName: entry.toolName,
              input: entry.input,
            };
          } else if (entry?.type === "text" && entry.text) {
            yield {
              type: "text-delta",
              delta: String(entry.text || ""),
            };
          }
        }
        yield {
          type: "finish",
          usage: generated?.usage || createEmptyUsageTotals(),
        };
      }
      return { stream: buildStream() };
    },
  };
}

function detectCraftBootstrapIntent(text = "") {
  const source = String(text || "").trim().toLowerCase();
  return {
    translation: /(uebersetz|übersetz|translate|translation)/i.test(source),
    correction: /(rechtschreib|grammatik|stil|korrigier|korrektur)/i.test(source),
    rewrite: /(rewrite|rephrase|paraphras|proofread|polish|shorter|clearer|summari[sz]e|zusammenfass|umschreib|umformulier)/i.test(source),
    browserSelection: /(markiert|markierten|ausgew[aä]hlt|selection|selected text|eingabefeld|zwischenablage|clipboard)/i.test(source),
    currentTab: /(current tab|aktuell(?:e|en)? (?:tab|seite)|browserseite|geoeffnete seite|geöffnete seite)/i.test(source),
    inPlaceReplace: /(in[\s-]?place|direkt ersetzen|an derselben stelle|write back|zurueckschreib|replace(?: the)? text back)/i.test(source),
    noExtraInput: /(ohne (?:weitere|zus[aä]tzliche) (?:texteingabe|eingabe)|nur ausf(?:u|ü)hren|direkt im browser)/i.test(source),
  };
}

function normalizeActiveTextValidationTargetMode(value = "") {
  const normalized = asText(value)
    .toLowerCase()
    .replace(/[\s-]+/g, "_");
  if (normalized === "focused" || normalized === "focused_field" || normalized === "field") {
    return "focused_editable";
  }
  if (["selection", "focused_editable", "clipboard"].includes(normalized)) {
    return normalized;
  }
  return "selection";
}

function inferActiveTextValidationTargetMode(text = "") {
  const source = asText(text).toLowerCase();
  if (!source) return "selection";
  if (/(selection|selected text|markiert(?:e|en|er|es)?|auswahl)/i.test(source)) {
    return "selection";
  }
  if (/(focused(?:\s+editable|\s+field)?|eingabefeld|textfeld|fokussiert(?:e|en|er|es)?)/i.test(source)) {
    return "focused_editable";
  }
  if (/(clipboard|copied text|zwischenablage|kopiert(?:e|en|er|es)?)/i.test(source)) {
    return "clipboard";
  }
  return "selection";
}

function collectCraftBootstrapText(run, args = {}) {
  return [
    asText(args?.objective),
    asText(args?.goal),
    asText(run?.brief),
    asText(run?.craft?.summary),
    asText(run?.craft?.name),
  ].filter(Boolean).join("\n");
}

function resolveRequestedActiveTextValidationTargetMode(run, args = {}) {
  const explicitMode = normalizeActiveTextValidationTargetMode(args?.activeTextTargetMode);
  if (explicitMode && explicitMode !== "selection") {
    return explicitMode;
  }
  return inferActiveTextValidationTargetMode(collectCraftBootstrapText(run, args));
}

function buildSelectionBootstrapTemplate(mode = "rewrite") {
  const safeMode = mode === "translation" || mode === "correction" ? mode : "rewrite";
  const systemText =
    safeMode === "translation"
      ? "You are a browser-safe translation assistant."
      : "You are a browser-safe writing assistant.";
  const userText =
    safeMode === "translation"
      ? "Translate the active text directly and replace the selection, focused field, or clipboard content in place."
      : safeMode === "correction"
        ? "Correct the active German text directly and replace the selection, focused field, or clipboard content in place."
        : "Revise the active text directly and replace the selection, focused field, or clipboard content in place.";
  const sourceText =
    safeMode === "translation"
      ? "This are two short sentence with a mistake."
      : safeMode === "correction"
        ? "This is an sentence with a mistake."
        : "This text can be phrased more clearly.";
  const targetText =
    safeMode === "translation"
      ? "These are two short sentences with a mistake."
      : safeMode === "correction"
        ? "This is a sentence with a mistake."
        : "This text is phrased more clearly.";
  return {
    messages: [
      { role: "system", content: systemText },
      { role: "user", content: userText },
      {
        role: "assistant",
        content: "",
        tool_calls: [
          {
            id: "call_1",
            type: "function",
            function: {
              name: "read_active_text_target",
              arguments: "{}",
            },
          },
        ],
      },
      {
        role: "tool",
        tool_call_id: "call_1",
        name: "read_active_text_target",
        content: stringifyTrainingJson({
          ok: true,
          data: {
            targetType: "selection",
            text: sourceText,
          },
        }),
      },
      {
        role: "assistant",
        content: "",
        tool_calls: [
          {
            id: "call_2",
            type: "function",
            function: {
              name: "replace_active_text_target",
              arguments: stringifyTrainingJson({ text: targetText }),
            },
          },
        ],
      },
    ],
    tools: [
      {
        type: "function",
        function: {
          name: "read_active_text_target",
          description: "Read the active text target in priority order: browser selection, focused editable field, then clipboard.",
          parameters: {
            type: "object",
            additionalProperties: false,
          },
        },
      },
      {
        type: "function",
        function: {
          name: "replace_active_text_target",
          description: "Replace the active text target in priority order: browser selection, focused editable field, then clipboard.",
          parameters: {
            type: "object",
            properties: {
              text: { type: "string" },
            },
            required: ["text"],
            additionalProperties: false,
          },
        },
      },
    ],
    targetTurnIndex: 4,
  };
}

function buildCraftBootstrapHints(run, args = {}) {
  const objectiveText = collectCraftBootstrapText(run, args);
  const intent = detectCraftBootstrapIntent(objectiveText);
  const inputMode = asText(run?.craft?.inputMode).toLowerCase();
  const activeTextTransform = intent.translation || intent.correction || intent.rewrite || intent.inPlaceReplace;
  const selectionLike =
    inputMode === "selection" ||
    (activeTextTransform && (intent.browserSelection || intent.noExtraInput || intent.inPlaceReplace));
  if (!selectionLike || intent.currentTab) return null;
  const rewriteMode = intent.translation ? "translation" : intent.correction ? "correction" : "rewrite";
  return {
    genericSelectionBootstrap: true,
    browserEvidenceOptional: true,
    reviewedPortablePattern: "read_active_text_target -> replace_active_text_target",
    runtimeLimitations: [
      "Current reviewed contract resolves the active text target in priority order: browser selection, focused editable field, then clipboard.",
      "Do not require a live HTTP(S) tab just to seed generic rows for this craft.",
      "Keep the priority order deterministic in the reviewed tools instead of asking the runtime model to guess the target.",
    ],
    starterRow: buildSelectionBootstrapTemplate(rewriteMode),
  };
}

function requiresPreparedActiveTextValidationFixture(run, args = {}) {
  return Boolean(buildCraftBootstrapHints(run, args)?.genericSelectionBootstrap);
}

function assessPreparedActiveTextValidationFixture(run, args = {}) {
  const required = requiresPreparedActiveTextValidationFixture(run, args);
  const preparedFixture =
    run?.browserValidationFixture && typeof run.browserValidationFixture === "object"
      ? run.browserValidationFixture
      : null;
  const traces = Array.isArray(run?.toolTrace) ? run.toolTrace.slice().reverse() : [];
  const recentSuccessfulBrowserSteps = [];
  let hasScenarioSetup = false;
  let hasVisibleVerification = false;

  for (const trace of traces) {
    const action = asText(trace?.action);
    if (!action || trace?.ok === false || !BROWSER_TOOL_ACTIONS.has(action)) continue;
    if (recentSuccessfulBrowserSteps.length < 6) {
      recentSuccessfulBrowserSteps.push(action);
    }
    if (action === "browser_action" || action === "playwright_ctx") {
      hasScenarioSetup = true;
    }
    if (action === "browser_inspect") {
      hasVisibleVerification = true;
    }
    if (hasScenarioSetup && hasVisibleVerification && recentSuccessfulBrowserSteps.length >= 3) {
      break;
    }
  }

  const activeTabUrl = asText(run?.activeTabContext?.url || preparedFixture?.activeTabUrl);
  const hasPreparedFixture =
    preparedFixture?.prepared === true &&
    Boolean(activeTabUrl) &&
    Number(preparedFixture?.tabId || 0) > 0;
  return {
    required,
    ready: !required || hasPreparedFixture || (Boolean(activeTabUrl) && hasScenarioSetup && hasVisibleVerification),
    activeTabUrl,
    hasScenarioSetup,
    hasVisibleVerification,
    hasPreparedFixture,
    recentSuccessfulBrowserSteps: uniqueTextList(recentSuccessfulBrowserSteps, 6),
  };
}

async function ensurePreparedActiveTextValidationFixture(run, args = {}) {
  const bootstrapHints = buildCraftBootstrapHints(run, args);
  if (!bootstrapHints?.genericSelectionBootstrap) return null;
  const requestedTargetMode = resolveRequestedActiveTextValidationTargetMode(run, args);
  const currentAssessment = assessPreparedActiveTextValidationFixture(run, args);
  const existingFixture =
    run?.browserValidationFixture && typeof run.browserValidationFixture === "object"
      ? run.browserValidationFixture
      : null;
  const existingFixtureMode = normalizeActiveTextValidationTargetMode(existingFixture?.activeTextTargetMode);
  const canReusePreparedFixture =
    currentAssessment.hasPreparedFixture &&
    existingFixture &&
    existingFixtureMode === requestedTargetMode &&
    (requestedTargetMode !== "clipboard" || existingFixture?.clipboardReady === true);
  if (canReusePreparedFixture) {
    return run?.browserValidationFixture && typeof run.browserValidationFixture === "object"
      ? cloneJson(run.browserValidationFixture, {})
      : {
          prepared: true,
          activeTabUrl: currentAssessment.activeTabUrl,
          tabId: Number(run?.activeTabContext?.tabId || 0) || 0,
          source: "existing_browser_fixture",
        };
  }
  if (currentAssessment.ready && !currentAssessment.hasPreparedFixture && requestedTargetMode === "selection") {
    return {
      prepared: true,
      activeTabUrl: currentAssessment.activeTabUrl,
      tabId: Number(run?.activeTabContext?.tabId || 0) || 0,
      source: "existing_browser_fixture",
      activeTextTargetMode: requestedTargetMode,
      clipboardReady: false,
    };
  }

  let activeContext = await queryActiveTabContext().catch(() => null);
  if (!activeContext) {
    const createdTab = await pTabsCreate({
      url: DEFAULT_BROWSER_TOOL_BASE_URL,
      active: true,
    }).catch(() => null);
    activeContext = await waitForBrowserTabContext(run, {
      preferredTabId: Number(createdTab?.id || 0),
      allowActiveFallback: true,
      timeoutMs: 8_000,
      fallbackUrl: DEFAULT_BROWSER_TOOL_BASE_URL,
    }).catch(() => null);
  } else {
    syncRunBrowserContext(run, activeContext, { clearOnNull: false });
  }

  if (!activeContext || !Number(activeContext?.tabId || 0)) {
    return null;
  }

  const targetTabId = Number(activeContext.tabId || 0);
  await pTabsUpdate(targetTabId, { active: true }).catch(() => null);
  const scriptResults = await pExecuteScript(
    targetTabId,
    async ({ fixtureId, fixtureText, targetMode }) => {
      const root = globalThis.document?.body || globalThis.document?.documentElement;
      if (!root) {
        return {
          ok: false,
          error: "Document body was not ready for the active-text validation fixture.",
        };
      }
      const normalizeMode = (value) => {
        const normalized = String(value == null ? "" : value)
          .trim()
          .toLowerCase()
          .replace(/[\s-]+/g, "_");
        if (normalized === "focused" || normalized === "focused_field" || normalized === "field") {
          return "focused_editable";
        }
        return ["selection", "focused_editable", "clipboard"].includes(normalized)
          ? normalized
          : "selection";
      };
      const existing = globalThis.document.getElementById(fixtureId);
      const textarea =
        existing instanceof globalThis.HTMLTextAreaElement
          ? existing
          : globalThis.document.createElement("textarea");
      textarea.id = fixtureId;
      textarea.value = String(fixtureText || "");
      textarea.setAttribute("data-sinepanel-active-text-fixture", "true");
      textarea.setAttribute("spellcheck", "false");
      textarea.style.cssText = [
        "position: fixed",
        "top: 24px",
        "left: 24px",
        "width: min(720px, calc(100vw - 48px))",
        "height: 180px",
        "padding: 16px",
        "font: 18px/1.45 Georgia, serif",
        "color: #111",
        "background: #fffbe8",
        "border: 2px solid #111",
        "border-radius: 12px",
        "box-shadow: 0 18px 48px rgba(0, 0, 0, 0.18)",
        "z-index: 2147483646",
        "resize: none",
      ].join("; ");
      if (textarea.parentElement !== root) {
        root.append(textarea);
      }
      let passiveFocusTarget = globalThis.document.getElementById(`${fixtureId}-passive-focus`);
      if (!(passiveFocusTarget instanceof globalThis.HTMLButtonElement)) {
        passiveFocusTarget = globalThis.document.createElement("button");
        passiveFocusTarget.id = `${fixtureId}-passive-focus`;
        passiveFocusTarget.type = "button";
        passiveFocusTarget.textContent = "fixture-focus";
        passiveFocusTarget.setAttribute("aria-hidden", "true");
        passiveFocusTarget.tabIndex = -1;
        passiveFocusTarget.style.cssText = [
          "position: fixed",
          "top: 0",
          "left: 0",
          "width: 1px",
          "height: 1px",
          "opacity: 0",
          "pointer-events: none",
          "z-index: -1",
        ].join("; ");
        root.append(passiveFocusTarget);
      }
      const requestedMode = normalizeMode(targetMode);
      const clearWindowSelection = () => {
        try {
          const selection = globalThis.getSelection?.();
          selection?.removeAllRanges?.();
        } catch {}
      };
      const collapseTextareaSelection = () => {
        try {
          textarea.setSelectionRange?.(textarea.value.length, textarea.value.length, "none");
        } catch {}
      };
      const focusTextarea = () => {
        try {
          textarea.focus({ preventScroll: true });
        } catch {
          textarea.focus?.();
        }
      };
      const focusPassiveTarget = () => {
        try {
          passiveFocusTarget.focus({ preventScroll: true });
        } catch {
          passiveFocusTarget.focus?.();
        }
      };
      const setSelectionMode = () => {
        focusTextarea();
        if (typeof textarea.setSelectionRange === "function") {
          textarea.setSelectionRange(0, textarea.value.length, "forward");
        } else {
          textarea.select?.();
        }
      };
      const setFocusedEditableMode = () => {
        focusTextarea();
        collapseTextareaSelection();
        clearWindowSelection();
      };
      const setClipboardMode = async () => {
        collapseTextareaSelection();
        clearWindowSelection();
        textarea.blur?.();
        focusPassiveTarget();
        if (!globalThis.navigator?.clipboard?.writeText) {
          return false;
        }
        try {
          await globalThis.navigator.clipboard.writeText(String(fixtureText || ""));
          return true;
        } catch {
          return false;
        }
      };
      globalThis.scrollTo?.(0, 0);
      let clipboardReady = false;
      if (requestedMode === "focused_editable") {
        setFocusedEditableMode();
      } else if (requestedMode === "clipboard") {
        clipboardReady = await setClipboardMode();
      } else {
        setSelectionMode();
      }
      return {
        ok: true,
        activeId: textarea.id,
        value: textarea.value,
        selectionStart: typeof textarea.selectionStart === "number" ? textarea.selectionStart : null,
        selectionEnd: typeof textarea.selectionEnd === "number" ? textarea.selectionEnd : null,
        href: globalThis.location?.href || "",
        activeTextTargetMode: requestedMode,
        clipboardReady,
      };
    },
    [
      {
        fixtureId: ACTIVE_TEXT_SMOKE_FIXTURE_ELEMENT_ID,
        fixtureText: ACTIVE_TEXT_SMOKE_FIXTURE_TEXT,
        targetMode: requestedTargetMode,
      },
    ],
  ).catch(() => []);
  const scriptResult =
    Array.isArray(scriptResults) && scriptResults[0] && typeof scriptResults[0].result === "object"
      ? scriptResults[0].result
      : null;
  if (!scriptResult?.ok) return null;
  if (requestedTargetMode === "clipboard" && scriptResult?.clipboardReady !== true) {
    return null;
  }

  const refreshedContext =
    await waitForBrowserTabContext(run, {
      preferredTabId: targetTabId,
      allowActiveFallback: true,
      timeoutMs: 2_500,
      fallbackUrl: DEFAULT_BROWSER_TOOL_BASE_URL,
    }).catch(() => null) ||
    activeContext;

  run.browserValidationFixture = {
    prepared: true,
    source: "automatic_smoke_fixture",
    tabId: Number(refreshedContext?.tabId || targetTabId) || targetTabId,
    activeTabUrl: asText(refreshedContext?.url || scriptResult?.href || DEFAULT_BROWSER_TOOL_BASE_URL),
    activeId: asText(scriptResult?.activeId),
    value: asText(scriptResult?.value),
    selectionStart: Number.isInteger(scriptResult?.selectionStart) ? scriptResult.selectionStart : null,
    selectionEnd: Number.isInteger(scriptResult?.selectionEnd) ? scriptResult.selectionEnd : null,
    activeTextTargetMode: normalizeActiveTextValidationTargetMode(scriptResult?.activeTextTargetMode),
    clipboardReady: scriptResult?.clipboardReady === true,
    updatedAt: new Date().toISOString(),
  };
  return cloneJson(run.browserValidationFixture, {});
}

function buildPreparedActiveTextValidationFixtureResult(run, toolLabel = "Agentic smoke", args = {}) {
  const assessment = assessPreparedActiveTextValidationFixture(run, args);
  if (!assessment.required || assessment.ready) return null;

  const missingSignals = [
    assessment.activeTabUrl ? "" : "no active HTTP(S) tab",
    assessment.hasScenarioSetup || assessment.hasPreparedFixture ? "" : "no prepared input field or visible selection target",
    assessment.hasVisibleVerification ? "" : "no visible validation of the test state",
  ].filter(Boolean);

  return {
    ok: false,
    summary: `${asText(toolLabel) || "The validation run"} was skipped because no prepared browser test case exists yet for this direct text correction.`,
    error: "Missing prepared browser validation fixture.",
    errorDetail: {
      reason: "missing_prepared_browser_validation_fixture",
      toolLabel: asText(toolLabel),
      activeTabUrl: assessment.activeTabUrl,
      missingSignals,
      recentSuccessfulBrowserSteps: assessment.recentSuccessfulBrowserSteps,
    },
    report: {
      currentState:
        "The direct text correction has not been validated against a real prepared browser state yet.",
      nextAction:
        "Open or activate an HTTP(S) tab first. Use browser_action or playwright_ctx to place a short test string into an input field or visible selection, confirm the state with browser_inspect, then rerun the same test.",
      matchingSignals: [
        assessment.activeTabUrl,
        ...missingSignals,
        ...assessment.recentSuccessfulBrowserSteps,
      ].filter(Boolean).slice(0, 6),
    },
    provenance: [],
  };
}

function buildCraftingAgentInstructions(run) {
  const toolingLabel = formatCraftingAgentToolLabels(run.agentTooling) || "Web Search + Browser Inspect + Browser Action + Browser Tabs + Playwright CTX";
  return [
    "You are the Crafting Agent supervisor for a browser-first local capability factory.",
    "You operate as one supervisor agent with fixed tools and live local state.",
    "You are in capability-development mode. Your job is to expose and repair missing or broken tool paths, not to route around them.",
    `The sidepanel always gives you these fixed tools: ${toolingLabel}.`,
    "You also have local bundle tools for inspecting, saving, smoke-testing, and training reviewed capabilities.",
    "Do not ask the user to choose tools. The tool chooser is intentionally hidden.",
    "Inspect training data early when you need to understand the local prompt-to-JSON baseline.",
    "Inspect the current bundle before editing runtime scripts or reviewed browser capabilities.",
    "Treat the currently saved reviewed bundle as the champion path unless new evidence shows a challenger is stronger.",
    "When a candidate change can be tested without replacing the saved champion, prefer run_agentic_smoke or run_capability_eval with bundleOverride as a challenger check before you overwrite the live bundle.",
    "When you intentionally test a change, include an experiment block on the mutating or evaluation tool call with candidateId, hypothesis, expectedSignal, and exactly one mutationScope whenever possible.",
    "Prefer one artifact family per experiment round: tool_scripts, browser_capabilities, training_rows, training_config, or bundle_policy. Only use multi_artifact when repairing one directly blocking defect.",
    "Use workspace_code_dive when a smoke, eval, runtime, or reviewed-tool failure likely comes from this extension workspace itself.",
    "workspace_code_dive is read-only. It inspects packaged source files, narrows likely patch targets, and prepares a coding handoff.",
    "Use web_search first when you need quick candidate URLs, sources, or public references.",
    "Use browser_tabs to list, open, activate, or close tabs before doing any visible browser work.",
    "Use browser_inspect to read the visible UI state with the vision model before clicking or typing.",
    "Treat browser_inspect as the canonical local Qwen multimodal perception path for screenshot-visible browser evidence.",
    "When a reviewed browser task depends on what is visible in the tab, prefer browser_inspect or another reviewed vision path over guessing from DOM text or prior knowledge alone.",
    "Use browser_action for direct visible-tab interactions like click, type, scroll, keypress, wait, or drag.",
    "Use playwright_ctx for deterministic DOM extraction or scripted in-tab automation when visible actions are too brittle.",
    "Use save_tool_scripts to persist deterministic runtime scripts after you have authored or revised them.",
    "If a reviewed capability name is not a builtin runtime tool name, first save a matching reviewed tool script with the same id via save_tool_scripts so the published capability receives a real execute path.",
    "Use save_browser_capabilities to publish reviewed capabilities, bundle skills, and resource references that the runtime model may call.",
    "Prefer challenger smoke or eval evidence before overwriting a stable champion capability path.",
    "Prefer stable reviewed capability names over opaque one-off scripts whenever the task needs a reusable side effect or handoff output.",
    "For generic outputs and handoffs, prefer explicit reviewed contracts such as read_clipboard_text, write_clipboard_text, compose_email, or capture_bug_report_context when they fit the task.",
    "For domain workflows like classifieds, bug reports, newsletters, or restaurant search, keep the runtime surface compact with named reviewed capabilities such as search_listings, shortlist_candidates, generate_html_digest, or recommend_nearby_restaurants instead of exposing raw browser steps to the runtime model.",
    "When you publish domain-specific reviewed capabilities like search_listings or generate_html_digest, save the reviewed tool scripts first and then publish browser capabilities that reference those same tool names.",
    "If a workflow needs both content preparation and a user-facing handoff, separate those steps into reusable reviewed capabilities, for example capture_bug_report_context -> compose_email -> write_clipboard_text.",
    "Use run_agentic_smoke immediately after saving scripts or reviewed capabilities to validate the runtime path.",
    "Use run_capability_eval when you need a small manual test set for the capability agent before or after training.",
    "When you compare candidate paths, reuse a stable suiteId or evalSetId whenever the same frozen mini-suite should judge both runs.",
    "For selection-, focused-field-, or clipboard-based active-text crafts, run_agentic_smoke and run_capability_eval are not fixture builders.",
    "Before the first smoke or eval for such a craft, use browser_tabs plus browser_action or playwright_ctx to create a live test scenario in an HTTP(S) tab, then use browser_inspect to confirm the visible state.",
    "If no suitable page already exists, open a simple HTTP(S) tab and inject a temporary textarea or contenteditable fixture instead of testing against an empty page.",
    "Only then run smoke or eval, and visually verify the replacement result again afterwards.",
    "Treat failed run_agentic_smoke or run_capability_eval results as normal iteration on the artifacts you just wrote, not as an automatic stop condition.",
    "If the first candidate tool script, browser capability, or bundle behavior fails validation, revise it and retry in the same run while supported local edits remain.",
    "After a failed smoke or eval on a locally authored reviewed path, inspect the saved bundle or workspace code before you conclude that a fixed tool is broken.",
    "Distinguish between your own artifact bug and a real runtime or tool defect. The first failed script revision is not evidence that the underlying tool cannot work.",
    "Use generate_training_batch when the reviewed task contract is stable and you need many new grounded training rows from the local batch generator.",
    "When you generate training data for screenshot-dependent browser tasks, keep the supervised row multimodal with image-bearing messages instead of collapsing visible evidence into text-only paraphrases.",
    "Use draft_training_changes when you have enough evidence to add, update, or delete training rows.",
    "Prefer generate_training_batch for canary- or pilot-sized dataset growth and keep draft_training_changes for surgical edits on a few local rows.",
    `If smoke and eval passed, a trained capability already exists, but maturity is still below ${AUTO_DATASET_GROWTH_TARGET_MATURITY_PERCENT}% because the local dataset is thin, do not stop. Grow the dataset with generate_training_batch, get more rows into ready status, and retrain before you present the craft as mature.`,
    "Use the Test Vision, Test Self Use, and Test Vision FT flows as local proof that the Qwen vision pipeline is ready before scaling multimodal dataset growth or finetuning.",
    "Use start_training_run only after the reviewed capability smoke path is working.",
    "If a required reviewed capability, tool contract, bundle skill, or tool behavior is missing or does not work as intended, inspect the failing reviewed path first and repair the local artifact when the defect is actionable in this workspace.",
    "Before you end blocked on a likely workspace defect, run workspace_code_dive and leave a patch-oriented diagnosis for a coding agent.",
    "If a likely local workspace defect has a concrete patch-oriented diagnosis, you may use request_codex_repair to hand the workspace, prompt, and repo path to the local Codex repair bridge.",
    "After starting a local Codex repair, use get_codex_repair_status until the job either completes or fails, then rerun the same smoke, eval, or save path inside this agent run when feasible.",
    "Do not send Codex repair jobs for missing user details, missing external approvals, or failures that are clearly outside this workspace.",
    "Do not paper over missing or broken tool paths with fallback browser work, prompt-only behavior, synthetic dataset edits, or training runs.",
    "If smoke or eval shows that the runtime did not use the intended reviewed capability, debug the capability selection path, contract, and policy before you stop the run.",
    "Only stop after you have either corrected and revalidated the reviewed path or isolated the remaining concrete blocker with grounded evidence.",
    "Do not start or continue training when the target tool path is still missing or failing.",
    "Prefer continue over blocked while you can still repair the locally authored script, capability, skill text, or validation setup in this workspace.",
    "Use blocked only when a required external tool path, approval boundary, or genuinely missing user detail prevents the next supported repair step.",
    "Use get_training_run_status to follow local training progress instead of guessing whether a run has finished.",
    "When a local training run is still active, prefer get_training_run_status with a bounded wait instead of stopping immediately.",
    "If run_agentic_smoke and run_capability_eval passed, ready training rows already exist, and no trained capability artifact exists yet, do not stop. Start training and follow it with get_training_run_status in the same run.",
    "After a challenger smoke, eval, or training result is clear, call record_experiment_decision with keep, discard, or park so the run retains an explicit research decision.",
    "Prefer keep only when the challenger is measurably better or equally strong and simpler.",
    "Prefer park when the signal is interesting but not yet strong enough to replace the champion.",
    "Keep train, validation, and test evidence separate whenever enough samples exist so you can judge training quality honestly.",
    "Use update_craft_maturity whenever a provenance-worthy milestone materially changes the craft state shown in the UI.",
    "Do not call update_craft_maturity just because seed rows, saved capabilities, or smoke setup changed.",
    `Treat ${ASK_USER_TOOL_NAME} as a last resort, not a discovery shortcut.`,
    "Only ask the user when continuing would likely produce the wrong site target, wrong tool contract, or wrong output schema.",
    "If you can proceed with a reasonable default, generic seed rows, or a reviewed contract bootstrap, do that instead.",
    "Never ask for confirmation, convenience examples, or browser setup that the runtime can infer or recover from on its own.",
    `If critical details are missing, call ${ASK_USER_TOOL_NAME} with all blocking questions in one call.`,
    "Do not ask the user to open a page or mark example text just to bootstrap a generic selection, focused-field, or clipboard-fallback craft.",
    "If no active HTTP(S) tab is present, prefer browser_tabs or a generic contract bootstrap before interrupting the user.",
    "Keep your reasoning grounded in the visible dataset, browser evidence, and approved user answers.",
    "When refineContext indicates that the user edited the refine brief, assume the previous version was not good enough.",
    "Treat the latest refine brief as authoritative even when it deletes, replaces, or shortens earlier wording.",
    "If the new refine brief conflicts with the previous brief, follow the latest edited brief instead of trying to preserve the old description.",
    "Fill officialDescription with one concise read-only capability description for the UI only when the behavior is grounded by real evidence from working tools, saved artifacts, browser evidence, smoke results, eval results, or dataset evidence.",
    "Do not put wishes, planned future behavior, or unverified promises into officialDescription.",
    "If the current run has not verified the capability enough to support a precise description, leave officialDescription null.",
    "Treat provenance as milestone highlights only: meaningful research finds, grounded browser evidence, user decisions, and applied dataset edits.",
    "Do not use provenance for routine status updates, raw tool chatter, or bookkeeping.",
    "Use plain German for summary, responseText, report, and provenance titles shown in the UI.",
    "Avoid internal terms like bootstrap, starter row, contract, schema, pattern match, lexical variety, reviewed, or portable flow in user-facing text.",
    "Do not stop after the first useful tool call if another concrete high-confidence action is still available in the current run.",
    "Set status to continue when further supported work remains in the current run.",
    "Set status to done only when this pass reached a sensible stopping point for the user.",
    "Set status to blocked only when a missing detail truly prevents the next supported step.",
    "If the requested capability still depends on an unsupported or unapproved tool path, do not present it as finished. Explain the missing path clearly and prefer blocked over done.",
    "Craft maturity is never auto-derived by the runtime. You must set it explicitly.",
    "Until a trained capability artifact exists, any maturity above 0% is invalid and will be forced back to 0% by the runtime.",
    "Only raise craft maturity for durable state changes backed by real evidence, not because time passed or tools ran.",
    "Before a trained capability artifact exists, prefer leaving maturity unchanged unless a durable 0% rationale truly needs to be shown.",
    "Before there is evaluated capability evidence, keep maturity in phase crafting_progress.",
    "Do not claim capability_readiness from seed rows, naming, prompt drafts, or UI setup alone.",
    "When the current craft title is only a placeholder or a prompt slice, fill suggestedName with a short user-facing capability name.",
    "The final answer is validated against a structured output contract by the runtime.",
  ].join("\n");
}

function buildCraftingAgentRunInput(run) {
  return JSON.stringify(
    {
      craft: {
        id: run.craftId,
        name: run.craft?.name || "",
        nameSource: run.craft?.nameSource || "",
        summary: run.craft?.summary || "",
        agentPrompt: run.craft?.agentPrompt || "",
        stage: run.craft?.stage || "",
        inputMode: run.craft?.inputMode || "",
        actionLabel: run.craft?.actionLabel || "",
        inputHint: run.craft?.inputHint || "",
        inputExamples: Array.isArray(run.craft?.inputExamples) ? run.craft.inputExamples.slice(0, 6) : [],
        agentTooling: run.agentTooling,
      },
      objective: run.brief,
      refineContext:
        run.craft?.refineContext && typeof run.craft.refineContext === "object"
          ? cloneJson(run.craft.refineContext, null)
          : null,
      answeredQuestions: normalizeQuestions(run.previousQuestions)
        .filter((entry) => asText(entry.answer))
        .map((entry) => ({
          id: entry.id,
          question: entry.question,
          answer: entry.answer,
        })),
      compactedMemory: Array.isArray(run.compactedMemory)
        ? run.compactedMemory.slice(-3)
        : [],
      localDatasetPreview: summarizeTrainingSamples(run.samples, 4),
      bootstrapHints: buildCraftBootstrapHints(run),
      activeTabContext: run.activeTabContext,
      currentReport: run.report,
      currentMaturity: run.maturity,
      capabilityEvidence: run.capabilityEvidence,
    },
    null,
    2,
  );
}

function buildTrainingChangesAgentInstructions(run) {
  return [
    "You are a specialist subagent for deriving prompt-to-JSON training-data revisions.",
    "You do not browse or execute tools yourself. Work only from the provided local dataset preview, browser evidence, and web-search notes.",
    "You are for targeted row edits only, not for bulk batch generation.",
    "Keep the dataset seed-grounded and generic to the craft instead of overfitting to a single example.",
    "For agentic tool-use skills, prefer portable multi-turn rows with messages, reviewed tools, and a supervised assistant target turn.",
    "Prefer a small number of high-confidence edits over broad speculative rewrites.",
    "Use openQuestions only as a last resort for decisive ambiguities. Do not ask for convenience confirmations, extra examples, or recoverable browser setup.",
    "Use openQuestions only when a missing detail blocks reliable sample generation.",
    "When a generic selection, focused-field, or clipboard-fallback craft has no live browser tab, bootstrap from the reviewed active-text contract instead of blocking on user clarification.",
    "Use useSurface to describe how the resulting ability should be executed in the sidepanel.",
    "Use provenance only for milestone highlights such as concrete dataset edits, strong research findings, or grounded browser evidence.",
    "Write summary, report, and provenance in plain German for the user-facing UI.",
    "Avoid internal training jargon like bootstrap, starter row, contract, schema, reviewed pattern, or lexical variety.",
    "If you set maturity, treat it as an explicit agent statement, not a heuristic.",
    "Until a trained capability artifact exists, maturity must remain exactly 0%.",
    "Before there is evaluated capability evidence, keep maturity in phase crafting_progress.",
    "The runtime validates your result against a structured contract. Do not add extra wrapper objects or prose outside that result.",
  ].join("\n");
}

function buildTrainingChangesRunInput(run, args = {}) {
  return JSON.stringify(
    {
      craft: {
        id: run.craftId,
        name: run.craft?.name || "",
        summary: run.craft?.summary || "",
        inputMode: run.craft?.inputMode || "",
        actionLabel: run.craft?.actionLabel || "",
        inputHint: run.craft?.inputHint || "",
        inputExamples: Array.isArray(run.craft?.inputExamples) ? run.craft.inputExamples.slice(0, 6) : [],
      },
      objective: asText(args.objective) || run.brief,
      localDatasetPreview: summarizeTrainingSamples(run.samples, 12),
      datasetMeta: getTrainingDataMeta(run.samples),
      recentResearchNotes: Array.isArray(run.researchNotes) ? run.researchNotes.slice(-6) : [],
      recentBrowserNotes: Array.isArray(run.browserNotes) ? run.browserNotes.slice(-6) : [],
      bootstrapHints: buildCraftBootstrapHints(run, args),
      activeTabContext: run.activeTabContext,
      previousReport: run.report,
      currentMaturity: run.maturity,
      capabilityEvidence: run.capabilityEvidence,
      answeredQuestions: normalizeQuestions(run.previousQuestions)
        .filter((entry) => asText(entry.answer))
        .map((entry) => ({
          id: entry.id,
          question: entry.question,
          reason: entry.reason,
          answer: entry.answer,
        })),
      constraints: {
        maxOperations: 8,
        allowedOperationTypes: ["add", "update", "delete"],
        allowedInputModes: ["free_text", "mixed", "selection", "current_tab", "context_only"],
      },
    },
    null,
    2,
  );
}

function buildTrainingChangesFallbackBrief(run, args = {}) {
  return JSON.stringify(
    {
      objective: asText(args.objective) || run.brief,
      recentResearchNotes: Array.isArray(run.researchNotes) ? run.researchNotes.slice(-6) : [],
      recentBrowserNotes: Array.isArray(run.browserNotes) ? run.browserNotes.slice(-6) : [],
      bootstrapHints: buildCraftBootstrapHints(run, args),
      activeTabContext: run.activeTabContext,
      previousReport: run.report,
      currentMaturity: run.maturity,
      capabilityEvidence: run.capabilityEvidence,
      answeredQuestions: normalizeQuestions(run.previousQuestions)
        .filter((entry) => asText(entry.answer))
        .map((entry) => ({
          id: entry.id,
          question: entry.question,
          reason: entry.reason,
          answer: entry.answer,
        })),
    },
    null,
    2,
  );
}

function buildCraftNamingAgentInstructions() {
  return [
    "You are a naming specialist for a local browser capability factory.",
    "Return one concise user-facing capability name.",
    "Prefer 2 to 5 words. Focus on the capability outcome instead of copying the raw prompt.",
    "Avoid ellipses, trailing punctuation, quotation marks, and filler words like capability, assistant, tool, or helper unless they are essential.",
    "The runtime validates the result against a structured contract.",
  ].join("\n");
}

function buildCraftNamingRunInput(run) {
  return JSON.stringify(
    {
      craft: {
        id: run.craftId,
        currentName: run.craft?.name || "",
        currentNameSource: run.craft?.nameSource || "",
        summary: run.craft?.summary || "",
        inputMode: run.craft?.inputMode || "",
        actionLabel: run.craft?.actionLabel || "",
        inputHint: run.craft?.inputHint || "",
        inputExamples: Array.isArray(run.craft?.inputExamples) ? run.craft.inputExamples.slice(0, 4) : [],
      },
      objective: run.brief,
      finalReport: run.report,
      finalMaturity: run.maturity,
      capabilityEvidence: run.capabilityEvidence,
      finalResponse: trimText(run.responseText, 800),
      useSurface: run.useSurface,
      operationsSummary: run.operations.length ? summarizeOperationCounts(run.operations) : "",
      answeredQuestions: normalizeQuestions(run.previousQuestions)
        .filter((entry) => asText(entry.answer))
        .map((entry) => ({
          question: entry.question,
          answer: entry.answer,
        })),
    },
    null,
    2,
  );
}

function parseAskUserToolPayload(rawArguments) {
  const parsed =
    rawArguments && typeof rawArguments === "object"
      ? rawArguments
      : parseJsonLoose(rawArguments || "");
  const payload = parsed && typeof parsed === "object" ? parsed : {};
  return {
    questions: normalizeQuestions(payload.questions),
    report: normalizeReport(payload.report, {}),
    provenance: normalizeProvenance(payload.provenance),
    maturity: normalizeMaturity(payload.maturity, createEmptyCraftMaturity()),
    summary: asText(payload.summary || payload.reason),
  };
}

function collectClarificationAssessmentText(params = {}) {
  const report = params?.report && typeof params.report === "object" ? params.report : {};
  const questions = normalizeQuestions(params?.questions);
  return [
    asText(params?.summary),
    asText(report.currentState),
    asText(report.nextAction),
    ...questions.flatMap((entry) => [entry.question, entry.reason]),
  ]
    .filter(Boolean)
    .join("\n");
}

function assessClarificationRequest(run, params = {}) {
  const questions = normalizeQuestions(params?.questions);
  if (!questions.length) {
    return {
      requireApproval: false,
      reasonCode: "empty",
      summary: "Clarification skipped because no concrete blocking questions were provided.",
      nextAction: "Continue with the current context.",
    };
  }

  const contextText = collectClarificationAssessmentText(params);
  const bootstrapHints = buildCraftBootstrapHints(run, params);
  const missingReasons = questions.some((entry) => !asText(entry.reason));
  const looksLikeConvenienceBrowserRequest = CLARIFICATION_CONVENIENCE_HINT_RE.test(contextText);
  const craftNeedsLivePage = asText(run?.craft?.inputMode).toLowerCase() === "current_tab";
  const hasExplicitBlockingSignal =
    CLARIFICATION_BLOCKING_HINT_RE.test(contextText) ||
    (CLARIFICATION_DECISION_HINT_RE.test(contextText) &&
      /\b(ambig(?:uous|uity)|incompatible|wrong|required|critical|must|determin(?:e|es|ing)|block(?:ed|ing)?)\b/i.test(
        contextText,
      ));

  if (bootstrapHints?.genericSelectionBootstrap && looksLikeConvenienceBrowserRequest) {
    return {
      requireApproval: false,
      reasonCode: "generic_browser_bootstrap",
      summary:
        "Clarification skipped because the run can bootstrap the generic selection, focused-field, or clipboard-fallback craft without extra user input.",
      nextAction:
        "Continue with draft_training_changes and derive portable rows from the reviewed active-text contract.",
    };
  }

  if (missingReasons) {
    return {
      requireApproval: false,
      reasonCode: "missing_blocking_rationale",
      summary:
        "Clarification skipped because the request did not explain why the missing information is truly blocking.",
      nextAction: "Continue with the current context and use a reasonable default instead of interrupting the user.",
    };
  }

  if (craftNeedsLivePage && looksLikeConvenienceBrowserRequest) {
    return {
      requireApproval: true,
      reasonCode: "page_specific_browser_blocker",
      summary: asText(params?.summary) || "The run requires live page context before it can continue safely.",
      nextAction: "Wait for the user input because this craft is tied to the active page context.",
    };
  }

  if (!hasExplicitBlockingSignal) {
    return {
      requireApproval: false,
      reasonCode: "not_explicitly_blocking",
      summary:
        "Clarification skipped because the existing context is sufficient to continue without blocking the user.",
      nextAction: "Keep moving with the best supported default and only interrupt if a decisive blocker remains.",
    };
  }

  return {
    requireApproval: true,
    reasonCode: "explicit_blocker",
    summary: asText(params?.summary) || "Critical details are missing and the run is blocked pending user input.",
    nextAction: "Wait for the user answers before continuing the run.",
  };
}

function extractAskUserInterruptions(interruptions = []) {
  return (Array.isArray(interruptions) ? interruptions : [])
    .map((entry) => {
      const toolName = asText(entry?.toolName || entry?.name || entry?.rawItem?.name);
      if (toolName !== ASK_USER_TOOL_NAME) return null;
      const payload = parseAskUserToolPayload(entry?.arguments || entry?.rawItem?.arguments || "");
      if (!payload.questions.length) return null;
      return {
        callId: asText(entry?.rawItem?.callId || entry?.rawItem?.id),
        ...payload,
      };
    })
    .filter(Boolean);
}

function normalizeTrainingDraftOutput(rawOutput) {
  const parsed =
    rawOutput && typeof rawOutput === "object"
      ? rawOutput
      : parseJsonLoose(String(rawOutput || "").trim()) || {};
  return CRAFTING_AGENT_TRAINING_DRAFT_CONTRACT.parse({
    summary: asText(parsed?.summary || parsed?.rationale || "Training data review completed."),
    rationale: asText(parsed?.rationale || parsed?.summary || "No rationale provided."),
    report: normalizeStructuredReportContract(parsed?.report, {
      currentState: asText(parsed?.summary || parsed?.rationale),
    }),
    maturity:
      parsed?.maturity && typeof parsed.maturity === "object"
        ? normalizeStructuredMaturityContract(parsed.maturity)
        : null,
    openQuestions: normalizeStructuredQuestionContracts(parsed?.openQuestions),
    provenance: normalizeStructuredProvenanceContracts(parsed?.provenance, 12),
    useSurface:
      parsed?.useSurface && typeof parsed.useSurface === "object"
        ? normalizeStructuredUseSurfaceContract(parsed.useSurface)
        : null,
    operations: (Array.isArray(parsed?.operations) ? parsed.operations : [])
      .map((operation) => ({
        type: ["add", "update", "delete"].includes(asText(operation?.type).toLowerCase())
          ? asText(operation?.type).toLowerCase()
          : "update",
        sampleId: nullIfEmptyText(operation?.sampleId),
        reason: asText(operation?.reason || "No rationale provided."),
        fields:
          operation?.fields && typeof operation.fields === "object"
            ? {
                promptText:
                  operation.fields.promptText == null ? null : String(operation.fields.promptText || ""),
                messages:
                  operation.fields.messages == null ? null : normalizeTrainingMessages(operation.fields.messages),
                tools:
                  operation.fields.tools == null ? null : normalizeTrainingTools(operation.fields.tools),
                targetTurnIndex:
                  operation.fields.targetTurnIndex == null ? null : normalizeTrainingTargetTurnIndex(operation.fields.targetTurnIndex),
                split: operation.fields.split == null ? null : operation.fields.split,
                status: operation.fields.status == null ? null : operation.fields.status,
                source: nullIfEmptyText(operation.fields.source),
              }
            : null,
      }))
      .slice(0, 8),
  });
}

function normalizeCraftingAgentFinalOutput(rawOutput) {
  const parsed =
    rawOutput && typeof rawOutput === "object"
      ? rawOutput
      : parseJsonLoose(String(rawOutput || "").trim()) || {};
  const status = asText(parsed?.status).toLowerCase();
  return CRAFTING_AGENT_FINAL_OUTPUT_CONTRACT.parse({
    status: FINAL_OUTPUT_STATUSES.has(status) ? status : "done",
    summary: asText(parsed?.summary || parsed?.responseText || "Crafting agent run completed."),
    responseText: asText(parsed?.responseText || parsed?.summary || "Crafting agent run completed."),
    report: normalizeStructuredReportContract(parsed?.report, {
      currentState: asText(parsed?.summary || parsed?.responseText),
    }),
    maturity:
      parsed?.maturity && typeof parsed.maturity === "object"
        ? normalizeStructuredMaturityContract(parsed.maturity)
        : null,
    provenance: normalizeStructuredProvenanceContracts(parsed?.provenance),
    suggestedName:
      normalizeStructuredNameSuggestionContract(parsed?.suggestedName || parsed?.suggested_name) ||
      normalizeStructuredNameSuggestionContract(parsed?.capabilityName || parsed?.capability_name),
    officialDescription:
      normalizeStructuredOfficialDescriptionContract(
        parsed?.officialDescription ||
          parsed?.official_description ||
          parsed?.suggestedDescription ||
          parsed?.suggested_description,
      ),
    useSurface:
      parsed?.useSurface && typeof parsed.useSurface === "object"
        ? normalizeStructuredUseSurfaceContract(parsed.useSurface)
        : null,
  });
}

function summarizeToolStringOutput(rawOutput) {
  if (rawOutput && typeof rawOutput === "object") {
    const summary = asText(rawOutput.summary || rawOutput.message || rawOutput.error);
    if (summary) return summary;
    return trimText(JSON.stringify(rawOutput), 260);
  }
  const parsed = parseJsonLoose(rawOutput);
  if (parsed && typeof parsed === "object") {
    const summary = asText(parsed.summary || parsed.message || parsed.error);
    if (summary) return summary;
  }
  return trimText(rawOutput, 260);
}

function isToolFailureLikeText(value = "") {
  const haystack = asText(value).toLowerCase();
  return /an error occurred while running the tool|tool execution failed|(?:^|\b)error:|runtime failed|failed to call ortrun/.test(haystack);
}

async function executeInspectTrainingDataTool(run) {
  const meta = getTrainingDataMeta(run.samples);
  return {
    ok: true,
    summary: meta.totalSamples
      ? `${meta.totalSamples} training samples loaded. ${meta.readySamples} ready, ${meta.invalidSamples} invalid rows.`
      : "No training samples are stored yet.",
    data: {
      meta,
      samples: summarizeTrainingSamples(run.samples, 8),
    },
    report: {
      currentState: meta.totalSamples
        ? `${meta.totalSamples} local training rows are visible, and ${meta.readySamples} are training-ready.`
        : "No local training rows exist yet.",
    },
    provenance: [],
  };
}

async function readJsonResponse(response) {
  const text = await response.text();
  if (!text.trim()) return {};
  try {
    return JSON.parse(text);
  } catch {
    return {
      ok: false,
      error: `Invalid JSON from local training bridge: ${trimText(text, 300)}`,
    };
  }
}

async function fetchLocalTrainingBridge(pathname, init = {}) {
  const baseUrl = DEFAULT_LOCAL_TRAINING_BRIDGE_BASE_URL.replace(/\/$/, "");
  const timeoutMs = Math.max(0, Math.min(120_000, Number(init?.timeoutMs || 0) || 0));
  const controller = timeoutMs ? new AbortController() : null;
  const headers =
    init?.headers && typeof init.headers === "object"
      ? init.headers
      : {};
  const requestInit = {
    ...(init && typeof init === "object" ? init : {}),
    headers: {
      "content-type": "application/json",
      ...headers,
    },
  };
  if ("timeoutMs" in requestInit) {
    delete requestInit.timeoutMs;
  }
  if (controller) {
    requestInit.signal = controller.signal;
  }
  let timeoutId = 0;
  if (controller) {
    timeoutId = globalThis.setTimeout(() => {
      try {
        controller.abort();
      } catch {}
    }, timeoutMs);
  }
  let response;
  try {
    response = await fetch(`${baseUrl}${pathname}`, requestInit);
  } catch (error) {
    if (controller && (controller.signal.aborted || asText(error?.name) === "AbortError")) {
      const timeoutError = new Error(`Local training bridge request timed out after ${timeoutMs} ms.`);
      timeoutError.detail = {
        bridgeBaseUrl: baseUrl,
        pathname: asText(pathname),
        timeoutMs,
      };
      throw timeoutError;
    }
    throw error;
  } finally {
    if (timeoutId) {
      globalThis.clearTimeout(timeoutId);
    }
  }
  const payload = await readJsonResponse(response);
  if (!response.ok || (payload?.ok === false && asText(payload?.error || payload?.message))) {
    const error = new Error(
      asText(payload?.error || payload?.message) || `Local training bridge HTTP ${response.status}`,
    );
    error.detail = {
      bridgeBaseUrl: baseUrl,
      pathname: asText(pathname),
      status: Number(response.status || 0) || null,
      payload: payload && typeof payload === "object" ? cloneJson(payload, null) : null,
    };
    throw error;
  }
  return payload;
}

async function readCachedUiPreferences() {
  const now = Date.now();
  if (
    devModePreferenceCache.promise == null &&
    devModePreferenceCache.loadedAt &&
    now - devModePreferenceCache.loadedAt < DEV_MODE_PREFERENCE_CACHE_MS
  ) {
    return {
      showDevHeader: devModePreferenceCache.value === true,
    };
  }
  if (devModePreferenceCache.promise) {
    return await devModePreferenceCache.promise;
  }
  devModePreferenceCache.promise = (async () => {
    if (!craftSync?.getValue) {
      devModePreferenceCache.value = false;
      devModePreferenceCache.loadedAt = Date.now();
      return { showDevHeader: false };
    }
    const uiPreferences = await readUiPreferences(craftSync);
    devModePreferenceCache.value = uiPreferences?.showDevHeader === true;
    devModePreferenceCache.loadedAt = Date.now();
    return {
      showDevHeader: devModePreferenceCache.value,
    };
  })();
  try {
    return await devModePreferenceCache.promise;
  } catch (error) {
    devModePreferenceCache.value = false;
    devModePreferenceCache.loadedAt = Date.now();
    throw error;
  } finally {
    devModePreferenceCache.promise = null;
  }
}

async function isDevModeEnabled() {
  try {
    const uiPreferences = await readCachedUiPreferences();
    return uiPreferences?.showDevHeader === true;
  } catch (error) {
    console.warn("[crafting-agent-runner] failed to read ui preferences", error);
    return false;
  }
}

function summarizeRunLogEntry(entry = null) {
  const source = entry && typeof entry === "object" ? entry : {};
  const summary = {
    level: asText(source.level),
    message: trimText(source.message, 280),
    time: asText(source.time),
  };
  const toolName = asText(source.toolName);
  if (toolName) summary.toolName = toolName;
  const stageId = asText(source.stageId);
  if (stageId) summary.stageId = stageId;
  const status = asText(source.status);
  if (status) summary.status = status;
  return summary;
}

function summarizeToolTraceEntry(entry = null) {
  const source = entry && typeof entry === "object" ? entry : {};
  return {
    turn: Number(source.turn) || 0,
    action: asText(source.action),
    ok: source.ok !== false,
    summary: trimText(source.summary || source.error || "", 280),
    error: trimText(source.error, 400),
    recordedAt: asText(source.recordedAt),
  };
}

function summarizeWorkspaceDiagnosis(diagnosis = null) {
  const source = diagnosis && typeof diagnosis === "object" ? diagnosis : null;
  if (!source) return null;
  const codingHandoff =
    source.codingHandoff && typeof source.codingHandoff === "object"
      ? source.codingHandoff
      : null;
  return {
    failingTool: asText(source?.query?.failingTool || codingHandoff?.failingTool),
    errorText: trimText(asText(source?.query?.errorText || codingHandoff?.errorText), 320),
    patchTargets: Array.isArray(source?.patchTargets)
      ? source.patchTargets
          .map((entry) => trimText(asText(entry?.path || entry), 200))
          .filter(Boolean)
          .slice(0, 8)
      : [],
    validationTargets: Array.isArray(source?.validationTargets)
      ? source.validationTargets
          .map((entry) => trimText(asText(entry), 200))
          .filter(Boolean)
          .slice(0, 8)
      : [],
    bestHypothesis: trimText(asText(source?.bestHypothesis?.summary), 320),
  };
}

function buildRunObservabilitySnapshot(run) {
  const logs = Array.isArray(run?.logs) ? run.logs : [];
  const toolTrace = Array.isArray(run?.toolTrace) ? run.toolTrace : [];
  return {
    runId: asText(run?.jobId),
    craftId: asText(run?.craftId),
    status: asText(run?.status),
    phase: asText(run?.phase),
    finalStatus: normalizeFinalRunStatus(run?.finalStatus),
    error: trimText(asText(run?.error), 400),
    updatedAt: asText(run?.updatedAt),
    startedAt: asText(run?.startedAt),
    completedAt: asText(run?.completedAt),
    latestCodexRepairJobId: asText(run?.latestCodexRepairJobId),
    latestTrainingJobId: asText(run?.latestTrainingJobId),
    report: run?.report && typeof run.report === "object"
      ? {
          objective: trimText(asText(run.report.objective), 240),
          currentState: trimText(asText(run.report.currentState), 320),
          nextAction: trimText(asText(run.report.nextAction), 320),
        }
      : null,
    lastToolFailure:
      run?.lastToolFailure && typeof run.lastToolFailure === "object"
        ? {
            action: asText(run.lastToolFailure.action),
            error: trimText(asText(run.lastToolFailure.error), 400),
            recordedAt: asText(run.lastToolFailure.recordedAt),
          }
        : null,
    codexRepair:
      run?.codexRepair && typeof run.codexRepair === "object"
        ? {
            pendingJobId: asText(run.codexRepair.pendingJobId),
            pendingFingerprint: trimText(asText(run.codexRepair.pendingFingerprint), 120),
            recentHistory: Array.isArray(run.codexRepair.history)
              ? run.codexRepair.history.slice(-4).map((entry) => ({
                  fingerprint: trimText(asText(entry?.fingerprint), 120),
                  attemptCount: Number(entry?.attemptCount) || 0,
                  lastJobId: asText(entry?.lastJobId),
                  lastStatus: asText(entry?.lastStatus),
                  lastSummary: trimText(asText(entry?.lastSummary), 240),
                  updatedAt: asText(entry?.updatedAt),
                }))
              : [],
          }
        : null,
    reloadResume:
      run?.reloadResume && typeof run.reloadResume === "object"
        ? {
            pending: run.reloadResume.pending === true,
            reason: asText(run.reloadResume.reason),
            requestedAt: asText(run.reloadResume.requestedAt),
          }
        : null,
    lastLog: logs.length ? summarizeRunLogEntry(logs[logs.length - 1]) : null,
    logTail: logs.slice(-DEV_OBSERVABILITY_LOG_TAIL).map((entry) => summarizeRunLogEntry(entry)),
    toolTraceTail: toolTrace.slice(-DEV_OBSERVABILITY_TOOL_TRACE_TAIL).map((entry) => summarizeToolTraceEntry(entry)),
    workspaceDiagnosis: summarizeWorkspaceDiagnosis(run?.workspaceCodeDive),
  };
}

async function flushDevObservabilityEvents() {
  if (devObservabilityFlushPromise || !devObservabilityQueue.length) return;
  const batch = devObservabilityQueue.splice(0, devObservabilityQueue.length);
  devObservabilityFlushPromise = (async () => {
    try {
      await fetchLocalTrainingBridge("/api/dev/agent-events", {
        method: "POST",
        body: JSON.stringify({
          events: batch,
        }),
      });
    } catch (error) {
      console.warn("[crafting-agent-runner] failed to flush dev observability events", error);
      if (devObservabilityQueue.length < DEV_OBSERVABILITY_MAX_BUFFER) {
        devObservabilityQueue.unshift(...batch.slice(-Math.max(1, DEV_OBSERVABILITY_MAX_BUFFER - devObservabilityQueue.length)));
      }
    } finally {
      devObservabilityFlushPromise = null;
      if (devObservabilityQueue.length) {
        scheduleDevObservabilityFlush();
      }
    }
  })();
  await devObservabilityFlushPromise;
}

function scheduleDevObservabilityFlush() {
  if (devObservabilityFlushTimer) return;
  devObservabilityFlushTimer = globalThis.setTimeout(() => {
    devObservabilityFlushTimer = 0;
    void flushDevObservabilityEvents();
  }, DEV_OBSERVABILITY_FLUSH_MS);
}

function emitDevObservabilityEvent(run, type, payload = {}) {
  const eventPayload = payload && typeof payload === "object" ? cloneJson(payload, {}) : {};
  const runId = asText(run?.jobId || eventPayload?.runId);
  const craftId = asText(run?.craftId || eventPayload?.craftId);
  const status = asText(run?.status || eventPayload?.status);
  const phase = asText(run?.phase || eventPayload?.phase);
  const finalStatus = normalizeFinalRunStatus(run?.finalStatus || eventPayload?.finalStatus);
  const sequence = Number(run?.devObservabilitySequence || 0) + 1;
  if (run && typeof run === "object") {
    run.devObservabilitySequence = sequence;
  }
  void (async () => {
    if (!(await isDevModeEnabled())) return;
    devObservabilityQueue.push({
      eventId: `${runId || craftId || "event"}:${sequence}:${Date.now()}`,
      source: "crafting_agent_runner",
      type: asText(type) || "event",
      recordedAt: new Date().toISOString(),
      runId,
      craftId,
      status,
      phase,
      finalStatus,
      latestCodexRepairJobId: asText(run?.latestCodexRepairJobId),
      payload: eventPayload,
    });
    if (devObservabilityQueue.length > DEV_OBSERVABILITY_MAX_BUFFER) {
      devObservabilityQueue.splice(0, devObservabilityQueue.length - DEV_OBSERVABILITY_MAX_BUFFER);
    }
    scheduleDevObservabilityFlush();
  })();
}

function inferTrainingOutputSchema(samples = []) {
  for (const sample of Array.isArray(samples) ? samples : []) {
    if (!hasStructuredTrainingTrace(sample)) continue;
    const targetMessage = sample.messages[sample.targetTurnIndex];
    if (Array.isArray(targetMessage?.tool_calls) && targetMessage.tool_calls.length) {
      return "{ tool_name: string, arguments: object }";
    }
    if (typeof targetMessage?.content === "string" && targetMessage.content.trim()) {
      return "{ assistant_content: string }";
    }
  }
  return "{ assistant_content: string }";
}

function buildBridgeToolInterfaceFromCapability(capability = {}, craftId = "", bundleSkills = []) {
  const functionName = asText(
    capability?.toolName ||
    capability?.functionName ||
    capability?.function_name ||
    capability?.name ||
    capability?.id,
  );
  if (!functionName) return null;
  return {
    functionName,
    description: asText(capability?.description) || `Reviewed capability ${functionName}.`,
    parameterSchema:
      capability?.parameterSchema && typeof capability.parameterSchema === "object"
        ? cloneJson(capability.parameterSchema, {})
        : (parseJsonLoose(capability?.parameterSchema) || {
            type: "object",
            additionalProperties: false,
          }),
    returnSchema:
      capability?.returnSchema && typeof capability.returnSchema === "object"
        ? cloneJson(capability.returnSchema, {})
        : (parseJsonLoose(capability?.returnSchema) || {
            type: "object",
            additionalProperties: true,
          }),
    preconditions: uniqueTextList(capability?.preconditions || capability?.readsFrom || [], 16),
    siteAssumptions: uniqueTextList(capability?.siteAssumptions || capability?.writesTo || [], 16),
    bundleRef: getBrowserCapabilityBundleArtifactId(craftId),
    skillRef: asText(bundleSkills[0]) || "",
    resourceRefs: uniqueTextList(capability?.resourceRefs || capability?.resources || [], 16),
  };
}

function buildBridgeToolInterfaceFromPortableTool(toolEntry = {}, craftId = "") {
  const functionName = asText(toolEntry?.function?.name);
  if (!functionName) return null;
  return {
    functionName,
    description: asText(toolEntry?.function?.description) || `Reviewed tool ${functionName}.`,
    parameterSchema:
      toolEntry?.function?.parameters && typeof toolEntry.function.parameters === "object"
        ? cloneJson(toolEntry.function.parameters, {})
        : {
            type: "object",
            additionalProperties: false,
          },
    returnSchema: {
      type: "object",
      additionalProperties: true,
    },
    preconditions: [],
    siteAssumptions: [],
    bundleRef: getBrowserCapabilityBundleArtifactId(craftId),
    skillRef: "",
    resourceRefs: [],
  };
}

function buildTrainingBatchToolInterfaces(snapshot, craftId, bootstrapHints = null) {
  const bundleSkills = Array.isArray(snapshot?.bundle?.browserCapabilities?.payload?.skills)
    ? snapshot.bundle.browserCapabilities.payload.skills
    : [];
  const fromCapabilities = (Array.isArray(snapshot?.bundle?.browserCapabilities?.payload?.capabilities)
    ? snapshot.bundle.browserCapabilities.payload.capabilities
    : [])
    .map((capability) => buildBridgeToolInterfaceFromCapability(capability, craftId, bundleSkills))
    .filter(Boolean);
  if (fromCapabilities.length) return fromCapabilities;
  const bootstrapTools = Array.isArray(bootstrapHints?.starterRow?.tools)
    ? bootstrapHints.starterRow.tools
    : [];
  return bootstrapTools
    .map((entry) => buildBridgeToolInterfaceFromPortableTool(entry, craftId))
    .filter(Boolean);
}

function buildTrainingBatchSeedRecords(run, objective, bootstrapHints = null) {
  const out = [];
  const seen = new Set();
  const pushSeed = (seed) => {
    const title = trimText(asText(seed?.title), 180);
    const bodyText = trimText(asText(seed?.body_text), 4_000);
    if (!title && !bodyText) return;
    const dedupeKey = JSON.stringify({
      title,
      bodyText,
      categories: seed?.categories || {},
    });
    if (seen.has(dedupeKey)) return;
    seen.add(dedupeKey);
    out.push({
      id: asText(seed?.id) || `seed-${out.length + 1}`,
      title: title || trimText(asText(run?.craft?.name || objective || "Craft seed"), 180),
      body_text: bodyText,
      url: asText(seed?.url),
      attributes: seed?.attributes && typeof seed.attributes === "object" ? cloneJson(seed.attributes, {}) : {},
      categories: seed?.categories && typeof seed.categories === "object" ? cloneJson(seed.categories, {}) : {},
    });
  };

  for (const sample of Array.isArray(run?.samples) ? run.samples.slice(0, 24) : []) {
    pushSeed({
      id: `sample-${asText(sample?.id) || out.length + 1}`,
      title: sample?.promptText || run?.craft?.name || objective || "Training sample",
      body_text: [
        asText(sample?.promptText) ? `Prompt: ${sample.promptText}` : "",
        hasStructuredTrainingTrace(sample) ? `Messages: ${JSON.stringify(sample.messages.slice(0, 6))}` : "",
        hasStructuredTrainingTrace(sample) ? `Target turn: ${renderStructuredTrainingTargetSummary(sample)}` : "",
      ].filter(Boolean).join("\n"),
      attributes: {
        prompt_text: asText(sample?.promptText),
        messages: Array.isArray(sample?.messages) ? cloneJson(sample.messages, []) : [],
        tools: Array.isArray(sample?.tools) ? cloneJson(sample.tools, []) : [],
        target_turn_index: Number.isInteger(sample?.targetTurnIndex) ? sample.targetTurnIndex : null,
        split: asText(sample?.split),
        status: asText(sample?.status),
        source: asText(sample?.source),
      },
      categories: {
        seed_source: "training_sample",
        split: asText(sample?.split),
        mode: "multiturn",
      },
    });
  }

  const pushNoteSeeds = (entries, sourceKind) => {
    for (const entry of Array.isArray(entries) ? entries.slice(-8) : []) {
      if (typeof entry === "string") {
        pushSeed({
          id: `${sourceKind}-${out.length + 1}`,
          title: `${run?.craft?.name || "Craft"} ${sourceKind}`,
          body_text: entry,
          attributes: { note: entry },
          categories: { seed_source: sourceKind },
        });
        continue;
      }
      if (!entry || typeof entry !== "object") continue;
      pushSeed({
        id: `${sourceKind}-${asText(entry.id || entry.url || out.length + 1)}`,
        title: asText(entry.title || entry.name || `${sourceKind} seed`),
        body_text: [
          asText(entry.snippet || entry.summary || entry.detail || entry.text),
          asText(entry.url),
        ].filter(Boolean).join("\n"),
        url: asText(entry.url),
        attributes: cloneJson(entry, {}),
        categories: { seed_source: sourceKind },
      });
    }
  };

  pushNoteSeeds(run?.researchNotes, "research_note");
  pushNoteSeeds(run?.browserNotes, "browser_note");

  if (bootstrapHints?.starterRow) {
    const starterRow = bootstrapHints.starterRow;
    pushSeed({
      id: "bootstrap-starter-row",
      title: `${run?.craft?.name || "Craft"} bootstrap`,
      body_text: [
        Array.isArray(starterRow.messages) ? `Messages: ${JSON.stringify(starterRow.messages)}` : "",
        Array.isArray(starterRow.tools) ? `Tools: ${JSON.stringify(starterRow.tools)}` : "",
      ].filter(Boolean).join("\n"),
      attributes: {
        messages: Array.isArray(starterRow.messages) ? cloneJson(starterRow.messages, []) : [],
        tools: Array.isArray(starterRow.tools) ? cloneJson(starterRow.tools, []) : [],
        target_turn_index: Number.isInteger(starterRow.targetTurnIndex) ? starterRow.targetTurnIndex : null,
      },
      categories: {
        seed_source: "bootstrap",
        mode: "multiturn",
      },
    });
  }

  if (!out.length) {
    pushSeed({
      id: "craft-summary-seed",
      title: run?.craft?.name || "Craft summary",
      body_text: [
        asText(objective),
        asText(run?.craft?.summary),
        asText(run?.brief),
      ].filter(Boolean).join("\n"),
      attributes: {
        craft_name: asText(run?.craft?.name),
        craft_summary: asText(run?.craft?.summary),
      },
      categories: {
        seed_source: "craft_summary",
      },
    });
  }

  return out.slice(0, 32);
}

function buildTrainingSampleDedupKey(sample) {
  const normalized = normalizeTrainingSample(sample, 0);
  if (hasStructuredTrainingTrace(normalized)) {
    return JSON.stringify({
      messages: normalized.messages,
      tools: normalized.tools,
      targetTurnIndex: normalized.targetTurnIndex,
      split: normalized.split,
    });
  }
  return JSON.stringify({
    promptText: normalized.promptText,
    targetTurnIndex: normalized.targetTurnIndex,
    split: normalized.split,
  });
}

function mergeTrainingBatchSamples(existingSamples = [], incomingSamples = []) {
  const merged = (Array.isArray(existingSamples) ? existingSamples : []).map((sample, index) =>
    normalizeTrainingSample(sample, index),
  );
  const seen = new Set(merged.map((sample) => buildTrainingSampleDedupKey(sample)));
  let addedCount = 0;
  let skippedCount = 0;

  for (const sample of Array.isArray(incomingSamples) ? incomingSamples : []) {
    const normalized = normalizeTrainingSample(sample, merged.length + addedCount);
    const key = buildTrainingSampleDedupKey(normalized);
    if (seen.has(key)) {
      skippedCount += 1;
      continue;
    }
    seen.add(key);
    merged.unshift(normalized);
    addedCount += 1;
  }

  return {
    samples: merged,
    addedCount,
    skippedCount,
  };
}

function buildTrainingBatchBridgeRequest(run, args, snapshot, bootstrapHints) {
  const craftId = asText(run?.craftId || run?.craft?.id);
  const objective = asText(args?.objective || run?.brief || run?.craft?.summary || run?.craft?.name);
  const toolInterfaces = buildTrainingBatchToolInterfaces(snapshot, craftId, bootstrapHints);
  const bundleSkills = Array.isArray(snapshot?.bundle?.browserCapabilities?.payload?.skills)
    ? snapshot.bundle.browserCapabilities.payload.skills
    : [];
  const seedRecords = buildTrainingBatchSeedRecords(run, objective, bootstrapHints);
  const requestedJobKind = asText(args?.jobKind).toLowerCase();
  const defaultJobKind = run?.samples?.length >= 12 ? "pair_generation_pilot" : "pair_generation_canary";
  const jobKind = [
    "pair_generation_probe",
    "pair_generation_canary",
    "pair_generation_pilot",
    "augmentation_canary",
    "augmentation_pilot",
  ].includes(requestedJobKind)
    ? requestedJobKind
    : defaultJobKind;
  const rawMaxItems = Number(args?.maxItems);
  const maxItems = Number.isFinite(rawMaxItems) && rawMaxItems > 0
    ? Math.max(1, Math.min(256, Math.floor(rawMaxItems)))
    : (jobKind === "pair_generation_pilot" || jobKind === "augmentation_pilot" ? 64 : 24);
  return {
    craftId,
    objective,
    jobKind,
    maxItems,
    modelRef: asText(args?.modelRef),
    notes: asText(args?.notes || run?.brief),
    toolInterfaces,
    seedRecords,
    taskSpec: {
      taskName: asText(run?.craft?.name) || "Craft batch generation",
      taskGoal: objective || "Generate grounded training rows for the current craft.",
      taskType: toolInterfaces.length ? "tool_routing" : "generation",
      inputSchema: "{ prompt_text: string }",
      outputSchema: toolInterfaces.length
        ? "{ tool_name: string, arguments: object }"
        : inferTrainingOutputSchema(run?.samples),
      successMetric: "adapt_test_acc",
      acceptanceThreshold: 0.55,
      locale: "de-DE",
      guardrails: [
        "Only use reviewed tool signatures and grounded seed facts.",
        "Keep the rows generic to the craft instead of memorizing one source phrase.",
        "When visible browser state determines the next action, preserve that evidence in multimodal rows instead of flattening it into text-only context.",
      ],
      stopConditions: [
        "task_completed",
        "missing_required_context",
      ],
      preferredAugmentationIntensity: jobKind.startsWith("augmentation")
        ? "balanced"
        : "minimal",
    },
    categoryPlan: {
      categoryFields: [
        {
          name: "seed_source",
          values: uniqueTextList(seedRecords.map((entry) => entry?.categories?.seed_source), 12),
          exhaustive: true,
        },
        {
          name: "input_mode",
          values: uniqueTextList([run?.craft?.inputMode || "unknown"], 4),
          exhaustive: true,
        },
      ],
    },
    fewShotPolicy: {
      dynamicSampling: true,
      minShots: 0,
      maxShots: run?.samples?.length ? 2 : 0,
      allowCrossCategoryStructureReference: true,
    },
    samplingPlan: {
      mode: "random",
      categoryFields: ["seed_source", "input_mode"],
      randomizationSeed: `${craftId || "craft"}:${jobKind}:${Date.now()}`,
      includeCounterexamples: false,
      includeNearNeighbors: true,
    },
    passChain: [{
      id: "craft-batch-pass",
      kind: "generate",
      systemPrompt: [
        objective ? `Objective: ${objective}` : "",
        bundleSkills.length ? `Bundle skill hints:\n${bundleSkills.join("\n")}` : "",
        bootstrapHints?.reviewedPortablePattern
          ? `Reviewed workflow: ${bootstrapHints.reviewedPortablePattern}`
          : "",
        "Return grounded training rows only. Keep the task contract stable while varying the user phrasing.",
        "If the reviewed next step depends on screenshot-visible UI state, generate native Qwen multimodal rows with text plus image content so the model learns the vision path.",
        toolInterfaces.length
          ? "For tool-use tasks, supervise the next assistant turn against the reviewed tool list."
          : "For plain supervision tasks, return schema-bound outputs only.",
      ].filter(Boolean).join("\n\n"),
    }],
  };
}

async function executeGenerateTrainingBatchTool(run, args = {}) {
  const craftId = asText(run?.craftId || run?.craft?.id);
  if (!craftId) {
    return {
      ok: false,
      summary: "Batch training data could not be generated because no craft ID is available.",
      error: "Missing craftId.",
      errorDetail: {
        reason: "missing_craft_id",
      },
    };
  }

  const snapshot = await readCraftBundleSnapshot(run.craft);
  const bootstrapHints = buildCraftBootstrapHints(run, args);
  const request = buildTrainingBatchBridgeRequest(run, args, snapshot, bootstrapHints);
  if (!request.seedRecords.length) {
    return {
      ok: false,
      summary: "Batch training data could not be generated because no reliable seed data was available.",
      error: "Missing seed records.",
      errorDetail: {
        reason: "missing_seed_records",
        jobKind: asText(request?.jobKind),
      },
    };
  }

  let bridgeResult;
  try {
    bridgeResult = await fetchLocalTrainingBridge("/api/model-factory/generate-training-batch", {
      method: "POST",
      body: JSON.stringify(request),
    });
  } catch (error) {
    return {
      ok: false,
      summary: "The local training-batch generator is unavailable.",
      error: asText(error?.message || error) || "Local training bridge unavailable.",
      errorDetail: cloneStructuredErrorDetail(error && typeof error === "object" && "detail" in error ? error.detail : null),
      errorStack: readStructuredErrorStack(error),
      report: {
        currentState: "The local batch generator could not be reached.",
        nextAction: "Start the local bridge process or check whether http://127.0.0.1:8765 is available.",
      },
      provenance: [],
    };
  }

  const now = new Date().toISOString();
  const sourceLabel = [
    "model_factory_batch",
    asText(request.jobKind),
    asText(bridgeResult?.artifacts?.generatedPairs?.artifactId),
  ].filter(Boolean).join(":");
  const incomingSamples = (Array.isArray(bridgeResult?.rows) ? bridgeResult.rows : [])
    .map((row, index) =>
      createTrainingSampleDraft(index, {
        id: asText(row?.pair_id) || `batch-row-${Date.now()}-${index + 1}`,
        promptText: asText(row?.prompt_text),
        expectedJsonText: "",
        messages: Array.isArray(row?.messages) ? cloneJson(row.messages, []) : [],
        tools: Array.isArray(row?.tools) ? cloneJson(row.tools, []) : [],
        targetTurnIndex: Number.isInteger(row?.target_turn_index) ? row.target_turn_index : null,
        split: normalizeTrainingSampleSplit(args?.split || "train"),
        status: normalizeTrainingSampleStatus(args?.status || "review"),
        source: sourceLabel,
        createdAt: now,
        updatedAt: now,
      }))
    .filter((sample) => validateTrainingSample(sample).runnable);

  const mergeResult = mergeTrainingBatchSamples(run.samples, incomingSamples);
  if (mergeResult.addedCount > 0) {
    run.samples = await writeTrainingSamplesForCraft(craftId, mergeResult.samples);
  }

  const totalRows = Array.isArray(bridgeResult?.rows) ? bridgeResult.rows.length : 0;
  return {
    ok: true,
    summary: mergeResult.addedCount
      ? `The batch generator produced ${totalRows} training rows; ${mergeResult.addedCount} were imported locally.`
      : `The batch generator produced ${totalRows} training rows, but no additional new row was imported.`,
    data: {
      generatedRows: totalRows,
      importedRows: mergeResult.addedCount,
      skippedDuplicates: mergeResult.skippedCount,
      stats: bridgeResult?.stats || {},
      coverage: bridgeResult?.coverage || null,
      recipeWarnings: Array.isArray(bridgeResult?.recipeWarnings) ? bridgeResult.recipeWarnings : [],
      samples: summarizeTrainingSamples(
        mergeResult.addedCount ? mergeResult.samples.slice(0, 8) : run.samples.slice(0, 8),
        8,
      ),
      jobKind: request.jobKind,
      seedCount: request.seedRecords.length,
      toolCount: request.toolInterfaces.length,
    },
    report: {
      currentState: mergeResult.addedCount
        ? `${mergeResult.addedCount} new training rows were generated from existing evidence and saved via the local batch generator.`
        : "The local batch generator did not produce a distinct new training row.",
      nextAction: mergeResult.addedCount
        ? "Inspect the updated training data or start a training run next."
        : "If the batch output stayed too close to the current data, refine the reviewed capabilities, seeds, or objective first.",
      matchingSignals: [
        `${totalRows} erzeugt`,
        `${mergeResult.addedCount} importiert`,
        request.toolInterfaces.length ? `${request.toolInterfaces.length} reviewed Tools` : "",
      ].filter(Boolean),
    },
    provenance: normalizeProvenance([
      {
        title: "Batch training data generated",
        detail: trimText(
          `${mergeResult.addedCount}/${totalRows || 0} neue Zeilen · ${request.jobKind} · ${request.seedRecords.length} Seeds`,
          220,
        ),
        kind: "operation",
      },
    ]),
  };
}

function buildCodexRepairBridgeRequest(run, args = {}) {
  const codingHandoff =
    run?.workspaceCodeDive?.codingHandoff && typeof run.workspaceCodeDive.codingHandoff === "object"
      ? run.workspaceCodeDive.codingHandoff
      : null;
  const prompt =
    asText(args?.prompt) ||
    asText(codingHandoff?.prompt) ||
    "";
  const cwd =
    asText(args?.cwd) ||
    asText(codingHandoff?.repoPathHint) ||
    WORKSPACE_REPO_ROOT;
  return {
    jobId: "",
    cwd,
    prompt,
    model: asText(args?.model),
    metadata: {
      craftId: asText(run?.craftId || run?.craft?.id),
      agentRunId: asText(run?.jobId),
      objective: trimText(asText(run?.brief || run?.report?.objective), 240),
      failingTool: asText(run?.lastToolFailure?.action || codingHandoff?.failingTool),
      errorText: trimText(asText(run?.lastToolFailure?.error || codingHandoff?.errorText || run?.error), 800),
    },
  };
}

async function executeRequestCodexRepairTool(run, args = {}) {
  const request = buildCodexRepairBridgeRequest(run, args);
  if (!asText(request.prompt)) {
    return {
      ok: false,
      summary: "The local Codex repair run could not be started because no patchable workspace diagnosis exists yet.",
      error: "Missing coding handoff prompt.",
      errorDetail: {
        reason: "missing_coding_handoff_prompt",
      },
      report: {
        currentState: "There is no reliable patch handoff for the local Codex repair run yet.",
        nextAction: "Run workspace_code_dive first or provide an explicit patch prompt.",
      },
    };
  }

  let bridgeResult;
  try {
    bridgeResult = await fetchLocalTrainingBridge("/api/codex-repair/jobs", {
      method: "POST",
      body: JSON.stringify(request),
      timeoutMs: 15_000,
    });
  } catch (error) {
    return {
      ok: false,
      summary: "The local Codex repair service is unreachable.",
      error: asText(error?.message || error) || "Local Codex repair bridge unavailable.",
      errorDetail: cloneStructuredErrorDetail(error && typeof error === "object" && "detail" in error ? error.detail : null),
      errorStack: readStructuredErrorStack(error),
      report: {
        currentState: "The local Codex repair service could not be reached.",
        nextAction: "Start the Python repair service and then check again whether the local bridge path is available.",
      },
      provenance: [],
    };
  }

  const job = bridgeResult?.job && typeof bridgeResult.job === "object" ? bridgeResult.job : null;
  const jobId = asText(job?.jobId);
  return {
    ok: Boolean(jobId),
    summary: jobId
      ? "The local Codex repair run was started."
      : "The local Codex repair run could not be confirmed.",
    data: {
      job: job ? cloneJson(job, null) : null,
    },
    report: {
      currentState: jobId
        ? "A local Codex repair run is now patching the workspace based on the narrowed diagnosis."
        : "The local Codex repair run returned no valid job ID.",
      nextAction: jobId
        ? "Query the job status and, after successful completion, rerun the same smoke or eval path."
        : "Check the local repair service and then start the repair run again.",
      matchingSignals: [
        jobId ? `Repair-Job ${jobId}` : "",
        asText(job?.status),
      ].filter(Boolean),
    },
    provenance: normalizeProvenance(jobId ? [
      {
        title: "Started local Codex repair run",
        detail: trimText(`${jobId} · ${asText(job?.status || "queued")} · ${asText(job?.cwd)}`, 220),
        kind: "operation",
      },
    ] : []),
  };
}

async function executeGetCodexRepairStatusTool(run, args = {}) {
  const explicitJobId = asText(args?.jobId);
  const fallbackJobId = asText(run?.latestCodexRepairJobId);
  const jobId = explicitJobId || fallbackJobId;
  if (!jobId) {
    return {
      ok: false,
      summary: "The local Codex repair status could not be queried because no job ID is known yet.",
      error: "Missing Codex repair job id.",
      errorDetail: {
        reason: "missing_codex_repair_job_id",
      },
      report: {
        currentState: "No local Codex repair job is registered yet.",
        nextAction: "Start request_codex_repair first or provide an explicit job ID.",
      },
    };
  }

  const waitForMs = Math.max(0, Math.min(MAX_CODEX_REPAIR_STATUS_WAIT_MS, Number(args?.waitForMs || 0) || 0));
  const pollIntervalMs = Math.max(500, Math.min(10_000, Number(args?.pollIntervalMs || 2_000) || 2_000));
  const deadline = Date.now() + waitForMs;
  let bridgeResult = null;
  while (true) {
    try {
      bridgeResult = await fetchLocalTrainingBridge(`/api/codex-repair/jobs/${encodeURIComponent(jobId)}`, {
        method: "GET",
        timeoutMs: Math.min(8_000, Math.max(4_000, waitForMs || 8_000)),
      });
    } catch (error) {
      const errorMessage = asText(error?.message || error);
      if (/timed out/i.test(errorMessage)) {
        return {
          ok: true,
          summary: "The local Codex repair run is still running.",
          data: {
            job: {
              jobId,
              status: "running",
              summary: "The local status query is not responding right now; the repair run will be checked again on the next poll.",
            },
          },
          report: {
            currentState:
              "The local status query is not responding right now. The Codex repair run is being treated as still running for now.",
            nextAction:
              "Wait for the next status poll; afterward the same runtime path will automatically continue or be marked cleanly as failed.",
            matchingSignals: [jobId, "Statusabfrage-Timeout"],
          },
        };
      }
      return {
        ok: false,
        summary: "The local Codex repair status could not be read.",
        error: asText(error?.message || error) || "Local Codex repair status unavailable.",
        errorDetail: cloneStructuredErrorDetail(error && typeof error === "object" && "detail" in error ? error.detail : null),
        errorStack: readStructuredErrorStack(error),
        report: {
          currentState: "The local Codex repair service is not responding to the status query.",
          nextAction: "Check the repair service or restart the job afterward.",
        },
      };
    }
    const job = bridgeResult?.job && typeof bridgeResult.job === "object" ? bridgeResult.job : null;
    const status = asText(job?.status).toLowerCase();
    if (!waitForMs || ["completed", "failed"].includes(status) || Date.now() >= deadline) {
      const completed = status === "completed";
      const failed = status === "failed";
      return {
        ok: !failed,
        summary: completed
          ? "The local Codex repair run is complete."
          : failed
            ? "The local Codex repair run failed."
            : "The local Codex repair run is still running.",
        data: {
          job: job ? cloneJson(job, null) : null,
        },
        report: {
          currentState: completed
            ? "The local Codex repair run updated the workspace."
            : failed
              ? "The local Codex repair run ended with an error."
              : "The local Codex repair run is still working on the workspace patch.",
          nextAction: completed
            ? "Reload the updated artifact and then rerun the same smoke or eval path."
            : failed
              ? "Review the repair-run feedback and then decide whether another repair attempt or a manual patch is needed."
              : "Keep waiting for the job or poll the status again later.",
          matchingSignals: [
            jobId,
            asText(job?.status),
            Number.isFinite(Number(job?.exitCode)) ? `exit ${Number(job.exitCode)}` : "",
          ].filter(Boolean),
        },
        provenance: normalizeProvenance(completed ? [
          {
            title: "Local Codex repair run completed",
            detail: trimText(`${jobId} · ${asText(job?.status)} · ${asText(job?.summary || job?.lastMessage)}`, 220),
            kind: "operation",
          },
        ] : []),
      };
    }
    await sleepMs(Math.min(pollIntervalMs, Math.max(0, deadline - Date.now())));
  }
}

async function executeCodexRepairLifecycleTool(run, action, args = {}) {
  const meta = getCraftingAgentToolLogMetadata(action);
  pushRunLog(run, "info", `Tool starts: ${action}.`, {
    kind: "tool",
    toolName: action,
    title: meta.title,
    detail: meta.startDetail,
    stageId: meta.stageId,
    status: "running",
  });

  let result;
  try {
    result =
      action === "request_codex_repair"
        ? await executeRequestCodexRepairTool(run, args)
        : await executeGetCodexRepairStatusTool(run, args);
  } catch (error) {
    result = buildToolExecutionFailureResult(action, error);
  }
  await applyToolOutcomeToRun(run, action, args, result);
  pushRunLog(
    run,
    result?.ok === false ? "error" : "success",
    `${action}: ${asText(result?.summary || result?.error || meta.startDetail)}`,
    {
      kind: "tool",
      toolName: action,
      title: meta.title,
      detail: asText(result?.summary || result?.error || meta.startDetail),
      stageId: meta.stageId,
      status: result?.ok === false ? "error" : "done",
    },
  );
  return result;
}

async function isAutomaticCodexRepairEnabled() {
  return await isDevModeEnabled();
}

async function maybeRunAutomaticCodexRepair(run, finalOutput = null, effectiveStatus = "") {
  const normalizedStatus = asText(effectiveStatus).toLowerCase();
  if (!["blocked", "continue", "failed"].includes(normalizedStatus)) return null;
  if (!(await isAutomaticCodexRepairEnabled())) return null;
  const existingPendingRepair = ensureCodexRepairState(run);
  if (asText(existingPendingRepair.pendingJobId)) {
    return {
      nextStatus: "continue",
      reportOverride: {
        currentState:
          "The local Codex repair run is still working on the workspace patch.",
        nextAction:
          "Keep waiting for the job or poll the status again later.",
      },
      responseText:
        "The local Codex repair run is still working on the workspace patch.",
    };
  }
  const lastFailureReason = asText(run?.lastToolFailure?.errorDetail?.reason).toLowerCase();
  if (lastFailureReason === "missing_prepared_browser_validation_fixture") {
    return null;
  }
  const lastFailureAction = asText(run?.lastToolFailure?.action);
  if (!lastFailureAction || lastFailureAction === ASK_USER_TOOL_NAME || CODEX_REPAIR_TOOL_ACTIONS.has(lastFailureAction)) {
    return null;
  }
  const codingHandoff =
    run?.workspaceCodeDive?.codingHandoff && typeof run.workspaceCodeDive.codingHandoff === "object"
      ? run.workspaceCodeDive.codingHandoff
      : null;
  if (!codingHandoff || !asText(codingHandoff?.prompt)) return null;

  const fingerprint = buildCodexRepairFingerprint(run);
  if (!fingerprint) return null;
  if (getCodexRepairAttemptCount(run, fingerprint) >= MAX_CODEX_REPAIR_ATTEMPTS_PER_FINGERPRINT) {
    return null;
  }

  const requestArgs = {
    prompt: asText(codingHandoff?.prompt) || undefined,
    cwd: asText(codingHandoff?.repoPathHint || WORKSPACE_REPO_ROOT) || undefined,
  };
  const requestResult = await executeCodexRepairLifecycleTool(run, "request_codex_repair", requestArgs);
  const requestJob = requestResult?.data?.job && typeof requestResult.data.job === "object"
    ? requestResult.data.job
    : null;
  const requestedJobId = asText(requestJob?.jobId);

  updateCodexRepairHistory(run, fingerprint, {
    incrementAttempt: Boolean(requestedJobId),
    lastJobId: requestedJobId,
    lastStatus: asText(requestJob?.status || (requestResult?.ok === false ? "failed" : "")),
    lastSummary: asText(requestResult?.summary || requestResult?.error),
  });
  emitDevObservabilityEvent(run, "codex_repair_requested", {
    fingerprint: trimText(fingerprint, 120),
    jobId: requestedJobId,
    summary: trimText(asText(requestResult?.summary || requestResult?.error), 320),
    ok: requestResult?.ok !== false,
  });

  if (requestResult?.ok === false || !requestedJobId) {
    clearPendingCodexRepair(run);
    return {
      nextStatus: "blocked",
      reportOverride: {
        currentState:
          asText(requestResult?.report?.currentState) ||
          "The local Codex repair run could not be started.",
        nextAction:
          asText(requestResult?.report?.nextAction) ||
          "Check the Python repair service and then start the repair run again.",
      },
      responseText: asText(requestResult?.summary || requestResult?.error),
    };
  }

  setPendingCodexRepair(run, {
    jobId: requestedJobId,
    fingerprint,
  });

  const statusResult = await executeCodexRepairLifecycleTool(run, "get_codex_repair_status", {
    jobId: requestedJobId,
    waitForMs: 4_000,
    pollIntervalMs: 1_000,
  });
  const statusJob = statusResult?.data?.job && typeof statusResult.data.job === "object"
    ? statusResult.data.job
    : null;
  const status = asText(statusJob?.status).toLowerCase();
  updateCodexRepairHistory(run, fingerprint, {
    lastJobId: requestedJobId,
    lastStatus: status || asText(statusResult?.ok === false ? "failed" : ""),
    lastSummary: asText(statusResult?.summary || statusResult?.error || statusJob?.summary || statusJob?.lastMessage),
  });
  emitDevObservabilityEvent(run, "codex_repair_status", {
    fingerprint: trimText(fingerprint, 120),
    jobId: requestedJobId,
    status,
    summary: trimText(asText(statusJob?.summary || statusJob?.lastMessage || statusResult?.summary || statusResult?.error), 320),
    exitCode: Number.isFinite(Number(statusJob?.exitCode)) ? Number(statusJob.exitCode) : null,
  });

  if (status === "completed") {
    clearPendingCodexRepair(run);
    run.workspaceCodeDive = null;
    return {
      nextStatus: "continue",
      requestExtensionReload: true,
      reportOverride: {
        currentState:
          "The local Codex repair run updated the workspace. The agent now checks the same runtime path again.",
        nextAction:
          "Run the same smoke, eval, or storage path again now in the same run.",
      },
      responseText: asText(statusJob?.summary || statusJob?.lastMessage || statusResult?.summary),
    };
  }

  if (status === "queued" || status === "running" || !status) {
    return {
      nextStatus: "continue",
      reportOverride: {
        currentState:
          "A local Codex repair run is patching the workspace right now. After it completes, the agent will continue the same validation path.",
        nextAction:
          "Wait for repair completion; afterward the same smoke, eval, or storage path will be started again.",
      },
      responseText: asText(statusJob?.summary || statusJob?.lastMessage || statusResult?.summary),
    };
  }

  clearPendingCodexRepair(run);
  return {
    nextStatus: "blocked",
    reportOverride: {
      currentState:
        asText(statusResult?.report?.currentState) ||
        "The local Codex repair run failed.",
      nextAction:
        asText(statusResult?.report?.nextAction) ||
        "Review the repair-run feedback and then decide whether another repair attempt or a manual patch is needed.",
    },
    responseText: asText(statusJob?.summary || statusJob?.lastMessage || statusResult?.summary || statusResult?.error),
  };
}

async function maybeAdvancePendingCodexRepair(run) {
  const state = ensureCodexRepairState(run);
  const jobId = asText(state.pendingJobId);
  if (!jobId) return { pending: false };

  run.phase = "repairing";
  const statusResult = await executeCodexRepairLifecycleTool(run, "get_codex_repair_status", {
    jobId,
    waitForMs: AUTO_CODEX_REPAIR_STATUS_WAIT_MS,
    pollIntervalMs: AUTO_CODEX_REPAIR_POLL_INTERVAL_MS,
  });
  const job = statusResult?.data?.job && typeof statusResult.data.job === "object" ? statusResult.data.job : null;
  const status = asText(job?.status).toLowerCase();
  const fingerprint = asText(state.pendingFingerprint);
  if (fingerprint) {
    updateCodexRepairHistory(run, fingerprint, {
      lastJobId: jobId,
      lastStatus: status || asText(statusResult?.ok === false ? "failed" : ""),
      lastSummary: asText(statusResult?.summary || statusResult?.error || job?.summary || job?.lastMessage),
    });
  }

  if (status === "completed") {
    clearPendingCodexRepair(run);
    run.workspaceCodeDive = null;
    setRunReport(run, run.report, {
      currentState: "The local Codex repair run is complete. The agent is now checking the same runtime path again.",
      nextAction: "Run the same smoke, eval, or storage path again now.",
    });
    run.responseText = asText(job?.summary || job?.lastMessage || statusResult?.summary || run.responseText);
    return { pending: false, completed: true };
  }

  if (statusResult?.ok === false || status === "failed") {
    clearPendingCodexRepair(run);
    const message =
      asText(statusResult?.report?.currentState) ||
      asText(statusResult?.summary || statusResult?.error) ||
      "The local Codex repair run failed.";
    run.status = "blocked";
    run.phase = "blocked";
    run.finalStatus = "blocked";
    run.completedAt = new Date().toISOString();
    run.error = message;
    run.errorDetail = resolveRunErrorDetail(run, statusResult?.errorDetail);
    setRunReport(run, run.report, {
      currentState: message,
      nextAction:
        asText(statusResult?.report?.nextAction) ||
        "Review the repair-run feedback and then decide whether a manual patch or a new repair attempt is needed.",
    });
    pushRunLog(run, "error", message);
    return { pending: false, blocked: true };
  }

  setRunReport(run, run.report, {
    currentState:
      "A local Codex repair run continues patching the workspace. The agent is waiting for completion.",
    nextAction:
      "Wait for repair completion; afterward the same run will continue the smoke, eval, or storage path.",
  });
  run.responseText = asText(job?.summary || job?.lastMessage || statusResult?.summary || run.responseText);
  return { pending: true, waiting: true };
}

function canRequestExtensionRuntimeReload() {
  return typeof globalThis.chrome?.runtime?.reload === "function";
}

async function scheduleExtensionReloadForAutoRepair(run, state = null) {
  if (!canRequestExtensionRuntimeReload()) return false;
  if (state) {
    prepareRunForFreshSegment(
      run,
      state,
      "Workspace-Patch abgeschlossen. Die Extension wird neu geladen, damit derselbe Run mit dem aktualisierten Code fortgesetzt wird.",
    );
  } else {
    run.updatedAt = new Date().toISOString();
    pushRunLog(
      run,
      "info",
      "Workspace-Patch abgeschlossen. Die Extension wird neu geladen, damit derselbe Run mit dem aktualisierten Code fortgesetzt wird.",
    );
  }
  run.reloadResume = {
    pending: true,
    reason: "codex_repair_completed",
    requestedAt: new Date().toISOString(),
  };
  run.status = "reloading";
  run.phase = "reloading";
  run.finalStatus = "";
  run.completedAt = "";
  run.error = "";
  setRunReport(run, run.report, {
    currentState: "The extension is being reloaded after the local Codex repair.",
    nextAction: "After reload, the same agent run will automatically continue the previous validation path.",
  });
  emitDevObservabilityEvent(run, "extension_reload_requested", {
    run: buildRunObservabilitySnapshot(run),
    reason: "codex_repair_completed",
  });
  await persistRunStateNow(run);
  globalThis.setTimeout(() => {
    try {
      globalThis.chrome?.runtime?.reload?.();
    } catch (error) {
      console.warn("[crafting-agent-runner] runtime reload failed", error);
    }
  }, 60);
  return true;
}

let reloadResumePromise = null;

export async function resumePendingCraftingReloadRuns() {
  if (reloadResumePromise) return reloadResumePromise;
  reloadResumePromise = (async () => {
    await ensureRunsHydrated();
    for (const run of RUNS.values()) {
      const reloadResume = normalizeReloadResumeState(run?.reloadResume);
      if (!reloadResume.pending) continue;
      run.reloadResume = createEmptyReloadResumeState();
      run.status = "running";
      run.phase = "planning";
      run.finalStatus = "";
      run.completedAt = "";
      run.error = "";
      run.activeTab = "progress";
      run.serializedState = "";
      run.pendingApprovalCallId = "";
      run.pendingUserAnswersByCallId = {};
      setRunReport(run, run.report, {
        currentState: "The reloaded extension is resuming the agent run after the local Codex repair.",
        nextAction: "The previous smoke, eval, or save path is now being executed again.",
      });
      pushRunLog(run, "info", "Extension reload detected. The same agent run is now resuming automatically.");
      emitDevObservabilityEvent(run, "extension_reload_resumed", {
        run: buildRunObservabilitySnapshot(run),
        reason: asText(reloadResume.reason),
      });
      await persistRunStateNow(run);
      void executeRun(run);
    }
  })().finally(() => {
    reloadResumePromise = null;
  });
  return reloadResumePromise;
}

function normalizeSearchResultItem(rawResult = {}) {
  const source = rawResult && typeof rawResult === "object" ? rawResult : {};
  const url = asText(source.url || source.link || source.href);
  const title = trimText(asText(source.title || source.name || url), 180);
  const snippet = trimText(
    asText(
      source.snippet ||
        source.summary ||
        source.description ||
        source.excerpt ||
        source.text ||
        source.page_age,
    ),
    280,
  );
  if (!url || !title) return null;
  return { title, url, snippet };
}

function collectProviderWebSearchResults(value, results = [], seen = new WeakSet()) {
  if (!value || (typeof value !== "object" && !Array.isArray(value))) return results;
  if (typeof value === "object") {
    if (seen.has(value)) return results;
    seen.add(value);
  }

  if (Array.isArray(value)) {
    for (const entry of value) {
      collectProviderWebSearchResults(entry, results, seen);
    }
    return results;
  }

  const sourceArrays = [];
  if (Array.isArray(value?.sources)) sourceArrays.push(value.sources);
  if (Array.isArray(value?.result?.sources)) sourceArrays.push(value.result.sources);
  if (value?.type === "web_search_tool_result" && Array.isArray(value?.content)) {
    sourceArrays.push(value.content);
  }

  for (const sourceItems of sourceArrays) {
    for (const item of sourceItems) {
      const normalized = normalizeSearchResultItem(item);
      if (normalized) results.push(normalized);
    }
  }

  for (const nested of Object.values(value)) {
    collectProviderWebSearchResults(nested, results, seen);
  }
  return results;
}

async function executeProviderWebSearch(run, {
  query = "",
  maxResults = SEARCH_RESULT_LIMIT,
  allowedDomains = [],
} = {}) {
  const providerType = asText(run?.resolved?.provider?.type).toLowerCase();
  if (!PROVIDER_WEB_SEARCH_TYPES.has(providerType)) return null;

  try {
    const model = await getLanguageModelForSlot({
      slotId: run?.resolved?.slotId || run?.requestedSlotId || "agent",
      modelRef: run?.modelRef,
      providerId: run?.resolved?.providerId,
      modelName: run?.resolved?.modelName,
      parameters: run?.parameters,
      reasoningEffort: run?.reasoningEffort,
    });

    const response = await generateText({
      model,
      toolChoice: { type: "tool", toolName: "web_search" },
      tools: {
        web_search: webSearchTool({
          filters: Array.isArray(allowedDomains) && allowedDomains.length
            ? { allowedDomains }
            : undefined,
          searchContextSize: "medium",
        }),
      },
      messages: [
        {
          role: "system",
          content: [
            "Use the provider web_search tool to find strong public sources for the user query.",
            "After searching, answer tersely.",
          ].join("\n"),
        },
        {
          role: "user",
          content: JSON.stringify(
            {
              query,
              max_results: Math.max(1, Math.min(8, Number(maxResults || SEARCH_RESULT_LIMIT) || SEARCH_RESULT_LIMIT)),
            },
            null,
            2,
          ),
        },
      ],
    });

    const results = [];
    collectProviderWebSearchResults(response, results);
    const deduped = Array.from(new Map(results.map((entry) => [entry.url, entry])).values())
      .slice(0, Math.max(1, Math.min(8, Number(maxResults || SEARCH_RESULT_LIMIT) || SEARCH_RESULT_LIMIT)));
    if (!deduped.length) return null;
    return {
      backend: "provider_api",
      text: asText(response?.text),
      results: deduped,
    };
  } catch {
    return null;
  }
}

async function executeWebSearchTool(run, args = {}, executionOptions = {}) {
  const signal = executionOptions?.signal || null;
  const query = trimText(
    [asText(args.query), asText(args.focus)].filter(Boolean).join(" "),
    240,
  );
  if (!query) {
    return {
      ok: false,
      summary: "Web search skipped because no query was provided.",
      error: "Missing search query.",
      errorDetail: {
        reason: "missing_search_query",
      },
    };
  }

  const maxResults = Math.max(1, Math.min(8, Number(args.maxResults || SEARCH_RESULT_LIMIT) || SEARCH_RESULT_LIMIT));
  const allowedDomains = (Array.isArray(args.allowedDomains) ? args.allowedDomains : [])
    .map((entry) => asText(entry))
    .filter(Boolean)
    .slice(0, 8);
  const providerSearch = await executeProviderWebSearch(run, {
    query,
    maxResults,
    allowedDomains,
  });
  if (providerSearch?.results?.length) {
    const results = providerSearch.results;
    return {
      ok: true,
      summary: `Web search found ${results.length} candidate sources for "${query}" via provider API.`,
      data: {
        query,
        results,
        backend: providerSearch.backend,
        text: providerSearch.text || "",
      },
      report: {
        currentState: `Web search found ${results.length} public candidate sources for the craft.`,
        nextAction: "Use browser_tabs to open the best candidate, or go directly to browser_inspect/browser_action if the source is already clear.",
        matchingSignals: results.slice(0, 3).map((entry) => `${entry.title} · ${entry.url}`),
      },
      provenance: buildResearchMilestoneProvenance(query, results),
    };
  }

  const url = `${DDG_SEARCH_ENDPOINT}?q=${encodeURIComponent(query)}`;
  const response = await fetch(url, {
    method: "GET",
    headers: {
      accept: "text/html,application/xhtml+xml",
    },
    signal,
  });
  if (!response.ok) {
    return {
      ok: false,
      summary: `Web search failed with status ${response.status}.`,
      error: `DuckDuckGo HTML search returned ${response.status}.`,
      errorDetail: {
        backend: "duckduckgo_html",
        status: Number(response.status || 0) || null,
        url,
      },
    };
  }
  const html = await response.text();
  const results = extractDuckDuckGoResults(html, maxResults);
  if (!results.length) {
    return {
      ok: false,
      summary: "Web search returned no parseable results.",
      error: "No search results could be parsed.",
      errorDetail: {
        backend: "duckduckgo_html",
        url,
        resultCount: 0,
      },
    };
  }
  return {
    ok: true,
    summary: `Web search found ${results.length} candidate sources for "${query}".`,
    data: {
      query,
      results,
      backend: "duckduckgo_html",
    },
    report: {
      currentState: `Web search found ${results.length} public candidate sources for the craft.`,
      nextAction: "Use browser_tabs to open the best candidate, or go directly to browser_inspect/browser_action if the source is already clear.",
      matchingSignals: results.slice(0, 3).map((entry) => `${entry.title} · ${entry.url}`),
    },
    provenance: buildResearchMilestoneProvenance(query, results),
  };
}

function buildBrowserToolProvenance(title, detail) {
  return normalizeProvenance([
    {
      title,
      detail,
      kind: "sample",
    },
  ]);
}

async function executeBrowserInspectTool(run, args = {}, executionOptions = {}) {
  const signal = executionOptions?.signal || null;
  const session = await resolveBrowserToolSession(run, args, {
    toolLabel: "Browser inspect",
    allowBootstrapFallback: true,
  });
  if (session.kind !== "session") return session.result;

  const result = await runRestrictedBrowserInspection(session.runtime, {
    url: session.requestedUrl || session.tabContext?.url || "",
    active: args.active === true,
    baseUrl: session.baseUrl,
    allowedHosts: session.allowedHosts.length ? session.allowedHosts : collectBrowserHosts(session.baseUrl),
    question:
      asText(args.question || args.objective || args.goal) ||
      asText(run.brief) ||
      "Inspect the visible browser state and identify the next grounded action.",
    visionModelRef: asText(args.visionModelRef),
    imageDetail: asText(args.imageDetail),
    signal,
  });
  await syncRunBrowserContextFromResult(run, result);

  const uiTargets = Array.isArray(result?.data?.ui_targets) ? result.data.ui_targets : [];
  const blockers = Array.isArray(result?.data?.blockers) ? result.data.blockers : [];
  const errorDetail =
    result?.ok === false
      ? cloneStructuredErrorDetail(result?.errorDetail) || {
          backend: "vision_inspect",
          finalUrl: asText(result?.data?.url || ""),
          pageState:
            result?.data?.page_state && typeof result.data.page_state === "object"
              ? cloneJson(result.data.page_state, null)
              : null,
          screenshotMode: asText(result?.data?.screenshot_mode || ""),
          visionRaw: asText(result?.data?.vision_raw || ""),
        }
      : null;
  return {
    ok: result?.ok !== false,
    summary:
      result?.ok !== false
        ? `Browser inspect captured ${uiTargets.length} actionable UI target${uiTargets.length === 1 ? "" : "s"}${blockers.length ? ` with ${blockers.length} blocker${blockers.length === 1 ? "" : "s"}` : ""}.`
        : asText(result?.error) || "Browser inspect failed.",
    error: asText(result?.error),
    errorDetail,
    errorStack: readStructuredErrorStack(result),
    data: {
      ...(result?.data && typeof result.data === "object" ? result.data : {}),
      backend: "vision_inspect",
    },
    report: {
      currentState:
        result?.ok !== false
          ? `Browser inspect identified ${uiTargets.length} visible UI targets.`
          : "Browser inspect could not read the visible state reliably.",
      nextAction:
        result?.ok !== false
          ? "Use browser_action only for a clearly justified visible step or use playwright_ctx for deterministic DOM work."
          : "Check tab context, target URL, or switch to browser_tabs before planning more browser steps.",
      matchingSignals: [
        ...uiTargets.slice(0, 2).map((entry) => trimText([entry.text, entry.role, entry.selector_hint].filter(Boolean).join(" · "), 160)),
        ...blockers.slice(0, 1).map((entry) => trimText(entry, 160)),
      ].filter(Boolean),
    },
    provenance:
      result?.ok !== false
        ? buildBrowserToolProvenance(
            `Browser inspect grounded${asText(run?.activeTabContext?.hostname) ? ` on ${run.activeTabContext.hostname}` : ""}`,
            trimText(
              [
                asText(result?.data?.title),
                uiTargets.slice(0, 2).map((entry) => asText(entry?.text || entry?.selector_hint)).filter(Boolean).join(" | "),
                blockers.slice(0, 1).join(" | "),
              ].filter(Boolean).join(" | "),
              260,
            ),
          )
        : [],
  };
}

async function executeBrowserActionTool(run, args = {}, executionOptions = {}) {
  const signal = executionOptions?.signal || null;
  const session = await resolveBrowserToolSession(run, args, {
    toolLabel: "Browser action",
    allowBootstrapFallback: true,
  });
  if (session.kind !== "session") return session.result;

  const result = await runRestrictedBrowserAction(session.runtime, {
    url: session.requestedUrl || session.tabContext?.url || "",
    active: args.active === true,
    baseUrl: session.baseUrl,
    allowedHosts: session.allowedHosts.length ? session.allowedHosts : collectBrowserHosts(session.baseUrl),
    action: args.action,
    target: args.target,
    destination: args.destination,
    deltaX: args.deltaX,
    deltaY: args.deltaY,
    textValue: args.textValue,
    keys: args.keys,
    clear: args.clear,
    waitMs: args.waitMs,
    timeoutMs: args.timeoutMs,
    button: args.button,
    steps: args.steps,
    signal,
  });
  await syncRunBrowserContextFromResult(run, result);

  const errorDetail =
    result?.ok === false
      ? cloneStructuredErrorDetail(result?.errorDetail) || {
          backend: "crx_playwright_native",
          action: asText(args.action || result?.data?.action || ""),
          finalUrl: asText(result?.data?.url || ""),
          pageState:
            result?.data?.page_state && typeof result.data.page_state === "object"
              ? cloneJson(result.data.page_state, null)
              : null,
        }
      : null;
  return {
    ok: result?.ok !== false,
    summary:
      result?.ok !== false
        ? `Browser action ${asText(args.action || result?.data?.action || "step")} completed on ${asText(result?.data?.title || result?.data?.url || "the current tab")}.`
        : asText(result?.error) || "Browser action failed.",
    error: asText(result?.error),
    errorDetail,
    errorStack: readStructuredErrorStack(result),
    data: {
      ...(result?.data && typeof result.data === "object" ? result.data : {}),
      backend: "crx_playwright_native",
    },
    report: {
      currentState:
        result?.ok !== false
          ? `Browser action ${asText(args.action || result?.data?.action || "step")} was executed in the active tab.`
          : "Browser action failed.",
      nextAction:
        result?.ok !== false
          ? "If the visible state changed, run browser_inspect again or switch to playwright_ctx for deterministic follow-up steps."
          : "Use browser_inspect to reread the visible state before the next click or input step.",
      matchingSignals: [
        trimText(asText(result?.data?.title || result?.data?.url), 180),
      ].filter(Boolean),
    },
    provenance:
      result?.ok !== false
        ? buildBrowserToolProvenance(
            `Browser action executed${asText(run?.activeTabContext?.hostname) ? ` on ${run.activeTabContext.hostname}` : ""}`,
            trimText(
              [asText(args.action || result?.data?.action), asText(result?.data?.title), asText(result?.data?.url)]
                .filter(Boolean)
                .join(" | "),
              260,
            ),
          )
        : [],
  };
}

async function executeBrowserTabsTool(run, args = {}) {
  const runtime = ensureRunBrowserRuntime(run);
  const operation = asText(args.operation || args.action || "current").toLowerCase();

  try {
    if (operation === "list") {
      const tabs = summarizeBrowserTabs(await pTabsQuery({}), Number(args.limit || 12) || 12);
      return {
        ok: true,
        summary: `Browser tabs listed ${tabs.length} HTTP(S) tab${tabs.length === 1 ? "" : "s"}.`,
        data: {
          operation,
          tabs,
          currentTabId: Number(runtime.currentTabId || 0) || null,
        },
        report: {
          currentState: `${tabs.length} HTTP(S) tabs are available.`,
          nextAction: "Activate the correct tab or open a target URL before using browser_inspect or browser_action.",
          matchingSignals: tabs.slice(0, 3).map((entry) => trimText([entry.title, entry.url].filter(Boolean).join(" · "), 180)),
        },
        provenance: [],
      };
    }

    if (operation === "current") {
      const currentTab = await refreshRunActiveTabContext(run, runtime.currentTabId);
      return currentTab
        ? {
            ok: true,
            summary: `Current browser tab ready: ${currentTab.url}`,
            data: {
              operation,
              tab: currentTab,
            },
            report: {
              currentState: "A current HTTP(S) tab is available.",
              nextAction: "Use browser_inspect, browser_action, or playwright_ctx on this tab.",
              matchingSignals: [trimText([currentTab.title, currentTab.url].filter(Boolean).join(" · "), 180)],
            },
            provenance: [],
          }
        : {
            ok: false,
            summary: "No current HTTP(S) browser tab is available.",
            error: "No active HTTP(S) tab is available.",
            errorDetail: {
              operation,
              reason: "missing_active_http_tab",
            },
          };
    }

    if (operation === "open") {
      const url = asText(args.url);
      if (!url) {
        return {
          ok: false,
          summary: "Browser tab open skipped because no URL was provided.",
          error: "Missing target URL.",
          errorDetail: {
            operation,
            reason: "missing_target_url",
          },
        };
      }
      const created = await pTabsCreate({
        url,
        active: args.active !== false,
      });
      const createdTabId = Number(created?.id || 0);
      const fallbackContext =
        normalizeBrowserTabContext(created, url) ||
        normalizeBrowserTabContext({
          id: createdTabId || null,
          url,
          pendingUrl: url,
          title: created?.title,
          active: args.active !== false,
          windowId: created?.windowId,
        }, url);
      const context =
        await waitForBrowserTabContext(run, {
          preferredTabId: createdTabId,
          fallbackUrl: url,
          allowActiveFallback: true,
        }) ||
        syncRunBrowserContext(run, fallbackContext, {
          clearOnNull: false,
          fallbackUrl: url,
        }) ||
        fallbackContext;
      return {
        ok: true,
        summary: `Opened browser tab: ${asText(context?.url || url)}`,
        error: "",
        data: {
          operation,
          tab: context,
          contextResolved: !!context,
        },
        report: {
          currentState: context
            ? "A browser tab was opened."
            : "A browser tab was opened. Read the visible state next.",
          nextAction: "Use browser_inspect to read the visible state of this tab.",
          matchingSignals: context
            ? [trimText([context.title, context.url].filter(Boolean).join(" · "), 180)]
            : [trimText(url, 180)],
        },
        provenance: [],
      };
    }

    if (operation === "activate") {
      const tabId = Number(args.tabId || runtime.currentTabId || 0);
      if (!Number.isFinite(tabId) || tabId <= 0) {
        return {
          ok: false,
          summary: "Browser tab activation skipped because no tabId was provided.",
          error: "Missing tabId.",
          errorDetail: {
            operation,
            reason: "missing_tab_id",
          },
        };
      }
      const updated = await pTabsUpdate(tabId, { active: true });
      const fallbackUrl = asText(updated?.url || updated?.pendingUrl || run?.activeTabContext?.url);
      const fallbackContext =
        normalizeBrowserTabContext(updated, fallbackUrl) ||
        normalizeBrowserTabContext({
          id: tabId,
          url: fallbackUrl,
          pendingUrl: fallbackUrl,
          title: updated?.title,
          active: true,
          windowId: updated?.windowId,
        }, fallbackUrl);
      const context =
        await waitForBrowserTabContext(run, {
          preferredTabId: tabId,
          fallbackUrl,
          allowActiveFallback: true,
        }) ||
        syncRunBrowserContext(run, fallbackContext, {
          clearOnNull: false,
          fallbackUrl,
        }) ||
        fallbackContext;
      return {
        ok: true,
        summary: context ? `Activated browser tab: ${context.url}` : `Activated browser tab ${tabId}.`,
        error: "",
        data: {
          operation,
          tab: context,
          contextResolved: !!context,
        },
        report: {
          currentState: context ? "The target tab is now active." : "The target tab was activated.",
          nextAction: "Use browser_inspect or browser_action on this tab.",
          matchingSignals: context ? [trimText([context.title, context.url].filter(Boolean).join(" · "), 180)] : [],
        },
        provenance: [],
      };
    }

    if (operation === "close") {
      const tabId = Number(args.tabId || runtime.currentTabId || 0);
      if (!Number.isFinite(tabId) || tabId <= 0) {
        return {
          ok: false,
          summary: "Browser tab close skipped because no tabId was provided.",
          error: "Missing tabId.",
          errorDetail: {
            operation,
            reason: "missing_tab_id",
          },
        };
      }
      await pTabsRemove(tabId);
      if (Number(runtime.currentTabId || 0) === tabId) {
        runtime.currentTabId = 0;
        run.activeTabContext = null;
      }
      const currentTab = await refreshRunActiveTabContext(run, 0);
      return {
        ok: true,
        summary: `Closed browser tab ${tabId}.`,
        data: {
          operation,
          closedTabId: tabId,
          currentTab,
        },
        report: {
          currentState: `Tab ${tabId} was closed.`,
          nextAction: currentTab
            ? "Continue using the remaining current tab or open a new tab."
            : "No current HTTP(S) tab is available anymore. Open a new tab or continue with generic bootstrap work.",
          matchingSignals: currentTab ? [trimText([currentTab.title, currentTab.url].filter(Boolean).join(" · "), 180)] : [],
        },
        provenance: [],
      };
    }

    return {
      ok: false,
      summary: `Unsupported browser tab operation "${operation || "unknown"}".`,
      error: "Unsupported browser tab operation.",
      errorDetail: {
        operation,
        reason: "unsupported_browser_tab_operation",
      },
    };
  } catch (error) {
    const errorInfo = buildErrorInfo(error);
    return {
      ok: false,
      summary: "Browser tab management failed.",
      error: errorInfo.message,
      errorDetail: errorInfo.detail,
      errorStack: errorInfo.stack,
    };
  }
}

async function executePlaywrightCtxTool(run, args = {}, executionOptions = {}) {
  const signal = executionOptions?.signal || null;
  const session = await resolveBrowserToolSession(run, args, {
    toolLabel: "Playwright CTX",
    allowBootstrapFallback: true,
  });
  if (session.kind !== "session") return session.result;

  const code = asText(args.code);
  if (!code) {
    return {
      ok: false,
      summary: "Playwright CTX skipped because no code was provided.",
      error: "Missing Playwright code.",
      errorDetail: {
        reason: "missing_playwright_code",
      },
    };
  }

  const result = await runRestrictedBrowserScript(session.runtime, {
    url: session.requestedUrl || session.tabContext?.url || "",
    active: args.active === true,
    baseUrl: session.baseUrl,
    allowedHosts: session.allowedHosts.length ? session.allowedHosts : collectBrowserHosts(session.baseUrl),
    code,
    timeoutMs: args.timeoutMs,
    signal,
  });
  await syncRunBrowserContextFromResult(run, result);

  const errorDetail =
    result?.ok === false
      ? cloneStructuredErrorDetail(result?.errorDetail) || {
          backend: "playwright_ctx",
          finalUrl: asText(result?.data?.url || ""),
          pageState:
            result?.data?.page_state && typeof result.data.page_state === "object"
              ? cloneJson(result.data.page_state, null)
              : null,
          rawPreview: asText(result?.data?.raw_preview || ""),
        }
      : null;
  return {
    ok: result?.ok !== false,
    summary:
      result?.ok !== false
        ? `Playwright CTX completed in ${asText(result?.data?.title || result?.data?.url || "the current tab")}.`
        : asText(result?.error) || "Playwright CTX failed.",
    error: asText(result?.error),
    errorDetail,
    errorStack: readStructuredErrorStack(result),
    data: {
      ...(result?.data && typeof result.data === "object" ? result.data : {}),
      backend: "playwright_ctx",
    },
    report: {
      currentState:
        result?.ok !== false
          ? "Playwright CTX was executed in the current tab."
          : "Playwright CTX failed.",
      nextAction:
        result?.ok !== false
          ? "Use browser_inspect for visible verification or derive training data directly from the structured result."
          : "Check tab context or simplify the code before the next Playwright CTX run.",
      matchingSignals: [
        trimText(asText(result?.data?.raw_preview || result?.data?.title || result?.data?.url), 180),
      ].filter(Boolean),
    },
    provenance:
      result?.ok !== false
        ? buildBrowserToolProvenance(
            `Playwright CTX executed${asText(run?.activeTabContext?.hostname) ? ` on ${run.activeTabContext.hostname}` : ""}`,
            trimText([asText(result?.data?.title), asText(result?.data?.url), asText(result?.data?.raw_preview)].filter(Boolean).join(" | "), 260),
          )
        : [],
  };
}

async function executeBrowserAutomationTool(run, args = {}) {
  const inspectResult = await executeBrowserInspectTool(run, {
    ...args,
    question:
      asText(args.question || args.objective || args.goal) ||
      "Inspect the current page and collect grounded browser evidence.",
  });
  return inspectResult?.ok === false
    ? inspectResult
    : {
        ...inspectResult,
        summary:
          "Browser automation alias used. Prefer browser_inspect, browser_action, browser_tabs, or playwright_ctx for clearer tool intent.",
      };
}

function summarizeBundleSnapshotData(bundle = null) {
  const summary = bundle?.summary && typeof bundle.summary === "object" ? bundle.summary : {};
  const toolScripts = Array.isArray(bundle?.toolScripts?.payload?.scripts)
    ? bundle.toolScripts.payload.scripts
    : [];
  const capabilities = Array.isArray(bundle?.browserCapabilities?.payload?.capabilities)
    ? bundle.browserCapabilities.payload.capabilities
    : [];
  return {
    artifactIds: {
      trainingData: asText(bundle?.trainingData?.artifactId),
      toolScripts: asText(bundle?.toolScripts?.artifactId),
      browserCapabilities: asText(bundle?.browserCapabilities?.artifactId),
      weights: asText(bundle?.weights?.artifactId),
      policy: asText(bundle?.policy?.artifactId),
    },
    counts: {
      toolScripts: Number(summary.toolScriptCount || 0),
      browserCapabilities: Number(summary.browserCapabilityCount || 0),
      trainingSamples: Number(summary.sampleCount || 0),
    },
    availability: {
      hasWeights: Boolean(summary.hasAdapter),
      hasPolicy: Boolean(summary.hasPolicy),
    },
    toolScripts: toolScripts.slice(0, 8).map((entry) => ({
      id: asText(entry?.id),
      name: asText(entry?.name),
      reviewStatus: asText(entry?.reviewStatus),
      updatedAt: asText(entry?.updatedAt),
    })),
    browserCapabilities: capabilities.slice(0, 8).map((entry) => ({
      id: asText(entry?.id),
      name: asText(entry?.name),
      toolName: asText(entry?.toolName),
      version: asText(entry?.version),
    })),
  };
}

async function executeInspectBundleSnapshotTool(run) {
  const snapshot = await readCraftBundleSnapshot(run.craft);
  const bundle = snapshot?.bundle || buildCapabilityBundle({ craft: run.craft });
  const summary = bundle?.summary && typeof bundle.summary === "object" ? bundle.summary : {};
  return {
    ok: true,
    summary: `Bundle snapshot loaded: ${Number(summary.sampleCount || 0)} samples, ${Number(summary.toolScriptCount || 0)} tool scripts, ${Number(summary.browserCapabilityCount || 0)} browser functions.`,
    data: summarizeBundleSnapshotData(bundle),
    report: {
      currentState: `Bundle snapshot loaded: ${Number(summary.toolScriptCount || 0)} tool scripts and ${Number(summary.browserCapabilityCount || 0)} browser functions are locally available.`,
      nextAction: "Save tool scripts or browser functions before starting the agentic smoke.",
      matchingSignals: [
        `${Number(summary.sampleCount || 0)} training samples`,
        `${Number(summary.toolScriptCount || 0)} tool scripts`,
        `${Number(summary.browserCapabilityCount || 0)} browser functions`,
      ],
    },
    provenance: [],
  };
}

async function executeSaveToolScriptsTool(run, args = {}) {
  const craftId = asText(run.craftId || run?.craft?.id);
  if (!craftId) {
    return {
      ok: false,
      summary: "Tool scripts could not be saved because no craft ID is available.",
      error: "Missing craftId.",
      errorDetail: {
        reason: "missing_craft_id",
      },
    };
  }
  if (!craftSync?.putLocalArtifact) {
    return {
      ok: false,
      summary: "Tool scripts could not be saved because the local artifact store is unavailable.",
      error: "Local artifact store unavailable.",
      errorDetail: {
        reason: "local_artifact_store_unavailable",
      },
    };
  }

  return await withCraftArtifactWriteBarrier(craftId, async () => {
    const snapshot = await readCraftBundleSnapshot(run.craft);
    const basePayload = args.replaceAll === true
      ? { schemaVersion: 1, scripts: [], declaredTools: [] }
      : normalizeToolScriptsPayload(
          snapshot?.toolScriptsRecord?.payload || null,
          run.craft,
          STRICT_RUNTIME_TOOL_SCRIPTS_OPTIONS,
        );
    const incomingScripts = Array.isArray(args.scripts) ? args.scripts : [];
    const mergedScripts = mergeRecordsById(basePayload.scripts, incomingScripts);
    const declaredTools = uniqueTextList(
      [
        ...(Array.isArray(basePayload.declaredTools) ? basePayload.declaredTools : []),
        ...(Array.isArray(args.declaredTools) ? args.declaredTools : []),
        ...mergedScripts.map((entry) => asText(entry?.name)),
      ],
      64,
    );
    const payload = normalizeToolScriptsPayload({
      ...basePayload,
      scripts: mergedScripts,
      declaredTools,
    }, run.craft, STRICT_RUNTIME_TOOL_SCRIPTS_OPTIONS);

    const record = await craftSync.putLocalArtifact({
      id: getToolScriptsArtifactId(craftId),
      craftId,
      kind: TOOL_SCRIPTS_ARTIFACT_KIND,
      payload,
      meta: {
        scriptCount: Array.isArray(payload?.scripts) ? payload.scripts.length : 0,
        updatedAt: Date.now(),
        updatedBy: "crafting_agent",
      },
    });
    let browserRuntimeRepublished = false;
    let browserRuntimeStatus = "";
    const currentBrowserCapabilitiesPayload =
      snapshot?.browserCapabilitiesRecord?.payload && typeof snapshot.browserCapabilitiesRecord.payload === "object"
        ? normalizeBrowserCapabilityBundlePayload(
            snapshot.browserCapabilitiesRecord.payload,
            run.craft,
            payload,
            STRICT_RUNTIME_BROWSER_CAPABILITY_OPTIONS,
          )
        : null;
    if (currentBrowserCapabilitiesPayload) {
      const compiledBrowserCapabilitiesPayload = compilePublishedBrowserCapabilityBundlePayload(
        currentBrowserCapabilitiesPayload,
        {
          craft: run.craft,
          toolScriptsPayload: payload,
          publishedBy: "crafting_agent",
        },
      );
      if (!compiledBrowserCapabilitiesPayload.ok) {
        browserRuntimeStatus =
          compiledBrowserCapabilitiesPayload.error ||
          "Browser function package could not be republished.";
      } else {
        await craftSync.putLocalArtifact({
          id: getBrowserCapabilityBundleArtifactId(craftId),
          craftId,
          kind: BROWSER_CAPABILITY_BUNDLE_ARTIFACT_KIND,
          payload: compiledBrowserCapabilitiesPayload.payload,
          meta: {
            capabilityCount: Array.isArray(compiledBrowserCapabilitiesPayload.payload?.capabilities)
              ? compiledBrowserCapabilitiesPayload.payload.capabilities.length
              : 0,
            updatedAt: Date.now(),
            updatedBy: "crafting_agent",
            republishedFrom: TOOL_SCRIPTS_ARTIFACT_KIND,
          },
        });
        browserRuntimeRepublished = true;
      }
    }

    return {
      ok: true,
      summary: `${Array.isArray(payload?.scripts) ? payload.scripts.length : 0} tool scripts saved locally.`,
      data: {
        artifactId: asText(record?.id) || getToolScriptsArtifactId(craftId),
        payload,
        scriptCount: Array.isArray(payload?.scripts) ? payload.scripts.length : 0,
        updatedAt: asText(record?.updatedAt),
      },
      report: {
        currentState: browserRuntimeRepublished
          ? `${Array.isArray(payload?.scripts) ? payload.scripts.length : 0} tool scripts are saved and the browser function package was republished.`
          : `${Array.isArray(payload?.scripts) ? payload.scripts.length : 0} tool scripts are now saved locally.`,
        nextAction: browserRuntimeStatus
          ? "Review and save the browser functions again next so the runtime package becomes valid again."
          : "Save the browser functions next or start the agentic smoke directly.",
        matchingSignals: (Array.isArray(payload?.scripts) ? payload.scripts : [])
          .slice(0, 3)
          .map((entry) => asText(entry?.name))
          .filter(Boolean),
      },
      provenance: [
        {
          title: "Tool scripts saved",
          detail: trimText(
            (Array.isArray(payload?.scripts) ? payload.scripts : [])
              .slice(0, 3)
              .map((entry) => asText(entry?.name || entry?.id))
              .filter(Boolean)
              .join(" | "),
            240,
          ),
          kind: "match",
        },
        browserRuntimeStatus
          ? {
              title: "Browser function package not republished",
              detail: trimText(browserRuntimeStatus, 240),
              kind: "constraint",
            }
          : null,
      ].filter(Boolean),
    };
  });
}

async function executeSaveBrowserCapabilitiesTool(run, args = {}) {
  const craftId = asText(run.craftId || run?.craft?.id);
  if (!craftId) {
    return {
      ok: false,
      summary: "Browser functions could not be saved because no craft ID is available.",
      error: "Missing craftId.",
      errorDetail: {
        reason: "missing_craft_id",
      },
    };
  }
  if (!craftSync?.putLocalArtifact) {
    return {
      ok: false,
      summary: "Browser functions could not be saved because the local artifact store is unavailable.",
      error: "Local artifact store unavailable.",
      errorDetail: {
        reason: "local_artifact_store_unavailable",
      },
    };
  }

  return await withCraftArtifactWriteBarrier(craftId, async () => {
    const snapshot = await readCraftBundleSnapshot(run.craft);
    const toolScriptsPayload = normalizeToolScriptsPayload(
      snapshot?.toolScriptsRecord?.payload || null,
      run.craft,
      STRICT_RUNTIME_TOOL_SCRIPTS_OPTIONS,
    );
    const basePayload = args.replaceAll === true
      ? { schemaVersion: 1, actionProtocolVersion: 1, capabilities: [], resources: [], skills: [] }
      : normalizeBrowserCapabilityBundlePayload(
          snapshot?.browserCapabilitiesRecord?.payload || null,
          run.craft,
          toolScriptsPayload,
          STRICT_RUNTIME_BROWSER_CAPABILITY_OPTIONS,
        );
    const mergedCapabilities = mergeRecordsById(basePayload.capabilities, Array.isArray(args.capabilities) ? args.capabilities : []);
    const payload = buildPublishedBrowserCapabilitiesPayload(
      {
        ...basePayload,
        capabilities: mergedCapabilities,
        resources: uniqueTextList([...(basePayload.resources || []), ...(Array.isArray(args.resources) ? args.resources : [])], 64),
        skills: uniqueTextList([...(basePayload.skills || []), ...(Array.isArray(args.skills) ? args.skills : [])], 48),
      },
      run.craft,
      toolScriptsPayload,
      {
        publishedBy: "crafting_agent",
      },
    );

    const record = await craftSync.putLocalArtifact({
      id: getBrowserCapabilityBundleArtifactId(craftId),
      craftId,
      kind: BROWSER_CAPABILITY_BUNDLE_ARTIFACT_KIND,
      payload,
      meta: {
        capabilityCount: Array.isArray(payload?.capabilities) ? payload.capabilities.length : 0,
        updatedAt: Date.now(),
        updatedBy: "crafting_agent",
      },
    });

    return {
      ok: true,
      summary: `${Array.isArray(payload?.capabilities) ? payload.capabilities.length : 0} reviewed browser functions saved locally.`,
      data: {
        artifactId: asText(record?.id) || getBrowserCapabilityBundleArtifactId(craftId),
        payload,
        capabilityCount: Array.isArray(payload?.capabilities) ? payload.capabilities.length : 0,
        updatedAt: asText(record?.updatedAt),
      },
      report: {
        currentState: `${Array.isArray(payload?.capabilities) ? payload.capabilities.length : 0} browser functions are now available in the local bundle.`,
        nextAction: requiresPreparedActiveTextValidationFixture(run, {
          objective: (Array.isArray(payload?.capabilities) ? payload.capabilities : [])
            .map((entry) => asText(entry?.name || entry?.description || entry?.toolName))
            .filter(Boolean)
            .join(" "),
        })
          ? "First set up a short test text as a selection or in an input field in an HTTP(S) tab, inspect the visible state with browser_inspect, and then start run_agentic_smoke."
          : "Inspect the Browser Functions tab or start run_agentic_smoke to test the runtime path.",
        matchingSignals: (Array.isArray(payload?.capabilities) ? payload.capabilities : [])
          .slice(0, 3)
          .map((entry) => asText(entry?.name || entry?.id))
          .filter(Boolean),
      },
      provenance: [
        {
          title: "Browser functions saved",
          detail: trimText(
            (Array.isArray(payload?.capabilities) ? payload.capabilities : [])
              .slice(0, 3)
              .map((entry) => asText(entry?.name || entry?.id))
              .filter(Boolean)
              .join(" | "),
            240,
          ),
          kind: "match",
        },
      ],
    };
  });
}

async function executeRunAgenticSmokeTool(run, args = {}, executionOptions = {}) {
  const signal = executionOptions?.signal || null;
  pushRunLog(run, "info", "Agentic smoke is waiting for open artifact writes.");
  updateSmokeDebug("run_agentic_smoke:wait_artifacts:start", {
    craftId: asText(run?.craftId || run?.craft?.id),
  });
  await waitForCraftArtifactWrites(run?.craftId || run?.craft?.id);
  pushRunLog(run, "info", "Agentic smoke passed the artifact barrier.");
  updateSmokeDebug("run_agentic_smoke:wait_artifacts:done", {
    craftId: asText(run?.craftId || run?.craft?.id),
  });
  const prompt = asText(args.prompt || run.brief);
  if (!prompt) {
    return {
      ok: false,
      summary: "Agentic smoke was skipped because no prompt was provided.",
      error: "Missing smoke prompt.",
      errorDetail: {
        reason: "missing_smoke_prompt",
      },
    };
  }
  pushRunLog(run, "info", "Agentic smoke is preparing the active-text validation fixture.");
  updateSmokeDebug("run_agentic_smoke:prepare_fixture:start", {
    craftId: asText(run?.craftId || run?.craft?.id),
  });
  await ensurePreparedActiveTextValidationFixture(run, args).catch(() => null);
  pushRunLog(run, "info", "Agentic smoke prepared the active-text validation fixture.");
  updateSmokeDebug("run_agentic_smoke:prepare_fixture:done", {
    fixturePrepared: run?.browserValidationFixture?.prepared === true,
    tabId: Number(run?.browserValidationFixture?.tabId || 0),
    activeTabUrl: asText(run?.browserValidationFixture?.activeTabUrl),
  });
  const validationFixture = buildPreparedActiveTextValidationFixtureResult(run, "Agentic Smoke", args);
  if (validationFixture) {
    return validationFixture;
  }

  pushRunLog(run, "info", "Agentic smoke is now starting runCraftUse through the reviewed runtime path.");
  updateSmokeDebug("run_agentic_smoke:capability_eval:start", {
    craftId: asText(run?.craftId || run?.craft?.id),
  });
  const evaluation = await runCapabilityEvalCase(run, {
    prompt,
    maxTurns: args.maxTurns,
    bundleOverride: args.bundleOverride,
    signal,
  });
  updateSmokeDebug("run_agentic_smoke:capability_eval:done", {
    finalStatus: asText(evaluation?.finalStatus),
    usedReviewedCapability: evaluation?.usedReviewedCapability === true,
    underlyingFailingTool: asText(evaluation?.underlyingFailingTool),
  });
  const usedReviewedCapability = evaluation.usedReviewedCapability;
  const failed = Boolean(asText(evaluation.error)) || asText(evaluation.finalStatus).toLowerCase() === "failed";

  if (failed) {
    const errorText =
      describeLocalQwenRuntimeFailure(
        asText(evaluation.error) || "Agentic smoke failed on the reviewed runtime path.",
        evaluation.errorDetail,
      ) || "Agentic smoke failed on the reviewed runtime path.";
    return {
      ok: false,
      summary: `Agentic smoke failed: ${trimText(errorText, 220)}`,
      error: errorText,
      errorDetail: evaluation.errorDetail && typeof evaluation.errorDetail === "object"
        ? cloneJson(evaluation.errorDetail, null)
        : null,
      errorStack: trimText(evaluation.errorStack, 4_000),
      data: {
        run: evaluation.run,
        stepCount: evaluation.stepCount,
        capabilityNames: evaluation.capabilityNames,
        usedReviewedCapability,
        finalStatus: evaluation.finalStatus,
        modelRef: evaluation.modelRef,
        underlyingFailingTool: asText(evaluation.underlyingFailingTool),
      },
      report: {
        currentState: `Agentic smoke failed on the reviewed runtime path: ${trimText(errorText, 220)}`,
        nextAction: "Fix the affected reviewed tool or script path first and then restart the same smoke.",
        matchingSignals: [
          asText(evaluation.underlyingFailingTool),
          trimText(errorText, 180),
        ].filter(Boolean),
      },
      provenance: [],
    };
  }

  return {
    ok: usedReviewedCapability && evaluation.finalStatus.toLowerCase() !== "failed",
    summary: usedReviewedCapability
      ? `Agentic smoke completed: ${evaluation.capabilityNames.join(", ")}`
      : "Agentic smoke ran, but no reviewed capability was used.",
    data: {
      run: evaluation.run,
      stepCount: evaluation.stepCount,
      capabilityNames: evaluation.capabilityNames,
      usedReviewedCapability,
      finalStatus: evaluation.finalStatus,
      modelRef: evaluation.modelRef,
      underlyingFailingTool: asText(evaluation.underlyingFailingTool),
    },
    report: {
      currentState: usedReviewedCapability
        ? `Agentic smoke used ${evaluation.capabilityNames.join(", ")} through the reviewed runtime path.`
        : "Agentic smoke did not execute any reviewed capability call.",
      nextAction: usedReviewedCapability
        ? "If the behavior is correct, derive training data from it now, start run_capability_eval, or begin a training run."
        : "Review tool scripts and browser functions before starting training.",
      matchingSignals: [
        ...evaluation.capabilityNames,
        trimText(evaluation.text, 180),
      ].filter(Boolean).slice(0, 4),
    },
    provenance: usedReviewedCapability
      ? [
          {
            title: "Agentic smoke confirmed",
            detail: trimText(evaluation.capabilityNames.join(" | "), 220),
            kind: "sample",
          },
        ]
      : [],
  };
}

function sleepMs(ms = 0) {
  return new Promise((resolve) => {
    globalThis.setTimeout(resolve, Math.max(0, Number(ms || 0)));
  });
}

function formatWaitDurationMs(ms = 0) {
  const totalMs = Math.max(0, Number(ms || 0));
  if (!totalMs) return "";
  if (totalMs < 1000) return `${totalMs} ms`;
  const totalSeconds = Math.round(totalMs / 1000);
  if (totalSeconds < 60) return `${totalSeconds} s`;
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return seconds ? `${minutes} min ${seconds} s` : `${minutes} min`;
}

function getSupervisorToolTimeoutMs(action, args = {}) {
  const normalizedAction = asText(action);
  if (normalizedAction === "get_training_run_status") {
    const waitForMs = Math.max(0, Number(args?.waitForMs || 0) || 0);
    return Math.min(135_000, Math.max(15_000, waitForMs + 15_000));
  }
  const configured = Number(SUPERVISOR_TOOL_TIMEOUTS_MS[normalizedAction] || 0);
  if (Number.isFinite(configured) && configured > 0) {
    return configured;
  }
  return DEFAULT_SUPERVISOR_TOOL_TIMEOUT_MS;
}

function createSupervisorToolTimeoutError(action, timeoutMs = 0) {
  const durationLabel = formatWaitDurationMs(timeoutMs) || `${Math.max(0, Number(timeoutMs || 0))} ms`;
  const error = new Error(`${describeToolActionForUser(action)} timed out after ${durationLabel}.`);
  error.name = "TimeoutError";
  error.detail = {
    reason: "supervisor_tool_timeout",
    action: asText(action),
    timeoutMs: Math.max(0, Number(timeoutMs || 0)),
  };
  return error;
}

function isSupervisorToolTimeoutError(error) {
  return asText(error?.detail?.reason) === "supervisor_tool_timeout";
}

function buildSupervisorToolTimeoutResult(action, timeoutMs = 0) {
  const durationLabel = formatWaitDurationMs(timeoutMs) || `${Math.max(0, Number(timeoutMs || 0))} ms`;
  const userLabel = describeToolActionForUser(action);
  const timedOutAction = asText(action);
  return {
    ok: false,
    summary: `${userLabel} did not respond after ${durationLabel}.`,
    error: `${userLabel} timed out after ${durationLabel}.`,
    errorDetail: {
      reason: "supervisor_tool_timeout",
      action: timedOutAction,
      timeoutMs: Math.max(0, Number(timeoutMs || 0)),
    },
    data: {
      failingTool: timedOutAction,
      underlyingFailingTool: timedOutAction,
    },
    report: {
      currentState: `${userLabel} is hanging or no longer responding after ${durationLabel}.`,
      nextAction:
        timedOutAction === "run_agentic_smoke" || timedOutAction === "run_capability_eval"
          ? "Inspect the reviewed runtime path and the model or browser calls, fix the blocker, then rerun the same test."
          : "Inspect the hanging tool path first, then rerun the same step.",
      matchingSignals: [`${timedOutAction} timeout`, durationLabel].filter(Boolean),
    },
    provenance: [],
  };
}

function createSupervisorToolExecutionContext(run, action, args = {}) {
  const timeoutMs = getSupervisorToolTimeoutMs(action, args);
  const parentSignal = run?.abortController?.signal || null;
  const controller = typeof AbortController === "function" ? new AbortController() : null;
  let timeoutId = 0;
  let rejectTimeout = null;
  let detachParentAbort = () => {};
  const timeoutError = createSupervisorToolTimeoutError(action, timeoutMs);

  if (controller && parentSignal) {
    const onParentAbort = () => {
      if (controller.signal.aborted) return;
      try {
        controller.abort(parentSignal.reason);
      } catch {
        controller.abort();
      }
    };
    if (parentSignal.aborted) {
      onParentAbort();
    } else {
      parentSignal.addEventListener("abort", onParentAbort, { once: true });
      detachParentAbort = () => {
        parentSignal.removeEventListener("abort", onParentAbort);
      };
    }
  }

  const timeoutPromise = timeoutMs > 0
    ? new Promise((_, reject) => {
        rejectTimeout = reject;
      })
    : null;

  if (timeoutMs > 0) {
    timeoutId = globalThis.setTimeout(() => {
      if (controller && !controller.signal.aborted) {
        try {
          controller.abort(timeoutError);
        } catch {
          controller.abort();
        }
      }
      if (typeof rejectTimeout === "function") {
        rejectTimeout(timeoutError);
      }
    }, timeoutMs);
  }

  return {
    signal: controller?.signal || parentSignal || null,
    timeoutMs,
    timeoutPromise,
    cleanup() {
      detachParentAbort();
      if (timeoutId) {
        globalThis.clearTimeout(timeoutId);
      }
    },
  };
}

async function runCapabilityEvalCase(run, args = {}) {
  const snapshot = await readCraftBundleSnapshot(run.craft);
  const bundle = createBundleFromSnapshot(run.craft, snapshot, args.bundleOverride);
  const prompt = asText(args.prompt || run.brief);
  pushRunLog(run, "info", "runCraftUse is being invoked for the reviewed smoke path.");
  const smokeRun = await runCraftUse({
    craft: {
      ...(run.craft && typeof run.craft === "object" ? cloneJson(run.craft, {}) : {}),
      bundle,
    },
    prompt,
    maxTurns:
      Number.isFinite(Number(args.maxTurns)) && Number(args.maxTurns) > 0
        ? Math.max(1, Math.min(8, Number(args.maxTurns)))
        : undefined,
    signal: args.signal || null,
  });
  pushRunLog(run, "info", `runCraftUse ist mit Status ${asText(smokeRun?.status || "unknown")} zurueckgekehrt.`);
  const steps = Array.isArray(smokeRun?.steps) ? smokeRun.steps : [];
  const capabilityNames = uniqueTextList(
    steps.map((step) => asText(step?.execution?.capability?.name || step?.execution?.capability?.id)),
    12,
  );
  const underlyingFailingTool = asText(
    smokeRun?.failingTool ||
      steps[steps.length - 1]?.action?.tool_name ||
      steps[steps.length - 1]?.execution?.capability?.id ||
      steps[steps.length - 1]?.execution?.capability?.name,
  );
  const activeTextOutcome = extractCapabilityEvalActiveTextOutcome(steps);
  return {
    run: smokeRun,
    stepCount: steps.length,
    capabilityNames,
    usedReviewedCapability: capabilityNames.length > 0,
    finalStatus: asText(smokeRun?.status),
    modelRef: asText(smokeRun?.modelRef),
    text: asText(smokeRun?.text || smokeRun?.result?.text || ""),
    error: asText(smokeRun?.error),
    errorDetail:
      smokeRun?.errorDetail && typeof smokeRun.errorDetail === "object"
        ? cloneJson(smokeRun.errorDetail, null)
        : null,
    errorStack: trimText(smokeRun?.errorStack, 4_000),
    underlyingFailingTool,
    activeTextOutcome,
  };
}

function extractCapabilityEvalActiveTextOutcome(steps = []) {
  const normalizedSteps = Array.isArray(steps) ? steps : [];
  for (let index = normalizedSteps.length - 1; index >= 0; index -= 1) {
    const step = normalizedSteps[index];
    const toolName = asText(
      step?.action?.tool_name ||
        step?.execution?.capability?.id ||
        step?.execution?.capability?.name,
    );
    if (toolName !== "replace_active_text_target") continue;
    const finalOutput =
      step?.execution?.finalOutput && typeof step.execution.finalOutput === "object"
        ? step.execution.finalOutput
        : step?.result && typeof step.result === "object"
          ? step.result
          : null;
    const payload =
      finalOutput?.data && typeof finalOutput.data === "object"
        ? finalOutput.data
        : finalOutput;
    if (!payload || typeof payload !== "object") continue;
    return {
      ok: payload?.ok !== false,
      targetType: asText(payload?.targetType),
      text: asText(payload?.text),
    };
  }
  return null;
}

function buildCapabilityEvalTextHaystack(evaluation = {}) {
  const capabilityNames = uniqueTextList(evaluation?.capabilityNames || [], 12);
  const finalStatus = asText(evaluation?.finalStatus).toLowerCase();
  const activeTextOutcome =
    evaluation?.activeTextOutcome && typeof evaluation.activeTextOutcome === "object"
      ? evaluation.activeTextOutcome
      : null;
  const targetType = asText(activeTextOutcome?.targetType).toLowerCase();
  const targetTypeVariants = targetType
    ? uniqueTextList([
        targetType,
        targetType.replace(/_/g, " "),
        targetType === "focused_editable" ? "focused field" : "",
        targetType === "focused_editable" ? "focused editable" : "",
      ], 6)
    : [];
  const textFragments = [
    asText(evaluation?.text),
    asText(activeTextOutcome?.text),
    ...targetTypeVariants,
  ];
  const completedActiveTextFlow =
    finalStatus === "done" &&
    capabilityNames.includes("read_active_text_target") &&
    capabilityNames.includes("replace_active_text_target");
  if (completedActiveTextFlow) {
    textFragments.push(
      "execute completed",
      "active text updated.",
      "local reviewed tool workflow completed.",
      "replace_active_text_target completed.",
      "writeback completed.",
    );
  }
  return textFragments
    .filter(Boolean)
    .join("\n")
    .toLowerCase();
}

async function executeRunCapabilityEvalTool(run, args = {}, executionOptions = {}) {
  const signal = executionOptions?.signal || null;
  const cases = (Array.isArray(args.cases) ? args.cases : []).slice(0, 8);
  if (!cases.length) {
    return {
      ok: false,
      summary: "Capability eval was skipped because no test cases were provided.",
      error: "Missing eval cases.",
      errorDetail: {
        reason: "missing_eval_cases",
      },
    };
  }
  pushRunLog(run, "info", "Capability eval is preparing the active-text test state.");
  await ensurePreparedActiveTextValidationFixture(run, {
    ...args,
    objective: cases[0]?.prompt || run?.brief,
  }).catch(() => null);
  const validationFixture = buildPreparedActiveTextValidationFixtureResult(run, "Capability-Eval", {
    ...args,
    objective: cases[0]?.prompt || run?.brief,
  });
  if (validationFixture) {
    return validationFixture;
  }

  const results = [];
  for (const [index, entry] of cases.entries()) {
    await ensurePreparedActiveTextValidationFixture(run, {
      ...args,
      objective: asText(entry?.prompt || run?.brief),
      activeTextTargetMode: resolveRequestedActiveTextValidationTargetMode(run, {
        ...args,
        objective: asText(entry?.prompt || run?.brief),
      }),
    }).catch(() => null);
    const evaluation = await runCapabilityEvalCase(run, {
      prompt: asText(entry?.prompt || run.brief),
      maxTurns: entry?.maxTurns,
      bundleOverride: args.bundleOverride,
      signal,
    });
    const expectedCapabilityNames = uniqueTextList(entry?.expectedCapabilityNames || [], 8);
    const expectedFinalStatus = asText(entry?.expectedFinalStatus).toLowerCase();
    const expectedTextIncludes = uniqueTextList(entry?.expectedTextIncludes || [], 8);
    const textHaystack = buildCapabilityEvalTextHaystack(evaluation);
    const capabilityMatch =
      !expectedCapabilityNames.length ||
      expectedCapabilityNames.every((name) => evaluation.capabilityNames.includes(name));
    const statusMatch = !expectedFinalStatus || evaluation.finalStatus.toLowerCase() === expectedFinalStatus;
    const textMatch =
      !expectedTextIncludes.length ||
      expectedTextIncludes.every((snippet) => textHaystack.includes(asText(snippet).toLowerCase()));
    const passed =
      evaluation.usedReviewedCapability &&
      evaluation.finalStatus.toLowerCase() !== "failed" &&
      capabilityMatch &&
      statusMatch &&
      textMatch;
    results.push({
      index: index + 1,
      prompt: asText(entry?.prompt),
      capabilityNames: evaluation.capabilityNames,
      finalStatus: evaluation.finalStatus,
      usedReviewedCapability: evaluation.usedReviewedCapability,
      modelRef: evaluation.modelRef,
      text: evaluation.text,
      error: asText(evaluation.error),
      errorDetail:
        evaluation.errorDetail && typeof evaluation.errorDetail === "object"
          ? cloneJson(evaluation.errorDetail, null)
          : null,
      errorStack: trimText(evaluation.errorStack, 4_000),
      underlyingFailingTool: asText(evaluation.underlyingFailingTool),
      passed,
      expectedCapabilityNames,
      expectedFinalStatus,
      expectedTextIncludes,
    });
  }

  const passedCount = results.filter((entry) => entry.passed).length;
  const capabilityNames = uniqueTextList(results.flatMap((entry) => entry.capabilityNames), 12);
  const failedCases = results.filter((entry) => !entry.passed);
  const primaryFailure = failedCases.find((entry) => asText(entry.error)) || failedCases[0] || null;

  return {
    ok: passedCount === results.length,
    summary:
      passedCount === results.length
        ? `Capability eval passed: ${passedCount}/${results.length} cases.`
        : `Capability eval incomplete: ${passedCount}/${results.length} cases passed.`,
    error:
      passedCount === results.length
        ? ""
        : describeLocalQwenRuntimeFailure(
            asText(primaryFailure?.error) || `Capability eval still has ${failedCases.length} open or failing cases.`,
            primaryFailure?.errorDetail || null,
          ) || `Capability eval still has ${failedCases.length} open or failing cases.`,
    errorDetail:
      passedCount === results.length
        ? null
        : primaryFailure?.errorDetail && typeof primaryFailure.errorDetail === "object"
          ? cloneJson(primaryFailure.errorDetail, null)
          : {
              failedCases: failedCases.length,
              primaryFailureIndex: Number(primaryFailure?.index || 0) || null,
              underlyingFailingTool: asText(primaryFailure?.underlyingFailingTool),
            },
    errorStack: passedCount === results.length ? "" : trimText(asText(primaryFailure?.errorStack || ""), 4_000),
    data: {
      results,
      totalCases: results.length,
      passedCases: passedCount,
      failedCases: failedCases.length,
      capabilityNames,
      underlyingFailingTool: asText(primaryFailure?.underlyingFailingTool),
    },
    report: {
      currentState:
        passedCount === results.length
          ? `The capability passed all ${results.length} test cases through the reviewed runtime path.`
          : `${failedCases.length} of ${results.length} test cases are still open or failing.`,
      nextAction:
        passedCount === results.length
          ? "Use these cases as training references or start the training run directly."
          : "Revise the tools, the capability, or the training data and rerun the same cases.",
      matchingSignals: [
        ...capabilityNames,
        ...failedCases.slice(0, 2).map((entry) => `Case ${entry.index}: ${entry.finalStatus || "no status"}`),
      ].filter(Boolean).slice(0, 4),
    },
    provenance: capabilityNames.length
      ? [
          {
            title: "Capability manually validated",
            detail: trimText(`${passedCount}/${results.length} cases · ${capabilityNames.join(" | ")}`, 220),
            kind: passedCount === results.length ? "sample" : "constraint",
          },
        ]
      : [],
  };
}

async function executeStartTrainingRunTool(run, args = {}) {
  const craftId = asText(args.craftId || run.craftId);
  const resolvedTrainingModelName = await resolveLocalBrowserTrainingModelName(asText(args.modelName));
  const resolvedDatasetPayload =
    args.datasetPayload && typeof args.datasetPayload === "object"
      ? cloneJson(args.datasetPayload, null)
      : await buildTrainingDatasetPayloadForCraft(craftId, run?.samples);
  if (!resolvedDatasetPayload) {
    return {
      ok: false,
      summary: "The local training run could not start because this craft does not have valid native Qwen training rows yet.",
      error: "Missing craft training dataset.",
      errorDetail: {
        reason: "missing_craft_training_dataset",
        craftId,
      },
    };
  }
  const response = await sendMessageToOffscreen("OFFSCREEN_LOCAL_QWEN_TRAINING_START", {
    craftId,
    shardId: asText(args.shardId),
    modelName: resolvedTrainingModelName,
    datasetPayload: resolvedDatasetPayload,
    persistBundle: args.persistBundle !== false,
    smokeMode: asText(args.smokeMode),
    configOverrides: args.configOverrides && typeof args.configOverrides === "object" ? cloneJson(args.configOverrides, null) : null,
  });
  if (!response?.ok || !response?.run) {
    const errorDetail = cloneStructuredErrorDetail(response?.errorDetail);
    const trainingFailure = summarizeLocalQwenTrainingStartFailure(
      asText(response?.error) || "Training start failed.",
      errorDetail,
    );
    return {
      ok: false,
      summary: asText(trainingFailure?.summary) || "The local training run could not be started.",
      error: asText(trainingFailure?.error) || asText(response?.error) || "Training start failed.",
      errorDetail,
      errorStack: readStructuredErrorStack(response),
      report: trainingFailure
        ? {
            currentState: trainingFailure.currentState,
            nextAction: trainingFailure.nextAction,
            matchingSignals: [
              asText(errorDetail?.runtimeModelId),
              asText(errorDetail?.expectedRelativePath),
            ].filter(Boolean),
          }
        : null,
    };
  }
  return {
    ok: true,
    summary: `Local training run started: ${asText(response.run.jobId)}`,
    data: {
      run: response.run,
    },
    report: {
      currentState: "The local training run was handed off to the offscreen runtime.",
      nextAction: "Check the run with get_training_run_status until it completes or fails.",
      matchingSignals: [
        asText(response.run.jobId),
        asText(response.run.phaseLabel || response.run.phase),
      ].filter(Boolean),
    },
    provenance: [],
  };
}

async function executeRecordExperimentDecisionTool(run, args = {}) {
  const normalized = normalizeExperimentMetadata({
    ...(args && typeof args === "object" ? args : {}),
    candidateId: asText(args?.candidateId) || asText(run.latestExperimentCandidateId),
  }, {
    requireCandidateId: true,
    requireDecision: true,
  });
  if (!normalized || !["keep", "discard", "park"].includes(normalized.decision)) {
    return {
      ok: false,
      summary: "The experiment decision could not be saved because candidateId or decision is missing.",
      error: "Missing experiment candidateId or final decision.",
      errorDetail: {
        reason: "missing_experiment_decision_fields",
      },
    };
  }
  const decisionLabel =
    normalized.decision === "keep"
      ? "kept"
      : normalized.decision === "discard"
        ? "discarded"
        : "parked";
  const summary =
    asText(args.summary) ||
    `Experiment ${normalized.candidateId} was recorded as ${decisionLabel}.`;
  const ledgerEntry = appendRunExperimentLedger(run, {
    ...normalized,
    action: "record_experiment_decision",
    outcome:
      normalized.decision === "keep"
        ? "accepted"
        : normalized.decision === "discard"
          ? "rejected"
          : "parked",
    summary,
    recordedAt: new Date().toISOString(),
  });
  return {
    ok: true,
    summary,
    data: {
      entry: ledgerEntry,
    },
    report: {
      currentState: `Experiment ${normalized.candidateId} is now marked as ${decisionLabel}.`,
      nextAction:
        normalized.decision === "keep"
          ? "Use the stronger challenger now as the new reference or promotion-ready baseline."
          : normalized.decision === "discard"
            ? "Stay with the current champion and only try a new challenger when there is a new hypothesis."
            : "Keep the champion active and check later with the same mini suite whether the parked challenger becomes more reliable.",
      matchingSignals: uniqueTextList([
        normalized.candidateId,
        normalized.compareAgainst,
        normalized.suiteId,
        normalized.evalSetId,
        normalized.mutationScope,
      ], 4),
    },
    provenance: normalizeProvenance([
      {
        title: `Experiment ${decisionLabel}`,
        detail: trimText([
          normalized.candidateId,
          normalized.compareAgainst ? `against ${normalized.compareAgainst}` : "",
          normalized.hypothesis,
          normalized.rationale,
        ].filter(Boolean).join(" · "), 220),
        kind: "operation",
      },
    ]),
  };
}

async function hasPackagedLocalBrowserTrainingManifest(runtimeModelId = "") {
  const normalizedRuntimeModelId = asText(runtimeModelId);
  if (!normalizedRuntimeModelId) return false;
  try {
    const manifestUrl = getLocalQwenModelRepoUrl(
      normalizedRuntimeModelId,
      "lora_training_manifest.json",
    );
    if (!manifestUrl) return false;
    const response = await fetch(manifestUrl, { cache: "no-store" });
    return response.ok;
  } catch (_error) {
    return false;
  }
}

async function resolveLocalBrowserTrainingModelName(modelName = "") {
  const explicitModelName = asText(modelName);
  if (explicitModelName) {
    try {
      getLocalQwenRuntimePlan(explicitModelName);
      return getPreferredLocalQwenVanillaModelName(
        explicitModelName,
        DEFAULT_LOCAL_BROWSER_TRAINING_MODEL,
      );
    } catch (_error) {
      // Ignore unsupported artifact or slot models and fall back to the packaged local browser model set.
    }
  }

  const candidates = [];
  const seen = new Set();
  const pushCandidate = (value) => {
    const normalizedValue = asText(value);
    if (!normalizedValue || seen.has(normalizedValue)) return;
    seen.add(normalizedValue);
    candidates.push(normalizedValue);
  };

  pushCandidate(DEFAULT_LOCAL_BROWSER_TRAINING_MODEL);
  for (const entry of getSupportedLocalQwenModels()) {
    pushCandidate(entry?.vanillaModelName);
    pushCandidate(entry?.canonicalModelName);
    pushCandidate(entry?.runtimeModelId);
    pushCandidate(entry?.remoteRuntimeModelId);
  }

  let fallbackTrainingModelName = getPreferredLocalQwenVanillaModelName(
    "",
    DEFAULT_LOCAL_BROWSER_TRAINING_MODEL,
  );
  for (const candidate of candidates) {
    try {
      const runtimePlan = getLocalQwenRuntimePlan(candidate);
      fallbackTrainingModelName = getPreferredLocalQwenVanillaModelName(
        candidate,
        fallbackTrainingModelName,
      );
      if (await hasPackagedLocalBrowserTrainingManifest(runtimePlan.runtimeModelId)) {
        return fallbackTrainingModelName;
      }
    } catch (_error) {
      continue;
    }
  }

  return fallbackTrainingModelName || DEFAULT_LOCAL_BROWSER_TRAINING_MODEL;
}

async function executeGetTrainingRunStatusTool(_run, args = {}) {
  let jobId = asText(args.jobId);
  if (!jobId || jobId === asText(_run?.craftId)) {
    jobId = asText(_run?.latestTrainingJobId);
  }
  if (!jobId) {
    return {
      ok: false,
      summary: "Training status could not be read because no job ID was available.",
      error: "Missing jobId.",
      errorDetail: {
        reason: "missing_job_id",
      },
    };
  }

  const waitForMs = Math.max(0, Math.min(120_000, Number(args.waitForMs || 0) || 0));
  const pollIntervalMs = Math.max(500, Math.min(15_000, Number(args.pollIntervalMs || 2_000) || 2_000));
  const deadlineAt = Date.now() + waitForMs;
  let attempts = 0;
  let response = null;

  while (true) {
    attempts += 1;
    response = await sendMessageToOffscreen("OFFSCREEN_LOCAL_QWEN_TRAINING_STATUS", {
      jobId,
    });
    if (!response?.ok || !response?.run) {
      return {
        ok: false,
        summary: `Training status for ${jobId} could not be read.`,
        error: asText(response?.error) || "Training status failed.",
        errorDetail: cloneStructuredErrorDetail(response?.errorDetail),
        errorStack: readStructuredErrorStack(response),
      };
    }
    const runStatus = asText(response.run.status).toLowerCase();
    if (["completed", "failed"].includes(runStatus) || !waitForMs || Date.now() >= deadlineAt) {
      break;
    }
    const estimatedRemainingMs = Math.max(0, Number(response.run.estimatedRemainingMs || 0));
    const nextWaitMs = Math.min(
      pollIntervalMs,
      estimatedRemainingMs > 0 ? estimatedRemainingMs : pollIntervalMs,
      Math.max(0, deadlineAt - Date.now()),
    );
    if (nextWaitMs <= 0) break;
    await sleepMs(nextWaitMs);
  }

  const estimatedRemainingMs = Math.max(0, Number(response.run.estimatedRemainingMs || 0));
  const shouldPollAgain = !["completed", "failed"].includes(asText(response.run.status).toLowerCase());
  const pollAfterMs = shouldPollAgain
    ? Math.max(2_000, Math.min(30_000, estimatedRemainingMs > 0 ? estimatedRemainingMs : 5_000))
    : 0;

  return {
    ok: true,
    summary: `Training status ${jobId}: ${asText(response.run.status)} / ${asText(response.run.phaseLabel || response.run.phase)}`,
    data: {
      run: response.run,
      pollAttempts: attempts,
      shouldPollAgain,
      pollAfterMs,
      estimatedRemainingMs,
    },
    report: {
      currentState: `Training run ${jobId} is currently ${asText(response.run.status)}.`,
      nextAction:
        asText(response.run.status).toLowerCase() === "completed"
          ? "Inspect the saved bundle or start run_capability_eval with multiple cases."
          : asText(response.run.status).toLowerCase() === "failed"
            ? "Inspect the failure and only restart the run after a targeted fix."
            : `Wait ${formatWaitDurationMs(pollAfterMs) || "briefly"} and check the status again.`,
      matchingSignals: [
        asText(response.run.phaseLabel || response.run.phase),
        `${Number(response.run.completedSamples || 0)}/${Number(response.run.totalSamples || 0)} steps`,
        estimatedRemainingMs > 0 ? `ETA ${formatWaitDurationMs(estimatedRemainingMs)}` : "",
      ],
    },
    provenance: [],
  };
}

function buildPlanSummaryText(plan) {
  const summary = asText(plan?.summary);
  const opCount = Array.isArray(plan?.operations) ? plan.operations.length : 0;
  const nextAction = asText(plan?.report?.nextAction);
  const operationSummary = opCount ? `${describeOperationMilestone(plan.operations)}.` : "";
  return [summary, operationSummary, nextAction]
    .filter(Boolean)
    .join("\n\n");
}

function buildFinalDecisionLogMessage(status, report, fallbackText = "") {
  const currentState = trimText(asText(report?.currentState || fallbackText), 220);
  if (status === "continue") {
    return currentState
      ? `Agent continues with another pass: ${currentState}`
      : "Agent continues with another pass.";
  }
  if (status === "blocked") {
    return currentState
      ? `Agent stops here because of a blocker: ${currentState}`
      : "Agent stops here because of a blocker.";
  }
  return currentState
    ? `Agent stops after this pass: ${currentState}`
    : "Agent stops after this pass.";
}

function didRunToolSucceed(run, action = "") {
  const normalizedAction = asText(action);
  return Array.isArray(run?.toolTrace) &&
    run.toolTrace.some((entry) => asText(entry?.action) === normalizedAction && entry?.ok !== false);
}

function countSuccessfulToolActions(run, action = "") {
  const normalizedAction = asText(action);
  if (!normalizedAction) return 0;
  return Array.isArray(run?.toolTrace)
    ? run.toolTrace.filter((entry) => asText(entry?.action) === normalizedAction && entry?.ok !== false).length
    : 0;
}

function getRunMaturityPercent(run) {
  return Math.max(
    0,
    Math.min(
      100,
      Number(
        normalizeMaturity(run?.maturity, createEmptyCraftMaturity())?.percent || 0,
      ) || 0,
    ),
  );
}

function shouldForceTrainingContinuation(run, effectiveStatus = "") {
  if (asText(effectiveStatus).toLowerCase() !== "done") return false;
  if (run?.capabilityEvidence?.hasTrainedCapability === true) return false;
  if (getTrainingDataMeta(run?.samples).readySamples < 1) return false;
  return didRunToolSucceed(run, "run_agentic_smoke") && didRunToolSucceed(run, "run_capability_eval");
}

function shouldForceDatasetGrowthContinuation(run, effectiveStatus = "") {
  if (asText(effectiveStatus).toLowerCase() !== "done") return false;
  if (run?.capabilityEvidence?.hasTrainedCapability !== true) return false;
  if (!didRunToolSucceed(run, "run_agentic_smoke") || !didRunToolSucceed(run, "run_capability_eval")) return false;
  if (countSuccessfulToolActions(run, "generate_training_batch") >= AUTO_DATASET_GROWTH_MAX_BATCH_ATTEMPTS) return false;
  const meta = getTrainingDataMeta(run?.samples);
  if (meta.invalidSamples > 0) return false;
  const maturityPercent = getRunMaturityPercent(run);
  if (
    meta.readySamples >= AUTO_DATASET_GROWTH_MIN_READY_SAMPLES &&
    meta.runnableSamples >= AUTO_DATASET_GROWTH_MIN_RUNNABLE_SAMPLES &&
    maturityPercent >= AUTO_DATASET_GROWTH_TARGET_MATURITY_PERCENT
  ) {
    return false;
  }
  return (
    meta.readySamples < AUTO_DATASET_GROWTH_MIN_READY_SAMPLES ||
    meta.runnableSamples < AUTO_DATASET_GROWTH_MIN_RUNNABLE_SAMPLES ||
    maturityPercent < AUTO_DATASET_GROWTH_TARGET_MATURITY_PERCENT
  );
}

async function startTrainingRunWithLogging(run, args = {}) {
  const meta = getCraftingAgentToolLogMetadata("start_training_run");
  pushRunLog(run, "info", "Tool starts: start_training_run.", {
    kind: "tool",
    toolName: "start_training_run",
    title: meta.title,
    detail: meta.startDetail,
    stageId: meta.stageId,
    status: "running",
  });
  let result;
  try {
    result = await executeStartTrainingRunTool(run, args);
  } catch (error) {
    result = buildToolExecutionFailureResult("start_training_run", error);
  }
  await applyToolOutcomeToRun(run, "start_training_run", args, result);
  pushRunLog(
    run,
    result?.ok === false ? "error" : "success",
    `start_training_run: ${asText(result?.summary || result?.error || "Training run processed.")}`,
    {
      kind: "tool",
      toolName: "start_training_run",
      title: meta.title,
      detail: asText(result?.summary || result?.error || meta.startDetail),
      stageId: meta.stageId,
      status: result?.ok === false ? "error" : "done",
    },
  );
  return result;
}

async function generateTrainingBatchWithLogging(run, args = {}) {
  const meta = getCraftingAgentToolLogMetadata("generate_training_batch");
  pushRunLog(run, "info", "Tool starts: generate_training_batch.", {
    kind: "tool",
    toolName: "generate_training_batch",
    title: meta.title,
    detail: meta.startDetail,
    stageId: meta.stageId,
    status: "running",
  });
  let result;
  try {
    result = await executeGenerateTrainingBatchTool(run, args);
  } catch (error) {
    result = buildToolExecutionFailureResult("generate_training_batch", error);
  }
  await applyToolOutcomeToRun(run, "generate_training_batch", args, result);
  pushRunLog(
    run,
    result?.ok === false ? "error" : "success",
    `generate_training_batch: ${asText(result?.summary || result?.error || "Training data expanded.")}`,
    {
      kind: "tool",
      toolName: "generate_training_batch",
      title: meta.title,
      detail: asText(result?.summary || result?.error || meta.startDetail),
      stageId: meta.stageId,
      status: result?.ok === false ? "error" : "done",
    },
  );
  return result;
}

function promoteRunnableSamplesToReady(samples = [], limit = 0) {
  const normalizedSamples = (Array.isArray(samples) ? samples : []).map((sample, index) =>
    normalizeTrainingSample(sample, index),
  );
  const maxPromotions = Math.max(0, Number(limit || 0) || 0);
  if (maxPromotions < 1) {
    return {
      samples: normalizedSamples,
      promotedCount: 0,
    };
  }
  const now = new Date().toISOString();
  let promotedCount = 0;
  for (const sample of normalizedSamples) {
    if (promotedCount >= maxPromotions) break;
    if (sample.status === "ready" || sample.status === "blocked") continue;
    const validation = validateTrainingSample(sample);
    if (!validation.runnable) continue;
    sample.status = "ready";
    sample.updatedAt = now;
    promotedCount += 1;
  }
  return {
    samples: normalizedSamples,
    promotedCount,
  };
}

async function maybeForceTrainingContinuation(run, effectiveStatus = "") {
  if (!shouldForceTrainingContinuation(run, effectiveStatus)) return null;
  if (asText(run?.latestTrainingJobId)) {
    return {
      nextStatus: "continue",
      responseText: "The reviewed runtime path is confirmed. The training run is already continuing.",
      reportOverride: {
        currentState: "The reviewed runtime path is confirmed and a local training run is already active.",
        nextAction:
          "Check the run with get_training_run_status until a trained artifact appears or a real training error becomes visible.",
      },
    };
  }
  const result = await startTrainingRunWithLogging(run, {
    persistBundle: true,
  });
  if (result?.ok === false) {
    return {
      nextStatus: "blocked",
      responseText: asText(result?.error || result?.summary),
      reportOverride: {
        currentState: `The reviewed runtime path is confirmed, but the training run could not be started: ${trimText(asText(result?.error || result?.summary), 220)}`,
        nextAction: "Fix the local training path first, then restart the same training run.",
      },
    };
  }
  return {
    nextStatus: "continue",
    responseText: "The reviewed runtime path is confirmed. The local training run has now started.",
    reportOverride: {
      currentState: "The reviewed runtime path is confirmed and the local training run has started.",
      nextAction:
        "Check the run with get_training_run_status until a trained artifact appears or a real training error becomes visible.",
    },
  };
}

async function maybeForceDatasetGrowthContinuation(run, effectiveStatus = "") {
  if (!shouldForceDatasetGrowthContinuation(run, effectiveStatus)) return null;
  const metaBefore = getTrainingDataMeta(run?.samples);
  const readyGap = Math.max(0, AUTO_DATASET_GROWTH_MIN_READY_SAMPLES - metaBefore.readySamples);
  const batchResult = await generateTrainingBatchWithLogging(run, {
    objective: asText(run?.brief || run?.craft?.summary || run?.craft?.name),
    maxItems: Math.max(4, Math.min(AUTO_DATASET_GROWTH_BATCH_SIZE, readyGap || AUTO_DATASET_GROWTH_BATCH_SIZE)),
    split: "train",
    status: "ready",
  });
  if (batchResult?.ok === false) {
    return {
      nextStatus: "blocked",
      responseText: asText(batchResult?.error || batchResult?.summary),
      reportOverride: {
        currentState: `The capability already works, but the training base could not be expanded automatically: ${trimText(asText(batchResult?.error || batchResult?.summary), 220)}`,
        nextAction: "Fix the local batch generator or the seed evidence first, then retry the same expansion.",
      },
    };
  }

  let promotedCount = 0;
  const metaAfterBatch = getTrainingDataMeta(run?.samples);
  if (metaAfterBatch.readySamples < AUTO_DATASET_GROWTH_MIN_READY_SAMPLES) {
    const promotionLimit = AUTO_DATASET_GROWTH_MIN_READY_SAMPLES - metaAfterBatch.readySamples;
    const promoted = promoteRunnableSamplesToReady(run?.samples, promotionLimit);
    promotedCount = promoted.promotedCount;
    if (promotedCount > 0) {
      run.samples = await writeTrainingSamplesForCraft(run.craftId, promoted.samples);
      pushRunLog(run, "info", `${promotedCount} existing training rows were promoted to ready.`);
      pushRunProvenance(run, [
        {
          title: "Training rows promoted to ready",
          detail: `${promotedCount} existing rows were automatically marked training-ready after successful smoke/eval validation.`,
          kind: "operation",
          operationType: "update",
        },
      ]);
    }
  }

  const importedRows = Math.max(0, Number(batchResult?.data?.importedRows || 0) || 0);
  if (importedRows < 1 && promotedCount < 1) {
    return null;
  }

  if (asText(run?.latestTrainingJobId)) {
    return {
      nextStatus: "continue",
      responseText: "The training base was expanded. The existing training run should now continue to be monitored.",
      reportOverride: {
        currentState: `The training base was expanded (${importedRows} new ready rows, ${promotedCount} automatic ready promotions).`,
        nextAction: "Check the running training job with get_training_run_status and then update maturity based on the stronger dataset.",
      },
    };
  }

  const trainingResult = await startTrainingRunWithLogging(run, {
    persistBundle: true,
  });
  if (trainingResult?.ok === false) {
    return {
      nextStatus: "blocked",
      responseText: asText(trainingResult?.error || trainingResult?.summary),
      reportOverride: {
        currentState: `The training base was expanded, but the new training run could not be started: ${trimText(asText(trainingResult?.error || trainingResult?.summary), 220)}`,
        nextAction: "Fix the local training path and then retry the same expansion with retraining.",
      },
    };
  }

  return {
    nextStatus: "continue",
    responseText: "The training base was still too thin. New ready rows were generated and a new training run was started.",
    reportOverride: {
      currentState: `The training base was expanded (${importedRows} new ready rows, ${promotedCount} automatic ready promotions) and the local training run was restarted.`,
      nextAction: "Check the training job with get_training_run_status and then update maturity based on the stronger dataset.",
    },
  };
}

function withToolingProvenance(run, provenanceEntries) {
  return normalizeProvenance(provenanceEntries);
}

async function executeDraftTrainingChangesTool(run, args = {}) {
  let rawDraft = null;
  let fallbackReason = "";

  try {
    const plannerAgent = await createTrainingChangesAgent(run);
    const plannerRunner = new Runner();
    const plannerResult = await plannerRunner.run(
      plannerAgent,
      buildTrainingChangesRunInput(run, args),
    );
    rawDraft = plannerResult.finalOutput || null;
  } catch (error) {
    const errorMessage = asText(error?.message || error);
    if (!/Invalid output type/i.test(errorMessage)) {
      const errorInfo = buildErrorInfo(error);
      return {
        ok: false,
        summary: "Training-data draft planner failed before it could produce a valid plan.",
        error: errorInfo.message || errorMessage || "Unknown planner error.",
        errorDetail: errorInfo.detail,
        errorStack: errorInfo.stack,
        report: normalizeReport(run.report, {
          currentState: "The training-data planner failed.",
          nextAction: "Inspect the model path or restart the run with more precise context.",
        }),
      };
    }

    fallbackReason = errorMessage || "Invalid output type";
    pushRunLog(
      run,
      "warn",
      "Training-change subagent returned invalid structured output. Falling back to direct structured planning.",
    );

    try {
      const fallbackPlan = await planTrainingSampleOps({
        slotId: run.resolved?.slotId || run.requestedSlotId || "agent",
        modelRef: run.modelRef,
        providerId: run.resolved?.providerId || run.requestedProviderId,
        modelName: run.resolved?.modelName || run.requestedModelName,
        craft: run.craft,
        brief: buildTrainingChangesFallbackBrief(run, args),
        currentSamples: run.samples,
        parameters: run.parameters,
        reasoningEffort: run.reasoningEffort,
      });
      rawDraft = fallbackPlan?.object || null;
    } catch (fallbackError) {
      const fallbackMessage = asText(fallbackError?.message || fallbackError);
      const fallbackInfo = buildErrorInfo(fallbackError);
      return {
        ok: false,
        summary: "Training-data draft planner failed and the structured fallback could not recover.",
        error: [fallbackReason, fallbackMessage].filter(Boolean).join(" | "),
        errorDetail: fallbackInfo.detail || serializeErrorDetail({ fallbackReason }),
        errorStack: fallbackInfo.stack,
        report: normalizeReport(run.report, {
          currentState: "The training-data planner could not produce a valid structured draft.",
          nextAction: "Refine the craft text or inspect the model provider before restarting the run.",
        }),
      };
    }
  }

  const draft = normalizeTrainingDraftOutput(rawDraft || null);
  const operations = Array.isArray(draft.operations) ? draft.operations : [];
  let workingSamples = run.samples.slice();
  const appliedMessages = [];

  for (const operation of operations) {
    const next = applyTrainingSampleOperation(workingSamples, operation);
    workingSamples = next.samples;
    appliedMessages.push(next.message);
  }
  if (operations.length) {
    run.samples = await writeTrainingSamplesForCraft(run.craftId, workingSamples);
  }

  return {
    ok: true,
    summary:
      operations.length
        ? `Drafted and applied ${operations.length} training data operations.`
        : "Drafted a training-data revision pass without concrete row edits yet.",
    data: {
      operations,
      appliedMessages,
      draft,
      fallbackReason: fallbackReason || null,
    },
    report: normalizeReport(draft.report, {
      objective: trimText(asText(args.objective) || run.brief, 180),
      currentState: operations.length
        ? `The agent derived ${summarizeOperationCounts(operations)} from the visible training data.`
        : "The agent has not derived a reliable edit plan for the training data yet.",
      nextAction: operations.length
        ? "Inspect the live-updated training data in the separate view."
        : "If evidence is still missing, use web search, browser inspect/action, or Playwright CTX before the next planning pass.",
    }),
    maturity: await resolveRunMaturity(run, draft.maturity, run.maturity),
    questions: mergeQuestionsWithAnswers(draft.openQuestions, run.previousQuestions),
    provenance: normalizeProvenance([
      ...buildOperationMilestoneProvenance(operations, appliedMessages),
      ...withToolingProvenance(run, draft.provenance),
    ]),
    operations,
    useSurface: draft.useSurface && typeof draft.useSurface === "object" ? cloneJson(draft.useSurface, null) : null,
    responseText: buildPlanSummaryText(draft),
  };
}

async function executeSupervisorActionInner(run, planner, executionOptions = {}) {
  if (planner.action === "inspect_training_data") {
    return await executeInspectTrainingDataTool(run);
  }
  if (planner.action === "inspect_bundle_snapshot") {
    return await executeInspectBundleSnapshotTool(run);
  }
  if (planner.action === "workspace_code_dive") {
    return await executeWorkspaceCodeDiveTool(run, planner.args);
  }
  if (planner.action === "web_search") {
    return await executeWebSearchTool(run, planner.args, executionOptions);
  }
  if (planner.action === "browser_inspect") {
    return await executeBrowserInspectTool(run, planner.args, executionOptions);
  }
  if (planner.action === "browser_action") {
    return await executeBrowserActionTool(run, planner.args, executionOptions);
  }
  if (planner.action === "browser_tabs") {
    return await executeBrowserTabsTool(run, planner.args);
  }
  if (planner.action === "playwright_ctx") {
    return await executePlaywrightCtxTool(run, planner.args, executionOptions);
  }
  if (planner.action === "browser_automation") {
    return await executeBrowserAutomationTool(run, planner.args);
  }
  if (planner.action === "save_tool_scripts") {
    return await executeSaveToolScriptsTool(run, planner.args);
  }
  if (planner.action === "save_browser_capabilities") {
    return await executeSaveBrowserCapabilitiesTool(run, planner.args);
  }
  if (planner.action === "request_codex_repair") {
    return await executeRequestCodexRepairTool(run, planner.args);
  }
  if (planner.action === "get_codex_repair_status") {
    return await executeGetCodexRepairStatusTool(run, planner.args);
  }
  if (planner.action === "run_agentic_smoke") {
    return await executeRunAgenticSmokeTool(run, planner.args, executionOptions);
  }
  if (planner.action === "run_capability_eval") {
    return await executeRunCapabilityEvalTool(run, planner.args, executionOptions);
  }
  if (planner.action === "generate_training_batch") {
    return await executeGenerateTrainingBatchTool(run, planner.args);
  }
  if (planner.action === "start_training_run") {
    return await executeStartTrainingRunTool(run, planner.args);
  }
  if (planner.action === "get_training_run_status") {
    return await executeGetTrainingRunStatusTool(run, planner.args);
  }
  if (planner.action === "record_experiment_decision") {
    return await executeRecordExperimentDecisionTool(run, planner.args);
  }
  if (planner.action === "draft_training_changes") {
    return await executeDraftTrainingChangesTool(run, planner.args);
  }
  return {
    ok: true,
    summary: `Planner selected ${planner.action}.`,
    data: null,
  };
}

async function executeSupervisorAction(run, planner) {
  const executionContext = createSupervisorToolExecutionContext(run, planner?.action, planner?.args);
  try {
    const toolPromise = executeSupervisorActionInner(run, planner, executionContext);
    return executionContext.timeoutPromise
      ? await Promise.race([toolPromise, executionContext.timeoutPromise])
      : await toolPromise;
  } catch (error) {
    if (isSupervisorToolTimeoutError(error)) {
      return buildSupervisorToolTimeoutResult(planner?.action, executionContext.timeoutMs);
    }
    throw error;
  } finally {
    executionContext.cleanup();
  }
}

function createRun({
  slotId = "agent",
  modelRef = "",
  providerId = "",
  modelName = "",
  craft,
  brief,
  currentSamples,
  resolved = null,
  previousQuestions,
  parameters,
  reasoningEffort,
}) {
  const startedAt = new Date().toISOString();
  const agentTooling = normalizeCraftingAgentToolingPayload(craft?.agentTooling);
  const resolvedProviderType = asText(resolved?.provider?.type).toLowerCase();
  const run = {
    jobId: `craft-agent-${Date.now()}-${Math.random().toString(16).slice(2, 10)}`,
    craftId: asText(craft?.id),
    craft: cloneJson(craft, {}),
    brief: asText(brief),
    previousQuestions: normalizeQuestions(previousQuestions),
    samples: (Array.isArray(currentSamples) ? currentSamples : []).map((sample, index) =>
      normalizeTrainingSample(sample, index),
    ),
    agentTooling,
    status: "starting",
    phase: "starting",
    mode: resolvedProviderType === "local_qwen" ? "local" : resolvedProviderType ? "remote" : "",
    modelRef: asText(resolved?.modelRef),
    resolved: resolved ? cloneJson(resolved, {}) : null,
    requestedSlotId: asText(slotId) || "agent",
    requestedModelRef: asText(modelRef),
    requestedProviderId: asText(providerId),
    requestedModelName: asText(modelName),
    parameters: parameters && typeof parameters === "object" ? cloneJson(parameters, {}) : {},
    reasoningEffort: asText(reasoningEffort || resolved?.reasoningEffort),
    responseText: "",
    finalStatus: "",
    tokens: 0,
    costUsd: 0,
    error: "",
    errorDetail: null,
    report: {
      objective: trimText(brief, 180),
      currentState: "Resolving the configured model and provider.",
      nextAction: "",
      matchingSignals: [],
    },
    questions: [],
    provenance: withToolingProvenance({ agentTooling }, []),
    suggestedName: "",
    officialDescription: null,
    activeTab: "progress",
    logs: [],
    operations: [],
    useSurface: null,
    turnsUsed: 0,
    maxTurns: DEFAULT_MAX_TURNS,
    startedAt,
    updatedAt: startedAt,
    completedAt: "",
    activeTabContext: null,
    browserRuntime: {
      currentTabId: 0,
    },
    toolTrace: [],
    experimentLedger: [],
    latestExperimentCandidateId: "",
    lastToolFailure: null,
    workspaceCodeDive: null,
    researchNotes: [],
    browserNotes: [],
    compactedMemory: [],
    compactionCount: 0,
    turnOffset: 0,
    segmentCount: 0,
    segmentMaxTurns: DEFAULT_SEGMENT_MAX_TURNS,
    initialized: false,
    serializedState: "",
    pendingApprovalCallId: "",
    pendingUserAnswersByCallId: {},
    latestSuggestedQuestions: [],
    latestTrainingJobId: "",
    latestCodexRepairJobId: "",
    codexRepair: createEmptyCodexRepairState(),
    reloadResume: createEmptyReloadResumeState(),
    accountedUsage: createEmptyUsageTotals(),
    liveUsage: createEmptyUsageTotals(),
    abortController: null,
    stopRequested: false,
    stopReason: "",
    executionPromise: null,
  };
  run.maturity = createEmptyCraftMaturity();
  run.capabilityEvidence = createEmptyCapabilityEvidence(run.craftId);
  return run;
}

function addUsageTotals(baseUsage, deltaUsage) {
  const base = normalizeUsageTotals(baseUsage);
  const delta = normalizeUsageTotals(deltaUsage);
  return {
    requests: base.requests + delta.requests,
    inputTokens: base.inputTokens + delta.inputTokens,
    outputTokens: base.outputTokens + delta.outputTokens,
    totalTokens: base.totalTokens + delta.totalTokens,
  };
}

function syncRunCheckpoint(run, state) {
  if (!state) return;
  const serializedState = safeSerializeRunState(state);
  if (serializedState) {
    run.serializedState = serializedState;
  }
  run.liveUsage = readRunStateUsageTotals(state);
  run.turnsUsed = run.turnOffset + readRunStateCurrentTurn(state);
  run.tokens = run.accountedUsage.totalTokens + run.liveUsage.totalTokens;
  run.updatedAt = new Date().toISOString();
  scheduleRunPersistence(run);
}

function buildCompactionMemory(run, state) {
  return {
    compactionIndex: run.compactionCount + 1,
    turnsConsumed: readRunStateCurrentTurn(state),
    report: cloneJson(run.report, {}),
    answeredQuestions: normalizeQuestions(run.previousQuestions)
      .filter((entry) => asText(entry.answer))
      .map((entry) => ({
        id: entry.id,
        question: entry.question,
        answer: entry.answer,
      })),
    operations: Array.isArray(run.operations) ? cloneJson(run.operations, []).slice(0, 6) : [],
    researchNotes: Array.isArray(run.researchNotes) ? run.researchNotes.slice(-6) : [],
    browserNotes: Array.isArray(run.browserNotes) ? run.browserNotes.slice(-6) : [],
    provenance: normalizeProvenance(run.provenance).slice(-8),
    trainingDataMeta: getTrainingDataMeta(run.samples),
  };
}

function applyCompactionToRun(run, state, reason = "") {
  const turnsInState = readRunStateCurrentTurn(state);
  run.compactedMemory.push(buildCompactionMemory(run, state));
  if (run.compactedMemory.length > MAX_COMPACT_MEMORIES) {
    run.compactedMemory = run.compactedMemory.slice(-MAX_COMPACT_MEMORIES);
  }
  run.compactionCount += 1;
  run.turnOffset += turnsInState;
  run.turnsUsed = run.turnOffset;
  run.accountedUsage = addUsageTotals(run.accountedUsage, run.liveUsage);
  run.liveUsage = createEmptyUsageTotals();
  run.tokens = run.accountedUsage.totalTokens;
  run.serializedState = "";
  run.pendingApprovalCallId = "";
  run.pendingUserAnswersByCallId = {};
  run.phase = "compacting";
  pushRunLog(
    run,
    "info",
    `Context compacted after ${turnsInState} turns${reason ? ` (${reason})` : ""}. A fresh SDK segment continues with compact memory only.`,
  );
}

function prepareRunForFreshSegment(run, state, reason = "") {
  const turnsInState = Math.max(0, readRunStateCurrentTurn(state));
  if (turnsInState > 0) {
    run.turnOffset += turnsInState;
  }
  run.turnsUsed = run.turnOffset;
  run.accountedUsage = addUsageTotals(run.accountedUsage, run.liveUsage);
  run.liveUsage = createEmptyUsageTotals();
  run.tokens = run.accountedUsage.totalTokens;
  run.serializedState = "";
  run.pendingApprovalCallId = "";
  run.pendingUserAnswersByCallId = {};
  run.updatedAt = new Date().toISOString();
  if (reason) {
    pushRunLog(run, "info", reason);
  }
}

async function applyToolOutcomeToRun(run, action, args, toolResult) {
  const result = toolResult && typeof toolResult === "object" ? toolResult : {};
  const experiment = normalizeExperimentMetadata(
    args?.experiment || (action === "record_experiment_decision" ? args : null),
  );
  const normalizedFailureDetail =
    result.ok === false
      ? serializeErrorDetail(result.errorDetail) || buildSyntheticToolFailureDetail(action, result)
      : null;
  const normalizedErrorStack = readStructuredErrorStack(result);
  if (result.report) {
    setRunReport(run, result.report, run.report);
  }
  if (result.maturity) {
    await setRunMaturity(run, result.maturity, run.maturity);
  }
  if (Array.isArray(result.provenance) && result.provenance.length) {
    pushRunProvenance(run, result.provenance);
  }
  if (action === "web_search") {
    const results = Array.isArray(result?.data?.results) ? result.data.results : [];
    run.researchNotes.push(...results.slice(0, 3).map((entry) => `${entry.title} · ${entry.url}`));
    run.researchNotes = run.researchNotes.slice(-8);
  }
  if (BROWSER_TOOL_ACTIONS.has(action) || action === "browser_automation") {
    const noteCandidates = [
      asText(result?.summary),
      asText(result?.data?.title),
      asText(result?.data?.url),
      asText(result?.data?.raw_preview),
      ...(Array.isArray(result?.data?.ui_targets)
        ? result.data.ui_targets
          .slice(0, 2)
          .map((entry) => asText(entry?.text || entry?.selector_hint || entry?.role))
        : []),
    ].filter(Boolean);
    run.browserNotes.push(...noteCandidates.slice(0, 3).map((entry) => trimText(entry, 220)));
    run.browserNotes = run.browserNotes.slice(-8);
  }
  if (action === "draft_training_changes") {
    run.responseText = asText(result.responseText || run.responseText);
    run.latestSuggestedQuestions = mergeQuestionsWithAnswers(result.questions, run.previousQuestions);
    run.operations = Array.isArray(result.operations) ? cloneJson(result.operations, []) : [];
    if (result.useSurface && typeof result.useSurface === "object") {
      run.useSurface = cloneJson(result.useSurface, null);
    }
    if (run.operations.length) {
      pushRunLog(run, "info", `${describeOperationMilestone(run.operations)}.`);
    }
  }
  if (action === "start_training_run" || action === "get_training_run_status") {
    run.latestTrainingJobId = asText(result?.data?.run?.jobId || run.latestTrainingJobId);
  }
  if (action === "request_codex_repair" || action === "get_codex_repair_status") {
    const jobId = asText(result?.data?.job?.jobId || run.latestCodexRepairJobId);
    run.latestCodexRepairJobId = jobId;
    if (jobId && action === "request_codex_repair") {
      setPendingCodexRepair(run, {
        jobId,
        fingerprint: asText(run?.codexRepair?.pendingFingerprint) || buildCodexRepairFingerprint(run),
      });
    } else if (jobId && action === "get_codex_repair_status") {
      const jobStatus = asText(result?.data?.job?.status).toLowerCase();
      if (jobStatus === "completed" || jobStatus === "failed") {
        clearPendingCodexRepair(run);
      } else {
        setPendingCodexRepair(run, {
          jobId,
          fingerprint: asText(run?.codexRepair?.pendingFingerprint) || buildCodexRepairFingerprint(run),
        });
      }
    }
  }
  if (action === "workspace_code_dive" && result?.data && typeof result.data === "object") {
    run.workspaceCodeDive = cloneJson(result.data, null);
  }
  if (experiment?.candidateId) {
    run.latestExperimentCandidateId = experiment.candidateId;
  }
  if (experiment && action !== "record_experiment_decision") {
    appendRunExperimentLedger(run, {
      ...experiment,
      action,
      outcome: result.ok === false ? "tool_failed" : "tool_ok",
      summary: trimText(result.summary || result.error || "", 240),
      recordedAt: new Date().toISOString(),
    });
  }
  if (result.ok === false) {
    run.lastToolFailure = {
      action: asText(action),
      underlyingAction: asText(result?.data?.underlyingFailingTool || result?.data?.failingTool),
      args: cloneJson(args, {}),
      summary: trimText(result.summary || result.error || "", 260),
      error: trimText(result.error || result.summary || "", 600),
      errorDetail: normalizedFailureDetail,
      errorStack: normalizedErrorStack,
      report: result.report && typeof result.report === "object" ? cloneJson(result.report, null) : null,
      recordedAt: new Date().toISOString(),
    };
  } else if (asText(run?.lastToolFailure?.action) === asText(action)) {
    run.lastToolFailure = null;
  }
  if (action === ASK_USER_TOOL_NAME) {
    const questions = mergeQuestionsWithAnswers(result.questions, run.previousQuestions);
    run.previousQuestions = questions;
    run.questions = questions;
    run.pendingApprovalCallId = "";
    run.pendingUserAnswersByCallId = {};
  }
  const toolTraceEntry = {
    turn: Math.max(1, run.turnsUsed || run.turnOffset + 1),
    thought: "",
    action,
    args: cloneJson(args, {}),
    experiment: experiment ? cloneJson(experiment, null) : null,
    summary: trimText(result.summary || result.error || "", 240),
    ok: result.ok !== false,
    error: trimText(result.error, 600),
    errorDetail: normalizedFailureDetail,
    errorStack: normalizedErrorStack,
    recordedAt: new Date().toISOString(),
  };
  run.toolTrace.push(toolTraceEntry);
  if (run.toolTrace.length > MAX_TOOL_TRACE) {
    run.toolTrace = run.toolTrace.slice(-MAX_TOOL_TRACE);
  }
  emitDevObservabilityEvent(run, "tool_result", {
    tool: summarizeToolTraceEntry(toolTraceEntry),
    latestFailure:
      run?.lastToolFailure && typeof run.lastToolFailure === "object"
        ? {
            action: asText(run.lastToolFailure.action),
            error: trimText(asText(run.lastToolFailure.error), 320),
            recordedAt: asText(run.lastToolFailure.recordedAt),
          }
        : null,
  });
}

function buildWorkspaceCodeDiveArgs(run, finalOutput = null) {
  const lastFailure = run?.lastToolFailure && typeof run.lastToolFailure === "object" ? run.lastToolFailure : {};
  const focusFiles = uniqueTextList(
    buildWorkspaceCodeDiveFocusReferences({}, run).map((entry) => entry.runtimePath),
    8,
  );
  const problem = [
    asText(run?.brief),
    asText(finalOutput?.summary),
    asText(run?.report?.currentState),
    asText(run?.error),
    asText(lastFailure?.error),
  ].filter(Boolean).join("\n\n");
  return {
    problem: problem || "Blocked crafting-agent run with a likely workspace defect.",
    failingTool: asText(lastFailure?.underlyingAction || lastFailure?.action),
    errorText: asText(lastFailure?.error || run?.error),
    focusFiles,
    maxFiles: WORKSPACE_CODE_DIVE_MAX_FILES,
  };
}

async function maybeRunWorkspaceDiagnosis(run, finalOutput = null, effectiveStatus = "") {
  const normalizedStatus = asText(effectiveStatus || run?.status).toLowerCase();
  if (!["blocked", "failed"].includes(normalizedStatus)) return null;
  if (run?.workspaceCodeDive && typeof run.workspaceCodeDive === "object") return run.workspaceCodeDive;
  const failingTool = asText(run?.lastToolFailure?.action);
  if (!failingTool || NON_WORKSPACE_DIAGNOSIS_TOOL_ACTIONS.has(failingTool) || failingTool === ASK_USER_TOOL_NAME) {
    return null;
  }

  const args = buildWorkspaceCodeDiveArgs(run, finalOutput);
  const meta = getCraftingAgentToolLogMetadata("workspace_code_dive");
  pushRunLog(run, "info", "Tool starts: workspace_code_dive.", {
    kind: "tool",
    toolName: "workspace_code_dive",
    title: meta.title,
    detail: meta.startDetail,
    stageId: meta.stageId,
    status: "running",
  });
  let result;
  try {
    result = await executeWorkspaceCodeDiveTool(run, args);
  } catch (error) {
    result = buildToolExecutionFailureResult("workspace_code_dive", error);
  }
  await applyToolOutcomeToRun(run, "workspace_code_dive", args, result);
  pushRunLog(
    run,
    result?.ok === false ? "error" : "success",
    `workspace_code_dive: ${asText(result?.summary || result?.error || "Workspace diagnosis completed.")}`,
    {
      kind: "tool",
      toolName: "workspace_code_dive",
      title: meta.title,
      detail: asText(result?.summary || result?.error || meta.startDetail),
      stageId: meta.stageId,
      status: result?.ok === false ? "error" : "done",
    },
  );
  return result?.data || null;
}

function buildRunContextValue(run) {
  return {
    jobId: run.jobId,
    craftId: run.craftId,
    compactedMemory: cloneJson(run.compactedMemory, []),
    approvedQuestionAnswersByCallId: cloneJson(run.pendingUserAnswersByCallId, {}),
    answeredQuestions: normalizeQuestions(run.previousQuestions)
      .filter((entry) => asText(entry.answer))
      .map((entry) => ({
        id: entry.id,
        question: entry.question,
        answer: entry.answer,
      })),
  };
}

function applyRunContextToState(run, state) {
  if (!state?._context) return;
  const existingContext =
    state._context.context && typeof state._context.context === "object"
      ? state._context.context
      : {};
  state._context.context = {
    ...existingContext,
    ...buildRunContextValue(run),
  };
}

const EMPTY_TOOL_SCHEMA = z.object({}).strict();
const REPORT_TOOL_SCHEMA = CRAFTING_AGENT_REPORT_CONTRACT;
const MATURITY_TOOL_SCHEMA = CRAFTING_AGENT_MATURITY_CONTRACT.extend({
  updatedAt: z.string().max(80).nullable().optional(),
  isExplicit: z.boolean().nullable().optional(),
}).strict();
const QUESTION_TOOL_SCHEMA = CRAFTING_AGENT_QUESTION_CONTRACT;
const PROVENANCE_TOOL_ENTRY_SCHEMA = CRAFTING_AGENT_PROVENANCE_CONTRACT;
function createBrowserTargetSchema() {
  return z.object({
    selector: z.string().max(400).nullable().optional(),
    text: z.string().max(240).nullable().optional(),
    role: z.string().max(80).nullable().optional(),
    name: z.string().max(160).nullable().optional(),
    exact: z.boolean().nullable().optional(),
    frameSelectors: z.array(z.string().max(240)).max(6).nullable().optional(),
    x: z.number().min(0).nullable().optional(),
    y: z.number().min(0).nullable().optional(),
    coordSpace: z.enum(["viewport_css", "image_px"]).nullable().optional(),
    imageWidth: z.number().int().min(1).max(8_000).nullable().optional(),
    imageHeight: z.number().int().min(1).max(8_000).nullable().optional(),
  }).strict();
}

function createLooseJsonObjectSchema() {
  return z.record(z.any());
}

function createExperimentMetadataSchema() {
  return z.object({
    candidateId: z.string().max(120).nullable().optional(),
    mode: z.enum(["baseline", "champion", "challenger"]).nullable().optional(),
    compareAgainst: z.string().max(120).nullable().optional(),
    hypothesis: z.string().max(600).nullable().optional(),
    expectedSignal: z.string().max(400).nullable().optional(),
    mutationScope: z.enum([
      "none",
      "tool_scripts",
      "browser_capabilities",
      "training_rows",
      "training_config",
      "bundle_policy",
      "multi_artifact",
    ]).nullable().optional(),
    decisionPolicy: z.enum([
      "keep_if_better",
      "keep_if_equal_or_better",
      "park_unless_clear_win",
      "manual_review",
    ]).nullable().optional(),
    suiteId: z.string().max(120).nullable().optional(),
    evalSetId: z.string().max(120).nullable().optional(),
    decision: z.enum(["pending", "keep", "discard", "park"]).nullable().optional(),
    rationale: z.string().max(600).nullable().optional(),
    tags: z.array(z.string().max(80)).max(12).nullable().optional(),
    metrics: createLooseJsonObjectSchema().nullable().optional(),
  }).strict();
}

function createExperimentDecisionSchema() {
  return createExperimentMetadataSchema().extend({
    candidateId: z.string().min(1).max(120),
    decision: z.enum(["keep", "discard", "park"]),
    summary: z.string().max(240).nullable().optional(),
    rationale: z.string().max(600).nullable().optional(),
    metrics: createLooseJsonObjectSchema().nullable().optional(),
  }).strict();
}

function createToolScriptEntrySchema() {
  return z.object({
    id: z.string().min(1).max(120),
    name: z.string().min(1).max(160),
    description: z.string().max(600).nullable().optional(),
    language: z.string().max(40).nullable().optional(),
    entrypoint: z.string().max(160).nullable().optional(),
    source: z.string().min(1).max(60_000),
  }).strict();
}

function createBrowserCapabilityEntrySchema() {
  return z.object({
    id: z.string().min(1).max(120),
    name: z.string().min(1).max(160),
    toolName: z.string().max(160).nullable().optional(),
    tool_name: z.string().max(160).nullable().optional(),
    functionName: z.string().max(160).nullable().optional(),
    function_name: z.string().max(160).nullable().optional(),
    canonicalToolName: z.string().max(160).nullable().optional(),
    canonical_tool_name: z.string().max(160).nullable().optional(),
    version: z.string().max(40).nullable().optional(),
    description: z.string().max(800).nullable().optional(),
    parameterSchema: createLooseJsonObjectSchema().nullable().optional(),
    parameters: createLooseJsonObjectSchema().nullable().optional(),
    inputSchema: createLooseJsonObjectSchema().nullable().optional(),
    returnSchema: createLooseJsonObjectSchema().nullable().optional(),
    outputSchema: createLooseJsonObjectSchema().nullable().optional(),
    preconditions: z.array(z.string().max(300)).max(24).nullable().optional(),
    readsFrom: z.array(z.string().max(300)).max(24).nullable().optional(),
    writesTo: z.array(z.string().max(300)).max(24).nullable().optional(),
    examples: z.array(z.string().max(400)).max(12).nullable().optional(),
    tags: z.array(z.string().max(80)).max(24).nullable().optional(),
    skillRef: z.string().max(160).nullable().optional(),
    skill_ref: z.string().max(160).nullable().optional(),
    resourceRefs: z.array(z.string().max(160)).max(24).nullable().optional(),
    resource_refs: z.array(z.string().max(160)).max(24).nullable().optional(),
    scripts: z.object({
      pre: z.string().max(60_000).nullable().optional(),
      execute: z.string().max(60_000).nullable().optional(),
      post: z.string().max(60_000).nullable().optional(),
    }).strict().nullable().optional(),
    preScript: z.string().max(60_000).nullable().optional(),
    pre_script: z.string().max(60_000).nullable().optional(),
    executeScript: z.string().max(60_000).nullable().optional(),
    execute_script: z.string().max(60_000).nullable().optional(),
    postScript: z.string().max(60_000).nullable().optional(),
    post_script: z.string().max(60_000).nullable().optional(),
    source: z.string().max(60_000).nullable().optional(),
  }).strict();
}

function createBundleOverrideSchema() {
  return z.object({
    trainingData: createLooseJsonObjectSchema().nullable().optional(),
    toolScripts: createLooseJsonObjectSchema().nullable().optional(),
    browserCapabilities: createLooseJsonObjectSchema().nullable().optional(),
    weights: createLooseJsonObjectSchema().nullable().optional(),
    policy: createLooseJsonObjectSchema().nullable().optional(),
  }).strict();
}

function createCapabilityEvalCaseSchema() {
  return z.object({
    prompt: z.string().min(2).max(4_000),
    maxTurns: z.number().int().min(1).max(8).nullable().optional(),
    expectedCapabilityNames: z.array(z.string().max(160)).max(8).nullable().optional(),
    expectedFinalStatus: z.enum(["done", "blocked", "continue", "failed"]).nullable().optional(),
    expectedTextIncludes: z.array(z.string().min(1).max(240)).max(8).nullable().optional(),
  }).strict();
}

function buildCraftingAgentTools(run) {
  const createToolExecutor = (action) =>
    async (params = {}) => {
      const normalizedParams = params && typeof params === "object" ? params : {};
      let result;
      try {
        result = await executeSupervisorAction(run, {
          action,
          args: normalizedParams,
        });
      } catch (error) {
        if (isAbortLikeError(error) && (run.stopRequested || run.abortController?.signal?.aborted)) {
          throw error;
        }
        result = buildToolExecutionFailureResult(action, error);
      }
      await applyToolOutcomeToRun(run, action, normalizedParams, result);
      return result;
    };

  return [
    tool({
      name: "inspect_training_data",
      description: "Inspect the current local prompt-to-JSON dataset and summarize sample health.",
      parameters: EMPTY_TOOL_SCHEMA,
      execute: createToolExecutor("inspect_training_data"),
    }),
    tool({
      name: "inspect_bundle_snapshot",
      description: "Inspect the currently stored local bundle artifacts before editing runtime scripts, reviewed capabilities, or training state.",
      parameters: EMPTY_TOOL_SCHEMA,
      execute: createToolExecutor("inspect_bundle_snapshot"),
    }),
    tool({
      name: "workspace_code_dive",
      description: "Read packaged extension source files to isolate likely patch targets, execution paths, and validation files for a workspace defect.",
      parameters: z.object({
        problem: z.string().min(3).max(2_000),
        failingTool: z.string().max(160).nullable().optional(),
        errorText: z.string().max(4_000).nullable().optional(),
        focusFiles: z.array(z.string().max(240)).max(8).nullable().optional(),
        focusTags: z.array(z.string().max(120)).max(8).nullable().optional(),
        maxFiles: z.number().int().min(1).max(WORKSPACE_CODE_DIVE_MAX_FILES).nullable().optional(),
      }).strict(),
      execute: createToolExecutor("workspace_code_dive"),
    }),
    tool({
      name: "web_search",
      description: "Search the public web for likely sources, examples, or reference pages for this craft.",
      parameters: z.object({
        query: z.string().min(2).max(240),
        focus: z.string().max(160).nullable(),
        maxResults: z.number().int().min(1).max(8).nullable(),
        allowedDomains: z.array(z.string().max(200)).max(8).nullable().optional(),
      }).strict(),
      execute: createToolExecutor("web_search"),
    }),
    tool({
      name: "browser_inspect",
      description: "Use the vision model to inspect the visible browser UI and identify grounded next actions.",
      parameters: z.object({
        tabId: z.number().int().min(1).nullable().optional(),
        url: z.string().max(1_200).nullable().optional(),
        question: z.string().max(600).nullable().optional(),
        objective: z.string().max(600).nullable(),
        goal: z.string().max(600).nullable(),
        active: z.boolean().nullable().optional(),
        visionModelRef: z.string().max(120).nullable().optional(),
        imageDetail: z.enum(["auto", "low", "high"]).nullable().optional(),
      }).strict(),
      execute: createToolExecutor("browser_inspect"),
    }),
    tool({
      name: "browser_action",
      description: "Execute visible browser interactions such as click, type, scroll, keypress, wait, or drag in the current tab.",
      parameters: z.object({
        tabId: z.number().int().min(1).nullable().optional(),
        url: z.string().max(1_200).nullable().optional(),
        active: z.boolean().nullable().optional(),
        action: z.enum(["click", "double_click", "move", "drag", "scroll", "type", "keypress", "wait"]),
        target: createBrowserTargetSchema().nullable().optional(),
        destination: createBrowserTargetSchema().nullable().optional(),
        deltaX: z.number().nullable().optional(),
        deltaY: z.number().nullable().optional(),
        textValue: z.string().max(8_000).nullable().optional(),
        keys: z.string().max(240).nullable().optional(),
        clear: z.boolean().nullable().optional(),
        waitMs: z.number().int().min(0).max(120_000).nullable().optional(),
        timeoutMs: z.number().int().min(250).max(60_000).nullable().optional(),
        button: z.enum(["left", "middle", "right"]).nullable().optional(),
        steps: z.number().int().min(1).max(60).nullable().optional(),
      }).strict(),
      execute: createToolExecutor("browser_action"),
    }),
    tool({
      name: "browser_tabs",
      description: "List, open, activate, or close browser tabs before running inspect, action, or Playwright steps.",
      parameters: z.object({
        operation: z.enum(["list", "current", "open", "activate", "close"]),
        tabId: z.number().int().min(1).nullable().optional(),
        url: z.string().max(1_200).nullable().optional(),
        active: z.boolean().nullable().optional(),
        limit: z.number().int().min(1).max(30).nullable().optional(),
      }).strict(),
      execute: createToolExecutor("browser_tabs"),
    }),
    tool({
      name: "playwright_ctx",
      description: "Run deterministic Playwright-like DOM or automation code inside the current browser tab context.",
      parameters: z.object({
        tabId: z.number().int().min(1).nullable().optional(),
        url: z.string().max(1_200).nullable().optional(),
        active: z.boolean().nullable().optional(),
        code: z.string().min(3).max(20_000),
        timeoutMs: z.number().int().min(500).max(180_000).nullable().optional(),
      }).strict(),
      execute: createToolExecutor("playwright_ctx"),
    }),
    tool({
      name: "save_tool_scripts",
      description: "Persist one or more reviewed runtime tool scripts into the local craft bundle.",
      parameters: z.object({
        scriptCount: z.number().int().min(1).max(12).nullable().optional(),
        scripts: z.array(createToolScriptEntrySchema()).min(1).max(12),
        declaredTools: z.array(z.string().max(160)).max(48).nullable().optional(),
        replaceAll: z.boolean().nullable().optional(),
        experiment: createExperimentMetadataSchema().nullable().optional(),
      }).strict(),
      execute: createToolExecutor("save_tool_scripts"),
    }),
    tool({
      name: "save_browser_capabilities",
      description: "Persist reviewed browser capabilities, bundle skills, and resources into the local craft bundle.",
      parameters: z.object({
        capabilityCount: z.number().int().min(1).max(24).nullable().optional(),
        capabilities: z.array(createBrowserCapabilityEntrySchema()).min(1).max(24),
        skills: z.array(z.string().max(400)).max(24).nullable().optional(),
        resources: z.array(z.string().max(160)).max(48).nullable().optional(),
        replaceAll: z.boolean().nullable().optional(),
        experiment: createExperimentMetadataSchema().nullable().optional(),
      }).strict(),
      execute: createToolExecutor("save_browser_capabilities"),
    }),
    tool({
      name: "request_codex_repair",
      description: "Submit a local workspace defect to the Codex repair bridge so it can patch and test the workspace outside the current extension runtime.",
      parameters: z.object({
        prompt: z.string().min(20).max(12_000).nullable().optional(),
        cwd: z.string().max(400).nullable().optional(),
        model: z.string().max(160).nullable().optional(),
      }).strict(),
      execute: createToolExecutor("request_codex_repair"),
    }),
    tool({
      name: "get_codex_repair_status",
      description: "Poll the local Codex repair bridge for the status and result of a previously submitted repair job.",
      parameters: z.object({
        jobId: z.string().min(1).max(160).nullable().optional(),
        waitForMs: z.number().int().min(0).max(120_000).nullable().optional(),
        pollIntervalMs: z.number().int().min(500).max(10_000).nullable().optional(),
      }).strict(),
      execute: createToolExecutor("get_codex_repair_status"),
    }),
    tool({
      name: "run_agentic_smoke",
      description: "Run the reviewed capability bundle through the real craft runtime to verify tool-calling behavior before training.",
      parameters: z.object({
        prompt: z.string().min(2).max(4_000),
        maxTurns: z.number().int().min(1).max(8).nullable().optional(),
        bundleOverride: createBundleOverrideSchema().nullable().optional(),
        experiment: createExperimentMetadataSchema().nullable().optional(),
      }).strict(),
      execute: createToolExecutor("run_agentic_smoke"),
    }),
    tool({
      name: "run_capability_eval",
      description: "Run a small manual multi-case evaluation against the reviewed capability bundle through the real craft runtime.",
      parameters: z.object({
        cases: z.array(createCapabilityEvalCaseSchema()).min(1).max(8),
        bundleOverride: createBundleOverrideSchema().nullable().optional(),
        experiment: createExperimentMetadataSchema().nullable().optional(),
      }).strict(),
      execute: createToolExecutor("run_capability_eval"),
    }),
    tool({
      name: "generate_training_batch",
      description: "Generate a grounded canary or pilot batch of training rows through the local model-factory bridge and merge new rows into the craft dataset.",
      parameters: z.object({
        objective: z.string().max(600).nullable().optional(),
        jobKind: z.enum([
          "pair_generation_probe",
          "pair_generation_canary",
          "pair_generation_pilot",
          "augmentation_canary",
          "augmentation_pilot",
        ]).nullable().optional(),
        maxItems: z.number().int().min(1).max(256).nullable().optional(),
        split: z.enum(["train", "validation", "test"]).nullable().optional(),
        status: z.enum(["draft", "review", "ready", "blocked"]).nullable().optional(),
        modelRef: z.string().max(160).nullable().optional(),
        notes: z.string().max(600).nullable().optional(),
        experiment: createExperimentMetadataSchema().nullable().optional(),
      }).strict(),
      execute: createToolExecutor("generate_training_batch"),
    }),
    tool({
      name: "draft_training_changes",
      description: "Draft and apply concrete training-data operations for the current craft.",
      parameters: z.object({
        objective: z.string().max(600).nullable(),
        experiment: createExperimentMetadataSchema().nullable().optional(),
      }).strict(),
      execute: createToolExecutor("draft_training_changes"),
    }),
    tool({
      name: "start_training_run",
      description: "Start the local Qwen training run for this craft through the existing offscreen training runtime.",
      parameters: z.object({
        craftId: z.string().max(120).nullable().optional(),
        shardId: z.string().max(160).nullable().optional(),
        modelName: z.string().max(240).nullable().optional(),
        datasetPayload: createLooseJsonObjectSchema().nullable().optional(),
        persistBundle: z.boolean().nullable().optional(),
        smokeMode: z.string().max(80).nullable().optional(),
        configOverrides: createLooseJsonObjectSchema().nullable().optional(),
        experiment: createExperimentMetadataSchema().nullable().optional(),
      }).strict(),
      execute: createToolExecutor("start_training_run"),
    }),
    tool({
      name: "get_training_run_status",
      description: "Poll the status of an already started local Qwen training run.",
      parameters: z.object({
        jobId: z.string().min(1).max(160).nullable().optional(),
        waitForMs: z.number().int().min(0).max(120_000).nullable().optional(),
        pollIntervalMs: z.number().int().min(500).max(15_000).nullable().optional(),
        experiment: createExperimentMetadataSchema().nullable().optional(),
      }).strict(),
      execute: createToolExecutor("get_training_run_status"),
    }),
    tool({
      name: "record_experiment_decision",
      description: "Record whether a challenger should be kept, discarded, or parked after a smoke, eval, or training comparison.",
      parameters: createExperimentDecisionSchema(),
      execute: createToolExecutor("record_experiment_decision"),
    }),
    tool({
      name: "update_craft_maturity",
      description:
        "Set the explicit craft maturity percentage shown in the UI. Use this only for real milestone updates, never as a heuristic. Before a trained capability artifact exists, prefer skipping this tool unless a durable 0% rationale must be shown.",
      parameters: z.object({
        summary: z.string().max(240).nullable().optional(),
        report: REPORT_TOOL_SCHEMA.nullable().optional(),
        maturity: MATURITY_TOOL_SCHEMA,
        provenance: z.array(PROVENANCE_TOOL_ENTRY_SCHEMA).max(4).optional(),
      }).strict(),
      execute: async (params = {}) => {
        const maturity = await resolveRunMaturity(run, params?.maturity, run.maturity);
        return {
          ok: true,
          summary: asText(params?.summary) || `Craft maturity updated to ${maturity.percent}%.`,
          data: {
            maturity,
          },
          report: normalizeReport(params?.report, run.report),
          maturity,
          provenance: normalizeProvenance(Array.isArray(params?.provenance) ? params.provenance : []),
        };
      },
    }),
    tool({
      name: ASK_USER_TOOL_NAME,
      description: "Interrupt the run with blocking user questions and continue only after answers are approved.",
      parameters: z.object({
        summary: z.string().max(240).nullable(),
        report: REPORT_TOOL_SCHEMA.nullable(),
        maturity: MATURITY_TOOL_SCHEMA.nullable().optional(),
        provenance: z.array(PROVENANCE_TOOL_ENTRY_SCHEMA).max(12),
        questions: z.array(QUESTION_TOOL_SCHEMA).min(1).max(4),
      }).strict(),
      needsApproval: async (runContext, params, callId) => {
        const answersByCallId =
          runContext?.context?.approvedQuestionAnswersByCallId &&
          typeof runContext.context.approvedQuestionAnswersByCallId === "object"
            ? runContext.context.approvedQuestionAnswersByCallId
            : {};
        const approvedAnswers = Array.isArray(answersByCallId?.[callId]) ? answersByCallId[callId] : [];
        const approvedById = new Map(
          approvedAnswers.map((entry) => [asText(entry?.id), asText(entry?.answer)]),
        );
        const clarificationAssessment = assessClarificationRequest(run, params);
        if (!clarificationAssessment.requireApproval) {
          return false;
        }
        return normalizeQuestions(params?.questions).some((entry) => !approvedById.get(entry.id));
      },
      execute: async (params = {}, runContext, details) => {
        const callId = asText(details?.toolCall?.callId);
        const answersByCallId =
          runContext?.context?.approvedQuestionAnswersByCallId &&
          typeof runContext.context.approvedQuestionAnswersByCallId === "object"
            ? runContext.context.approvedQuestionAnswersByCallId
            : {};
        const approvedAnswers = Array.isArray(answersByCallId?.[callId]) ? answersByCallId[callId] : [];
        const approvedById = new Map(
          approvedAnswers.map((entry) => [asText(entry?.id), asText(entry?.answer)]),
        );
        const clarificationAssessment = assessClarificationRequest(run, params);
        if (!clarificationAssessment.requireApproval) {
          const result = {
            ok: true,
            summary: clarificationAssessment.summary,
            data: {
              callId,
              suppressed: true,
              reasonCode: clarificationAssessment.reasonCode,
              answers: [],
            },
            report: normalizeReport(params?.report, {
              nextAction: clarificationAssessment.nextAction,
            }),
            maturity: await resolveRunMaturity(run, params?.maturity, run.maturity),
            provenance: normalizeProvenance(Array.isArray(params?.provenance) ? params.provenance : []),
            questions: [],
          };
          await applyToolOutcomeToRun(run, ASK_USER_TOOL_NAME, params, result);
          return result;
        }
        const questions = normalizeQuestions(params?.questions).map((entry) => ({
          ...entry,
          answer: approvedById.get(entry.id) || "",
        }));
        const result = {
          ok: true,
          summary:
            asText(params?.summary) ||
            `User answered ${questions.filter((entry) => asText(entry.answer)).length} clarification prompts.`,
          data: {
            callId,
            answers: questions.map((entry) => ({
              id: entry.id,
              question: entry.question,
              reason: entry.reason,
              answer: entry.answer,
            })),
          },
          report: normalizeReport(params?.report, {
            nextAction: "Use the approved user answers to continue the craft run.",
          }),
          maturity: await resolveRunMaturity(run, params?.maturity, run.maturity),
          provenance: normalizeProvenance(Array.isArray(params?.provenance) ? params.provenance : []),
          questions,
        };
        await applyToolOutcomeToRun(run, ASK_USER_TOOL_NAME, params, result);
        return result;
      },
    }),
  ];
}

async function createRunBaseModel(run) {
  const providerType = asText(run.resolved?.provider?.type).toLowerCase();
  return providerType === "local_qwen"
    ? aisdk(createLocalQwenAiSdkLanguageModel(run))
    : aisdk(
        await getLanguageModelForSlot({
          slotId: run.resolved?.slotId || "agent",
          modelRef: run.modelRef,
          parameters: run.parameters,
          reasoningEffort: run.reasoningEffort,
        }),
      );
}

async function createTrainingChangesAgent(run) {
  return new Agent({
    name: "TrainingChangePlanner",
    model: await createRunBaseModel(run),
    instructions: buildTrainingChangesAgentInstructions(run),
    outputType: CRAFTING_AGENT_TRAINING_DRAFT_CONTRACT,
  });
}

async function createCraftNamingAgent(run) {
  return new Agent({
    name: "CraftNamePlanner",
    model: await createRunBaseModel(run),
    instructions: buildCraftNamingAgentInstructions(),
    outputType: CRAFTING_AGENT_NAME_CONTRACT,
  });
}

async function createCraftingAgent(run) {
  const baseModel = await createRunBaseModel(run);

  return new Agent({
    name: "CraftingAgentSupervisor",
    model: baseModel,
    instructions: buildCraftingAgentInstructions(run),
    tools: buildCraftingAgentTools(run),
    outputType: CRAFTING_AGENT_FINAL_OUTPUT_CONTRACT,
    modelSettings: {
      parallelToolCalls: false,
    },
  });
}

function attachRunnerListeners(runner, run) {
  runner.on("agent_tool_start", (_runContext, _agent, toolDef) => {
    const toolName = asText(toolDef?.name || "tool");
    const meta = getCraftingAgentToolLogMetadata(toolName);
    run.phase = "tooling";
    pushRunLog(run, "info", `Tool starts: ${toolName}.`, {
      kind: "tool",
      toolName,
      title: meta.title,
      detail: meta.startDetail,
      stageId: meta.stageId,
      status: "running",
    });
  });
  runner.on("agent_tool_end", (_runContext, _agent, toolDef, output) => {
    const toolName = asText(toolDef?.name || "tool");
    const meta = getCraftingAgentToolLogMetadata(toolName);
    const summary = summarizeToolStringOutput(output);
    const outputObject = output && typeof output === "object" ? output : null;
    const outputPayload = outputObject || (typeof output === "string" ? parseJsonLoose(output) : null);
    const toolFailed = outputPayload?.ok === false || isToolFailureLikeText(summary || outputPayload?.error);
    const status =
      toolFailed
        ? "error"
        : toolName === ASK_USER_TOOL_NAME
          ? "done"
          : "done";
    const level =
      status === "error"
        ? "error"
        : toolName === ASK_USER_TOOL_NAME
          ? "info"
          : "success";
    pushRunLog(
      run,
      level,
      summary ? `${toolName}: ${summary}` : `Tool completed: ${toolName}.`,
      {
        kind: "tool",
        toolName,
        title: meta.title,
        detail: summary || meta.startDetail,
        stageId: meta.stageId,
        status,
      },
    );
  });
  runner.on("agent_end", () => {
    pushRunLog(run, "success", "Agent segment completed.");
  });
}

function handleSdkStreamEvent(run, stream, event) {
  if (!event || typeof event !== "object") return;
  if (event.type === "raw_model_stream_event" && event.data?.type === "response_done") {
    syncRunCheckpoint(run, stream.state);
    return;
  }
  if (event.type !== "run_item_stream_event") return;

  if (event.name === "reasoning_item_created") {
    const reasoningText = trimText(
      event?.item?.rawItem?.rawContent?.[0]?.text ||
        event?.item?.rawItem?.content?.[0]?.text ||
        "",
      220,
    );
    if (reasoningText) {
      pushRunLog(run, "info", `Reasoning: ${reasoningText}`);
    }
    syncRunCheckpoint(run, stream.state);
    return;
  }

  if (event.name === "tool_approval_requested") {
    const pending = extractAskUserInterruptions([event.item])[0];
    if (pending) {
      run.pendingApprovalCallId = pending.callId;
      run.questions = mergeQuestionsWithAnswers(pending.questions, run.previousQuestions);
      run.activeTab = "questions";
      setRunReport(run, pending.report, {
        nextAction: "Answer the open questions and continue the same agent run.",
      });
      if (pending.provenance.length) {
        pushRunProvenance(run, pending.provenance);
      }
      pushRunLog(
        run,
        "warn",
        `${run.questions.length || 1} user clarification${run.questions.length === 1 ? "" : "s"} requested.`,
      );
    }
    syncRunCheckpoint(run, stream.state);
    return;
  }

  if (event.name === "tool_output" || event.name === "message_output_created") {
    syncRunCheckpoint(run, stream.state);
  }
}

async function applyInterruptionToRun(run, state, interruptions) {
  const pendingInterruptions = extractAskUserInterruptions(interruptions);
  if (!pendingInterruptions.length) {
    throw new Error("Unsupported agent interruption. Only user clarification interruptions are expected.");
  }
  const pending = pendingInterruptions[0];
  syncRunCheckpoint(run, state);
  run.pendingApprovalCallId = pending.callId;
  run.questions = mergeQuestionsWithAnswers(pending.questions, run.previousQuestions);
  run.status = "needs_input";
  run.phase = "complete";
  run.finalStatus = "";
  run.completedAt = "";
  run.activeTab = "questions";
  setRunReport(run, pending.report, {
    nextAction: "Answer the open questions and continue the same agent run.",
  });
  if (pending.provenance.length) {
    pushRunProvenance(run, pending.provenance);
  }
  if (pending.maturity) {
    await setRunMaturity(run, pending.maturity, run.maturity);
  }
}

async function applyFinalOutputToRun(run, rawFinalOutput, state = null) {
  const finalOutput = normalizeCraftingAgentFinalOutput(rawFinalOutput);
  const finishedAt = new Date().toISOString();
  const recoverableIteration = detectRecoverableValidationIteration(run, finalOutput);
  const toolingBlocker = recoverableIteration ? null : detectDevelopmentToolingBlocker(run, finalOutput);
  let effectiveStatus = toolingBlocker ? "blocked" : recoverableIteration ? "continue" : finalOutput.status;
  if (finalOutput.report || toolingBlocker || recoverableIteration) {
    setRunReport(run, finalOutput.report, {
      currentState: toolingBlocker?.currentState || recoverableIteration?.currentState || undefined,
      nextAction:
        toolingBlocker?.nextAction ||
        recoverableIteration?.nextAction ||
        (
          run.operations.length
            ? "Review the live training data in the separate view."
            : "Training data can now be reviewed in the separate view."
        ),
    });
  }
  if (finalOutput.provenance.length) {
    pushRunProvenance(run, finalOutput.provenance);
  }
  if (finalOutput.maturity) {
    await setRunMaturity(run, finalOutput.maturity, run.maturity);
  }
  if (finalOutput.suggestedName?.name) {
    run.suggestedName = normalizeCraftNameCandidate(finalOutput.suggestedName.name);
  }
  if (finalOutput.officialDescription?.text) {
    run.officialDescription = finalOutput.officialDescription.text;
  }
  if (finalOutput.useSurface && typeof finalOutput.useSurface === "object") {
    run.useSurface = cloneJson(finalOutput.useSurface, null);
  }
  run.responseText =
    asText(finalOutput.responseText) ||
    asText(finalOutput.summary) ||
    asText(toolingBlocker?.currentState) ||
    asText(run.responseText) ||
    "The crafting agent completed the current run.";
  const forcedTrainingContinuation = await maybeForceTrainingContinuation(run, effectiveStatus);
  if (forcedTrainingContinuation?.reportOverride) {
    setRunReport(run, run.report, forcedTrainingContinuation.reportOverride);
  }
  if (forcedTrainingContinuation?.responseText) {
    run.responseText = forcedTrainingContinuation.responseText;
  }
  if (forcedTrainingContinuation?.nextStatus) {
    effectiveStatus = forcedTrainingContinuation.nextStatus;
  }
  const forcedDatasetGrowthContinuation = await maybeForceDatasetGrowthContinuation(run, effectiveStatus);
  if (forcedDatasetGrowthContinuation?.reportOverride) {
    setRunReport(run, run.report, forcedDatasetGrowthContinuation.reportOverride);
  }
  if (forcedDatasetGrowthContinuation?.responseText) {
    run.responseText = forcedDatasetGrowthContinuation.responseText;
  }
  if (forcedDatasetGrowthContinuation?.nextStatus) {
    effectiveStatus = forcedDatasetGrowthContinuation.nextStatus;
  }
  await maybeRunWorkspaceDiagnosis(run, finalOutput, effectiveStatus);
  const autoRepair = await maybeRunAutomaticCodexRepair(run, finalOutput, effectiveStatus);
  if (autoRepair?.reportOverride) {
    setRunReport(run, run.report, autoRepair.reportOverride);
  }
  if (autoRepair?.responseText) {
    run.responseText = autoRepair.responseText;
  }
  if (autoRepair?.nextStatus) {
    effectiveStatus = autoRepair.nextStatus;
  }
  const decisionMessage = buildFinalDecisionLogMessage(
    effectiveStatus,
    run.report,
    finalOutput.summary || finalOutput.responseText || run.responseText,
  );
  const nextActionText = trimText(asText(run.report?.nextAction), 220);
  pushRunLog(
    run,
    effectiveStatus === "blocked" ? "warn" : effectiveStatus === "continue" ? "info" : "success",
    decisionMessage,
  );
  if (toolingBlocker && effectiveStatus === "blocked" && finalOutput.status !== "blocked") {
    pushRunLog(
      run,
      "warn",
      "Development mode: the run will not continue through workarounds when tool paths are missing or broken.",
    );
  }
  if (recoverableIteration && effectiveStatus === "continue" && finalOutput.status !== "continue") {
    pushRunLog(
      run,
      "info",
      "Development mode: a failed validation run of the internal script path is treated as normal iteration.",
    );
  }
  if (nextActionText && (!toolingBlocker && !recoverableIteration || autoRepair?.reportOverride)) {
    pushRunLog(run, "info", `Next suggested step: ${nextActionText}`);
  } else if (toolingBlocker?.nextAction) {
    pushRunLog(run, "info", `Next suggested step: ${trimText(toolingBlocker.nextAction, 220)}`);
  } else if (recoverableIteration?.nextAction) {
    pushRunLog(run, "info", `Next suggested step: ${trimText(recoverableIteration.nextAction, 220)}`);
  } else if (effectiveStatus !== "continue" && finalOutputHasLimitationNote(finalOutput)) {
    pushRunLog(
      run,
      "warn",
      "The run stops with a known limitation: one requested part remains open because no approved tool path exists for it.",
    );
  }
  if (effectiveStatus === "continue") {
    if (autoRepair?.requestExtensionReload) {
      const reloading = await scheduleExtensionReloadForAutoRepair(run, state);
      if (reloading) {
        return;
      }
    }
    run.status = "running";
    run.phase = "planning";
    run.finalStatus = "continue";
    run.completedAt = "";
    run.activeTab = "progress";
    return;
  }
  if (effectiveStatus === "blocked") {
    run.status = "blocked";
    run.phase = "blocked";
    run.finalStatus = "blocked";
    run.completedAt = finishedAt;
    run.error = asText(run.report?.currentState || toolingBlocker?.currentState || finalOutput.summary || finalOutput.responseText || "The crafting agent is blocked.");
    run.errorDetail = resolveRunErrorDetail(run, run?.lastToolFailure?.errorDetail);
    pushRunLog(run, "error", run.error);
    finalizeRunRetention(run);
    return;
  }
  run.status = "done";
  run.phase = "complete";
  run.finalStatus = "done";
  run.completedAt = finishedAt;
  run.activeTab = "progress";
  finalizeRunRetention(run);
}

async function maybeSuggestCraftName(run) {
  if (!shouldDeriveCraftNameSuggestion(run) || asText(run.suggestedName)) return;
  try {
    const namingAgent = await createCraftNamingAgent(run);
    const namingRunner = new Runner();
    const namingResult = await namingRunner.run(
      namingAgent,
      buildCraftNamingRunInput(run),
    );
    const suggestion = normalizeStructuredNameSuggestionContract(namingResult.finalOutput || null);
    if (!suggestion?.name) return;
    const suggestedName = normalizeCraftNameCandidate(suggestion.name);
    if (
      !suggestedName ||
      normalizeNameComparison(suggestedName) === normalizeNameComparison(run.craft?.name) ||
      isPlaceholderCraftName(suggestedName)
    ) {
      return;
    }
    run.suggestedName = suggestedName;
    pushRunLog(run, "success", `Suggested craft name: ${suggestedName}.`);
    pushRunProvenance(run, [
      {
        title: "Craft-Name vorgeschlagen",
        detail: suggestion.reason ? `${suggestedName} · ${suggestion.reason}` : suggestedName,
        kind: "match",
      },
    ]);
  } catch (error) {
    pushRunLog(
      run,
      "warn",
      `Name suggestion skipped: ${error instanceof Error ? error.message : String(error || "Unknown naming error.")}`,
    );
  }
}

function isAbortLikeError(error) {
  if (!error) return false;
  if (asText(error?.name) === "AbortError") return true;
  return /abort(ed)?/i.test(asText(error?.message || error));
}

function markRunFailed(run, errorMessage, nextAction = "", errorDetail = null) {
  run.status = "failed";
  run.phase = "failed";
  run.error = asText(errorMessage) || "Crafting agent run failed.";
  run.errorDetail = resolveRunErrorDetail(run, errorDetail);
  run.completedAt = new Date().toISOString();
  setRunReport(run, run.report, {
    nextAction:
      nextAction ||
      "Inspect the blocker, add context if needed, and start the run again.",
  });
  pushRunLog(run, "error", run.error);
  emitDevObservabilityEvent(run, "run_failed", {
    run: buildRunObservabilitySnapshot(run),
    error: trimText(run.error, 400),
  });
  finalizeRunRetention(run);
}

function markRunStopped(run, reasonMessage = "") {
  const message = asText(reasonMessage) || asText(run.stopReason) || "The crafting agent was stopped.";
  run.status = "stopped";
  run.phase = "stopped";
  run.error = message;
  run.finalStatus = "";
  run.completedAt = new Date().toISOString();
  run.activeTab = "progress";
  run.stopRequested = false;
  run.stopReason = message;
  run.serializedState = "";
  run.pendingApprovalCallId = "";
  run.pendingUserAnswersByCallId = {};
  run.questions = [];
  setRunReport(
    run,
    {
      ...(run.report && typeof run.report === "object" ? run.report : {}),
      currentState: message,
      nextAction: "Adjust the brief and start a fresh run when you want to continue.",
    },
    {},
  );
  if (asText(run.logs?.[run.logs.length - 1]?.message) !== message) {
    pushRunLog(run, "warn", message);
  }
  finalizeRunRetention(run);
}

function requestRunStop(run, reasonMessage = "") {
  const message = asText(reasonMessage) || "The crafting agent was stopped by the user.";
  run.stopRequested = true;
  run.stopReason = message;
  if (run.abortController && !run.abortController.signal.aborted) {
    try {
      run.abortController.abort(new DOMException(message, "AbortError"));
    } catch {
      run.abortController.abort();
    }
  }
  markRunStopped(run, message);
  run.updatedAt = new Date().toISOString();
  emitDevObservabilityEvent(run, "run_stopped", {
    run: buildRunObservabilitySnapshot(run),
    reason: trimText(message, 240),
  });
  return message;
}

async function ensureRunResolved(run) {
  if (run?.resolved?.provider && asText(run?.modelRef)) return;

  run.status = "starting";
  run.phase = "resolving";
  setRunReport(run, run.report, {
    currentState: "Resolving the configured model and provider.",
    nextAction: "",
  });
  pushRunLog(run, "info", "Resolving the configured model and provider.");

  const resolved = await resolveProviderAndModel({
    slotId: run?.requestedSlotId || run?.resolved?.slotId || "agent",
    modelRef: run?.requestedModelRef || run?.modelRef || "",
    providerId: run?.requestedProviderId || run?.resolved?.providerId || "",
    modelName: run?.requestedModelName || run?.resolved?.modelName || "",
    parameters: run?.parameters,
    reasoningEffort: run?.reasoningEffort,
  });

  run.resolved = cloneJson(resolved, {});
  run.modelRef = asText(resolved.modelRef);
  run.mode = asText(resolved?.provider?.type).toLowerCase() === "local_qwen" ? "local" : "remote";
  run.reasoningEffort = asText(resolved.reasoningEffort || run.reasoningEffort);
  run.updatedAt = new Date().toISOString();
}

async function ensureRunInitialized(run) {
  if (run.initialized) return;
  run.samples = await readTrainingSamplesForCraft(run.craftId, run.samples);
  await refreshRunCapabilityEvidence(run);
  syncRunBrowserContext(run, await queryActiveTabContext(), { clearOnNull: true });
  setRunReport(
    run,
    buildDefaultReport(run, {
      nextAction: `Wait for the next agent step. Supervisor tools: ${formatCraftingAgentToolLabels(run.agentTooling)}.`,
    }),
  );
  pushRunLog(run, "info", "Local training data is loaded for the agent run.");
  pushRunLog(
    run,
    "info",
    run.samples.length
      ? `${run.samples.length} training rows are available as local reference.`
      : "No seed rows exist yet. The agent must first stabilize the target structure.",
  );
  if (run.activeTabContext?.url) {
    pushRunLog(run, "info", `Active browser context detected: ${run.activeTabContext.url}`);
  }
  pushRunLog(
    run,
    "info",
    `Supervisor tools are fixed: ${formatCraftingAgentToolLabels(run.agentTooling)}.`,
  );
  run.initialized = true;
}

function buildApprovedAnswersForPendingCall(run, providedQuestions = []) {
  const pendingQuestions = normalizeQuestions(run.questions);
  const merged = mergeQuestionsWithAnswers(pendingQuestions, providedQuestions);
  const unanswered = merged.filter((entry) => !asText(entry.answer));
  if (unanswered.length) {
    throw new Error(`Open clarification answers are still missing (${unanswered.length}).`);
  }
  return merged.map((entry) => ({
    id: entry.id,
    question: entry.question,
    reason: entry.reason,
    answer: entry.answer,
  }));
}

async function executeSdkSegment(run) {
  if (run.stopRequested || asText(run.status) === "stopped") {
    return {
      type: "stopped",
      state: null,
      error: null,
    };
  }
  const agent = await createCraftingAgent(run);
  const runner = new Runner();
  attachRunnerListeners(runner, run);
  const abortController = new AbortController();
  run.abortController = abortController;

  let input = buildCraftingAgentRunInput(run);
  const segmentTurnBudget = Math.max(1, Math.min(run.segmentMaxTurns, run.maxTurns - run.turnOffset));
  const startingFromSerializedState = Boolean(asText(run.serializedState));

  if (startingFromSerializedState) {
    const state = await RunState.fromString(agent, run.serializedState);
    applyRunContextToState(run, state);
    const answerPayload = Array.isArray(run.pendingUserAnswersByCallId?.[run.pendingApprovalCallId])
      ? run.pendingUserAnswersByCallId[run.pendingApprovalCallId]
      : [];
    if (answerPayload.length) {
      const interruptions =
        typeof state.getInterruptions === "function" ? state.getInterruptions() : [];
      const approvalItem = interruptions.find((entry) => {
        const toolName = asText(entry?.toolName || entry?.name || entry?.rawItem?.name);
        const callId = asText(entry?.rawItem?.callId || entry?.rawItem?.id);
        return toolName === ASK_USER_TOOL_NAME && callId === run.pendingApprovalCallId;
      });
      if (approvalItem) {
        state.approve(approvalItem);
      }
    }
    input = state;
  } else {
    run.segmentCount += 1;
  }

  run.phase = startingFromSerializedState ? "resuming" : "planning";
  pushRunLog(
    run,
    "info",
    startingFromSerializedState
      ? "SDK run resumes from the serialized agent state."
      : `SDK segment ${run.segmentCount} starts with up to ${segmentTurnBudget} turns.`,
  );

  try {
    const stream = await runner.run(
      agent,
      input,
      startingFromSerializedState
        ? { stream: true, signal: abortController.signal }
        : {
            stream: true,
            maxTurns: segmentTurnBudget,
            context: buildRunContextValue(run),
            signal: abortController.signal,
          },
    );
    for await (const event of stream) {
      handleSdkStreamEvent(run, stream, event);
    }
    await stream.completed;
    syncRunCheckpoint(run, stream.state);
    const interruptions =
      typeof stream.state?.getInterruptions === "function"
        ? stream.state.getInterruptions()
        : Array.isArray(stream.interruptions)
          ? stream.interruptions
          : [];
    if (interruptions.length) {
      return {
        type: "interruption",
        state: stream.state,
        interruptions,
      };
    }
    return {
      type: "final",
      state: stream.state,
      finalOutput: stream.finalOutput ?? null,
    };
  } catch (error) {
    if (error?.state) {
      syncRunCheckpoint(run, error.state);
    }
    if (run.stopRequested || abortController.signal.aborted || isAbortLikeError(error)) {
      return {
        type: "stopped",
        state: error?.state || null,
        error,
      };
    }
    if (error instanceof Error && /Max turns/i.test(error.message)) {
      return {
        type: "compact",
        state: error.state || null,
        error,
      };
    }
    return {
      type: "error",
      state: error?.state || null,
      error,
    };
  } finally {
    if (run.abortController === abortController) {
      run.abortController = null;
    }
  }
}

async function executeRun(run) {
  if (run.executionPromise) return run.executionPromise;

  run.executionPromise = (async () => {
    try {
      await ensureRunResolved(run);
      if (run.stopRequested || asText(run.status) === "stopped") {
        markRunStopped(run, run.stopReason);
        return;
      }
      await ensureRunInitialized(run);
      if (run.stopRequested || asText(run.status) === "stopped") {
        markRunStopped(run, run.stopReason);
        return;
      }
      run.status = "running";
      run.phase = "planning";
      run.error = "";
      while (run.status === "running") {
        if (run.stopRequested || asText(run.status) === "stopped") {
          markRunStopped(run, run.stopReason);
          return;
        }
        if (run.turnOffset >= run.maxTurns) {
          markRunFailed(
            run,
            `Agent reached maxTurns=${run.maxTurns} without a stable result.`,
            "Give the agent more context or answer the open clarifications and try again.",
          );
          return;
        }

        const pendingRepair = await maybeAdvancePendingCodexRepair(run);
        if (pendingRepair?.blocked) {
          return;
        }
        if (pendingRepair?.waiting) {
          continue;
        }

        run.samples = await readTrainingSamplesForCraft(run.craftId, run.samples);
        await refreshRunCapabilityEvidence(run);
        run.activeTabContext = await queryActiveTabContext();

        const segmentResult = await executeSdkSegment(run);
        if (segmentResult.type === "stopped" || run.stopRequested || asText(run.status) === "stopped") {
          markRunStopped(run, run.stopReason);
          return;
        }
        if (segmentResult.type === "interruption") {
          await applyInterruptionToRun(run, segmentResult.state, segmentResult.interruptions);
          return;
        }
        if (segmentResult.type === "final") {
          await applyFinalOutputToRun(run, segmentResult.finalOutput, segmentResult.state);
          if (run.status === "done") {
            await maybeSuggestCraftName(run);
            return;
          }
          if (run.status === "reloading") {
            return;
          }
          if (run.status === "running") {
            prepareRunForFreshSegment(
              run,
              segmentResult.state,
              "Der Agent startet auf Basis des letzten Ergebnisses einen weiteren Durchgang.",
            );
            continue;
          }
          return;
        }
        if (segmentResult.type === "compact") {
          if (!segmentResult.state) {
            throw segmentResult.error || new Error("Context compaction failed without a recoverable state.");
          }
          if (run.compactionCount >= MAX_COMPACTION_PASSES) {
            markRunFailed(
              run,
              "The crafting agent exceeded the compaction budget before reaching a stable result.",
              "Narrow the brief or provide more explicit examples before retrying.",
            );
            return;
          }
          applyCompactionToRun(run, segmentResult.state, "turn budget reached");
          continue;
        }
        if (segmentResult.type === "error") {
          throw segmentResult.error || new Error("Crafting agent segment failed.");
        }
        throw new Error(`Unknown SDK segment result: ${segmentResult?.type || "unknown"}`);
      }
    } catch (error) {
      if (run.stopRequested || asText(run.status) === "stopped" || isAbortLikeError(error)) {
        markRunStopped(run, run.stopReason || (error instanceof Error ? error.message : String(error || "")));
      } else {
        markRunFailed(
          run,
          error instanceof Error ? error.message : String(error || "Crafting agent run failed."),
          "",
          error && typeof error === "object" && "detail" in error ? error.detail : null,
        );
        await maybeRunWorkspaceDiagnosis(run, {
          summary: run.error,
          responseText: run.error,
          report: run.report,
        }, "failed");
      }
    } finally {
      run.updatedAt = new Date().toISOString();
      run.stopRequested = false;
      run.executionPromise = null;
      scheduleRunPersistence(run);
    }
  })();

  return run.executionPromise;
}

export async function startCraftingAgentRun({
  slotId = "agent",
  modelRef = "",
  providerId = "",
  modelName = "",
  craft = null,
  brief = "",
  currentSamples = [],
  previousQuestions = [],
  questionAnswers = [],
  resumeJobId = "",
  parameters = {},
  reasoningEffort = "",
} = {}) {
  const normalizedResumeJobId = asText(resumeJobId);
  if (normalizedResumeJobId) {
    await ensureRunsHydrated();
    const existingRun = RUNS.get(normalizedResumeJobId);
    if (!existingRun) {
      throw new Error("The requested crafting run could not be found anymore.");
    }
    if (!existingRun.serializedState || !existingRun.pendingApprovalCallId) {
      throw new Error("This crafting run is not waiting for a resumable clarification step.");
    }
    const suppliedQuestions = normalizeQuestions(
      Array.isArray(questionAnswers) && questionAnswers.length ? questionAnswers : previousQuestions,
    );
    const approvedAnswers = buildApprovedAnswersForPendingCall(existingRun, suppliedQuestions);
    existingRun.pendingUserAnswersByCallId = {
      ...(
        existingRun.pendingUserAnswersByCallId &&
        typeof existingRun.pendingUserAnswersByCallId === "object"
          ? existingRun.pendingUserAnswersByCallId
          : {}
      ),
      [existingRun.pendingApprovalCallId]: approvedAnswers,
    };
    existingRun.previousQuestions = approvedAnswers;
    existingRun.questions = approvedAnswers;
    existingRun.stopRequested = false;
    existingRun.stopReason = "";
    existingRun.status = "running";
    existingRun.phase = "resuming";
    existingRun.finalStatus = "";
    existingRun.completedAt = "";
    existingRun.error = "";
    existingRun.activeTab = "progress";
    pushRunLog(
      existingRun,
      "info",
      "Clarification answers were recorded. The agent resumes from the serialized SDK state.",
    );
    emitDevObservabilityEvent(existingRun, "run_resumed", {
      run: buildRunObservabilitySnapshot(existingRun),
      reason: "clarification_answers_recorded",
    });
    scheduleRunPersistence(existingRun);
    void executeRun(existingRun);
    return {
      run: createRunSnapshot(existingRun),
    };
  }

  const run = createRun({
    slotId,
    modelRef,
    providerId,
    modelName,
    craft,
    brief,
    currentSamples,
    resolved: null,
    previousQuestions,
    parameters,
    reasoningEffort,
  });
  RUNS.set(run.jobId, run);
  emitDevObservabilityEvent(run, "run_started", {
    run: buildRunObservabilitySnapshot(run),
    brief: trimText(asText(run.brief), 320),
  });
  scheduleRunPersistence(run);
  void executeRun(run).catch((error) => {
    markRunFailed(
      run,
      error instanceof Error ? error.message : String(error || "Crafting agent run failed."),
      "",
      error && typeof error === "object" && "detail" in error ? error.detail : null,
    );
  });
  return {
    run: createRunSnapshot(run),
  };
}

export async function stopCraftingAgentRun(jobId = "", reason = "") {
  await ensureRunsHydrated();
  const run = RUNS.get(asText(jobId));
  if (!run) {
    return {
      run: null,
    };
  }
  if (["done", "needs_input", "blocked", "failed", "stopped"].includes(asText(run.status))) {
    return {
      run: createRunSnapshot(run),
    };
  }
  requestRunStop(run, reason);
  return {
    run: createRunSnapshot(run),
  };
}

export async function getCraftingAgentRun(jobId = "") {
  await ensureRunsHydrated();
  const normalizedJobId = asText(jobId);
  let run = RUNS.get(normalizedJobId);
  if (!run) {
    run = await readPersistedRunByJobId(normalizedJobId);
    if (run) {
      RUNS.set(run.jobId, run);
      pruneRetainedRuns();
    }
  }
  return {
    run: run ? createRunSnapshot(run) : null,
  };
}
