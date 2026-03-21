import "./vendor/react-libs.js";
import {
  BROWSER_CAPABILITY_BUNDLE_ARTIFACT_KIND,
  buildCapabilityBundle,
  createBasePolicyPayload,
  createBaseWeightsPayload,
  getBrowserCapabilityBundleArtifactId,
  getPolicyBundleArtifactId,
  POLICY_BUNDLE_ARTIFACT_KIND,
  TOOL_SCRIPTS_ARTIFACT_KIND,
  getWeightsArtifactId,
  describeCapabilityBundle,
  getToolScriptsArtifactId,
  getTrainingDataArtifactId as getBundleTrainingDataArtifactId,
  normalizeToolScriptsPayload,
  WEIGHTS_ARTIFACT_KIND,
} from "./shared/capability-bundle.mjs";
import {
  compilePublishedBrowserCapabilityBundlePayload,
  normalizeBrowserCapabilityBundlePayload,
} from "./shared/browser-capability-bundle.mjs";
import {
  formatCraftingAgentToolLabels,
  normalizeCraftingAgentToolingPayload,
} from "./shared/crafting-agent-tooling.mjs";
import {
  createEmptyCraftMaturity,
  formatCraftMaturityPhase,
  formatCraftMaturityPercent,
  gateCraftMaturityForCapability,
  normalizeCraftMaturity,
} from "./shared/craft-maturity.mjs";
import {
  LOCAL_QWEN_PIPELINE_SMOKE_CONFIG,
  createLocalQwenPipelineSmokeDatasetPayload,
} from "./shared/local_qwen_pipeline_smoke.mjs";
import {
  LOCAL_QWEN_MULTIMODAL_SMOKE_CONFIG,
  createLocalQwenMultimodalSmokeDatasetPayload,
} from "./shared/local_qwen_multimodal_smoke.mjs";
import {
  inspectPortableTrainingTrace,
  normalizePortableTrainingMessages,
  renderQwen35TrainingInspection,
} from "./shared/local_qwen_training_contract.mjs";
import {
  createDefaultUiPreferences,
  readUiPreferences,
  UI_PREFERENCES_KEY,
} from "./shared/ui-preferences.mjs";
import {
  AGENT_RUN_STATE_ARTIFACT_KIND,
  deleteAgentRunState,
  listAgentRunStates,
  readAgentRunState,
  upsertAgentRunState,
} from "./shared/agent-run-store.mjs";

const __g =
  typeof globalThis !== "undefined"
    ? globalThis
    : typeof window !== "undefined"
      ? window
      : self;

var react = __g.react || { exports: {} };
var reactDom = __g.reactDom || { exports: {} };
var client = __g.client || {};
var hasRequiredReact;
var hasRequiredReactDom;
var hasRequiredClient;
var requireReact_production_min = __g.requireReact_production_min;
var requireReactDom_production_min = __g.requireReactDom_production_min;
var getDefaultExportFromCjs =
  __g.getDefaultExportFromCjs ||
  function getDefaultExportFromCjsLocal(module) {
    return module && module.__esModule ? module.default : module;
  };

var requireReact =
  typeof requireReact !== "undefined"
    ? requireReact
    : function requireReactLocal() {
        if (hasRequiredReact) return react.exports;
        hasRequiredReact = 1;
        react.exports = requireReact_production_min();
        return react.exports;
      };

function requireReactDom() {
  if (hasRequiredReactDom) return reactDom.exports;
  hasRequiredReactDom = 1;
  reactDom.exports = requireReactDom_production_min();
  return reactDom.exports;
}

function requireClient() {
  if (hasRequiredClient) return client;
  hasRequiredClient = 1;
  const module = requireReactDom();
  client.createRoot = module.createRoot;
  client.hydrateRoot = module.hydrateRoot;
  return client;
}

const clientExports = requireClient();
const reactExports = requireReact();
const React = getDefaultExportFromCjs(reactExports);
const h = React.createElement;
const ReactFragment = React.Fragment;

const craftStore = globalThis.SinepanelCraftStorage;
const craftSync = globalThis.SinepanelCraftSync;
const configApi = globalThis.SinepanelModelConfig;
const themeApi = globalThis.SinepanelAppTheme;
const TRAINING_STATUS_POLL_MS = 1000;
const CRAFT_AGENT_START_TIMEOUT_MS = 12000;
const LOCAL_STORAGE_TIMEOUT_MS = 1800;
const LOCAL_STORAGE_TIMEOUT_RETRY_MS = 6000;
const LOCAL_STORAGE_DELETE_TIMEOUT_MS = 6000;
const SETUP_SAVE_TIMEOUT_MS = 20000;
const SIDEPANEL_STATE_ARTIFACT_KIND = "sidepanel_state";
const SIDEPANEL_STATE_ARTIFACT_ID = "sidepanel-state:local";
const SIDEPANEL_STATE_ARTIFACT_CRAFT_ID = "__sidepanel__";
const SIDEPANEL_SETUP_COMPLETED_KEY = "sinepanel.sidepanel.setup-completed.v1";
const STARTER_TUTORIAL_SEEN_KEY = "sinepanel.sidepanel.starter-tutorial-seen.v1";
const STARTER_TUTORIAL_EXAMPLES = Object.freeze([
  {
    title: "Text correction in place",
    prompt:
      "Fix spelling and grammar for the selected, focused, or copied text (in that priority order) and replace it in place.",
  },
  {
    title: "Classified listing outreach",
    prompt:
      "Search classified listings for offers that match predefined criteria, identify strong opportunities, and contact the seller with a prepared message of interest. The user provides the search target and the criteria to look for.",
  },
  {
    title: "Bug report from current page",
    prompt:
      "Analyze the current page, capture the relevant context, and create a prefilled bug report email with the necessary details copied to the clipboard.",
  },
  {
    title: "Newsletter digest",
    prompt:
      "Search news sites for a given topic and generate a newsletter-style HTML digest with selected articles, short summaries, and links to the original sources. The user provides the topic or subject area to cover.",
  },
  {
    title: "Walkable restaurant picker",
    prompt:
      "Find walkable nearby restaurants on Google Maps, verify opening hours and menus, filter by saved preferences such as vegan options, and recommend the best matches based on reviews when many options are available. The user can provide additional filters such as cuisine, distance, or dietary preferences.",
  },
]);
const CRAFT_INPUT_MODES = ["free_text", "mixed", "selection", "current_tab", "context_only"];
const CRAFT_SYNC_SETTINGS_KEY = craftSync?.SETTINGS_KEY || "sinepanel.craft-sync.settings.v1";
const DEFAULT_SYNC_SETTINGS = {
  displayName: "",
  signalingUrls: craftSync?.DEFAULT_SIGNALING_URL || "wss://api.metricspace.org/signal",
  token: "",
  tokenAutoGenerated: true,
  mode: "off",
};
const TRAINING_DATA_ARTIFACT_KIND = "training_samples";
const TRAINING_DATA_PAGE_SIZE = 5;
const DEFAULT_DEBUG_FIXED_TRAINING_CRAFT_ID = "__debug_fixed_training__";
const INITIAL_PROVIDERS = configApi?.createDefaultProviders?.() || {};
const INITIAL_SLOTS = configApi?.createDefaultModelSlots?.() || {};
const EXTENSION_VERSION = globalThis.chrome?.runtime?.getManifest?.().version || "0.1.0";
const INITIAL_REQUIRED_SETUP = configApi?.getRequiredSetupStatus?.(INITIAL_SLOTS, INITIAL_PROVIDERS) || {
  ready: true,
  items: [],
  missingItems: [],
};
const DEBUG_FIXED_TRAINING = {
  craftId: String(craftStore?.DEBUG_FIXED_TRAINING?.craftId || "").trim() || DEFAULT_DEBUG_FIXED_TRAINING_CRAFT_ID,
  shardId: String(craftStore?.DEBUG_FIXED_TRAINING?.shardId || "").trim() || "debug-fixed-run",
  modelName: String(craftStore?.DEBUG_FIXED_TRAINING?.modelName || "").trim() || "unsloth/Qwen3.5-0.8B",
};
const DEBUG_PROOFREAD_TRAINING = Object.freeze({
  relativeDatasetPath: "assets/training/proofread_toolcall_qwen35_dataset.json",
  shardIdSuffix: "proofread-webgpu-smoke",
  maxTrainPairs: 32,
  maxValidationPairs: 8,
  maxTestPairs: 8,
  maxSeqLen: 256,
  modelBatchSize: 1,
  batchTokens: 256,
  epochs: 1,
});
const DEV_PROOFREAD_WORKSPACE_CRAFT_ID = "__dev_proofread_workspace__";
const DEV_PROOFREAD_WORKSPACE_CRAFT_NAME = "Proofread Workspace Smoke";
const DEV_WORKSPACE_ROUNDTRIP_CRAFT_ID = "__dev_workspace_roundtrip__";
const DEV_WORKSPACE_ROUNDTRIP_CRAFT_NAME = "Workspace Roundtrip Smoke";
const HEADER_TEST_SUCCESS_HIDE_MS = 1400;
const HEADER_TEST_SESSION_KEY = "sinepanel.sidepanel.header-tests.session.v1";
const HEADER_TEST_ROW_DEFS = Object.freeze([
  {
    id: "qwen",
    label: "Qwen 3.5",
    tests: ["inference", "vision", "selfUse", "agentic"],
    includeMenu: false,
  },
  {
    id: "training",
    label: "Training",
    tests: ["finetuning", "proofreadFinetuning", "visionFinetuning"],
    includeMenu: false,
  },
  {
    id: "workspace",
    label: "Workspace",
    tests: ["workspaceRoundtrip"],
    includeMenu: false,
  },
  {
    id: "tools",
    label: "Tool testing",
    tests: ["toolTabs", "toolVisual", "toolCode", "toolIntegrated"],
    includeMenu: true,
  },
]);
const HEADER_TEST_DEFS = Object.freeze({
  inference: {
    label: "Test Inference",
    action: "run-inference-smoke",
    variant: "quiet",
  },
  agentic: {
    label: "Test Agentic",
    action: "run-agent-smoke",
    variant: "quiet",
  },
  vision: {
    label: "Test Vision",
    action: "run-vision-smoke",
    variant: "quiet",
  },
  selfUse: {
    label: "Test Self Use",
    action: "run-self-use-smoke",
    variant: "quiet",
  },
  finetuning: {
    label: "Test Finetuning",
    action: "start-header-training",
    variant: "default",
  },
  proofreadFinetuning: {
    label: "Test Proofread FT",
    action: "start-header-training",
    variant: "default",
  },
  visionFinetuning: {
    label: "Test Vision FT",
    action: "start-header-training",
    variant: "default",
  },
  workspaceRoundtrip: {
    label: "Test Workspace",
    action: "run-workspace-roundtrip-smoke",
    variant: "default",
  },
  toolTabs: {
    label: "Test Tabs",
    action: "run-tool-tabs-smoke",
    variant: "quiet",
  },
  toolVisual: {
    label: "Test Visual",
    action: "run-tool-visual-smoke",
    variant: "quiet",
  },
  toolCode: {
    label: "Test Code",
    action: "run-tool-code-smoke",
    variant: "quiet",
  },
  toolIntegrated: {
    label: "Test Integrated",
    action: "run-tool-integrated-smoke",
    variant: "quiet",
  },
});

function createEmptySmokeTestState() {
  return {
    status: "idle",
    message: "",
    detail: "",
    helpText: "",
    failureKind: "",
    updatedAt: "",
    reportText: "",
    copyStatus: "idle",
    testId: "",
  };
}

function createHeaderTestEntry(status = "idle", progress = 0, indeterminate = false) {
  return {
    status: String(status || "idle"),
    progress: Math.max(0, Math.min(1, Number(progress) || 0)),
    indeterminate: indeterminate === true,
  };
}

function createHeaderTestUiState() {
  return {
    inference: createHeaderTestEntry(),
    agentic: createHeaderTestEntry(),
    vision: createHeaderTestEntry(),
    selfUse: createHeaderTestEntry(),
    finetuning: createHeaderTestEntry(),
    visionFinetuning: createHeaderTestEntry(),
    workspaceRoundtrip: createHeaderTestEntry(),
    toolTabs: createHeaderTestEntry(),
    toolVisual: createHeaderTestEntry(),
    toolCode: createHeaderTestEntry(),
    toolIntegrated: createHeaderTestEntry(),
  };
}

const state = {
  crafts: [],
  activeCraftId: null,
  craftingCraftId: null,
  createOpen: false,
  draftName: "",
  draftGoal: "",
  promptDrafts: {},
  agentPromptDrafts: {},
  agentRuns: {},
  trainingRuns: {},
  useMessages: {},
  craftResponses: {},
  trainingDataCraftId: null,
  trainingDataStates: {},
  toolScriptStates: {},
  capabilityWeightsStates: {},
  debugTrainingRun: null,
  smokeTest: createEmptySmokeTestState(),
  headerTests: createHeaderTestUiState(),
  syncSettings: { ...DEFAULT_SYNC_SETTINGS },
  syncSnapshot: craftSync?.getState?.() || null,
  providers: INITIAL_PROVIDERS,
  slots: INITIAL_SLOTS,
  setupLoaded: true,
  requiredSetup: INITIAL_REQUIRED_SETUP,
  setupCompleted: false,
  setupSaving: false,
  setupError: "",
  startupError: "",
  tutorialOverlay: null,
  starterTutorialSeen: false,
  themeId: themeApi?.DEFAULT_THEME_ID || "copyshop",
  uiPreferences: createDefaultUiPreferences(),
};

const panelRoot = document.querySelector("#panel-root");
let sidepanelReactRoot = null;
let sidepanelReactMounted = false;
let sidepanelRenderVersion = 0;
let pendingRenderUiState = null;
const sidepanelRenderListeners = new Set();
let sidepanelStatePersistenceReady = false;
let sidepanelStatePersistTimer = 0;
let lastPersistedSidepanelStateSignature = "";
const agentRunPollTimers = new Map();
const agentRunPersistTimers = new Map();
const trainingDataSaveTimers = new Map();
const toolScriptSaveTimers = new Map();
const headerTestHideTimers = new Map();
let backgroundKeepAlivePort = null;
let backgroundKeepAliveTimer = 0;
let renderPassId = 0;
let craftSyncSubscription = null;
let activeInferenceBenchmarkId = "";
const STICKY_LOG_SCROLL_THRESHOLD_PX = 2;
const BUTTON_PRESS_FEEDBACK_MS = 180;
const BUTTON_PRESS_HOLD_CLASS = "sidepanel-button-pressing";
const BUTTON_PRESS_BURST_CLASS = "sidepanel-button-feedback";
const INFERENCE_BENCHMARK_PROGRESS_START = 0.56;
const INFERENCE_BENCHMARK_PROGRESS_END = 0.94;
let activePressedButton = null;
const buttonPressFeedbackTimers = new WeakMap();

function escapeAttributeSelectorValue(value) {
  if (globalThis.CSS?.escape) {
    return globalThis.CSS.escape(String(value == null ? "" : value));
  }
  return String(value == null ? "" : value).replace(/["\\]/g, "\\$&");
}

function getEventTargetElement(event) {
  const target = event?.target || null;
  if (typeof globalThis.Element === "function" && target instanceof globalThis.Element) {
    return target;
  }
  if (target && typeof target.parentElement !== "undefined") {
    return target.parentElement || null;
  }
  return null;
}

function getPressFeedbackButton(target) {
  if (!target || typeof target.closest !== "function") return null;
  const button = target.closest("button");
  if (!(button instanceof globalThis.HTMLElement)) return null;
  if (button.disabled || button.getAttribute("aria-disabled") === "true") return null;
  return button;
}

function clearButtonPressBurst(button) {
  if (!(button instanceof globalThis.HTMLElement)) return;
  const timerId = buttonPressFeedbackTimers.get(button);
  if (timerId) {
    globalThis.clearTimeout(timerId);
    buttonPressFeedbackTimers.delete(button);
  }
  button.classList.remove(BUTTON_PRESS_BURST_CLASS);
}

function beginButtonPressFeedback(button) {
  if (!(button instanceof globalThis.HTMLElement)) return;
  if (activePressedButton && activePressedButton !== button) {
    activePressedButton.classList.remove(BUTTON_PRESS_HOLD_CLASS);
  }
  activePressedButton = button;
  clearButtonPressBurst(button);
  button.classList.add(BUTTON_PRESS_HOLD_CLASS);
}

function burstButtonPressFeedback(button) {
  if (!(button instanceof globalThis.HTMLElement)) return;
  clearButtonPressBurst(button);
  button.classList.remove(BUTTON_PRESS_HOLD_CLASS);
  void button.offsetWidth;
  button.classList.add(BUTTON_PRESS_BURST_CLASS);
  const timerId = globalThis.setTimeout(() => {
    button.classList.remove(BUTTON_PRESS_BURST_CLASS);
    buttonPressFeedbackTimers.delete(button);
  }, BUTTON_PRESS_FEEDBACK_MS);
  buttonPressFeedbackTimers.set(button, timerId);
}

function endButtonPressFeedback(button = activePressedButton, { burst = false } = {}) {
  if (!(button instanceof globalThis.HTMLElement)) {
    activePressedButton = null;
    return;
  }
  button.classList.remove(BUTTON_PRESS_HOLD_CLASS);
  if (activePressedButton === button) {
    activePressedButton = null;
  }
  if (burst) {
    burstButtonPressFeedback(button);
  }
}

function isSetupGateBlocked() {
  return state.setupLoaded && (state.setupCompleted !== true || state.requiredSetup?.ready !== true);
}

function clearCraftSyncSubscription() {
  if (typeof craftSyncSubscription === "function") {
    try {
      craftSyncSubscription();
    } catch {}
  }
  craftSyncSubscription = null;
}

function ensureCraftSyncSubscription() {
  if (craftSyncSubscription || !craftSync?.subscribe || isSetupGateBlocked()) return;
  craftSyncSubscription = craftSync.subscribe(handleCraftSyncChange);
}

function syncCraftSyncSubscriptionState() {
  if (isSetupGateBlocked()) {
    clearCraftSyncSubscription();
    return;
  }
  ensureCraftSyncSubscription();
}

function ensureSidepanelReactRoot() {
  if (!panelRoot) return null;
  if (!sidepanelReactRoot) {
    sidepanelReactRoot = clientExports.createRoot(panelRoot);
  }
  return sidepanelReactRoot;
}

function subscribeSidepanelRender(listener) {
  if (typeof listener !== "function") {
    return () => {};
  }
  sidepanelRenderListeners.add(listener);
  return () => {
    sidepanelRenderListeners.delete(listener);
  };
}

function getSidepanelRenderVersion() {
  return sidepanelRenderVersion;
}

function notifySidepanelRender() {
  sidepanelRenderVersion += 1;
  for (const listener of Array.from(sidepanelRenderListeners)) {
    try {
      listener();
    } catch (error) {
      console.warn("[sidepanel] render listener failed", error);
    }
  }
}

function useSidepanelRenderVersion() {
  if (typeof React.useSyncExternalStore === "function") {
    return React.useSyncExternalStore(
      subscribeSidepanelRender,
      getSidepanelRenderVersion,
      getSidepanelRenderVersion,
    );
  }

  const [version, setVersion] = React.useState(getSidepanelRenderVersion());
  React.useEffect(() => subscribeSidepanelRender(() => setVersion(getSidepanelRenderVersion())), []);
  return version;
}

function mountSidepanelReactApp() {
  const root = ensureSidepanelReactRoot();
  if (!root || sidepanelReactMounted) return;
  root.render(h(SidepanelRootView));
  sidepanelReactMounted = true;
}

function buildRestorableTextFieldState(element) {
  if (typeof globalThis.Element !== "function" || !(element instanceof globalThis.Element)) return null;
  const selectionStart = typeof element.selectionStart === "number" ? element.selectionStart : null;
  const selectionEnd = typeof element.selectionEnd === "number" ? element.selectionEnd : null;
  const selectionDirection =
    typeof element.selectionDirection === "string" ? element.selectionDirection : "none";

  if (element.matches("[data-agent-question-answer]")) {
    const craftId = String(element.dataset.craftId || "").trim();
    const questionId = String(element.dataset.questionId || "").trim();
    if (!craftId || !questionId) return null;
    return {
      kind: "agent-question",
      craftId,
      questionId,
      selectionStart,
      selectionEnd,
      selectionDirection,
      scrollTop: element.scrollTop,
      scrollLeft: element.scrollLeft,
    };
  }

  if (element.matches("[data-agent-prompt-field]")) {
    const craftId = String(element.dataset.craftId || "").trim();
    if (!craftId) return null;
    return {
      kind: "agent-prompt",
      craftId,
      selectionStart,
      selectionEnd,
      selectionDirection,
      scrollTop: element.scrollTop,
      scrollLeft: element.scrollLeft,
    };
  }

  if (element.matches("[data-draft-field]")) {
    const draftField = String(element.dataset.draftField || "").trim();
    if (!draftField) return null;
    return {
      kind: "draft-field",
      draftField,
      selectionStart,
      selectionEnd,
      selectionDirection,
      scrollTop: element.scrollTop,
      scrollLeft: element.scrollLeft,
    };
  }

  if (element.matches("[data-setup-api-key]")) {
    return {
      kind: "setup-api-key",
      selectionStart,
      selectionEnd,
      selectionDirection,
      scrollTop: element.scrollTop,
      scrollLeft: element.scrollLeft,
    };
  }

  if (element.matches("[data-setup-sync-field]")) {
    const setupSyncField = String(element.dataset.setupSyncField || "").trim();
    if (!setupSyncField) return null;
    return {
      kind: "setup-sync-field",
      setupSyncField,
      selectionStart,
      selectionEnd,
      selectionDirection,
      scrollTop: element.scrollTop,
      scrollLeft: element.scrollLeft,
    };
  }

  if (element.matches("[data-setup-provider]")) {
    return { kind: "setup-provider" };
  }

  if (element.matches("[data-setup-theme]")) {
    return { kind: "setup-theme" };
  }

  if (element.matches("[data-setup-slot-model]")) {
    const slotId = String(element.dataset.setupSlotModel || "").trim();
    if (!slotId) return null;
    return {
      kind: "setup-slot-model",
      slotId,
    };
  }

  if (element.matches("[data-setup-slot-reasoning]")) {
    const slotId = String(element.dataset.setupSlotReasoning || "").trim();
    if (!slotId) return null;
    return {
      kind: "setup-slot-reasoning",
      slotId,
    };
  }

  return null;
}

function resolveRestorableTextField(fieldState) {
  if (!fieldState || !panelRoot) return null;
  if (fieldState.kind === "agent-question") {
    return panelRoot.querySelector(
      `[data-agent-question-answer][data-craft-id="${escapeAttributeSelectorValue(fieldState.craftId)}"][data-question-id="${escapeAttributeSelectorValue(fieldState.questionId)}"]`,
    );
  }
  if (fieldState.kind === "agent-prompt") {
    return panelRoot.querySelector(
      `[data-agent-prompt-field][data-craft-id="${escapeAttributeSelectorValue(fieldState.craftId)}"]`,
    );
  }
  if (fieldState.kind === "draft-field") {
    return panelRoot.querySelector(
      `[data-draft-field="${escapeAttributeSelectorValue(fieldState.draftField)}"]`,
    );
  }
  if (fieldState.kind === "setup-api-key") {
    return panelRoot.querySelector("[data-setup-api-key]");
  }
  if (fieldState.kind === "setup-sync-field") {
    return panelRoot.querySelector(
      `[data-setup-sync-field="${escapeAttributeSelectorValue(fieldState.setupSyncField)}"]`,
    );
  }
  if (fieldState.kind === "setup-provider") {
    return panelRoot.querySelector("[data-setup-provider]");
  }
  if (fieldState.kind === "setup-theme") {
    return panelRoot.querySelector("[data-setup-theme]");
  }
  if (fieldState.kind === "setup-slot-model") {
    return panelRoot.querySelector(
      `[data-setup-slot-model="${escapeAttributeSelectorValue(fieldState.slotId)}"]`,
    );
  }
  if (fieldState.kind === "setup-slot-reasoning") {
    return panelRoot.querySelector(
      `[data-setup-slot-reasoning="${escapeAttributeSelectorValue(fieldState.slotId)}"]`,
    );
  }
  return null;
}

function capturePreservedScrollPosition(element) {
  if (typeof globalThis.HTMLElement !== "function" || !(element instanceof globalThis.HTMLElement)) return null;
  const key = String(element.getAttribute("data-preserve-scroll-key") || "").trim();
  if (!key) return null;
  const anchor = String(element.getAttribute("data-scroll-anchor") || "").trim();
  const maxScrollTop = Math.max(0, element.scrollHeight - element.clientHeight);
  const distanceFromBottom = Math.max(0, maxScrollTop - element.scrollTop);
  return {
    key,
    top: element.scrollTop,
    left: element.scrollLeft,
    anchor,
    stickToBottom: anchor === "bottom" && distanceFromBottom <= STICKY_LOG_SCROLL_THRESHOLD_PX,
  };
}

function applyPreservedScrollPosition(element, entry) {
  if (typeof globalThis.HTMLElement !== "function" || !(element instanceof globalThis.HTMLElement)) return;
  const anchor = String(element.getAttribute("data-scroll-anchor") || "").trim();
  const shouldStartAtBottom =
    anchor === "bottom" && String(element.getAttribute("data-scroll-start") || "").trim() === "bottom";
  const maxScrollTop = Math.max(0, element.scrollHeight - element.clientHeight);
  const maxScrollLeft = Math.max(0, element.scrollWidth - element.clientWidth);

  if (entry) {
    element.scrollTop =
      anchor === "bottom" && entry.stickToBottom
        ? maxScrollTop
        : Math.max(0, Math.min(maxScrollTop, Number(entry.top) || 0));
    element.scrollLeft = Math.max(0, Math.min(maxScrollLeft, Number(entry.left) || 0));
    return;
  }

  if (shouldStartAtBottom) {
    element.scrollTop = maxScrollTop;
  }
}

function captureRenderUiState() {
  const scrollingElement = document.scrollingElement || document.documentElement;
  return {
    documentScrollTop: scrollingElement?.scrollTop || 0,
    documentScrollLeft: scrollingElement?.scrollLeft || 0,
    focusField: buildRestorableTextFieldState(document.activeElement),
    scrollPositions: panelRoot
      ? Array.from(panelRoot.querySelectorAll("[data-preserve-scroll-key]"))
          .map((element) => capturePreservedScrollPosition(element))
          .filter(Boolean)
      : [],
  };
}

function restoreRenderUiState(uiState, passId) {
  if (!uiState) return;
  const apply = () => {
    if (passId !== renderPassId) return;

    const scrollingElement = document.scrollingElement || document.documentElement;
    if (scrollingElement) {
      scrollingElement.scrollTop = Math.max(0, Number(uiState.documentScrollTop) || 0);
      scrollingElement.scrollLeft = Math.max(0, Number(uiState.documentScrollLeft) || 0);
    }

    if (panelRoot) {
      const preservedEntries = new Map(
        (Array.isArray(uiState.scrollPositions) ? uiState.scrollPositions : [])
          .filter((entry) => String(entry?.key || "").trim())
          .map((entry) => [String(entry.key).trim(), entry]),
      );
      for (const element of Array.from(panelRoot.querySelectorAll("[data-preserve-scroll-key]"))) {
        const key = String(element.getAttribute("data-preserve-scroll-key") || "").trim();
        if (!key) continue;
        applyPreservedScrollPosition(element, preservedEntries.get(key) || null);
      }
    }

    const focusField = resolveRestorableTextField(uiState.focusField);
    if (typeof globalThis.HTMLElement !== "function" || !(focusField instanceof globalThis.HTMLElement) || focusField.disabled) return;
    try {
      focusField.focus({ preventScroll: true });
    } catch {
      focusField.focus();
    }
    if (typeof uiState.focusField?.selectionStart === "number" && typeof focusField.setSelectionRange === "function") {
      try {
        focusField.setSelectionRange(
          uiState.focusField.selectionStart,
          typeof uiState.focusField.selectionEnd === "number"
            ? uiState.focusField.selectionEnd
            : uiState.focusField.selectionStart,
          uiState.focusField.selectionDirection || "none",
        );
      } catch {}
    }
    if (typeof uiState.focusField?.scrollTop === "number") {
      focusField.scrollTop = uiState.focusField.scrollTop;
    }
    if (typeof uiState.focusField?.scrollLeft === "number") {
      focusField.scrollLeft = uiState.focusField.scrollLeft;
    }
  };

  if (typeof globalThis.requestAnimationFrame === "function") {
    globalThis.requestAnimationFrame(apply);
    return;
  }
  apply();
}

async function callCraftSyncWithTimeout(
  methodName,
  args = [],
  fallbackValue = null,
  timeoutMessage = "",
  timeoutMs = LOCAL_STORAGE_TIMEOUT_MS,
  options = {},
) {
  const effectiveTimeoutMessage = timeoutMessage || `Craft sync ${methodName} timed out.`;
  const retryTimeoutMs = Math.max(timeoutMs, Number(options?.retryTimeoutMs) || LOCAL_STORAGE_TIMEOUT_RETRY_MS);
  const throwOnFailure = options?.throwOnFailure === true;
  if (!craftSync || typeof craftSync?.[methodName] !== "function") {
    if (throwOnFailure) {
      throw new Error(`Craft sync ${methodName} is unavailable.`);
    }
    return fallbackValue;
  }

  let pending = null;
  try {
    pending = Promise.resolve(craftSync[methodName](...args));
  } catch (error) {
    console.warn(`[sidepanel] craftSync.${methodName} failed`, error);
    if (throwOnFailure) {
      throw (error instanceof Error ? error : new Error(String(error || `${methodName} failed.`)));
    }
    return fallbackValue;
  }

  let lastError = null;
  try {
    return await withTimeout(pending, timeoutMs, effectiveTimeoutMessage);
  } catch (error) {
    lastError = error;
  }

  if (
    lastError instanceof Error &&
    lastError.message === effectiveTimeoutMessage &&
    retryTimeoutMs > timeoutMs
  ) {
    try {
      return await withTimeout(pending, retryTimeoutMs, effectiveTimeoutMessage);
    } catch (error) {
      lastError = error;
    }
  }

  console.warn(`[sidepanel] craftSync.${methodName} failed`, lastError);
  if (throwOnFailure) {
    throw (lastError instanceof Error
      ? lastError
      : new Error(String(lastError || `${methodName} failed.`)));
  }
  return fallbackValue;
}

async function readLocalArtifactFromStores(artifactId, fallback = null, options = {}) {
  const key = String(artifactId || "").trim();
  if (!key) return fallback;
  const synced = await callCraftSyncWithTimeout(
    "readLocalArtifact",
    [key, null],
    null,
    `Reading local artifact ${key} timed out.`,
    LOCAL_STORAGE_TIMEOUT_MS,
    options,
  );
  if (synced) return synced;
  return fallback;
}

async function writeLocalArtifactToStores(record, options = {}) {
  const artifactId = String(record?.id || "").trim();
  if (!artifactId) return null;
  const synced = await callCraftSyncWithTimeout(
    "putLocalArtifact",
    [record],
    null,
    `Writing local artifact ${artifactId} timed out.`,
    LOCAL_STORAGE_TIMEOUT_MS,
    options,
  );
  if (synced) return synced;
  return null;
}

async function readBrowserCapabilityArtifactRecord(craftId) {
  const key = String(craftId || "").trim();
  if (!key) return null;
  return await readLocalArtifactFromStores(getBrowserCapabilityBundleArtifactId(key), null, { throwOnFailure: true });
}

async function writeBrowserCapabilityArtifactRecord(craftId, payload) {
  const key = String(craftId || "").trim();
  if (!key) return null;
  return await writeLocalArtifactToStores({
    id: getBrowserCapabilityBundleArtifactId(key),
    craftId: key,
    kind: BROWSER_CAPABILITY_BUNDLE_ARTIFACT_KIND,
    payload,
    meta: {
      capabilityCount: Array.isArray(payload?.capabilities) ? payload.capabilities.length : 0,
      updatedAt: Date.now(),
    },
  }, { throwOnFailure: true });
}

async function readPolicyArtifactRecord(craftId) {
  const key = String(craftId || "").trim();
  if (!key) return null;
  return await readLocalArtifactFromStores(getPolicyBundleArtifactId(key), null, { throwOnFailure: true });
}

async function writePolicyArtifactRecord(craftId, payload) {
  const key = String(craftId || "").trim();
  if (!key) return null;
  return await writeLocalArtifactToStores({
    id: getPolicyBundleArtifactId(key),
    craftId: key,
    kind: POLICY_BUNDLE_ARTIFACT_KIND,
    payload,
    meta: {
      trainingMode: String(payload?.trainingMode || payload?.status || "").trim(),
      updatedAt: Date.now(),
    },
  }, { throwOnFailure: true });
}

async function writeCapabilityWeightsArtifactRecord(craftId, payload) {
  const key = String(craftId || "").trim();
  if (!key) return null;
  return await writeLocalArtifactToStores({
    id: getWeightsArtifactId(key),
    craftId: key,
    kind: WEIGHTS_ARTIFACT_KIND,
    payload,
    meta: {
      status: String(payload?.status || "").trim(),
      updatedAt: Date.now(),
    },
  }, { throwOnFailure: true });
}

function clearBackgroundKeepAlive() {
  if (backgroundKeepAliveTimer) {
    window.clearInterval(backgroundKeepAliveTimer);
    backgroundKeepAliveTimer = 0;
  }
  if (backgroundKeepAlivePort) {
    try {
      backgroundKeepAlivePort.disconnect();
    } catch {}
    backgroundKeepAlivePort = null;
  }
}

function ensureBackgroundKeepAlive() {
  if (!chrome?.runtime?.connect) return;
  if (backgroundKeepAlivePort) return;

  backgroundKeepAlivePort = chrome.runtime.connect({
    name: "sidepanel-keepalive",
  });

  const ping = () => {
    try {
      backgroundKeepAlivePort?.postMessage({
        type: "keepalive:ping",
        ts: Date.now(),
      });
    } catch {}
  };

  backgroundKeepAlivePort.onDisconnect.addListener(() => {
    clearBackgroundKeepAlive();
    window.setTimeout(() => {
      ensureBackgroundKeepAlive();
    }, 1000);
  });

  ping();
  backgroundKeepAliveTimer = window.setInterval(ping, 20_000);
}

const FULL_RENDER_CRAFT_SYNC_EVENT_TYPES = new Set([
  "manual-sync-starting",
  "publish-local-crafts",
  "remote-crafts",
  "remote-crafts-refresh-failed",
  "settings-updated",
  "sync-stopped",
]);

function mapCraftSyncDetailToSharedStateChange(detail = {}) {
  const type = String(detail?.type || "");
  if (type === "kv-upsert" || type === "kv-delete") {
    return {
      area: "kv",
      key: String(detail?.key || ""),
    };
  }
  if (type === "local-crafts") {
    return {
      area: "crafts",
    };
  }
  if (type === "local-artifacts" || type === "local-artifact-upsert" || type === "local-artifact-delete") {
    return {
      area: "artifacts",
      artifactId: String(detail?.artifactId || detail?.id || ""),
      craftId: String(detail?.craftId || ""),
      kind: String(detail?.kind || ""),
    };
  }
  return null;
}

function mergeRenderAction(currentAction, nextAction) {
  if (currentAction?.type === "full" || nextAction?.type === "full") {
    return { type: "full" };
  }
  if (nextAction?.type === "craft" && nextAction.craftId) {
    return nextAction;
  }
  return currentAction || nextAction || null;
}

function applyRenderAction(renderAction = null) {
  if (!renderAction) return;
  if (renderAction.type === "craft" && renderAction.craftId) {
    refreshCraftCard(renderAction.craftId);
    return;
  }
  render();
}

const handleSharedStateChange = async (change, options = {}) => {
  const area = String(change?.area || "");
  const key = String(change?.key || "");
  const artifactId = String(change?.artifactId || change?.id || "");
  const artifactCraftId = String(change?.craftId || "").trim();
  const artifactKind = String(change?.kind || "").trim();
  const setupGateBlocked = isSetupGateBlocked();
  let renderAction = null;

  if (
    configApi &&
    area === "kv" &&
    (key === configApi.PROVIDER_STORAGE_KEY || key === configApi.MODEL_SLOT_STORAGE_KEY)
  ) {
    await refreshSetupState();
    renderAction = mergeRenderAction(renderAction, { type: "full" });
  }

  if (area === "kv" && key === CRAFT_SYNC_SETTINGS_KEY) {
    state.syncSettings =
      (await (craftSync?.refreshSettings?.() || craftSync?.readSettings?.())) || state.syncSettings;
    state.syncSnapshot = craftSync?.getState?.() || state.syncSnapshot;
    renderAction = mergeRenderAction(renderAction, { type: "full" });
  }

  if (area === "kv" && key === themeApi?.STORAGE_KEY) {
    state.themeId = (await themeApi?.readThemeId?.()) || state.themeId;
    themeApi?.applyTheme?.(state.themeId);
    renderAction = mergeRenderAction(renderAction, { type: "full" });
  }

  if (area === "kv" && key === UI_PREFERENCES_KEY) {
    await refreshUiPreferences();
    renderAction = mergeRenderAction(renderAction, { type: "full" });
  }

  if (area === "kv" && key === STARTER_TUTORIAL_SEEN_KEY) {
    await refreshStarterTutorialState();
    renderAction = mergeRenderAction(renderAction, { type: "full" });
  }

  if (area === "crafts") {
    await refreshCraftsState();
    if (!setupGateBlocked) {
      renderAction = mergeRenderAction(renderAction, { type: "full" });
    }
  }

  if (
    !setupGateBlocked &&
    area === "artifacts" &&
    (!artifactKind || artifactKind === TRAINING_DATA_ARTIFACT_KIND) &&
    (artifactId.startsWith("training-samples:") || artifactCraftId)
  ) {
    const targetCraftId =
      artifactCraftId ||
      artifactId.replace(/^training-samples:/, "") ||
      (state.trainingDataCraftId && state.trainingDataStates[state.trainingDataCraftId]
        ? state.trainingDataCraftId
        : "");
    if (targetCraftId && (state.trainingDataStates[targetCraftId] || state.trainingDataCraftId === targetCraftId)) {
      await loadTrainingDataState(targetCraftId, { force: true });
      renderAction = mergeRenderAction(renderAction, { type: "craft", craftId: targetCraftId });
    }
  }

  if (options?.deferRender === true) {
    return renderAction;
  }

  applyRenderAction(renderAction);
  return renderAction;
};

render();
void init().catch(handleInitError);

async function init() {
  state.startupError = "";
  state.themeId = (await themeApi?.hydrateTheme?.()) || state.themeId;
  ensureBackgroundKeepAlive();
  await refreshSetupState({ syncSubscription: false });
  render();

  if (craftSync?.ensureStartedFromSettings) {
    await callCraftSyncWithTimeout(
      "ensureStartedFromSettings",
      [{ pageName: "Sidepanel" }],
      null,
      "Craft sync startup timed out.",
      LOCAL_STORAGE_TIMEOUT_MS,
      { retryTimeoutMs: LOCAL_STORAGE_TIMEOUT_RETRY_MS },
    );
    state.syncSnapshot = craftSync.getState?.() || state.syncSnapshot;
  }

  await Promise.all([
    refreshCraftsState(),
    refreshUiPreferences(),
    hydrateSidepanelState(),
    refreshStarterTutorialState(),
    hydrateHeaderTestSessionState(),
  ]);
  if (!Object.keys(state.agentRuns).length) {
    await hydratePersistedAgentRuns();
  }
  reconcileSidepanelStateWithCrafts();
  state.craftingCraftId = null;
  if (state.activeCraftId) {
    void Promise.all([
      loadTrainingDataState(state.activeCraftId),
      loadToolScriptsState(state.activeCraftId),
      loadCapabilityWeightsState(state.activeCraftId),
    ]);
  }
  resumePersistedTrainingPolls();
  resumePersistedAgentRunPolls();
  sidepanelStatePersistenceReady = true;
  syncCraftSyncSubscriptionState();
  render();
}

window.addEventListener("beforeunload", () => {
  clearCraftSyncSubscription();
  clearBackgroundKeepAlive();
});

function handleInitError(error) {
  state.startupError =
    error instanceof Error ? error.message : String(error || "Unknown startup error.");
  state.setupLoaded = true;
  state.requiredSetup = { ready: true, missingItems: [] };
  clearCraftSyncSubscription();
  render();
}

async function handleCraftSyncChange(payload = null) {
  state.syncSnapshot = craftSync?.getState?.() || state.syncSnapshot;
  const detail = payload?.detail && typeof payload.detail === "object" ? payload.detail : {};
  const type = String(detail.type || "");
  const artifactKind = String(detail.kind || "");
  const artifactCraftId = String(detail.craftId || "").trim();
  const setupGateBlocked = isSetupGateBlocked();
  let renderAction = await handleSharedStateChange(
    mapCraftSyncDetailToSharedStateChange(detail),
    { deferRender: true },
  );

  if (!type || ["remote-crafts", "publish-local-crafts"].includes(type)) {
    await refreshCraftsState();
    if (!setupGateBlocked && !type && state.activeCraftId) {
      try {
        await Promise.all([
          loadTrainingDataState(state.activeCraftId, { force: true }),
          loadToolScriptsState(state.activeCraftId, { force: true }),
          loadCapabilityWeightsState(state.activeCraftId, { force: true }),
        ]);
      } catch (error) {
        console.warn("[sidepanel] failed to reload local artifact state after craft sync startup", error);
      }
    }
    if (!setupGateBlocked) {
      renderAction = mergeRenderAction(renderAction, { type: "full" });
    }
  }

  if (
    !setupGateBlocked &&
    ["local-artifacts", "local-artifact-upsert", "local-artifact-delete"].includes(type) &&
    artifactKind === AGENT_RUN_STATE_ARTIFACT_KIND
  ) {
    if (shouldHydrateAgentRunsFromArtifactEvent(artifactCraftId)) {
      await hydratePersistedAgentRuns();
    }
    renderAction = mergeRenderAction(
      renderAction,
      artifactCraftId ? { type: "craft", craftId: artifactCraftId } : { type: "full" },
    );
  }

  if (
    !setupGateBlocked &&
    ["local-artifacts", "local-artifact-upsert", "local-artifact-delete"].includes(type) &&
    (!artifactKind || artifactKind === TRAINING_DATA_ARTIFACT_KIND)
  ) {
    const targetCraftId =
      artifactCraftId ||
      (state.trainingDataCraftId && state.trainingDataStates[state.trainingDataCraftId]
        ? state.trainingDataCraftId
        : "");
    if (targetCraftId && (state.trainingDataCraftId === targetCraftId || state.trainingDataStates[targetCraftId])) {
      await loadTrainingDataState(targetCraftId, { force: true });
      renderAction = mergeRenderAction(renderAction, { type: "craft", craftId: targetCraftId });
    }
  }

  if (
    !setupGateBlocked &&
    ["local-artifacts", "local-artifact-upsert", "local-artifact-delete"].includes(type) &&
    (!artifactKind || artifactKind === TOOL_SCRIPTS_ARTIFACT_KIND)
  ) {
    const targetCraftId = artifactCraftId || state.activeCraftId || "";
    if (targetCraftId && (state.activeCraftId === targetCraftId || state.toolScriptStates[targetCraftId])) {
      await loadToolScriptsState(targetCraftId, { force: true });
      renderAction = mergeRenderAction(renderAction, { type: "craft", craftId: targetCraftId });
    }
  }

  if (
    !setupGateBlocked &&
    ["local-artifacts", "local-artifact-upsert", "local-artifact-delete"].includes(type) &&
    (!artifactKind || artifactKind === WEIGHTS_ARTIFACT_KIND)
  ) {
    const targetCraftId = artifactCraftId || state.activeCraftId || "";
    if (targetCraftId && (state.activeCraftId === targetCraftId || state.capabilityWeightsStates[targetCraftId])) {
      await loadCapabilityWeightsState(targetCraftId, { force: true });
      renderAction = mergeRenderAction(renderAction, { type: "craft", craftId: targetCraftId });
    }
  }

  if (!setupGateBlocked && !renderAction && FULL_RENDER_CRAFT_SYNC_EVENT_TYPES.has(type)) {
    renderAction = { type: "full" };
  }

  applyRenderAction(renderAction);
}

function shouldHydrateAgentRunsFromArtifactEvent(craftId = "") {
  const key = String(craftId || "").trim();
  if (!key) return true;
  const run = state.agentRuns[key] || null;
  if (!run) return true;
  return !run.runId || isFinalAgentRunStatus(run.status);
}

function cloneJson(value, fallback) {
  try {
    return JSON.parse(JSON.stringify(value));
  } catch (_error) {
    return fallback;
  }
}

function asPlainObject(value) {
  return value && typeof value === "object" && !Array.isArray(value) ? value : {};
}

function createPersistedLog(level, message) {
  return {
    level: String(level || "info"),
    message: String(message || ""),
    time: formatClock(new Date()),
  };
}

function normalizeAgentLogStatus(value) {
  const status = String(value || "").trim().toLowerCase();
  return ["running", "done", "warn", "error"].includes(status) ? status : "";
}

function looksLikeAgentToolFailureMessage(value = "") {
  const haystack = String(value || "").trim().toLowerCase();
  return /an error occurred while running the tool|tool execution failed|(?:^|\b)error:|runtime failed|failed to call ortrun/.test(haystack);
}

function normalizeAgentLogEntry(rawEntry, agentRun = null, lastToolFailure = null) {
  const entry = asPlainObject(rawEntry);
  const message = String(entry.message || "").trim();
  const time = String(entry.time || "");
  const kind = String(entry.kind || "").trim().toLowerCase();
  const title = String(entry.title || "").trim();
  const detail = String(entry.detail || "").trim();
  const toolName = String(entry.toolName || "").trim();
  const stageId = String(entry.stageId || "").trim();
  const rawLevel = String(entry.level || "info").trim().toLowerCase() || "info";
  const rawStatus = normalizeAgentLogStatus(entry.status);
  const inferredToolError =
    kind === "tool" &&
    rawStatus !== "running" &&
    looksLikeAgentToolFailureMessage(`${message}\n${title}\n${detail}`);
  const semanticToolFailure =
    kind === "tool" &&
    rawStatus !== "running" &&
    ["blocked", "failed"].includes(String(agentRun?.status || "").trim().toLowerCase()) &&
    String(toolName || "").trim() &&
    String(lastToolFailure?.action || "").trim() === String(toolName || "").trim() &&
    rawStatus !== "error" &&
    rawLevel !== "error";
  const semanticFailureText = trimText(
    String(lastToolFailure?.error || lastToolFailure?.summary || detail || message),
    600,
  );
  const level = semanticToolFailure || inferredToolError || rawStatus === "error" ? "error" : rawLevel;
  const status = semanticToolFailure || inferredToolError ? "error" : rawStatus;
  const out = {
    level,
    message:
      semanticToolFailure && toolName && semanticFailureText
        ? `${toolName}: ${trimText(semanticFailureText, 260)}`
        : message,
    time,
  };
  if (kind) out.kind = kind;
  if (title) out.title = title;
  if (semanticToolFailure && semanticFailureText) {
    out.detail = semanticFailureText;
  } else if (detail) {
    out.detail = detail;
  }
  if (toolName) out.toolName = toolName;
  if (stageId) out.stageId = stageId;
  if (status) out.status = status;
  if (entry.data && typeof entry.data === "object") {
    out.data = cloneJson(entry.data, null);
  }
  return out;
}

function buildStableAgentEntryId(prefix, text, index = 0) {
  const normalized = String(text || "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 24);
  return `${prefix}-${index + 1}-${normalized || "entry"}`;
}

function normalizePromptExampleTexts(values, limit = 6) {
  const sourceValues = Array.isArray(values) ? values : [];
  const seen = new Set();
  const out = [];
  for (const value of sourceValues) {
    const text = trimText(String(value || "").trim(), 240);
    if (!text) continue;
    const key = text.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(text);
    if (out.length >= limit) break;
  }
  return out;
}

function getStarterTutorialExample(index = 0) {
  if (!STARTER_TUTORIAL_EXAMPLES.length) return null;
  const parsedIndex = Number.isInteger(index) ? index : Number.parseInt(index, 10);
  const safeIndex = Number.isFinite(parsedIndex) && parsedIndex >= 0 ? parsedIndex : 0;
  return STARTER_TUTORIAL_EXAMPLES[safeIndex] || STARTER_TUTORIAL_EXAMPLES[0] || null;
}

function seedDraftGoalFromStarterTutorial() {
  if (String(state.draftGoal || "").trim()) return;
  const starterExample = getStarterTutorialExample(0);
  if (!starterExample?.prompt) return;
  state.draftGoal = starterExample.prompt;
}

function focusDraftGoalField({ placeCursorAtEnd = true } = {}) {
  const apply = () => {
    const textarea = panelRoot?.querySelector?.('[data-draft-field="goal"]');
    if (typeof globalThis.HTMLTextAreaElement !== "function" || !(textarea instanceof globalThis.HTMLTextAreaElement)) {
      return;
    }
    try {
      textarea.focus({ preventScroll: true });
    } catch {
      textarea.focus();
    }
    if (!placeCursorAtEnd || typeof textarea.setSelectionRange !== "function") return;
    const end = textarea.value.length;
    try {
      textarea.setSelectionRange(end, end, "none");
    } catch {}
  };

  if (typeof globalThis.requestAnimationFrame === "function") {
    globalThis.requestAnimationFrame(apply);
    return;
  }
  apply();
}

function continueStarterTutorialCreateFlow() {
  rememberStarterTutorialSeen();
  state.tutorialOverlay = null;
  state.createOpen = true;
  render();
  focusDraftGoalField();
}

function useStarterTutorialExample(index = 0) {
  const example = getStarterTutorialExample(index);
  if (!example?.prompt) return;
  state.createOpen = true;
  state.draftGoal = example.prompt;
  rememberStarterTutorialSeen();
  state.tutorialOverlay = null;
  render();
  focusDraftGoalField();
}

function normalizeAgentQuestion(rawQuestion, index = 0) {
  const source = asPlainObject(rawQuestion);
  const question = String(source.question ?? source.text ?? "").trim();
  if (!question) return null;
  return {
    id: String(source.id || buildStableAgentEntryId("question", question, index)),
    question,
    reason: String(source.reason || "").trim(),
    answer: String(source.answer || ""),
  };
}

function normalizeAgentQuestions(rawQuestions) {
  return (Array.isArray(rawQuestions) ? rawQuestions : [])
    .map((entry, index) => normalizeAgentQuestion(entry, index))
    .filter(Boolean);
}

function normalizeAgentProvenanceEntry(rawEntry, index = 0) {
  const source = asPlainObject(rawEntry);
  const title = String(source.title || source.label || "").trim();
  const detail = String(source.detail || source.reason || source.message || "").trim();
  if (!title && !detail) return null;
  const kind = String(source.kind || "").trim().toLowerCase();
  const operationType = String(source.operationType || source.operation_type || "").trim().toLowerCase();
  return {
    id: String(source.id || buildStableAgentEntryId("prov", title || detail, index)),
    title: title || "Signal",
    detail,
    kind: ["match", "constraint", "operation", "sample"].includes(kind) ? kind : "match",
    sampleId: String(source.sampleId || source.sample_id || "").trim(),
    operationType: ["add", "update", "delete"].includes(operationType) ? operationType : "",
  };
}

function normalizeAgentProvenance(rawEntries) {
  return (Array.isArray(rawEntries) ? rawEntries : [])
    .map((entry, index) => normalizeAgentProvenanceEntry(entry, index))
    .filter(Boolean)
    .slice(0, 18);
}

function normalizeAgentReport(rawReport, fallback = {}) {
  const report = asPlainObject(rawReport);
  const fallbackReport = asPlainObject(fallback);
  const matchingSignals = Array.from(
    new Set(
      (Array.isArray(report.matchingSignals) ? report.matchingSignals : [])
        .map((entry) => String(entry || "").trim())
        .filter(Boolean),
    ),
  ).slice(0, 8);
  return {
    objective: String(report.objective || fallbackReport.objective || "").trim(),
    currentState: String(report.currentState || fallbackReport.currentState || "").trim(),
    nextAction: String(report.nextAction || fallbackReport.nextAction || "").trim(),
    matchingSignals,
  };
}

function getCraftMaturity(craft, agentRun = null) {
  const key = String(craft?.id || "").trim();
  const capabilityState = key ? getCapabilityWeightsState(key) : createCapabilityWeightsState();
  return gateCraftMaturityForCapability(
    normalizeCraftMaturity(agentRun?.maturity, createEmptyCraftMaturity()),
    {
      hasTrainedCapability: capabilityState.hasTrainedAdapter === true,
    },
  );
}

function getCraftMaturityText(craft, agentRun = null) {
  return formatCraftMaturityPercent(getCraftMaturity(craft, agentRun).percent);
}

function getCraftMaturityTitle(craft, agentRun = null) {
  const maturity = getCraftMaturity(craft, agentRun);
  const agentUiState = getAgentUiState(agentRun);
  const parts = [];
  if (agentUiState === "starting") {
    parts.push("Agent is starting.");
  } else if (agentUiState === "running") {
    parts.push("Agent is running.");
  }
  if (!maturity.isExplicit) {
    parts.push("No explicit agent maturity update yet.");
    return parts.join(" ");
  }
  parts.push(
    `Agent-reported maturity ${formatCraftMaturityPercent(maturity.percent)}`,
    `(${formatCraftMaturityPhase(maturity.phase)})`,
  );
  if (maturity.rationale) {
    parts.push(maturity.rationale);
  }
  return parts.join(" ");
}

function getCraftMaturityBadgeClassName(craft, agentRun = null) {
  const maturity = getCraftMaturity(craft, agentRun);
  const agentUiState = getAgentUiState(agentRun);
  const isRunning = agentUiState === "starting" || agentUiState === "running";
  return joinClassNames(
    "craft-maturity-badge",
    maturity.phase === "capability_readiness" ? "craft-maturity-badge-quality" : "craft-maturity-badge-progress",
    maturity.percent > 0 ? "craft-maturity-badge-has-progress" : "",
    isRunning ? "craft-maturity-badge-running" : "",
  );
}

function getCraftMaturityBadgeStyle(craft, agentRun = null) {
  const percent = Math.max(0, Math.min(100, Number(getCraftMaturity(craft, agentRun).percent) || 0));
  return { "--craft-maturity-progress": `${percent}%` };
}

function countOpenAgentQuestions(agentRun) {
  return normalizeAgentQuestions(agentRun?.questions).filter((question) => !String(question.answer || "").trim()).length;
}

function getAnsweredAgentQuestions(agentRun) {
  return normalizeAgentQuestions(agentRun?.questions).filter((question) => String(question.answer || "").trim());
}

function normalizePersistedTextMap(rawMap) {
  const out = {};
  for (const [key, value] of Object.entries(asPlainObject(rawMap))) {
    const normalizedKey = String(key || "").trim();
    if (!normalizedKey) continue;
    out[normalizedKey] = String(value || "");
  }
  return out;
}

function normalizePersistedAgentRuns(rawRuns, { interruptRunning = true } = {}) {
  const out = {};
  for (const [craftId, rawRun] of Object.entries(asPlainObject(rawRuns))) {
    const run = asPlainObject(rawRun);
    const craft = findCraft(craftId);
    const status = String(run.status || "");
    const logs = Array.isArray(run.logs)
      ? run.logs
          .map((entry) => normalizeAgentLogEntry(entry))
          .filter((entry) => entry.message || entry.title || entry.detail)
      : [];
    if (interruptRunning && status === "running") {
      logs.push(createPersistedLog("error", "Revision was interrupted by a panel reload."));
    }
    const questions = normalizeAgentQuestions(run.questions);
    const provenance = normalizeAgentProvenance(run.provenance);
    const report = normalizeAgentReport(run.report, {
      currentState: String(run.responseText || "").trim(),
    });
    const rawActiveTab = String(run.activeTab || "");
    const activeTab =
      rawActiveTab === "log"
        ? "progress"
        : ["progress", "questions", "provenance"].includes(rawActiveTab)
          ? rawActiveTab
          : "progress";
    out[String(craftId || "")] = {
      runId: String(run.runId || run.jobId || ""),
      status: interruptRunning && status === "running" ? "failed" : status || "idle",
      phase: String(run.phase || ""),
      mode: String(run.mode || ""),
      modelRef: String(run.modelRef || ""),
      responseText: String(run.responseText || ""),
      finalStatus: normalizeAgentFinalStatus(run.finalStatus),
      tokens: Math.max(0, Number(run.tokens || 0)),
      costUsd: Math.max(0, Number(run.costUsd || 0)),
      turnsUsed: Math.max(0, Number(run.turnsUsed || 0)),
      maxTurns: Math.max(0, Number(run.maxTurns || 0)),
      error:
        status === "running"
          ? String(run.error || "Revision was interrupted before completion.")
          : String(run.error || ""),
      errorDetail: run.errorDetail && typeof run.errorDetail === "object" ? cloneJson(run.errorDetail, null) : null,
      logs,
      toolTrace: Array.isArray(run.toolTrace) ? cloneJson(run.toolTrace, []) : [],
      lastToolFailure: run.lastToolFailure && typeof run.lastToolFailure === "object" ? cloneJson(run.lastToolFailure, null) : null,
      workspaceCodeDive: run.workspaceCodeDive && typeof run.workspaceCodeDive === "object" ? cloneJson(run.workspaceCodeDive, null) : null,
      report,
      questions,
      provenance,
      maturity: getCraftMaturity(craft, run),
      suggestedName: normalizeCraftNameCandidate(run.suggestedName),
      officialDescription: String(run.officialDescription || "").trim(),
      operations: Array.isArray(run.operations) ? cloneJson(run.operations, []) : [],
      useSurface: run.useSurface && typeof run.useSurface === "object" ? cloneJson(run.useSurface, null) : null,
      activeTab,
      clipboardStatus: String(run.clipboardStatus || "idle"),
      clipboardError: String(run.clipboardError || ""),
      activityRecorded: run.activityRecorded === true,
      useSurfaceApplied: run.useSurfaceApplied === true,
      officialDescriptionApplied: run.officialDescriptionApplied === true,
      completedAt: String(run.completedAt || ""),
    };
  }
  return out;
}

function normalizePersistedTrainingRun(rawRun) {
  const run = asPlainObject(rawRun);
  const status = String(run.status || "");
  const hasJobId = Boolean(String(run.jobId || "").trim());
  const wasInterrupted = ["queued", "starting", "running"].includes(status) && !hasJobId;

  return {
    jobId: String(run.jobId || ""),
    headerTestId: String(run.headerTestId || ""),
    shardId: String(run.shardId || ""),
    modelName: String(run.modelName || ""),
    status: wasInterrupted ? "failed" : status || "idle",
    phaseLabel: String(
      run.phaseLabel || (wasInterrupted ? "Run interrupted by reload" : run.phase || "Idle"),
    ),
    progress: Math.max(0, Math.min(1, Number(run.progress || 0))),
    totalSamples: Math.max(0, Number(run.totalSamples || 0)),
    completedSamples: Math.max(0, Number(run.completedSamples || 0)),
    phaseTotalSamples: Math.max(0, Number(run.phaseTotalSamples || 0)),
    phaseCompletedSamples: Math.max(0, Number(run.phaseCompletedSamples || 0)),
    phaseUnitLabel: String(run.phaseUnitLabel || ""),
    samplesPerSecond: Math.max(0, Number(run.samplesPerSecond || 0)),
    estimatedRemainingMs: Math.max(0, Number(run.estimatedRemainingMs || 0)),
    currentEpoch: Math.max(0, Number(run.currentEpoch || 0)),
    epochsTotal: Math.max(0, Number(run.epochsTotal || 0)),
    baseValidationAcc: Math.max(0, Number(run.baseValidationAcc || 0)),
    baseTestAcc: Math.max(0, Number(run.baseTestAcc || 0)),
    adaptValidationAcc: Math.max(0, Number(run.adaptValidationAcc || 0)),
    adaptTestAcc: Math.max(0, Number(run.adaptTestAcc || 0)),
    adapterSizeMb: Math.max(0, Number(run.adapterSizeMb || 0)),
    metrics: cloneJson(asPlainObject(run.metrics), {}),
    smokeMode: String(run.smokeMode || ""),
    smoke: cloneJson(asPlainObject(run.smoke), null),
    dataset: cloneJson(asPlainObject(run.dataset), null),
    runtime: cloneJson(asPlainObject(run.runtime), null),
    history: Array.isArray(run.history) ? cloneJson(run.history, []) : [],
    message: String(
      run.message || (wasInterrupted ? "The panel reloaded before the run received a job id." : ""),
    ),
    error: String(run.error || ""),
    workdir: String(run.workdir || ""),
    manifestPath: String(run.manifestPath || ""),
    startedAt: String(run.startedAt || ""),
    endedAt: String(run.endedAt || ""),
    autoCopyOnComplete: run.autoCopyOnComplete === true,
    clipboardStatus: String(run.clipboardStatus || "idle"),
    clipboardError: String(run.clipboardError || ""),
    lastReport:
      run.lastReport && typeof run.lastReport === "object"
        ? cloneJson(run.lastReport, null)
        : null,
    lastReportUpdatedAt: String(run.lastReportUpdatedAt || ""),
    activityRecorded: run.activityRecorded === true,
  };
}

function normalizePersistedTrainingRuns(rawRuns) {
  const out = {};
  for (const [craftId, rawRun] of Object.entries(asPlainObject(rawRuns))) {
    out[String(craftId || "")] = normalizePersistedTrainingRun(rawRun);
  }
  return out;
}

function normalizePersistedUseMessages(rawMessages) {
  const out = {};
  for (const [craftId, rawMessage] of Object.entries(asPlainObject(rawMessages))) {
    const message = asPlainObject(rawMessage);
    out[String(craftId || "")] = {
      time: String(message.time || ""),
      text: String(message.text || ""),
    };
  }
  return out;
}

function normalizePersistedCraftResponses(rawResponses) {
  const out = {};
  for (const [craftId, rawResponse] of Object.entries(asPlainObject(rawResponses))) {
    const response = asPlainObject(rawResponse);
    const status = String(response.status || "");
    out[String(craftId || "")] = {
      status: status === "running" ? "error" : status || "idle",
      text: String(response.text || ""),
      error:
        status === "running"
          ? String(response.error || "Vanilla chat was interrupted by a panel reload.")
          : String(response.error || ""),
      modelRef: String(response.modelRef || ""),
    };
  }
  return out;
}

function normalizePersistedTutorialOverlay(rawOverlay) {
  const overlay = asPlainObject(rawOverlay);
  const mode = String(overlay.mode || "").trim();
  if (mode === "create") {
    return { mode: "create" };
  }
  const craftId = String(overlay.craftId || "").trim();
  if (!craftId) return null;
  return {
    mode: "craft",
    craftId,
    craftName: String(overlay.craftName || ""),
    starterModelName: String(overlay.starterModelName || ""),
  };
}

function buildPersistedSidepanelState() {
  return {
    activeCraftId: String(state.activeCraftId || ""),
    craftingCraftId: "",
    createOpen: state.createOpen === true,
    draftName: String(state.draftName || ""),
    draftGoal: String(state.draftGoal || ""),
    promptDrafts: cloneJson(state.promptDrafts, {}),
    agentPromptDrafts: cloneJson(state.agentPromptDrafts, {}),
    trainingRuns: cloneJson(state.trainingRuns, {}),
    useMessages: cloneJson(state.useMessages, {}),
    craftResponses: cloneJson(state.craftResponses, {}),
    debugTrainingRun: state.debugTrainingRun ? cloneJson(state.debugTrainingRun, null) : null,
    tutorialOverlay: state.tutorialOverlay ? cloneJson(state.tutorialOverlay, null) : null,
  };
}

function getPersistedSidepanelStateSignature(snapshot) {
  try {
    return JSON.stringify(snapshot);
  } catch (_error) {
    return "";
  }
}

async function readPersistedSidepanelStateRecord() {
  const record = await readLocalArtifactFromStores(SIDEPANEL_STATE_ARTIFACT_ID, null);
  if (record?.payload && typeof record.payload === "object") {
    return record.payload;
  }
  return null;
}

async function writePersistedSidepanelStateRecord(snapshot) {
  return await writeLocalArtifactToStores({
    id: SIDEPANEL_STATE_ARTIFACT_ID,
    craftId: SIDEPANEL_STATE_ARTIFACT_CRAFT_ID,
    kind: SIDEPANEL_STATE_ARTIFACT_KIND,
    payload: snapshot,
    meta: {
      updatedAt: Date.now(),
    },
  });
}

function getPersistableAgentRunSnapshot(craftId) {
  const key = String(craftId || "").trim();
  const run = state.agentRuns[key];
  if (!run || typeof run !== "object") return null;
  const craft = findCraft(key);
  return cloneJson(
    {
      ...run,
      maturity: getCraftMaturity(craft, run),
    },
    null,
  );
}

async function persistAgentRunSnapshot(craftId) {
  const key = String(craftId || "").trim();
  if (!key) return;
  const snapshot = getPersistableAgentRunSnapshot(key);
  if (!snapshot) {
    await deleteAgentRunState(key);
    return;
  }
  await upsertAgentRunState({
    craftId: key,
    snapshot,
    meta: {
      source: "sidepanel",
    },
  });
}

function scheduleAgentRunPersist(craftId) {
  const key = String(craftId || "").trim();
  if (!key) return;
  globalThis.clearTimeout(agentRunPersistTimers.get(key) || 0);
  const timer = globalThis.setTimeout(() => {
    agentRunPersistTimers.delete(key);
    void persistAgentRunSnapshot(key).catch((error) => {
      console.warn("[sidepanel] failed to persist agent run", error);
    });
  }, 120);
  agentRunPersistTimers.set(key, timer);
}

async function hydratePersistedAgentRuns() {
  const records = await listAgentRunStates();
  const nextRuns = {};
  for (const record of records) {
    const snapshot = record?.snapshot || record?.runtime || null;
    const craftId = String(record?.craftId || snapshot?.craftId || "").trim();
    if (!craftId || !snapshot || typeof snapshot !== "object") continue;
    const normalized = normalizePersistedAgentRuns({
      [craftId]: snapshot,
    }, {
      interruptRunning: false,
    });
    if (normalized[craftId]) {
      nextRuns[craftId] = normalized[craftId];
    }
  }
  state.agentRuns = nextRuns;
}

async function readHeaderTestSessionState() {
  if (!globalThis.chrome?.storage?.session?.get) return null;
  try {
    const stored = await globalThis.chrome.storage.session.get(HEADER_TEST_SESSION_KEY);
    const value = stored?.[HEADER_TEST_SESSION_KEY];
    return value && typeof value === "object" ? value : null;
  } catch (error) {
    console.warn("[sidepanel] failed to read header test session state", error);
    return null;
  }
}

async function writeHeaderTestSessionState() {
  if (!globalThis.chrome?.storage?.session?.set) return;
  const completed = {};
  for (const testId of Object.keys(HEADER_TEST_DEFS)) {
    const status = String(state.headerTests?.[testId]?.status || "");
    if (status === "success" || status === "hidden") {
      completed[testId] = true;
    }
  }
  try {
    await globalThis.chrome.storage.session.set({
      [HEADER_TEST_SESSION_KEY]: completed,
    });
  } catch (error) {
    console.warn("[sidepanel] failed to persist header test session state", error);
  }
}

async function hydrateHeaderTestSessionState() {
  const persisted = await readHeaderTestSessionState();
  if (!persisted || typeof persisted !== "object") return;

  const nextState = createHeaderTestUiState();
  for (const testId of Object.keys(nextState)) {
    if (persisted[testId] === true) {
      nextState[testId] = createHeaderTestEntry("hidden", 1, false);
    }
  }
  state.headerTests = nextState;
}

async function hydrateSidepanelState() {
  const persisted = await readPersistedSidepanelStateRecord();
  if (!persisted || typeof persisted !== "object") return;

  state.activeCraftId = String(persisted.activeCraftId || "") || null;
  state.craftingCraftId = null;
  state.createOpen = persisted.createOpen === true;
  state.draftName = String(persisted.draftName || "");
  state.draftGoal = String(persisted.draftGoal || "");
  state.promptDrafts = normalizePersistedTextMap(persisted.promptDrafts);
  state.agentPromptDrafts = normalizePersistedTextMap(persisted.agentPromptDrafts);
  state.trainingRuns = normalizePersistedTrainingRuns(persisted.trainingRuns);
  state.useMessages = normalizePersistedUseMessages(persisted.useMessages);
  state.craftResponses = normalizePersistedCraftResponses(persisted.craftResponses);
  state.debugTrainingRun = persisted.debugTrainingRun
    ? normalizePersistedTrainingRun(persisted.debugTrainingRun)
    : null;
  state.tutorialOverlay = normalizePersistedTutorialOverlay(persisted.tutorialOverlay);
  await hydratePersistedAgentRuns();
  lastPersistedSidepanelStateSignature = getPersistedSidepanelStateSignature(
    buildPersistedSidepanelState(),
  );
}

function pruneMapToCrafts(rawMap, validCraftIds) {
  const out = {};
  for (const [craftId, value] of Object.entries(asPlainObject(rawMap))) {
    if (!validCraftIds.has(craftId)) continue;
    out[craftId] = value;
  }
  return out;
}

function reconcileSidepanelStateWithCrafts() {
  const validCraftIds = new Set(state.crafts.map((craft) => craft.id));
  if (state.activeCraftId && !validCraftIds.has(state.activeCraftId)) {
    state.activeCraftId = null;
  }
  if (state.craftingCraftId && !validCraftIds.has(state.craftingCraftId)) {
    state.craftingCraftId = null;
  }
  state.promptDrafts = pruneMapToCrafts(state.promptDrafts, validCraftIds);
  state.agentPromptDrafts = pruneMapToCrafts(state.agentPromptDrafts, validCraftIds);
  state.agentRuns = pruneMapToCrafts(state.agentRuns, validCraftIds);
  state.trainingRuns = pruneMapToCrafts(state.trainingRuns, validCraftIds);
  state.useMessages = pruneMapToCrafts(state.useMessages, validCraftIds);
  state.craftResponses = pruneMapToCrafts(state.craftResponses, validCraftIds);
  state.trainingDataStates = pruneMapToCrafts(state.trainingDataStates, validCraftIds);
  state.toolScriptStates = pruneMapToCrafts(state.toolScriptStates, validCraftIds);
  state.capabilityWeightsStates = pruneMapToCrafts(state.capabilityWeightsStates, validCraftIds);
  if (state.trainingDataCraftId && !validCraftIds.has(state.trainingDataCraftId)) {
    state.trainingDataCraftId = null;
  }
  if (state.tutorialOverlay?.craftId && !validCraftIds.has(state.tutorialOverlay.craftId)) {
    state.tutorialOverlay = null;
  }
  if (!state.crafts.length) {
    resetStarterTutorialSeen();
  }
}

function resumePersistedTrainingPolls() {
  for (const craftId of Object.keys(state.trainingRuns || {})) {
    if (isTrainingRunActive(state.trainingRuns[craftId])) {
      scheduleTrainingPoll(craftId);
    }
  }
  if (isTrainingRunActive(state.debugTrainingRun)) {
    scheduleDebugTrainingPoll();
  }
}

function resumePersistedAgentRunPolls() {
  for (const [craftId, run] of Object.entries(state.agentRuns || {})) {
    if (!run?.runId || isFinalAgentRunStatus(run.status)) continue;
    scheduleAgentRunPoll(craftId);
  }
}

function scheduleSidepanelStatePersist() {
  if (!sidepanelStatePersistenceReady) return;
  globalThis.clearTimeout(sidepanelStatePersistTimer);
  sidepanelStatePersistTimer = globalThis.setTimeout(async () => {
    const snapshot = buildPersistedSidepanelState();
    const signature = getPersistedSidepanelStateSignature(snapshot);
    if (!signature || signature === lastPersistedSidepanelStateSignature) return;
    try {
      await writePersistedSidepanelStateRecord(snapshot);
      lastPersistedSidepanelStateSignature = signature;
    } catch (error) {
      console.warn("[sidepanel] failed to persist UI state", error);
    }
  }, 120);
}

document.addEventListener("pointerdown", (event) => {
  if (event.defaultPrevented) return;
  if (event.isPrimary === false) return;
  if (typeof event.button === "number" && event.button !== 0) return;
  const button = getPressFeedbackButton(getEventTargetElement(event));
  if (!button) return;
  beginButtonPressFeedback(button);
});

document.addEventListener("pointerup", (event) => {
  if (event.isPrimary === false) return;
  const releasedButton = getPressFeedbackButton(getEventTargetElement(event));
  if (!activePressedButton) return;
  endButtonPressFeedback(activePressedButton, { burst: activePressedButton === releasedButton });
});

document.addEventListener("pointercancel", () => {
  endButtonPressFeedback(activePressedButton);
});

document.addEventListener("click", (event) => {
  const target = getEventTargetElement(event);
  if (!target) return;
  const button = getPressFeedbackButton(target);
  if (button && event.detail === 0) {
    burstButtonPressFeedback(button);
  }
  const actionButton = target.closest("[data-action]");
  if (!actionButton) return;

  const action = actionButton.dataset.action;

  if (action === "open-options") {
    openOptionsPage();
    return;
  }

  if (action === "open-share-settings") {
    openOptionsPage("share");
    return;
  }

  if (action === "open-navigator") {
    const { craftId } = actionButton.dataset;
    if (!craftId) return;
    openBundleTab(craftId);
    return;
  }

  if (action === "open-bundle") {
    const { craftId } = actionButton.dataset;
    if (!craftId) return;
    openBundleTab(craftId);
    return;
  }

  if (action === "copy-agent-run-debug") {
    const { craftId } = actionButton.dataset;
    if (!craftId) return;
    void copyAgentRunDebugReport(craftId);
    return;
  }

  if (action === "copy-training-run-report") {
    const { craftId } = actionButton.dataset;
    if (!craftId) return;
    void copyTrainingRunReport(craftId);
    return;
  }

  if (action === "sync-now") {
    void craftSync?.syncNow?.({ pageName: "Sidepanel" });
    return;
  }

  if (action === "start-header-training") {
    void startHeaderTrainingRun(actionButton.dataset.headerTestId || "finetuning");
    return;
  }

  if (action === "run-workspace-roundtrip-smoke") {
    void runWorkspaceRoundtripSmokeAndCopy();
    return;
  }

  if (action === "run-inference-smoke") {
    void runSmokeTestAndCopy();
    return;
  }

  if (action === "run-agent-smoke") {
    void runAgentSmokeTestAndCopy();
    return;
  }

  if (action === "run-vision-smoke") {
    void runVisionSmokeTestAndCopy();
    return;
  }

  if (action === "run-self-use-smoke") {
    void runSelfUseSmokeTestAndCopy();
    return;
  }

  if (action === "run-tool-tabs-smoke") {
    void runToolTabsSmokeTestAndCopy();
    return;
  }

  if (action === "run-tool-visual-smoke") {
    void runToolVisualSmokeTestAndCopy();
    return;
  }

  if (action === "run-tool-code-smoke") {
    void runToolCodeSmokeTestAndCopy();
    return;
  }

  if (action === "run-tool-integrated-smoke") {
    void runToolIntegratedSmokeTestAndCopy();
    return;
  }

  if (action === "copy-test-error-report") {
    void copyCurrentSmokeErrorReport();
    return;
  }

  if (action === "randomize-setup-sync-token") {
    randomizeInlineSetupSyncToken();
    return;
  }

  if (action === "save-inline-setup") {
    void saveInlineSetup();
    return;
  }

  if (action === "dismiss-tutorial") {
    rememberStarterTutorialSeen();
    state.tutorialOverlay = null;
    render();
    return;
  }

  if (action === "tutorial-open-craft") {
    const craftId = state.tutorialOverlay?.craftId;
    if (!craftId) return;
    rememberStarterTutorialSeen();
    state.activeCraftId = craftId;
    state.tutorialOverlay = null;
    openBundleTab(craftId);
    render();
    return;
  }

  if (action === "toggle-craft") {
    const { craftId } = actionButton.dataset;
    if (!craftId) return;
    const isSameCraft = state.activeCraftId === craftId;
    state.activeCraftId = isSameCraft ? null : craftId;
    if (!isSameCraft) {
      void loadTrainingDataState(craftId);
    }
    if (isSameCraft && state.craftingCraftId === craftId) {
      state.craftingCraftId = null;
    }
    render();
    return;
  }

  if (action === "toggle-create") {
    state.createOpen = true;
    if (!state.starterTutorialSeen && !state.tutorialOverlay) {
      seedDraftGoalFromStarterTutorial();
      state.tutorialOverlay = { mode: "create" };
    }
    render();
    return;
  }

  if (action === "cancel-create") {
    resetDraft();
    render();
    return;
  }

  if (action === "create-craft") {
    void createCraftFromDraft();
    return;
  }

  if (action === "tutorial-continue-create") {
    continueStarterTutorialCreateFlow();
    return;
  }

  if (action === "tutorial-use-create-example") {
    useStarterTutorialExample(actionButton.dataset.exampleIndex);
    return;
  }

  if (action === "toggle-detail") {
    const { craftId } = actionButton.dataset;
    if (!craftId) return;
    openBundleTab(craftId);
    return;
  }

  if (action === "start-agent-run") {
    const { craftId } = actionButton.dataset;
    if (!craftId) return;
    void startAgentRun(craftId);
    return;
  }

  if (action === "stop-agent-run") {
    const { craftId } = actionButton.dataset;
    if (!craftId) return;
    void stopAgentRun(craftId);
    return;
  }

  if (action === "set-agent-tab") {
    const { craftId, agentTab } = actionButton.dataset;
    if (!craftId || !agentTab) return;
    setAgentRunTab(craftId, agentTab);
    return;
  }

  if (action === "fork-craft") {
    const { craftId } = actionButton.dataset;
    if (!craftId) return;
    void forkCraftIntoLocalDraft(craftId);
    return;
  }

  if (action === "delete-craft") {
    const { craftId } = actionButton.dataset;
    if (!craftId) return;
    void deleteCraft(craftId);
    return;
  }

  if (action === "toggle-share") {
    const { craftId } = actionButton.dataset;
    if (!craftId) return;
    void toggleCraftShareFromSidepanel(craftId);
    return;
  }

  if (action === "start-training-run") {
    const { craftId } = actionButton.dataset;
    if (!craftId) return;
    void startTrainingRun(craftId);
    return;
  }

  if (action === "open-training-data") {
    const { craftId } = actionButton.dataset;
    if (!craftId) return;
    void openTrainingDataView(craftId);
    return;
  }

  if (action === "close-training-data") {
    state.trainingDataCraftId = null;
    render();
    return;
  }

  if (action === "training-data-prev-page") {
    const { craftId } = actionButton.dataset;
    if (!craftId) return;
    setTrainingDataPage(craftId, getTrainingDataPage(craftId) - 1);
    render();
    return;
  }

  if (action === "training-data-next-page") {
    const { craftId } = actionButton.dataset;
    if (!craftId) return;
    setTrainingDataPage(craftId, getTrainingDataPage(craftId) + 1);
    render();
    return;
  }

  if (action === "training-data-add-sample") {
    const { craftId } = actionButton.dataset;
    if (!craftId) return;
    insertTrainingSampleDraft(craftId);
    render();
    return;
  }

  if (action === "training-data-delete-sample") {
    const { craftId, sampleId } = actionButton.dataset;
    if (!craftId || !sampleId) return;
    deleteTrainingSampleDraft(craftId, sampleId);
    return;
  }

  if (action === "copy-training-smoke") {
    void runSmokeTestAndCopy();
    return;
  }
});

document.addEventListener("input", (event) => {
  const target = getEventTargetElement(event);
  if (!target) return;
  const draftField = target.closest("[data-draft-field]");
  if (draftField) {
    if (draftField.dataset.draftField === "name") state.draftName = draftField.value;
    if (draftField.dataset.draftField === "goal") state.draftGoal = draftField.value;
    render();
    return;
  }

  const setupApiKeyField = target.closest("[data-setup-api-key]");
  if (setupApiKeyField) {
    setInlineSetupApiKey(setupApiKeyField.value);
    return;
  }

  const setupSyncField = target.closest("[data-setup-sync-field]");
  if (setupSyncField) {
    setInlineSetupSyncField(setupSyncField.dataset.setupSyncField, setupSyncField.value);
    return;
  }

  const promptField = target.closest("[data-prompt-field]");
  if (promptField) {
    const { craftId } = promptField.dataset;
    if (!craftId) return;
    state.promptDrafts[craftId] = promptField.value;
    return;
  }

  const agentPromptField = target.closest("[data-agent-prompt-field]");
  if (agentPromptField) {
    const { craftId } = agentPromptField.dataset;
    if (!craftId) return;
    state.agentPromptDrafts[craftId] = agentPromptField.value;
    return;
  }

  const agentQuestionField = target.closest("[data-agent-question-answer]");
  if (agentQuestionField) {
    const { craftId, questionId } = agentQuestionField.dataset;
    if (!craftId || !questionId) return;
    const run = state.agentRuns[craftId];
    if (!run) return;
    run.questions = normalizeAgentQuestions(run.questions).map((entry) =>
      entry.id === questionId ? { ...entry, answer: agentQuestionField.value } : entry,
    );
    scheduleAgentRunPersist(craftId);
    return;
  }

  const trainingSampleField = target.closest("[data-training-sample-field]");
  if (trainingSampleField) {
    const { craftId, sampleId, trainingSampleField: field } = trainingSampleField.dataset;
    if (!craftId || !sampleId || !field) return;
    updateTrainingSampleDraft(craftId, sampleId, field, trainingSampleField.value);
    return;
  }

  const toolScriptsField = target.closest("[data-tool-scripts-field]");
  if (toolScriptsField) {
    const { craftId } = toolScriptsField.dataset;
    if (!craftId) return;
    updateToolScriptsDraft(craftId, toolScriptsField.value);
    return;
  }

});

document.addEventListener("change", (event) => {
  const target = getEventTargetElement(event);
  if (!target) return;

  const setupProviderField = target.closest("[data-setup-provider]");
  if (setupProviderField) {
    setInlineSetupProvider(setupProviderField.value);
    return;
  }

  const setupSlotModelField = target.closest("[data-setup-slot-model]");
  if (setupSlotModelField) {
    setInlineSetupSlotModel(setupSlotModelField.dataset.setupSlotModel, setupSlotModelField.value);
    return;
  }

  const setupSlotReasoningField = target.closest("[data-setup-slot-reasoning]");
  if (setupSlotReasoningField) {
    setInlineSetupSlotReasoning(
      setupSlotReasoningField.dataset.setupSlotReasoning,
      setupSlotReasoningField.value,
    );
    return;
  }

  const setupThemeField = target.closest("[data-setup-theme]");
  if (setupThemeField) {
    setInlineSetupTheme(setupThemeField.value);
    return;
  }

  const trainingSampleField = target.closest("[data-training-sample-field]");
  if (trainingSampleField) {
    const { craftId, sampleId, trainingSampleField: field } = trainingSampleField.dataset;
    if (!craftId || !sampleId || !field) return;
    updateTrainingSampleDraft(craftId, sampleId, field, trainingSampleField.value);
    return;
  }

});

document.addEventListener("keydown", (event) => {
  if (event.defaultPrevented) return;
});

function getTrainingDataArtifactId(craftId) {
  return `training-samples:${String(craftId || "").trim()}`;
}

function createTrainingDataState(overrides = {}) {
  return {
    artifactId: "",
    loaded: false,
    loading: false,
    saving: false,
    saveError: "",
    page: 1,
    samples: [],
    lastSavedAt: "",
    ...overrides,
  };
}

function getTrainingDataState(craftId) {
  const key = String(craftId || "").trim();
  if (!key) return createTrainingDataState();
  if (!state.trainingDataStates[key]) {
    state.trainingDataStates[key] = createTrainingDataState({
      artifactId: getTrainingDataArtifactId(key),
    });
  }
  return state.trainingDataStates[key];
}

function createToolScriptState(overrides = {}) {
  return {
    artifactId: "",
    loaded: false,
    loading: false,
    saving: false,
    saveError: "",
    text: "",
    lastSavedAt: "",
    ...overrides,
  };
}

function getToolScriptState(craftId) {
  const key = String(craftId || "").trim();
  if (!key) return createToolScriptState();
  if (!state.toolScriptStates[key]) {
    state.toolScriptStates[key] = createToolScriptState({
      artifactId: getToolScriptsArtifactId(key),
    });
  }
  return state.toolScriptStates[key];
}

function createCapabilityWeightsState(overrides = {}) {
  return {
    artifactId: "",
    loaded: false,
    loading: false,
    status: "base_model_only",
    hasTrainedAdapter: false,
    completedAt: "",
    lastCheckedAt: "",
    ...overrides,
  };
}

function getCapabilityWeightsState(craftId) {
  const key = String(craftId || "").trim();
  if (!key) return createCapabilityWeightsState();
  if (!state.capabilityWeightsStates[key]) {
    state.capabilityWeightsStates[key] = createCapabilityWeightsState({
      artifactId: getWeightsArtifactId(key),
    });
  }
  return state.capabilityWeightsStates[key];
}

async function readCapabilityWeightsArtifactRecord(craftId) {
  const key = String(craftId || "").trim();
  if (!key) return null;
  const artifactId = getWeightsArtifactId(key);
  return await readLocalArtifactFromStores(artifactId, null);
}

async function loadCapabilityWeightsState(craftId, { force = false } = {}) {
  const key = String(craftId || "").trim();
  if (!key) return createCapabilityWeightsState();
  const capabilityState = getCapabilityWeightsState(key);
  if (capabilityState.loading) return capabilityState;
  if (!force && capabilityState.loaded) return capabilityState;

  capabilityState.loading = true;
  try {
    const record = await readCapabilityWeightsArtifactRecord(key);
    const payload = record?.payload && typeof record.payload === "object" ? record.payload : {};
    const status = String(payload?.status || "").trim().toLowerCase() || "base_model_only";
    const hasTransformerLora =
      Array.isArray(payload?.adapter?.modules) &&
      payload.adapter.modules.some(
        (entry) =>
          Boolean(String(entry?.modulePath || "").trim()) &&
          Boolean(String(entry?.loraADataUrl || "").trim()) &&
          Boolean(String(entry?.loraBDataUrl || "").trim()),
      );
    capabilityState.status = status;
    capabilityState.hasTrainedAdapter = status === "trained_adapter" && hasTransformerLora;
    capabilityState.completedAt = String(payload?.run?.completedAt || "").trim();
    capabilityState.loaded = true;
    capabilityState.lastCheckedAt = new Date().toISOString();
  } finally {
    capabilityState.loading = false;
  }
  render();
  return capabilityState;
}

async function readToolScriptsArtifactRecord(craftId) {
  const key = String(craftId || "").trim();
  if (!key) return null;
  const artifactId = getToolScriptsArtifactId(key);
  return await readLocalArtifactFromStores(artifactId, null, { throwOnFailure: true });
}

async function writeToolScriptsArtifactRecord(craftId, payload) {
  const key = String(craftId || "").trim();
  if (!key) return null;
  const record = {
    id: getToolScriptsArtifactId(key),
    craftId: key,
    kind: TOOL_SCRIPTS_ARTIFACT_KIND,
    payload,
    meta: {
      scriptCount: Array.isArray(payload?.scripts) ? payload.scripts.length : 0,
      updatedAt: Date.now(),
    },
  };
  return await writeLocalArtifactToStores(record, { throwOnFailure: true });
}

const STRICT_SIDEPANEL_TOOL_SCRIPTS_OPTIONS = Object.freeze({
  inferPlaceholderScripts: false,
  allowToolFallback: false,
});

const STRICT_SIDEPANEL_BROWSER_CAPABILITY_OPTIONS = Object.freeze({
  allowSyntheticCapabilities: false,
  allowBundleFallback: false,
});

function shouldUseEmbeddedSidepanelBundleFallback(craft = null) {
  return isRemoteSharedCraft(craft);
}

function getSidepanelToolScriptsOptions(craft = null) {
  return shouldUseEmbeddedSidepanelBundleFallback(craft)
    ? { inferPlaceholderScripts: false }
    : STRICT_SIDEPANEL_TOOL_SCRIPTS_OPTIONS;
}

function getSidepanelBrowserCapabilityOptions(craft = null) {
  return shouldUseEmbeddedSidepanelBundleFallback(craft)
    ? null
    : STRICT_SIDEPANEL_BROWSER_CAPABILITY_OPTIONS;
}

function stringifyToolScriptsPayload(payload, craft = null) {
  return JSON.stringify(normalizeToolScriptsPayload(payload, craft, getSidepanelToolScriptsOptions(craft)), null, 2);
}

function parseToolScriptsDraft(text, craft = null) {
  const source = String(text || "").trim();
  if (!source) {
    return {
      ok: true,
      payload: normalizeToolScriptsPayload(null, craft, getSidepanelToolScriptsOptions(craft)),
      error: "",
    };
  }
  try {
    return {
      ok: true,
      payload: normalizeToolScriptsPayload(JSON.parse(source), craft, getSidepanelToolScriptsOptions(craft)),
      error: "",
    };
  } catch (error) {
    return {
      ok: false,
      payload: null,
      error: error instanceof Error ? error.message : String(error || "Invalid JSON."),
    };
  }
}

async function loadToolScriptsState(craftId, { force = false } = {}) {
  const key = String(craftId || "").trim();
  if (!key) return createToolScriptState();

  const toolState = getToolScriptState(key);
  if (toolState.loaded && !force) {
    return toolState;
  }

  toolState.loading = true;
  toolState.saveError = "";
  render();

  try {
    const craft = findCraft(key);
    const fallbackPayload =
      shouldUseEmbeddedSidepanelBundleFallback(craft) &&
      craft?.bundle?.toolScripts?.payload &&
      typeof craft.bundle.toolScripts.payload === "object"
        ? craft.bundle.toolScripts.payload
        : null;
    const record = (await readToolScriptsArtifactRecord(key)) || {
      id: getToolScriptsArtifactId(key),
      payload: fallbackPayload || normalizeToolScriptsPayload(null, craft, getSidepanelToolScriptsOptions(craft)),
    };
    const normalizedPayload = normalizeToolScriptsPayload(record?.payload, craft, getSidepanelToolScriptsOptions(craft));
    toolState.artifactId = String(record?.id || getToolScriptsArtifactId(key));
    toolState.text = stringifyToolScriptsPayload(normalizedPayload, craft);
    toolState.lastSavedAt = String(record?.updatedAt || record?.updated_at || toolState.lastSavedAt || "");
    toolState.loaded = true;
    toolState.loading = false;
  } catch (error) {
    toolState.loading = false;
    toolState.loaded = true;
    toolState.saveError = error instanceof Error ? error.message : String(error || "Tool scripts could not be loaded.");
  }

  return toolState;
}

async function persistToolScriptsState(craftId) {
  const key = String(craftId || "").trim();
  if (!key) return null;
  const craft = findCraft(key);
  const toolState = getToolScriptState(key);
  const parsed = parseToolScriptsDraft(toolState.text, craft);
  if (!parsed.ok) {
    toolState.saveError = parsed.error;
    toolState.saving = false;
    render();
    return null;
  }

  toolState.saving = true;
  toolState.saveError = "";
  render();

  try {
    const record = await writeToolScriptsArtifactRecord(key, parsed.payload);
    const currentBrowserCapabilitiesRecord = await readBrowserCapabilityArtifactRecord(key);
    if (currentBrowserCapabilitiesRecord?.payload && typeof currentBrowserCapabilitiesRecord.payload === "object") {
      const normalizedBrowserCapabilitiesPayload = normalizeBrowserCapabilityBundlePayload(
        currentBrowserCapabilitiesRecord.payload,
        craft,
        parsed.payload,
        getSidepanelBrowserCapabilityOptions(craft),
      );
      const compiledBrowserCapabilitiesPayload = compilePublishedBrowserCapabilityBundlePayload(
        normalizedBrowserCapabilitiesPayload,
        {
          craft,
          toolScriptsPayload: parsed.payload,
          publishedBy: "sidepanel_editor",
        },
      );
      if (!compiledBrowserCapabilitiesPayload.ok) {
        toolState.saveError =
          compiledBrowserCapabilitiesPayload.error || "Browser functions runtime package could not be republished.";
      } else {
        await writeBrowserCapabilityArtifactRecord(key, compiledBrowserCapabilitiesPayload.payload);
      }
    }
    toolState.artifactId = String(record?.id || toolState.artifactId || getToolScriptsArtifactId(key));
    toolState.lastSavedAt = String(record?.updatedAt || new Date().toISOString());
    toolState.text = stringifyToolScriptsPayload(parsed.payload, craft);
    toolState.saving = false;
    render();
    return record;
  } catch (error) {
    toolState.saving = false;
    toolState.saveError = error instanceof Error ? error.message : String(error || "Tool scripts could not be saved.");
    render();
    throw error;
  }
}

function scheduleToolScriptsSave(craftId) {
  const key = String(craftId || "").trim();
  if (!key) return;
  const currentTimer = toolScriptSaveTimers.get(key);
  if (currentTimer) {
    globalThis.clearTimeout(currentTimer);
  }
  const timer = globalThis.setTimeout(() => {
    toolScriptSaveTimers.delete(key);
    void persistToolScriptsState(key);
  }, 240);
  toolScriptSaveTimers.set(key, timer);
}

function updateToolScriptsDraft(craftId, value) {
  const toolState = getToolScriptState(craftId);
  toolState.text = String(value || "");
  toolState.saveError = "";
  scheduleToolScriptsSave(craftId);
}

function normalizeTrainingDataPage(value, totalSamples = 0) {
  const totalPages = Math.max(1, Math.ceil(Math.max(0, Number(totalSamples || 0)) / TRAINING_DATA_PAGE_SIZE));
  const page = Math.max(1, Math.trunc(Number(value) || 1));
  return Math.min(totalPages, page);
}

function setTrainingDataPage(craftId, nextPage) {
  const trainingState = getTrainingDataState(craftId);
  trainingState.page = normalizeTrainingDataPage(nextPage, trainingState.samples.length);
}

function getTrainingDataPage(craftId) {
  return normalizeTrainingDataPage(getTrainingDataState(craftId).page, getTrainingDataState(craftId).samples.length);
}

function normalizeTrainingSampleStatus(value) {
  const candidate = String(value || "").trim().toLowerCase();
  return ["draft", "review", "ready", "blocked"].includes(candidate) ? candidate : "draft";
}

function normalizeTrainingSampleSplit(value) {
  const candidate = String(value || "").trim().toLowerCase();
  return ["train", "validation", "test"].includes(candidate) ? candidate : "train";
}

function stringifyTrainingJson(value) {
  if (typeof value === "string") return value;
  if (value == null) return "{}";
  try {
    return JSON.stringify(value, null, 2);
  } catch (_error) {
    return "{}";
  }
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

function hasStructuredTrainingPayload(sample) {
  return (
    (Array.isArray(sample?.messages) && sample.messages.length > 0) ||
    Number.isInteger(sample?.targetTurnIndex)
  );
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

function describeStructuredTrainingTrace(sample) {
  const inspection = inspectPortableTrainingTrace(sample?.messages, sample?.targetTurnIndex);
  if (!inspection.ok) {
    return inspection.reason || "Multi-turn transcript is incomplete.";
  }
  const messageCount = Array.isArray(inspection.normalizedMessages) ? inspection.normalizedMessages.length : 0;
  const targetTurn = Number.isInteger(inspection.normalizedTargetTurnIndex) ? inspection.normalizedTargetTurnIndex + 1 : 0;
  const details = [];
  if (messageCount > 0) details.push(`${messageCount} messages`);
  if (targetTurn > 0) details.push(`target turn ${targetTurn}`);
  if (inspection.hasToolResponses) details.push("tool responses included");
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
  const hasLegacyPayload = String(sample?.promptText || "").trim() || String(sample?.expectedJsonText || "").trim();
  if (hasLegacyPayload) {
    throw new Error(
      "Local Qwen training rows must use messages + tools + targetTurnIndex. Legacy promptText/expectedJson rows are no longer supported.",
    );
  }
  return null;
}

function countTrainingTextareaRows(text, minimum = 6, maximum = 24) {
  const lineCount = String(text || "").split(/\r?\n/).length + 1;
  return Math.max(minimum, Math.min(maximum, lineCount));
}

function buildStructuredTrainingInspection(sample) {
  if (!hasStructuredTrainingTrace(sample)) return null;
  const normalized = normalizeTrainingSample(sample);
  const messages = normalizePortableTrainingMessages(normalized.messages);
  const targetTurnIndex = normalizeTrainingTargetTurnIndex(normalized.targetTurnIndex);
  if (!messages.length || !Number.isInteger(targetTurnIndex) || targetTurnIndex >= messages.length) return null;
  const rendered = renderQwen35TrainingInspection(messages, normalized.tools, targetTurnIndex);
  return {
    promptText: rendered.prompt || summarizeStructuredTrainingPrompt(normalized),
    targetText: rendered.target || summarizeStructuredTrainingTarget(normalized),
    rawText: stringifyTrainingJson({
      id: normalized.id,
      prompt_text: normalized.promptText,
      messages,
      tools: normalized.tools,
      target_turn_index: targetTurnIndex,
      split: normalized.split,
      status: normalized.status,
      source: normalized.source,
      createdAt: normalized.createdAt,
      updatedAt: normalized.updatedAt,
    }),
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
    source: "manual",
    createdAt: now,
    updatedAt: now,
    ...overrides,
  };
}

function normalizeTrainingSample(sample, index = 0) {
  const fallback = createTrainingSampleDraft(index);
  const messages = normalizeTrainingMessages(sample?.messages);
  const targetTurnIndex = normalizeTrainingTargetTurnIndex(sample?.targetTurnIndex ?? sample?.target_turn_index);
  const structuredPayload = messages.length > 0 || Number.isInteger(targetTurnIndex);
  return {
    id: String(sample?.id || fallback.id),
    promptText: String(sample?.promptText ?? sample?.prompt_text ?? sample?.prompt ?? ""),
    expectedJsonText: structuredPayload ? "" : String(resolveTrainingSampleTargetText(sample)),
    messages,
    tools: normalizeTrainingTools(sample?.tools ?? sample?.available_tools),
    targetTurnIndex,
    split: normalizeTrainingSampleSplit(sample?.split),
    status: normalizeTrainingSampleStatus(sample?.status),
    source: String(sample?.source ?? sample?.sourceRef ?? sample?.source_index ?? fallback.source),
    createdAt: String(sample?.createdAt || fallback.createdAt),
    updatedAt: String(sample?.updatedAt || sample?.createdAt || fallback.updatedAt),
  };
}

function parseTrainingSampleJson(text) {
  const source = String(text || "").trim();
  if (!source) {
    return {
      ok: false,
      value: null,
      error: "Expected JSON is empty.",
    };
  }
  try {
    return {
      ok: true,
      value: JSON.parse(source),
      error: "",
    };
  } catch (error) {
    return {
      ok: false,
      value: null,
      error: error instanceof Error ? error.message : String(error || "Invalid JSON."),
    };
  }
}

function validateTrainingSample(sample) {
  if (hasStructuredTrainingPayload(sample)) {
    const inspection = inspectPortableTrainingTrace(sample?.messages, sample?.targetTurnIndex);
    return {
      runnable: inspection.ok,
      invalidJson: false,
      invalidTrace: !inspection.ok,
      traceState: inspection,
      jsonState: {
        ok: true,
        value: null,
        error: "",
      },
      detail: inspection.ok ? describeStructuredTrainingTrace(sample) : inspection.reason,
    };
  }
  const hasLegacyPayload =
    String(sample?.promptText || "").trim().length > 0 || String(sample?.expectedJsonText || "").trim().length > 0;
  return {
    runnable: false,
    invalidJson: false,
    invalidTrace: hasLegacyPayload,
    traceState: null,
    jsonState: {
      ok: false,
      value: null,
      error: "",
    },
    detail: hasLegacyPayload
      ? "Local Qwen training rows must use messages + tools + targetTurnIndex."
      : "Add messages, tools, and a supervised assistant target turn for a native Qwen training row.",
  };
}

function isTrainingSampleRunnable(sample) {
  return sample && sample.status !== "blocked" && validateTrainingSample(sample).runnable;
}

function isTrainingSampleReady(sample) {
  return isTrainingSampleRunnable(sample);
}

function getTrainingDataMeta(samples = []) {
  const normalizedSamples = Array.isArray(samples) ? samples : [];
  const totalSamples = normalizedSamples.length;
  const draftSamples = normalizedSamples.filter((sample) => sample.status === "draft").length;
  const reviewSamples = normalizedSamples.filter((sample) => sample.status === "review").length;
  const readySamples = normalizedSamples.filter((sample) => sample.status === "ready" && isTrainingSampleReady(sample)).length;
  const runnableSamples = normalizedSamples.filter((sample) => sample.status !== "blocked" && validateTrainingSample(sample).runnable).length;
  const invalidJsonSamples = normalizedSamples.filter((sample) => {
    if (sample?.status === "blocked") return false;
    return validateTrainingSample(sample).invalidJson;
  }).length;
  const invalidTraceSamples = normalizedSamples.filter((sample) => {
    if (sample?.status === "blocked") return false;
    return validateTrainingSample(sample).invalidTrace;
  }).length;
  const lastUpdatedAt = normalizedSamples
    .map((sample) => String(sample.updatedAt || sample.createdAt || "").trim())
    .filter(Boolean)
    .sort()
    .slice(-1)[0] || "";

  return {
    totalSamples,
    draftSamples,
    reviewSamples,
    readySamples,
    runnableSamples,
    invalidJsonSamples,
    invalidTraceSamples,
    invalidSamples: invalidJsonSamples + invalidTraceSamples,
    lastUpdatedAt,
  };
}

async function readTrainingDataArtifactRecord(craftId) {
  const key = String(craftId || "").trim();
  if (!key) return null;
  const artifactId = getTrainingDataArtifactId(key);
  return await readLocalArtifactFromStores(artifactId, null, { throwOnFailure: true });
}

async function writeTrainingDataArtifactRecord(craftId, serializedSamples) {
  const key = String(craftId || "").trim();
  if (!key) return null;
  const record = {
    id: getTrainingDataArtifactId(key),
    craftId: key,
    kind: TRAINING_DATA_ARTIFACT_KIND,
    payload: {
      samples: serializedSamples,
    },
    meta: {
      sampleCount: serializedSamples.length,
      updatedAt: Date.now(),
    },
  };
  return await writeLocalArtifactToStores(record, { throwOnFailure: true });
}

async function loadTrainingDataState(craftId, { force = false } = {}) {
  const key = String(craftId || "").trim();
  if (!key) return createTrainingDataState();

  const trainingState = getTrainingDataState(key);
  if (trainingState.loaded && !force) {
    return trainingState;
  }

  trainingState.loading = true;
  trainingState.saveError = "";
  render();

  try {
    const artifact = (await readTrainingDataArtifactRecord(key)) || {
      id: getTrainingDataArtifactId(key),
      payload: {
        samples: [],
      },
    };
    const samples = Array.isArray(artifact?.payload?.samples)
      ? artifact.payload.samples.map((sample, index) => normalizeTrainingSample(sample, index))
      : [];
    trainingState.artifactId = String(artifact?.id || getTrainingDataArtifactId(key));
    trainingState.samples = samples;
    trainingState.loaded = true;
    trainingState.loading = false;
    trainingState.page = normalizeTrainingDataPage(trainingState.page, samples.length);
    trainingState.lastSavedAt = String(artifact?.updatedAt || artifact?.updated_at || trainingState.lastSavedAt || "");
  } catch (error) {
    trainingState.loading = false;
    trainingState.loaded = true;
    trainingState.saveError = error instanceof Error ? error.message : String(error || "Training data could not be loaded.");
  }

  return trainingState;
}

async function openTrainingDataView(craftId) {
  const key = String(craftId || "").trim();
  if (!key) return;
  openTrainingDataTab(key);
}

function updateTrainingSampleDraft(craftId, sampleId, field, value) {
  const trainingState = getTrainingDataState(craftId);
  const sample = trainingState.samples.find((entry) => entry.id === sampleId);
  if (!sample) return;
  if (!["promptText", "expectedJsonText", "split", "status", "source"].includes(field)) return;
  sample[field] = String(value || "");
  if (field === "split") sample[field] = normalizeTrainingSampleSplit(value);
  if (field === "status") sample[field] = normalizeTrainingSampleStatus(value);
  sample.updatedAt = new Date().toISOString();
  scheduleTrainingDataSave(craftId);
}

function insertTrainingSampleDraft(craftId) {
  const trainingState = getTrainingDataState(craftId);
  const nextSample = createTrainingSampleDraft(trainingState.samples.length);
  trainingState.samples.unshift(nextSample);
  trainingState.page = 1;
  scheduleTrainingDataSave(craftId);
}

function deleteTrainingSampleDraft(craftId, sampleId) {
  const trainingState = getTrainingDataState(craftId);
  trainingState.samples = trainingState.samples.filter((sample) => sample.id !== sampleId);
  trainingState.page = normalizeTrainingDataPage(trainingState.page, trainingState.samples.length);
  scheduleTrainingDataSave(craftId);
  render();
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
      split: normalized.split,
      status: normalized.status,
      source: normalized.source,
      createdAt: normalized.createdAt,
      updatedAt: normalized.updatedAt,
    };
  });
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

  const fallbackValidationSize = Math.max(1, Math.min(Math.floor(validRows.length * 0.15), Math.max(1, trainPool.length - 1)));
  const fallbackTestSize = Math.max(1, Math.min(Math.floor(validRows.length * 0.2), Math.max(1, trainPool.length - 1)));

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

async function buildTrainingDatasetPayload(craftId) {
  const trainingState = await loadTrainingDataState(craftId);
  const dataset = pickTrainingDatasetRows(trainingState.samples);
  if (!dataset.train.length || !dataset.validation.length || !dataset.test.length) {
    throw new Error("Add valid native Qwen multi-turn samples before starting a local run.");
  }
  return {
    train: dataset.train,
    validation: dataset.validation,
    test: dataset.test,
    meta: {
      source: "sidepanel_craft_samples",
      craftId,
      artifactId: trainingState.artifactId || getTrainingDataArtifactId(craftId),
      sampleCount: dataset.train.length + dataset.validation.length + dataset.test.length,
      splitCounts: {
        train: dataset.train.length,
        validation: dataset.validation.length,
        test: dataset.test.length,
      },
    },
  };
}

function summarizeTrainingSamplesForAgent(samples = [], limit = 16) {
  return (Array.isArray(samples) ? samples : [])
    .slice(0, limit)
    .map((sample) => {
      const structured = hasStructuredTrainingTrace(sample);
      return {
        id: String(sample.id || ""),
        mode: "multiturn",
        split: String(sample.split || "train"),
        status: String(sample.status || "draft"),
        source: String(sample.source || ""),
        promptText: structured
          ? trimText(summarizeStructuredTrainingPrompt(sample), 280)
          : trimText(sample.promptText || "", 280),
        targetTurnSummary: trimText(summarizeStructuredTrainingTarget(sample), 280),
        targetTurnIndex: structured && Number.isInteger(sample.targetTurnIndex) ? sample.targetTurnIndex : null,
        messageCount: structured && Array.isArray(sample.messages) ? sample.messages.length : 0,
        messages: structured && Array.isArray(sample.messages) ? sample.messages : [],
        tools: structured && Array.isArray(sample.tools) ? sample.tools : [],
        updatedAt: String(sample.updatedAt || sample.createdAt || ""),
      };
    });
}

function summarizeAgentOperationCounts(operations = []) {
  const counts = { add: 0, update: 0, delete: 0 };
  for (const operation of Array.isArray(operations) ? operations : []) {
    const type = String(operation?.type || "").trim().toLowerCase();
    if (Object.prototype.hasOwnProperty.call(counts, type)) {
      counts[type] += 1;
    }
  }
  return Object.entries(counts)
    .filter(([, count]) => count > 0)
    .map(([type, count]) => {
      if (type === "add") return `${formatCount(count)} hinzugefuegt`;
      if (type === "update") return `${formatCount(count)} aktualisiert`;
      if (type === "delete") return `${formatCount(count)} entfernt`;
      return `${formatCount(count)} ${type}`;
    })
    .join(" · ");
}

function getAgentOperationCounts(operations = []) {
  const counts = { add: 0, update: 0, delete: 0 };
  for (const operation of Array.isArray(operations) ? operations : []) {
    const type = String(operation?.type || "").trim().toLowerCase();
    if (Object.prototype.hasOwnProperty.call(counts, type)) {
      counts[type] += 1;
    }
  }
  return counts;
}

function describeAgentOperationMilestone(operations = []) {
  const counts = getAgentOperationCounts(operations);
  if (counts.add && !counts.update && !counts.delete) {
    return `${formatCount(counts.add)} ${counts.add === 1 ? "training sample" : "training samples"} added`;
  }
  if (counts.update && !counts.add && !counts.delete) {
    return `${formatCount(counts.update)} ${counts.update === 1 ? "training sample" : "training samples"} updated`;
  }
  if (counts.delete && !counts.add && !counts.update) {
    return `${formatCount(counts.delete)} ${counts.delete === 1 ? "training sample" : "training samples"} removed`;
  }
  const summary = summarizeAgentOperationCounts(operations);
  return summary ? `Training data revised (${summary})` : "Training data revised";
}

function describeAgentOperationMilestoneDetail(operations = []) {
  const counts = getAgentOperationCounts(operations);
  if (counts.add && !counts.update && !counts.delete) {
    return "The agent created new training samples for this capability.";
  }
  if (counts.update && !counts.add && !counts.delete) {
    return "The agent improved existing training samples for this capability.";
  }
  if (counts.delete && !counts.add && !counts.update) {
    return "The agent removed unsuitable training samples.";
  }
  return "The agent revised the local training data for this capability.";
}

function deriveAgentMatchingSignals(plan, prompt, samples = []) {
  const operations = Array.isArray(plan?.operations) ? plan.operations : [];
  const meta = getTrainingDataMeta(samples);
  const signals = [];
  if (samples.length) {
    signals.push(`${formatCount(samples.length)} local seed rows were available as reference.`);
  } else {
    signals.push("No local seed rows were available yet.");
  }
  if (meta.readySamples > 0) {
    signals.push(`${formatCount(meta.readySamples)} rows were already training-ready.`);
  }
  const toolingLabel = getCraftingAgentToolingLabel(plan?.craft || null);
  if (toolingLabel) {
    signals.push(`Feste Supervisor-Tools: ${toolingLabel}.`);
  }
  const operationSummary = summarizeAgentOperationCounts(operations);
  if (operationSummary) {
    signals.push(`Planned: ${operationSummary}.`);
  }
  if (/browser|markiert|ausgew[aä]hlt|eingabefeld/i.test(String(prompt || ""))) {
    signals.push("The brief points to a browser-related transformation task.");
  }
  return signals.slice(0, 6);
}

function deriveAgentQuestions(plan, prompt, samples = []) {
  const sourceQuestions = normalizeAgentQuestions(plan?.openQuestions);
  if (sourceQuestions.length) return sourceQuestions;

  const questions = [];
  if (!samples.length) {
    questions.push({
      question: "Provide a first concrete prompt-to-JSON seed example.",
      reason: "Without a seed row, the agent lacks a reliable reference for additional training data.",
    });
  }
  if (!Array.isArray(plan?.operations) || !plan.operations.length) {
    questions.push({
      question: "Which stable JSON target structure should this craft produce?",
      reason: "The current brief does not define a reliable output structure yet.",
    });
  }
  if (/browser|markiert|ausgew[aä]hlt|eingabefeld/i.test(String(prompt || "")) && !samples.length) {
    questions.push({
      question: "Which browser selection or source field should be described explicitly in the training JSON?",
      reason: "The request mentions browser context, but it does not define a concrete data structure yet.",
    });
  }
  return normalizeAgentQuestions(questions).slice(0, 4);
}

function deriveAgentProvenance(plan, samples = []) {
  const sourceEntries = normalizeAgentProvenance(plan?.provenance);
  const toolingEntry = createAgentToolingProvenanceEntry(plan?.craft || null);
  if (toolingEntry) {
    const alreadyTracked = sourceEntries.some((entry) =>
      /supervisor-tools|browser automation|web search/i.test(`${entry.title} ${entry.detail}`),
    );
    if (!alreadyTracked) {
      sourceEntries.unshift(toolingEntry);
    }
  }
  if (sourceEntries.length) return sourceEntries;

  const operations = Array.isArray(plan?.operations) ? plan.operations : [];
  const entries = operations.slice(0, 8).map((operation, index) => {
    const type = String(operation?.type || "").trim().toLowerCase();
    const sampleId = String(operation?.sampleId || "").trim();
    const reason = String(operation?.reason || "").trim();
    return {
      title:
        type === "add"
          ? "New seed row planned"
          : type === "delete"
            ? `Cleanup ${sampleId || "of one row"}`
            : `Revision ${sampleId || "of one row"}`,
      detail: reason || "Operation adopted from the structured agent plan.",
      kind: "operation",
      sampleId,
      operationType: ["add", "update", "delete"].includes(type) ? type : "",
    };
  });

  if (toolingEntry) {
    entries.unshift(toolingEntry);
  }

  if (!entries.length && samples.length) {
    entries.push({
      title: "Local dataset view inspected",
      detail: `${formatCount(samples.length)} existing rows were considered as context for the agent plan.`,
      kind: "sample",
      sampleId: "",
      operationType: "",
    });
  }

  return normalizeAgentProvenance(entries);
}

function normalizeAgentPlanResult(plan, prompt, samples = []) {
  const operations = Array.isArray(plan?.operations) ? plan.operations : [];
  const operationSummary = summarizeAgentOperationCounts(operations);
  const useSurface = normalizeAgentUseSurfaceSuggestion(plan?.useSurface, plan?.craft || null, prompt);
  const report = normalizeAgentReport(plan?.report, {
    objective: trimText(prompt || "", 180),
    currentState: operations.length
      ? `The agent derived ${operationSummary} from ${formatCount(samples.length)} visible seed rows.`
      : samples.length
        ? `The agent inspected ${formatCount(samples.length)} visible seed rows but has not found a reliable edit plan yet.`
        : "The agent is starting without local seed rows and must establish the target structure first.",
    nextAction: operations.length
      ? "Apply the planned training-data changes and inspect them in the dataset view."
      : "Answer the open questions or add a concrete seed example.",
  });
  if (!report.matchingSignals.length) {
    report.matchingSignals = deriveAgentMatchingSignals(plan, prompt, samples);
  }

  const questions = deriveAgentQuestions(plan, prompt, samples);
  const provenance = deriveAgentProvenance(plan, samples);

  return {
    summary: String(plan?.summary || "").trim(),
    rationale: String(plan?.rationale || "").trim(),
    report,
    questions,
    provenance,
    useSurface,
    operations,
    text: buildAgentTrainingSummaryText({
      summary: String(plan?.summary || "").trim(),
      rationale: String(plan?.rationale || "").trim(),
      report,
      operations,
    }),
  };
}

function buildAgentTrainingSummaryText(plan) {
  const summary = String(plan?.summary || "").trim();
  const rationale = String(plan?.rationale || "").trim();
  const operationCount = Array.isArray(plan?.operations) ? plan.operations.length : 0;
  const nextAction = String(plan?.report?.nextAction || "").trim();
  return [summary, rationale, operationCount ? `Operations: ${operationCount}.` : "", nextAction]
    .filter(Boolean)
    .join("\n\n");
}

function applyAgentTrainingSampleOperation(samples, operation) {
  const nextSamples = (Array.isArray(samples) ? samples : []).map((sample, index) =>
    normalizeTrainingSample(sample, index),
  );
  const type = String(operation?.type || "").trim().toLowerCase();
  const fields = operation?.fields && typeof operation.fields === "object" ? operation.fields : {};
  const sampleId = String(operation?.sampleId || "").trim();
  const reason = String(operation?.reason || "").trim();
  const now = new Date().toISOString();

  if (type === "add") {
    const promptText = String(fields.promptText || "").trim();
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
      source: String(fields.source || "agent"),
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

async function applyAgentTrainingOperations(craftId, runId, operations) {
  const ops = Array.isArray(operations) ? operations : [];
  const trainingState = await loadTrainingDataState(craftId, { force: true });
  let workingSamples = Array.isArray(trainingState.samples)
    ? trainingState.samples.map((sample, index) => normalizeTrainingSample(sample, index))
    : [];
  let applied = 0;
  let failed = 0;

  for (const [index, operation] of ops.entries()) {
    if (!appendAgentLog(craftId, runId, "info", `Applying ${operation?.type || "operation"} ${index + 1}/${ops.length}...`)) {
      break;
    }

    try {
      const result = applyAgentTrainingSampleOperation(workingSamples, operation);
      workingSamples = result.samples;
      await persistTrainingDataSnapshot(craftId, workingSamples, { updateLocalState: false });
      appendAgentLog(craftId, runId, "success", result.message);
      applied += 1;
      render();
      await sleep(120);
    } catch (error) {
      failed += 1;
      appendAgentLog(
        craftId,
        runId,
        "error",
        error instanceof Error ? error.message : String(error || "Operation failed."),
      );
    }
  }

  return { applied, failed };
}

async function syncCraftTrainingDataSummary(craftId, samples) {
  const craft = findCraft(craftId);
  if (!craft) return;
  const meta = getTrainingDataMeta(samples);
  craft.seedRows = meta.totalSamples;
  craft.datasetRows = meta.runnableSamples;
  touchCraft(craft, meta.lastUpdatedAt || new Date().toISOString());
  state.crafts = await persistCrafts(state.crafts);
}

async function persistTrainingDataSnapshot(craftId, samples, { updateLocalState = true } = {}) {
  const key = String(craftId || "").trim();
  if (!key) return null;
  const trainingState = getTrainingDataState(key);
  const normalizedSamples = (Array.isArray(samples) ? samples : []).map((sample, index) =>
    normalizeTrainingSample(sample, index),
  );
  for (const sample of normalizedSamples) {
    ensureNativeQwenTrainingSample(sample);
  }
  const serializedSamples = serializeTrainingSamples(normalizedSamples);

  if (updateLocalState) {
    trainingState.samples = normalizedSamples;
    trainingState.page = normalizeTrainingDataPage(trainingState.page, normalizedSamples.length);
    trainingState.saving = true;
    trainingState.saveError = "";
    render();
  }

  try {
    const record = await writeTrainingDataArtifactRecord(key, serializedSamples);
    if (updateLocalState) {
      trainingState.artifactId = String(record?.id || trainingState.artifactId || getTrainingDataArtifactId(key));
      trainingState.lastSavedAt = String(record?.updatedAt || new Date().toISOString());
      trainingState.saving = false;
    }
    await syncCraftTrainingDataSummary(key, normalizedSamples);
    return record;
  } catch (error) {
    if (updateLocalState) {
      trainingState.saving = false;
      trainingState.saveError = error instanceof Error ? error.message : String(error || "Training data could not be saved.");
    }
    throw error;
  } finally {
    if (updateLocalState) {
      render();
    }
  }
}

async function persistTrainingDataState(craftId) {
  const key = String(craftId || "").trim();
  if (!key) return null;
  const trainingState = getTrainingDataState(key);
  return await persistTrainingDataSnapshot(key, trainingState.samples, { updateLocalState: true });
}

function scheduleTrainingDataSave(craftId) {
  const key = String(craftId || "").trim();
  if (!key) return;
  const currentTimer = trainingDataSaveTimers.get(key);
  if (currentTimer) {
    globalThis.clearTimeout(currentTimer);
  }
  const timer = globalThis.setTimeout(() => {
    trainingDataSaveTimers.delete(key);
    void persistTrainingDataState(key);
  }, 180);
  trainingDataSaveTimers.set(key, timer);
}

async function flushTrainingDataSave(craftId) {
  const key = String(craftId || "").trim();
  if (!key) return;
  const currentTimer = trainingDataSaveTimers.get(key);
  if (currentTimer) {
    globalThis.clearTimeout(currentTimer);
    trainingDataSaveTimers.delete(key);
  }
  await persistTrainingDataState(key);
}

function sleep(ms) {
  return new Promise((resolve) => {
    globalThis.setTimeout(resolve, ms);
  });
}

async function withTimeout(promise, timeoutMs, timeoutMessage) {
  let timeoutId = null;
  const pending = Promise.resolve(promise);
  pending.catch(() => {});
  try {
    return await Promise.race([
      pending,
      new Promise((_, reject) => {
        timeoutId = globalThis.setTimeout(() => {
          reject(new Error(timeoutMessage));
        }, Math.max(0, Number(timeoutMs) || 0));
      }),
    ]);
  } finally {
    if (timeoutId) {
      globalThis.clearTimeout(timeoutId);
    }
  }
}

function joinClassNames(...parts) {
  return parts.filter(Boolean).join(" ");
}

function buildMetaTokenNodes(items = []) {
  const values = (Array.isArray(items) ? items : []).filter(Boolean);
  const nodes = [];
  values.forEach((value, index) => {
    if (index) {
      nodes.push(h("span", { className: "meta-sep", "aria-hidden": "true", key: `sep-${index}` }, "·"));
    }
    nodes.push(h("span", { className: "meta-token", key: `token-${index}` }, value));
  });
  return nodes;
}

function HeaderMenuButtonView() {
  return h(
    "button",
    {
      className: "header-menu",
      type: "button",
      "aria-label": "Open model settings",
      "data-action": "open-options",
    },
    [
      h("span", { key: "line-1" }),
      h("span", { key: "line-2" }),
      h("span", { key: "line-3" }),
    ],
  );
}

function HeaderTestButtonView({ testId, smokeBusy = false, trainingBusy = false }) {
  const def = HEADER_TEST_DEFS[testId];
  if (!def) return null;

  const testState = getHeaderTestState(testId);
  if (testState.status === "hidden") return null;

  const isRunning = testState.status === "running";
  const isSuccess = testState.status === "success";
  const disabled = isRunning || isSuccess || smokeBusy || trainingBusy;
  const label = isRunning ? `${def.label}...` : def.label;
  const progress = Math.max(0.04, Math.min(1, Number(testState.progress) || 0));
  const classes = joinClassNames(
    "header-smoke",
    def.variant === "quiet" ? "header-smoke-quiet" : "",
    isSuccess ? "header-smoke-success" : "",
  );

  return h("div", { className: joinClassNames("header-smoke-wrap", isRunning ? "header-smoke-wrap-running" : "") }, [
    h(
      "button",
      {
        className: classes,
        type: "button",
        disabled,
        "data-action": def.action,
        "data-header-test-id": testId,
        key: "button",
      },
      [
        isSuccess
          ? h("span", { className: "header-smoke-check", "aria-hidden": "true", key: "check" }, "✓")
          : null,
        h("span", { key: "label" }, label),
      ].filter(Boolean),
    ),
    isRunning
      ? h("span", { className: "header-smoke-progress", "aria-hidden": "true", key: "progress" }, [
          h("span", {
            className: joinClassNames(
              "header-smoke-progress-fill",
              testState.indeterminate ? "header-smoke-progress-fill-indeterminate" : "",
            ),
            style: testState.indeterminate ? undefined : { width: `${Math.round(progress * 100)}%` },
            key: "fill",
          }),
        ])
      : null,
  ]);
}

function HeaderActionRowView({ row, smokeBusy = false, trainingBusy = false }) {
  const visibleTestIds = row.tests.filter((testId) => {
    const def = HEADER_TEST_DEFS[testId];
    if (!def) return false;
    return getHeaderTestState(testId).status !== "hidden";
  });
  if (!visibleTestIds.length && !row.includeMenu) return null;

  return h(
    "div",
    { className: `panel-header-action-row panel-header-action-row-${row.id}` },
    [
      h("span", { className: "panel-header-action-label", key: "label" }, row.label),
      ...visibleTestIds.map((testId) => h(HeaderTestButtonView, { testId, smokeBusy, trainingBusy, key: testId })),
      row.includeMenu ? h(HeaderMenuButtonView, { key: "menu" }) : null,
    ].filter(Boolean),
  );
}

function PanelHeaderView({ headerTrainingRun = null }) {
  if (!shouldShowDevHeader()) {
    return h("header", { className: "panel-header panel-header-minimal" }, [
      h("div", { className: "panel-header-main", key: "main" }),
      h("div", { className: "panel-header-actions panel-header-actions-minimal", key: "actions" }, [
        h(HeaderMenuButtonView, { key: "menu" }),
      ]),
    ]);
  }

  const smokeBusy = isSmokeTestRunning();
  const trainingBusy = isTrainingRunActive(headerTrainingRun);

  return h("header", { className: "panel-header" }, [
    h("div", { className: "panel-header-main", key: "main" }),
    h(
      "div",
      { className: "panel-header-actions", key: "actions" },
      HEADER_TEST_ROW_DEFS
        .map((row) =>
          h(HeaderActionRowView, {
            row,
            smokeBusy,
            trainingBusy,
            key: row.id,
          }),
        )
        .filter(Boolean),
    ),
  ]);
}

function PanelFooterView() {
  return h("footer", { className: "panel-footer" }, [
    h("span", { className: "panel-footer-version", key: "version" }, `v${EXTENSION_VERSION}`),
    h("span", { className: "panel-footer-sync", key: "sync" }, buildFooterPeerStatus()),
  ]);
}

function TutorialCreateExampleCardView({ example, index, active = false }) {
  return h(
    "article",
    {
      className: joinClassNames("tutorial-example-card", active ? "tutorial-example-card-active" : ""),
    },
    [
      h("div", { className: "tutorial-example-head", key: "head" }, [
        h("div", { key: "copy" }, [
          h("div", { className: "tutorial-example-title", key: "title" }, example.title),
          h(
            "div",
            { className: "tutorial-example-meta", key: "meta" },
            active ? "Currently inserted into the input below." : "Starter prompt",
          ),
        ]),
        h(
          "button",
          {
            className: "workspace-secondary tutorial-example-button",
            type: "button",
            "data-action": "tutorial-use-create-example",
            "data-example-index": String(index),
            key: "button",
          },
          "Use example",
        ),
      ]),
      h("div", { className: "tutorial-example-copy", key: "prompt" }, example.prompt),
    ],
  );
}

function TutorialOverlayView() {
  const tutorial = state.tutorialOverlay;
  if (!tutorial) return null;

  if (tutorial.mode === "create") {
    const activePrompt = String(state.draftGoal || "").trim();
    return h("section", { className: "tutorial-overlay", "aria-label": "Capabilities" }, [
      h("div", { className: "tutorial-backdrop", key: "backdrop", "data-action": "dismiss-tutorial" }),
      h("div", { className: "tutorial-card", key: "card" }, [
        h("div", { className: "tutorial-brand", key: "brand" }, [
          h("img", {
            className: "tutorial-brand-mark",
            src: "assets/branding/logo-512.png",
            alt: "Fuck API, Train Local AI logo",
            width: 144,
            height: 144,
            key: "image",
          }),
        ]),
        h("div", { className: "tutorial-eyebrow", key: "eyebrow" }, "New capability"),
        h("h2", { className: "tutorial-title", key: "title" }, "Start from an example or write your own"),
        h(
          "p",
          { className: "tutorial-copy", key: "copy" },
          "The first example has already been inserted into the input below. Pick another example to replace it, or continue and edit the template.",
        ),
        h(
          "div",
          { className: "tutorial-example-list", key: "examples" },
          STARTER_TUTORIAL_EXAMPLES.map((example, index) =>
            h(TutorialCreateExampleCardView, {
              example,
              index,
              active: activePrompt === example.prompt,
              key: example.title,
            }),
          ),
        ),
        h("div", { className: "tutorial-steps", key: "steps" }, [
          h("article", { className: "tutorial-step", key: "step-1" }, [
            h("div", { className: "tutorial-step-title", key: "title" }, "1"),
            h("div", { className: "tutorial-step-copy", key: "copy" }, "Review or replace the starter prompt."),
          ]),
          h("article", { className: "tutorial-step", key: "step-2" }, [
            h("div", { className: "tutorial-step-title", key: "title" }, "2"),
            h("div", { className: "tutorial-step-copy", key: "copy" }, "Edit the text in the input if needed."),
          ]),
          h("article", { className: "tutorial-step", key: "step-3" }, [
            h("div", { className: "tutorial-step-title", key: "title" }, "3"),
            h(
              "div",
              { className: "tutorial-step-copy", key: "copy" },
              "Create the capability. The agent will take over and ask follow-up questions if needed.",
            ),
          ]),
        ]),
        h("div", { className: "tutorial-actions", key: "actions" }, [
          h(
            "button",
            {
              className: "workspace-primary",
              type: "button",
              "data-action": "tutorial-continue-create",
              key: "continue",
            },
            "Continue",
          ),
        ]),
      ]),
    ]);
  }

  return h("section", { className: "tutorial-overlay", "aria-label": "Craft tutorial" }, [
    h("div", { className: "tutorial-backdrop", key: "backdrop", "data-action": "dismiss-tutorial" }),
    h("div", { className: "tutorial-card", key: "card" }, [
      h("div", { className: "tutorial-brand", key: "brand" }, [
        h("img", {
          className: "tutorial-brand-mark",
          src: "assets/branding/logo-512.png",
          alt: "Fuck API, Train Local AI logo",
          width: 144,
          height: 144,
          key: "image",
        }),
      ]),
      h("div", { className: "tutorial-eyebrow", key: "eyebrow" }, "First Craft"),
      h("h2", { className: "tutorial-title", key: "title" }, `${tutorial.craftName} is your first capability`),
      h("p", { className: "tutorial-copy", key: "copy-1" }, [
        "A Craft is one narrow capability you can shape, test, tune locally, and later share as a bundle. The normal next step is to open ",
        h("strong", { key: "strong" }, "Craft"),
        " and define what this capability should do.",
      ]),
      h("p", { className: "tutorial-copy", key: "copy-2" }, [
        "This starter already runs on ",
        h("strong", { key: "strong" }, tutorial.starterModelName),
        ". Describe the capability now and the agent will handle the first build-out steps.",
      ]),
      h("div", { className: "tutorial-steps", key: "steps" }, [
        h("article", { className: "tutorial-step", key: "step-1" }, [
          h("div", { className: "tutorial-step-title", key: "title" }, "Craft now"),
          h(
            "div",
            { className: "tutorial-step-copy", key: "copy" },
            "Define the task. The agent will derive data, tools, and the first bundle setup from it.",
          ),
        ]),
      ]),
      h("div", { className: "tutorial-actions", key: "actions" }, [
        h(
          "button",
          {
            className: "workspace-primary",
            type: "button",
            "data-action": "tutorial-open-craft",
            key: "open",
          },
          "Craft now",
        ),
      ]),
    ]),
  ]);
}

function SmokeStatusView() {
  const headerTrainingRun = getHeaderTrainingRun();
  if (headerTrainingRun && isTrainingRunActive(headerTrainingRun)) {
    const progress = Math.max(0, Math.min(1, Number(headerTrainingRun.progress || 0)));
    const totalSamples = Number(headerTrainingRun.totalSamples || 0);
    const completedSamples = Number(headerTrainingRun.completedSamples || 0);
    const phaseTotalSamples = Number(headerTrainingRun.phaseTotalSamples || 0);
    const phaseCompletedSamples = Number(headerTrainingRun.phaseCompletedSamples || 0);
    const phaseUnitLabel = String(headerTrainingRun.phaseUnitLabel || "").trim() || "samples";
    const statParts = [];
    if (phaseTotalSamples > 0) {
      statParts.push(
        `${formatCompactCount(phaseCompletedSamples)} / ${formatCompactCount(phaseTotalSamples)} ${phaseUnitLabel}`,
      );
      statParts.push(formatItemsPerSecond(headerTrainingRun.samplesPerSecond || 0, phaseUnitLabel));
    } else if (Number(headerTrainingRun.samplesPerSecond || 0) > 0) {
      statParts.push(formatItemsPerSecond(headerTrainingRun.samplesPerSecond || 0, phaseUnitLabel));
    }
    const comparisonStat = getTrainingComparisonStat(headerTrainingRun);
    if (comparisonStat) {
      statParts.push(comparisonStat);
    }
    statParts.push(
      totalSamples > 0
        ? `job ${formatCompactCount(completedSamples)} / ${formatCompactCount(totalSamples)} steps`
        : `job ${formatCompactCount(completedSamples)} steps`,
    );
    if (Number(headerTrainingRun.currentEpoch || 0) > 0 && Number(headerTrainingRun.epochsTotal || 0) > 0) {
      statParts.push(`epoch ${headerTrainingRun.currentEpoch}/${headerTrainingRun.epochsTotal}`);
    }
    return h("section", { className: `smoke-status smoke-status-${getTrainingBannerTone(headerTrainingRun)}` }, [
      h("div", { className: "smoke-status-line", key: "line" }, headerTrainingRun.message || headerTrainingRun.phaseLabel || "Training run"),
      h("div", { className: "smoke-status-progress", "aria-hidden": "true", key: "progress" }, [
        h("span", {
          className: "smoke-status-progress-fill",
          style: { width: `${Math.round(progress * 100)}%` },
          key: "fill",
        }),
      ]),
      h("div", { className: "smoke-status-meta", key: "meta" }, statParts.join(" · ")),
      headerTrainingRun.error
        ? h("div", { className: "smoke-status-meta", key: "error" }, headerTrainingRun.error)
        : null,
    ]);
  }

  if (!["error", "success"].includes(String(state.smokeTest?.status || "")) || !state.smokeTest?.message) return null;

  return h("section", { className: `smoke-status smoke-status-${state.smokeTest.status || "idle"}` }, [
    h("div", { className: "smoke-status-line", key: "line" }, state.smokeTest.message),
    state.smokeTest.failureKind && state.smokeTest.status === "error"
      ? h(
          "div",
          { className: "smoke-status-meta smoke-status-kind", key: "kind" },
          `Type: ${formatFailureKindLabel(state.smokeTest.failureKind)}`,
        )
      : null,
    state.smokeTest.detail
      ? h("div", { className: "smoke-status-meta", key: "detail" }, state.smokeTest.detail)
      : null,
    state.smokeTest.helpText
      ? h("div", { className: "smoke-status-meta smoke-status-help", key: "help" }, `Next: ${state.smokeTest.helpText}`)
      : null,
    state.smokeTest.reportText
      ? h("div", { className: "smoke-status-actions", key: "actions" }, [
          h(
            "button",
            {
              className: "inline-mini-button smoke-status-copy",
              type: "button",
              "data-action": "copy-test-error-report",
              key: "copy",
            },
            state.smokeTest.status === "success" ? "Copy report" : "Copy error report",
          ),
          state.smokeTest.copyStatus === "copied"
            ? h("span", { className: "smoke-status-copy-note", key: "note" }, "Copied.")
            : null,
        ])
      : null,
  ].filter(Boolean));
}

function SyncStatusView() {
  const snapshot = state.syncSnapshot?.sync || null;
  if (!snapshot) return null;

  const tone =
    snapshot.lastError
      ? "error"
      : snapshot.connection === "connected"
      ? "success"
      : snapshot.running
        ? "running"
        : "idle";

  const modeLabel =
    snapshot.requestedMode === "continuous"
      ? "Live sync"
      : snapshot.requestedMode === "manual"
        ? "One-shot sync"
        : "Sync off";

  const isQuietLocalState =
    snapshot.requestedMode !== "continuous" &&
    !snapshot.remoteCraftCount &&
    !snapshot.transportPeerCount &&
    !snapshot.remotePeerCount &&
    !snapshot.lastError;
  if (isQuietLocalState) return null;
  const statusLabel = isQuietLocalState ? "Local only" : modeLabel;
  const meta = [];

  if (!isQuietLocalState) {
    meta.push(
      `${formatCount(snapshot.remoteCraftCount || 0)} shared`,
      `${formatCount(snapshot.transportPeerCount || 0)} live links`,
      `${formatCount(snapshot.remotePeerCount || 0)} visible peers`,
    );
  }

  if (snapshot.lastError) {
    meta.push(trimText(snapshot.lastError, 64));
  } else if (snapshot.lastSyncedAt) {
    meta.push(`updated ${formatTimeAgo(snapshot.lastSyncedAt)}`);
  }

  return h("section", { className: `sync-rail sync-rail-${tone}` }, [
    h("div", { className: "sync-rail-main", key: "main" }, [
      h("span", { className: "sync-rail-status", key: "status" }, statusLabel),
      meta.length
        ? h("span", { className: "sync-rail-meta", key: "meta" }, meta.join(" · "))
        : null,
    ]),
    h("div", { className: "sync-rail-actions", key: "actions" }, [
      snapshot.requestedMode !== "continuous"
        ? h(
            "button",
            {
              className: "inline-mini-button",
              type: "button",
              "data-action": "sync-now",
              key: "sync",
            },
            "Sync now",
          )
        : null,
      h(
        "button",
        {
          className: "inline-mini-button",
          type: "button",
          "data-action": "open-share-settings",
          key: "share",
        },
        "Share",
      ),
    ].filter(Boolean)),
  ]);
}

function AgentStepListView({ entries = [], emptyText = "" }) {
  const normalizedEntries = (Array.isArray(entries) ? entries : []).filter((entry) =>
    String(entry?.message || "").trim(),
  );
  if (!normalizedEntries.length) {
    return emptyText ? h("div", { className: "agent-log-empty" }, emptyText) : null;
  }

  return h(
    "div",
    { className: "agent-log-list agent-progress-step-list" },
    normalizedEntries.map((entry, index) =>
      h("div", {
        className: joinClassNames(
          "agent-log-entry",
          `agent-log-entry-${entry.level || "info"}`,
          entry.status ? `agent-log-entry-${entry.status}` : "",
          entry.kind ? `agent-log-entry-${entry.kind}` : "",
        ),
        key: `${entry.time || "step"}-${index}`,
      }, [
        h("span", { className: "agent-log-time", key: "time" }, entry.time || "Step"),
        h("span", { className: "agent-log-content", key: "content" }, [
          h("span", { className: "agent-log-title", key: "title" }, entry.title || entry.message),
          entry.detail && entry.detail !== entry.title
            ? h("span", { className: "agent-log-message agent-log-detail", key: "detail" }, entry.detail)
            : null,
        ]),
      ]),
    ),
  );
}

const AGENT_PROGRESS_STAGE_DEFINITIONS = Object.freeze([
  Object.freeze({ id: "inspect", label: "Inspect" }),
  Object.freeze({ id: "research", label: "Research" }),
  Object.freeze({ id: "tools", label: "Tools" }),
  Object.freeze({ id: "validate", label: "Validate" }),
  Object.freeze({ id: "diagnose", label: "Diagnose" }),
  Object.freeze({ id: "dataset", label: "Dataset" }),
  Object.freeze({ id: "train", label: "Train" }),
]);

const AGENT_PROGRESS_TOOL_METADATA = Object.freeze({
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

function humanizeAgentToolName(toolName = "") {
  const normalized = String(toolName || "")
    .trim()
    .replace(/[_-]+/g, " ");
  if (!normalized) return "Agent step";
  return normalized.replace(/\b[a-z]/g, (match) => match.toUpperCase());
}

function getAgentProgressToolMetadata(toolName = "") {
  const key = String(toolName || "").trim();
  const meta = AGENT_PROGRESS_TOOL_METADATA[key] || null;
  return {
    toolName: key,
    title: meta?.title || humanizeAgentToolName(key),
    startDetail: meta?.startDetail || "Working on this step.",
    stageId: meta?.stageId || "",
  };
}

function parseAgentToolStartMessage(message = "") {
  const match = /^Tool starts: ([^.]+)\.$/.exec(String(message || "").trim());
  if (!match) return null;
  return {
    toolName: String(match[1] || "").trim(),
  };
}

function parseAgentToolResultMessage(message = "") {
  const text = String(message || "").trim();
  const summaryMatch = /^([a-z0-9_]+): (.+)$/i.exec(text);
  if (summaryMatch) {
    return {
      toolName: String(summaryMatch[1] || "").trim(),
      summary: String(summaryMatch[2] || "").trim(),
    };
  }
  const completedMatch = /^Tool completed: ([^.]+)\.$/.exec(text);
  if (completedMatch) {
    return {
      toolName: String(completedMatch[1] || "").trim(),
      summary: "",
    };
  }
  return null;
}

function shouldHideAgentProgressLogMessage(message = "") {
  const text = String(message || "").trim().toLowerCase();
  if (!text) return true;
  return [
    "resolving the configured model and provider.",
    "local training data is loaded for the agent run.",
  ].includes(text) || text.startsWith("supervisor tools are fixed:");
}

function inferAgentStageIdFromText(text = "") {
  const haystack = String(text || "").trim().toLowerCase();
  if (!haystack) return "";
  if (/workspace code|quellcode-diagnose|patch target|coding handoff|workspace_code_dive/.test(haystack)) {
    return "diagnose";
  }
  if (/training run|training progress|eta|validation acc|adapter|epoch|get_training_run_status|start_training_run/.test(haystack)) {
    return "train";
  }
  if (/trainingsbeispiele|training data|draft_training_changes|seed rows|dataset|samples/.test(haystack)) {
    return "dataset";
  }
  if (/smoke|eval|capability/.test(haystack)) {
    return "validate";
  }
  if (/tool-skripte|tool scripts|browser-capabilities|browser-funktionen|browser functions|reviewed capabilities|bundle/.test(haystack)) {
    return "tools";
  }
  if (/browser|tab|seite|playwright|web search|websuche/.test(haystack)) {
    return "research";
  }
  if (/inspect|snapshot|initial|empty dataset|leerer ausgangsdatensatz/.test(haystack)) {
    return "inspect";
  }
  return "";
}

function buildAgentProgressMessage(title = "", detail = "") {
  const safeTitle = String(title || "").trim();
  const safeDetail = String(detail || "").trim();
  if (safeTitle && safeDetail && safeTitle !== safeDetail) {
    return `${safeTitle} · ${safeDetail}`;
  }
  return safeTitle || safeDetail;
}

function buildAgentProgressNoteTitle(message = "") {
  const text = String(message || "").trim();
  const haystack = text.toLowerCase();
  if (/sdk segment \d+ starts/.test(haystack)) return "Continue agent segment";
  if (/no seed rows exist yet/.test(haystack)) return "Start from an empty dataset";
  if (/workspace code|quellcode-diagnose|coding handoff|patch target/.test(haystack)) return "Workspace diagnosis";
  if (/invalid structured output/.test(haystack) && /falling back/.test(haystack)) return "Fallback planner used";
  if (/trainingsbeispiele hinzugefuegt|training data operations|training rows/.test(haystack)) return "Training data updated";
  if (/agent segment completed/.test(haystack)) return "Segment completed";
  if (/copied to clipboard/.test(haystack)) return "Copied report";
  return "Progress update";
}

function buildAgentProgressEntriesFromLogs(logs = [], agentRun = null) {
  const entries = [];
  const openByToolName = new Map();
  const lastToolFailure = buildAgentLastToolFailure(agentRun);

  for (const [index, rawEntry] of (Array.isArray(logs) ? logs : []).entries()) {
    const entry = normalizeAgentLogEntry(rawEntry, agentRun, lastToolFailure);
    const level = String(entry.level || "info");
    const time = String(entry.time || "");
    const message = String(entry.message || "").trim();
    if (!message) continue;
    if (!entry.kind && !entry.title && shouldHideAgentProgressLogMessage(message)) continue;

    if (entry.kind === "tool" && entry.toolName) {
      const meta = getAgentProgressToolMetadata(entry.toolName);
      const normalizedStatus = normalizeAgentLogStatus(entry.status || "done") || "done";
      const normalizedLevel =
        normalizedStatus === "error"
          ? "error"
          : normalizedStatus === "warn"
            ? "warn"
            : level === "success"
              ? "success"
              : level;
      const structuredEntry = {
        id: `tool-structured-${meta.toolName}-${index}`,
        kind: "tool",
        toolName: meta.toolName,
        stageId: entry.stageId || meta.stageId,
        level: normalizedLevel,
        status: normalizedStatus,
        time,
        title: entry.title || meta.title,
        detail: entry.detail || meta.startDetail,
        message: buildAgentProgressMessage(entry.title || meta.title, entry.detail || meta.startDetail),
      };
      entries.push(structuredEntry);
      if (normalizedStatus === "running") {
        openByToolName.set(meta.toolName, entries.length - 1);
      } else {
        openByToolName.delete(meta.toolName);
      }
      continue;
    }

    if (entry.kind && (entry.title || entry.detail)) {
      const title = entry.title || buildAgentProgressNoteTitle(message);
      const detail = entry.detail || message;
      entries.push({
        id: `note-structured-${index}`,
        kind: entry.kind || "note",
        toolName: entry.toolName || "",
        stageId: entry.stageId || inferAgentStageIdFromText(`${title} ${detail}`),
        level: level === "success" ? "success" : level,
        status: normalizeAgentLogStatus(entry.status) || (level === "error" ? "error" : level === "warn" ? "warn" : "done"),
        time,
        title,
        detail,
        message: buildAgentProgressMessage(title, detail),
      });
      continue;
    }

    const start = parseAgentToolStartMessage(message);
    if (start?.toolName) {
      const meta = getAgentProgressToolMetadata(start.toolName);
      entries.push({
        id: `tool-start-${meta.toolName}-${index}`,
        kind: "tool",
        toolName: meta.toolName,
        stageId: meta.stageId,
        level,
        status: "running",
        time,
        title: meta.title,
        detail: meta.startDetail,
        message: buildAgentProgressMessage(meta.title, meta.startDetail),
      });
      openByToolName.set(meta.toolName, entries.length - 1);
      continue;
    }

    const result = parseAgentToolResultMessage(message);
    if (result?.toolName) {
      const meta = getAgentProgressToolMetadata(result.toolName);
      const summary = String(result.summary || "").trim();
      const normalizedStatus =
        level === "error" || /failed|invalid json input|error occurred/i.test(summary)
          ? "error"
          : level === "warn"
            ? "warn"
            : "done";
      const normalizedLevel =
        normalizedStatus === "done"
          ? "success"
          : normalizedStatus === "error"
            ? "error"
            : "warn";
      const openIndex = openByToolName.get(meta.toolName);
      if (Number.isInteger(openIndex) && openIndex >= 0 && openIndex < entries.length) {
        const previousEntry = entries[openIndex];
        entries[openIndex] = {
          ...previousEntry,
          level: normalizedLevel,
          status: normalizedStatus,
          time: time || previousEntry.time,
          detail: summary || previousEntry.detail,
          message: buildAgentProgressMessage(previousEntry.title, summary || previousEntry.detail),
        };
        openByToolName.delete(meta.toolName);
      } else {
        entries.push({
          id: `tool-end-${meta.toolName}-${index}`,
          kind: "tool",
          toolName: meta.toolName,
          stageId: meta.stageId,
          level: normalizedLevel,
          status: normalizedStatus,
          time,
          title: meta.title,
          detail: summary || meta.startDetail,
          message: buildAgentProgressMessage(meta.title, summary || meta.startDetail),
        });
      }
      continue;
    }

    const title = buildAgentProgressNoteTitle(message);
    const detail = message;
    const stageId = inferAgentStageIdFromText(message);
    entries.push({
      id: `note-${index}`,
      kind: "note",
      toolName: "",
      stageId,
      level: level === "success" ? "success" : level,
      status: level === "error" ? "error" : level === "warn" ? "warn" : "done",
      time,
      title,
      detail,
      message: buildAgentProgressMessage(title, detail),
    });
  }

  const runActive = ["starting", "running"].includes(getAgentUiState(agentRun));
  if (!runActive) {
    for (const entry of entries) {
      if (entry.status === "running") {
        entry.status = "done";
        entry.level = entry.level === "error" ? "error" : "success";
        entry.message = buildAgentProgressMessage(entry.title, entry.detail);
      }
    }
  }

  return entries;
}

function buildAgentProgressStageItems(agentRun, progressEntries = []) {
  const entries = Array.isArray(progressEntries) ? progressEntries : [];
  const runActive = ["starting", "running"].includes(getAgentUiState(agentRun));
  const runningEntry = runActive
    ? [...entries].reverse().find((entry) => entry?.stageId && entry?.status === "running")
    : null;
  const activeStageId = runningEntry?.stageId || "";

  return AGENT_PROGRESS_STAGE_DEFINITIONS.map((stage) => {
    const hits = entries.filter((entry) => entry?.stageId === stage.id);
    const hasError = hits.some((entry) => entry?.status === "error");
    const hasWarn = hits.some((entry) => entry?.status === "warn");
    const isCurrent = Boolean(activeStageId) && activeStageId === stage.id;
    let status = "upcoming";
    if (isCurrent) {
      status = "current";
    } else if (hasError) {
      status = "error";
    } else if (hits.length) {
      status = "done";
    } else if (hasWarn) {
      status = "warn";
    }
    return {
      ...stage,
      status,
      count: hits.length,
      current: isCurrent,
    };
  });
}

function getAgentStageProgressValue(stageItems = []) {
  const items = Array.isArray(stageItems) ? stageItems : [];
  if (!items.length) return 0;
  const units = items.reduce((total, item) => {
    if (item.status === "done" || item.status === "error") return total + 1;
    if (item.status === "current") return total + 0.6;
    if (item.status === "warn") return total + 0.4;
    return total;
  }, 0);
  return Math.max(0, Math.min(1, units / items.length));
}

function getAgentTurnProgressValue(agentRun) {
  const turnsUsed = Math.max(0, Number(agentRun?.turnsUsed || 0));
  const maxTurns = Math.max(0, Number(agentRun?.maxTurns || 0));
  if (!maxTurns) return 0;
  return Math.max(0, Math.min(1, turnsUsed / maxTurns));
}

function formatAgentDurationCompact(ms = 0) {
  const totalMs = Math.max(0, Number(ms || 0));
  if (!totalMs) return "";
  const totalMinutes = Math.round(totalMs / 60000);
  if (totalMinutes < 1) return "<1m";
  if (totalMinutes < 60) return `${totalMinutes}m`;
  const hours = Math.floor(totalMinutes / 60);
  const minutes = totalMinutes % 60;
  return minutes ? `${hours}h ${minutes}m` : `${hours}h`;
}

function getAgentProgressOverview(agentRun, progressEntries = []) {
  const stageItems = buildAgentProgressStageItems(agentRun, progressEntries);
  const stageProgress = getAgentStageProgressValue(stageItems);
  const turnProgress = getAgentTurnProgressValue(agentRun);
  const turnsUsed = Math.max(0, Number(agentRun?.turnsUsed || 0));
  const maxTurns = Math.max(0, Number(agentRun?.maxTurns || 0));
  const estimatedRemainingMs = Math.max(0, Number(agentRun?.estimatedRemainingMs || 0));
  return {
    stageItems,
    stagePercent: Math.round(stageProgress * 100),
    turnPercent: Math.round(turnProgress * 100),
    flowLabel: stageItems.some((item) => item.status !== "upcoming")
      ? `${Math.round(stageProgress * 100)}% of the run flow covered`
      : "Waiting for the first concrete step",
    turnLabel: maxTurns > 0 ? `${formatCount(turnsUsed)} / ${formatCount(maxTurns)} turns` : "Turn budget not set",
    etaLabel: estimatedRemainingMs > 0 ? `ETA ${formatAgentDurationCompact(estimatedRemainingMs)}` : "",
  };
}

function AgentProgressOverviewView({ agentRun, progressEntries = [] }) {
  const overview = getAgentProgressOverview(agentRun, progressEntries);
  if (!overview.stageItems.length) return null;

  return h("div", { className: "agent-progress-overview" }, [
    h("div", { className: "agent-progress-meter-stack", key: "meters" }, [
      h("div", { className: "agent-progress-meter-row", key: "flow" }, [
        h("span", { className: "agent-progress-meter-label", key: "label" }, "Run flow"),
        h("span", { className: "agent-progress-meter-track", key: "track" }, [
          h("span", {
            className: joinClassNames(
              "agent-progress-meter-fill",
              overview.stagePercent > 0 && ["starting", "running"].includes(getAgentUiState(agentRun))
                ? "agent-progress-meter-fill-running"
                : "",
            ),
            style: { width: `${overview.stagePercent}%` },
            key: "fill",
          }),
        ]),
        h("span", { className: "agent-progress-meter-value", key: "value" }, overview.flowLabel),
      ]),
      h("div", { className: "agent-progress-meter-row", key: "turns" }, [
        h("span", { className: "agent-progress-meter-label", key: "label" }, "Turn budget"),
        h("span", { className: "agent-progress-meter-track", key: "track" }, [
          h("span", {
            className: "agent-progress-meter-fill agent-progress-meter-fill-subtle",
            style: { width: `${overview.turnPercent}%` },
            key: "fill",
          }),
        ]),
        h("span", { className: "agent-progress-meter-value", key: "value" }, overview.etaLabel || overview.turnLabel),
      ]),
    ]),
  ]);
}

function AgentQuestionsView({ craftId, agentRun }) {
  const questions = normalizeAgentQuestions(agentRun?.questions);
  const craft = findCraft(craftId);
  const currentPrompt = getCurrentAgentPromptValue(craftId, craft);
  const continueLabel = getCraftPrimaryActionLabel(craft, currentPrompt, agentRun, true);
  if (!questions.length) {
    return h("div", { className: "agent-log-empty" }, "No open questions for this craft.");
  }

  return h("div", { className: "agent-question-list" }, [
    ...questions.map((entry, index) =>
      h("label", { className: "agent-question-card", key: entry.id || `question-${index}` }, [
        h("div", { className: "agent-question-head", key: "head" }, [
          h("span", { className: "detail-eyebrow", key: "eyebrow" }, `Question ${index + 1}`),
          entry.reason ? h("span", { className: "drawer-log-status", key: "reason" }, entry.reason) : null,
        ].filter(Boolean)),
        h("div", { className: "agent-question-text", key: "text" }, entry.question),
        h("textarea", {
          className: "agent-question-input",
          "data-agent-question-answer": "true",
          "data-craft-id": craftId,
          "data-question-id": entry.id,
          rows: 3,
          placeholder: "Add the answer for the next agent run...",
          defaultValue: entry.answer || "",
          disabled: isAgentRunActive(agentRun),
          key: "input",
        }),
      ]),
    ),
    agentRun?.status === "needs_input"
      ? h("div", { className: "agent-panel-actions", key: "actions" }, [
          h(
            "button",
            {
                className: "detail-action",
                type: "button",
              "data-action": "start-agent-run",
              "data-craft-id": craftId,
              key: "continue",
            },
              continueLabel,
            ),
        ])
      : null,
  ]);
}

function AgentProvenanceView({ agentRun }) {
  const provenance = buildVisibleAgentProvenance(agentRun);
  if (!provenance.length) {
    return h("div", { className: "agent-log-empty" }, "No milestone highlights have been recorded for this run yet.");
  }

  return h(
    "div",
    { className: "agent-provenance-list" },
    provenance.map((entry, index) =>
      h("article", { className: "agent-provenance-card", key: entry.id || `${entry.title}-${index}` }, [
        h("div", { className: "agent-provenance-head", key: "head" }, [
          h("span", { className: "detail-eyebrow", key: "kind" }, entry.kind || "match"),
          entry.operationType
            ? h("span", { className: "drawer-log-status", key: "type" }, entry.operationType)
            : null,
        ].filter(Boolean)),
        h("div", { className: "agent-provenance-title", key: "title" }, entry.title),
        h("div", { className: "agent-provenance-detail", key: "detail" }, entry.detail || "No detail recorded."),
        entry.sampleId
          ? h("div", { className: "agent-provenance-meta", key: "sample" }, `Sample: ${entry.sampleId}`)
          : null,
      ]),
    ),
  );
}

function AgentProgressView({ craftId, agentRun, isEditable = true }) {
  const craft = findCraft(craftId);
  const report = normalizeAgentReport(agentRun?.report, {
    currentState: agentRun?.responseText || "No progress report yet.",
  });
  const headline = getAgentProgressHeadline(agentRun);
  const primaryText = getAgentProgressPrimaryText(agentRun, report);
  const secondaryText = getAgentProgressSecondaryText(agentRun, report, primaryText);
  const progressEntries = buildAgentProgressEntries(agentRun, report);
  const metaText = getAgentProgressMetaText(agentRun, progressEntries, craft);
  const clipboardStatus = String(agentRun?.clipboardStatus || "idle");
  const clipboardError = String(agentRun?.clipboardError || "");
  const clipboardMessage =
    clipboardStatus === "copied"
      ? "Debug JSON copied."
      : clipboardStatus === "copy_failed"
        ? (clipboardError || "Clipboard write failed.")
        : "";

  return h("div", { className: "agent-progress-view" }, [
    h("div", { className: "agent-progress-status", key: "headline" }, headline),
    metaText ? h("div", { className: "agent-progress-meta", key: "meta" }, metaText) : null,
    h(AgentProgressOverviewView, { agentRun, progressEntries, key: "overview" }),
    h("div", { className: "agent-progress-copy", key: "primary" }, primaryText),
    secondaryText ? h("div", { className: "agent-progress-next", key: "secondary" }, secondaryText) : null,
    h(AgentStepListView, { entries: progressEntries, key: "steps" }),
    clipboardMessage
      ? h(
          "div",
          {
            className: joinClassNames(
              "agent-progress-next",
              clipboardStatus === "copy_failed" ? "agent-progress-next-error" : "",
            ),
            key: "clipboard-status",
          },
          clipboardMessage,
        )
      : null,
  ].filter(Boolean));
}

function AgentInspectorView({ craftId, agentRun, isEditable = true }) {
  const activeTab = ["progress", "questions", "provenance"].includes(String(agentRun?.activeTab || ""))
    ? String(agentRun.activeTab)
    : "progress";
  const progressCount = countAgentProgressEntries(agentRun);
  const questionCount = countOpenAgentQuestions(agentRun);
  const provenanceCount = buildVisibleAgentProvenance(agentRun).length;
  const scrollKey = `agent-inspector:${craftId}:${activeTab}`;

  return h("section", { className: "drawer-log-panel agent-inspector-panel" }, [
    h("div", { className: "agent-tab-row", role: "tablist", "aria-label": "Agent views", key: "tabs" }, [
      h(
        "button",
        {
          className: joinClassNames("agent-tab-button", activeTab === "progress" ? "agent-tab-button-active" : ""),
          type: "button",
          "data-action": "set-agent-tab",
          "data-craft-id": craftId,
          "data-agent-tab": "progress",
          key: "progress",
        },
        ["Progress", h("span", { className: "agent-tab-count", key: "count" }, `(${progressCount})`)],
      ),
      h(
        "button",
        {
          className: joinClassNames("agent-tab-button", activeTab === "questions" ? "agent-tab-button-active" : ""),
          type: "button",
          "data-action": "set-agent-tab",
          "data-craft-id": craftId,
          "data-agent-tab": "questions",
          key: "questions",
        },
        ["Questions", h("span", { className: "agent-tab-count", key: "count" }, `(${questionCount})`)],
      ),
      h(
        "button",
        {
          className: joinClassNames("agent-tab-button", activeTab === "provenance" ? "agent-tab-button-active" : ""),
          type: "button",
          "data-action": "set-agent-tab",
          "data-craft-id": craftId,
          "data-agent-tab": "provenance",
          key: "provenance",
        },
        ["Provenance", h("span", { className: "agent-tab-count", key: "count" }, `(${provenanceCount})`)],
      ),
    ]),
    h(
      "div",
      {
        className: "drawer-log-scroll",
        "data-preserve-scroll-key": scrollKey,
        "data-scroll-anchor": activeTab === "progress" ? "bottom" : undefined,
        "data-scroll-start": activeTab === "progress" ? "bottom" : undefined,
        key: "body",
      },
      [
        activeTab === "questions"
          ? h(AgentQuestionsView, { craftId, agentRun, key: "questions" })
          : activeTab === "provenance"
            ? h(AgentProvenanceView, { agentRun, key: "provenance" })
            : h(AgentProgressView, { craftId, agentRun, isEditable, key: "progress" }),
      ],
    ),
  ]);
}

const CRAFT_WORKSPACE_BUTTON_LABEL = "Open craft workspace";

function DrawerWorkspaceButtonView({ craftId }) {
  return h(
    "button",
    {
      className: "drawer-icon-button",
      type: "button",
      title: CRAFT_WORKSPACE_BUTTON_LABEL,
      "aria-label": CRAFT_WORKSPACE_BUTTON_LABEL,
      "data-action": "open-bundle",
      "data-craft-id": craftId,
    },
    h("span", { className: "drawer-icon-glyph", "aria-hidden": "true" }, String.fromCodePoint(0x2699)),
  );
}

function CraftOfficialDescriptionView({ craft, statusLabel = "" }) {
  const description = buildCraftOfficialDescriptionPreview(craft, 220);
  return h("section", { className: "drawer-log-panel", key: "official-description" }, [
    h("div", { className: "drawer-log-head", key: "head" }, [
      h("span", { className: "detail-eyebrow", key: "eyebrow" }, "Official description"),
      statusLabel ? h("span", { className: "drawer-log-status", key: "status" }, statusLabel) : null,
    ].filter(Boolean)),
    description.full
      ? description.truncated
        ? h("details", { className: "shared-craft-summary-details", key: "details" }, [
            h("summary", { className: "shared-craft-summary-preview", key: "summary" }, [
              h("span", { className: "shared-craft-summary", key: "text" }, description.preview),
              h("span", { className: "shared-craft-summary-toggle", key: "toggle" }, "More"),
            ]),
            h("div", { className: "shared-craft-summary shared-craft-summary-full", key: "full" }, description.full),
          ])
        : h("div", { className: "shared-craft-summary", key: "text" }, description.full)
      : h(
          "div",
          { className: "shared-craft-note", key: "empty" },
          "No verified description yet. Use Refine to let the agent define this capability.",
        ),
  ]);
}

function CraftPromptEditorView({ craft, agentRun = null, currentPrompt = "" }) {
  const promptChanged = didCraftPromptChange(craft, currentPrompt, {
    countEmptyAsChange: hasExplicitAgentPromptDraft(craft?.id),
  });
  const promptNote = getCraftPromptEditorNote(agentRun, promptChanged);
  const promptNoteIsPendingChange = isCraftPromptEditorNotePendingChange(agentRun, promptChanged);
  return h("label", { className: "crafting-request crafting-definition", "aria-label": "Refine command" }, [
    h("div", { className: "detail-eyebrow", key: "eyebrow" }, "Refine command"),
    h("textarea", {
      "data-agent-prompt-field": "true",
      "data-craft-id": craft.id,
      rows: 3,
      placeholder: "Describe the correction, missing behavior, or the error the agent should address next.",
      defaultValue: currentPrompt,
      key: "textarea",
    }),
    promptNote
      ? h(
          "div",
          {
            className: joinClassNames(
              "crafting-request-note",
              promptNoteIsPendingChange ? "crafting-request-note-pending" : "",
            ),
            key: "note",
          },
          promptNote,
        )
      : null,
  ].filter(Boolean));
}

function CraftDrawerToolbarView({ craft, agentRun = null, isEditable = true, currentPrompt = "" }) {
  const trainingRun = state.trainingRuns[craft.id] || null;
  const refineDisabled =
    !isEditable ||
    !String(currentPrompt || "").trim() ||
    isAgentRunActive(agentRun) ||
    agentRun?.stopRequested === true;
  const showCopyDebugAction = shouldShowAgentRunDebugButton(agentRun);
  const showCopyTrainingReportAction = Boolean(
    trainingRun &&
      (
        (trainingRun.lastReport && typeof trainingRun.lastReport === "object") ||
        ["completed", "failed"].includes(String(trainingRun.status || ""))
      ),
  );
  const showStopAction = isEditable && isAgentRunActive(agentRun) && Boolean(String(agentRun?.runId || "").trim());
  return h("div", { className: "crafting-toolbar" }, [
    h("div", { className: "crafting-actions", key: "primary" }, [
      h(
        "button",
        {
          className: "detail-action",
          type: "button",
          disabled: refineDisabled,
          "data-action": "start-agent-run",
          "data-craft-id": craft.id,
          key: "refine",
        },
        getCraftPrimaryActionLabel(craft, currentPrompt, agentRun, isEditable),
      ),
      showCopyDebugAction
        ? h(
            "button",
            {
              className: "detail-secondary",
              type: "button",
              "data-action": "copy-agent-run-debug",
              "data-craft-id": craft.id,
              key: "copy-debug",
            },
            "Copy debug JSON",
          )
        : null,
      showCopyTrainingReportAction
        ? h(
            "button",
            {
              className: "detail-secondary",
              type: "button",
              "data-action": "copy-training-run-report",
              "data-craft-id": craft.id,
              key: "copy-training-report",
            },
            "Copy training JSON",
          )
        : null,
      showStopAction
        ? h(
            "button",
            {
              className: "detail-secondary",
              type: "button",
              disabled: agentRun?.stopRequested === true,
              "data-action": "stop-agent-run",
              "data-craft-id": craft.id,
              key: "stop",
            },
            agentRun?.stopRequested === true ? "Stopping..." : "Stop",
          )
        : null,
    ].filter(Boolean)),
    h("div", { className: "crafting-links crafting-links-utility", key: "utility" }, [
      h(DrawerWorkspaceButtonView, { craftId: craft.id, key: "workspace" }),
      isEditable
        ? h(
            "button",
            {
              className: "drawer-link drawer-link-danger",
              type: "button",
              "data-action": "delete-craft",
              "data-craft-id": craft.id,
              key: "delete",
            },
            "Delete",
          )
        : null,
    ].filter(Boolean)),
  ]);
}

function RemoteCraftDrawerView({ craft }) {
  const owner = getSharedOwnerName(craft);
  const syncMeta = craft?.sync || {};
  const metaItems = [owner, syncMeta.syncedAt ? `synced ${formatTimeAgo(syncMeta.syncedAt)}` : ""].filter(Boolean);

  return h("section", { className: "craft-drawer" }, [
    h("div", { className: "crafting-status-line", key: "status" }, buildMetaTokenNodes(metaItems)),
    h("div", { className: "shared-craft-note", key: "note" }, "Read only until you fork."),
    h("div", { className: "crafting-toolbar", key: "toolbar" }, [
      h("div", { className: "crafting-actions", key: "primary" }, [
        h(
          "button",
          {
            className: "detail-action",
            type: "button",
            "data-action": "fork-craft",
            "data-craft-id": craft.id,
            key: "fork",
          },
          "Fork",
        ),
      ]),
      h("div", { className: "crafting-links crafting-links-utility", key: "utility" }, [
        h(DrawerWorkspaceButtonView, { craftId: craft.id, key: "workspace" }),
      ]),
    ]),
    h(CraftOfficialDescriptionView, { craft, statusLabel: owner, key: "summary" }),
  ].filter(Boolean));
}

function CraftingDrawerView({ craft }) {
  const agentRun = state.agentRuns[craft.id] || null;
  const isEditable = craftStore?.isEditableCraft?.(craft) !== false;
  const currentPrompt = getCurrentAgentPromptValue(craft.id, craft);
  const showAgentSurface = Boolean(agentRun);
  return h("section", { className: "craft-drawer" }, [
    h(CraftOfficialDescriptionView, { craft, key: "official-description" }),
    h(CraftPromptEditorView, {
      craft,
      agentRun,
      currentPrompt,
      key: "definition",
    }),
    h(CraftDrawerToolbarView, {
      craft,
      agentRun,
      isEditable,
      currentPrompt,
      key: "toolbar",
    }),
    showAgentSurface
      ? h(AgentInspectorView, { craftId: craft.id, agentRun, isEditable, key: "inspector" })
      : null,
  ]);
}

function CraftLiveStripView({ signal }) {
  return h("div", { className: "craft-live-strip" }, [
    h("span", { className: "craft-live-dot", "aria-hidden": "true", key: "dot" }),
    h("span", { className: "craft-live-text", key: "message" }, signal.message),
    signal.meta ? h("span", { className: "craft-live-meta", key: "meta" }, signal.meta) : null,
  ]);
}

function WorkspaceView({ craft }) {
  return h("section", { className: "craft-workspace" }, [
    isRemoteSharedCraft(craft)
      ? h(RemoteCraftDrawerView, { craft, key: "remote" })
      : h(CraftingDrawerView, { craft, key: "local" }),
  ]);
}

function CraftCardView({ craft }) {
  const isActive = craft.id === state.activeCraftId;
  const agentRun = state.agentRuns[craft.id] || null;
  const liveSignal = isRemoteSharedCraft(craft) ? getCraftLiveSignal(craft) : null;
  const cardNote = getCraftCardSecondaryText(craft);
  const maturityText = getCraftMaturityText(craft, agentRun);

  return h("article", {
    className: joinClassNames("craft-card", isActive ? "craft-card-active" : ""),
    "data-craft-card-id": craft.id,
  }, [
    h(
      "button",
      {
        className: "craft-row",
        type: "button",
        "aria-expanded": isActive ? "true" : "false",
        "data-action": "toggle-craft",
        "data-craft-id": craft.id,
        key: "row",
      },
      [
        h("span", { className: "craft-main", key: "main" }, [
          h("span", { className: "craft-name-row", key: "name-row" }, [
            h("span", { className: "craft-name", key: "name" }, craft.name),
            h(
              "span",
              {
                className: getCraftMaturityBadgeClassName(craft, agentRun),
                title: getCraftMaturityTitle(craft, agentRun),
                style: getCraftMaturityBadgeStyle(craft, agentRun),
                key: "maturity",
              },
              h("span", { className: "craft-maturity-badge-label", key: "label" }, maturityText),
            ),
          ]),
          cardNote ? h("span", { className: "craft-note", key: "note" }, cardNote) : null,
        ]),
      ],
    ),
    liveSignal ? h(CraftLiveStripView, { signal: liveSignal, key: "signal" }) : null,
    isActive ? h(WorkspaceView, { craft, key: "workspace" }) : null,
  ].filter(Boolean));
}

function CreateRowView() {
  const description = String(state.draftGoal || "");
  const handleGoalChange = (event) => {
    state.draftGoal = String(event?.target?.value || "");
    render();
  };
  if (state.createOpen) {
    return h("section", { className: "composer-row", "aria-label": "New capability" }, [
      h("label", { className: "composer-definition", key: "definition" }, [
        h("span", { className: "detail-eyebrow", key: "eyebrow" }, "Describe the capability"),
        h("textarea", {
          className: "composer-definition-input",
          "data-draft-field": "goal",
          rows: 3,
          placeholder: "Describe what the capability should do in the browser.",
          value: description,
          onChange: handleGoalChange,
          key: "textarea",
        }),
      ]),
      h("div", { className: "composer-actions", key: "actions" }, [
        h(
          "button",
          {
            className: "composer-action",
            type: "button",
            disabled: !description.trim(),
            "data-action": "create-craft",
            key: "create",
          },
          "Create",
        ),
        h(
          "button",
          {
            className: "composer-cancel",
            type: "button",
            "aria-label": "Close",
            "data-action": "cancel-create",
            key: "cancel",
          },
          "×",
        ),
      ]),
    ]);
  }

  return h(
    "button",
    {
      className: "add-row",
      type: "button",
      "aria-label": "Create a new capability",
      title: "Create a new capability",
      "data-hover-label": "Create a new capability",
      "data-action": "toggle-create",
      key: "button",
    },
    [h("span", { className: "add-plus", key: "plus" }, "+")],
  );
}

function CraftSectionView({ title = "", crafts = [], tone = "", footer = null }) {
  return h(
    "section",
    { className: `craft-section ${tone ? `craft-section-${tone}` : ""}`.trim() },
    [
      title
        ? h("div", { className: "craft-section-head", key: "head" }, [
            h("span", { className: "craft-section-title", key: "title" }, title),
          ])
        : null,
      h(
        "div",
        { className: "craft-section-body", key: "body" },
        [
          ...crafts.map((craft) => h(CraftCardView, { craft, key: craft.id })),
          footer,
        ].filter(Boolean),
      ),
    ].filter(Boolean),
  );
}

function CraftListView({ installedCrafts = [], availableCrafts = [] }) {
  const hasAnyCrafts = installedCrafts.length > 0 || availableCrafts.length > 0;
  const createRow = h(CreateRowView, { key: "create-row" });
  if (!hasAnyCrafts) {
    return h(
      "section",
      { className: "craft-list craft-list-empty", "aria-label": "Crafts" },
      [createRow],
    );
  }
  return h("section", { className: "craft-list", "aria-label": "Crafts" }, [
    h(CraftSectionView, {
      crafts: installedCrafts,
      footer: createRow,
      key: "installed",
    }),
    availableCrafts.length
      ? h(CraftSectionView, {
          title: "Available",
          crafts: availableCrafts,
          tone: "available",
          key: "available",
        })
      : null,
  ].filter(Boolean));
}

function SidepanelAppView({
  setupBlocked = false,
  headerTrainingRun = null,
  installedCrafts = [],
  availableCrafts = [],
}) {
  const setupContent = state.startupError
    ? renderStartupError()
    : !state.setupLoaded
      ? renderSetupLoading()
      : setupBlocked
        ? renderSetupGate()
        : null;

  const mainContent = setupContent
    ? h(ReactFragment, null, [
        setupContent,
        h(SmokeStatusView, { key: "smoke-status" }),
      ])
    : h(ReactFragment, null, [
        h(SmokeStatusView, { key: "smoke-status" }),
        h(SyncStatusView, { key: "sync-status" }),
        h(CraftListView, {
          installedCrafts,
          availableCrafts,
          key: "craft-list",
        }),
      ]);

  return h(ReactFragment, null, [
    h("div", { className: "panel-main-stack", key: "stack" }, [
      h(PanelHeaderView, { headerTrainingRun, key: "header" }),
      mainContent,
    ]),
    h(PanelFooterView, { key: "footer" }),
    h(TutorialOverlayView, { key: "overlay" }),
  ]);
}

function SidepanelRootView() {
  const renderVersion = useSidepanelRenderVersion();
  const setupBlocked = state.setupLoaded && state.requiredSetup && !state.requiredSetup.ready;
  const headerTrainingRun = getHeaderTrainingRun();
  const installedCrafts = state.crafts.filter((craft) => !isRemoteSharedCraft(craft));
  const availableCrafts = state.crafts.filter((craft) => isRemoteSharedCraft(craft));

  React.useLayoutEffect(() => {
    if (!pendingRenderUiState) return;
    const nextUiState = pendingRenderUiState;
    pendingRenderUiState = null;
    restoreRenderUiState(nextUiState.uiState, nextUiState.passId);
  });

  void renderVersion;

  return h(SidepanelAppView, {
    setupBlocked,
    headerTrainingRun,
    installedCrafts,
    availableCrafts,
  });
}

function render() {
  const uiState = captureRenderUiState();
  const passId = ++renderPassId;
  pendingRenderUiState = { uiState, passId };
  mountSidepanelReactApp();
  if (sidepanelReactMounted) {
    notifySidepanelRender();
  }
  scheduleSidepanelStatePersist();
}

function patchCraftCardInDom(craftId) {
  return false;
}

function refreshCraftCard(craftId) {
  if (patchCraftCardInDom(craftId)) return;
  render();
}

function renderPanelHeader(headerTrainingRun) {
  if (!shouldShowDevHeader()) {
    return `
      <header class="panel-header panel-header-minimal">
        <div class="panel-header-main"></div>
        <div class="panel-header-actions panel-header-actions-minimal">
          ${renderHeaderMenuButton()}
        </div>
      </header>
    `;
  }

  const smokeBusy = isSmokeTestRunning();
  const trainingBusy = isTrainingRunActive(headerTrainingRun);
  return `
    <header class="panel-header">
      <div class="panel-header-main">
      </div>
      <div class="panel-header-actions">
        ${renderHeaderActionRows({ smokeBusy, trainingBusy })}
      </div>
    </header>
  `;
}

function renderCraftList(installedCrafts = [], availableCrafts = []) {
  const hasAnyCrafts = installedCrafts.length > 0 || availableCrafts.length > 0;
  const createMarkup = state.createOpen ? renderComposerRow() : renderAddRow();
  if (!hasAnyCrafts) {
    return `
      <section class="craft-list craft-list-empty" aria-label="Crafts">
        ${createMarkup}
      </section>
    `;
  }
  return `
    <section class="craft-list" aria-label="Crafts">
      ${renderCraftSection("", installedCrafts, {
        footerMarkup: createMarkup,
      })}
      ${
        availableCrafts.length
          ? renderCraftSection("Available", availableCrafts, {
              tone: "available",
            })
          : ""
      }
    </section>
  `;
}

function renderCraftSection(title, crafts = [], { tone = "", emptyMarkup = "", footerMarkup = "" } = {}) {
  return `
    <section class="craft-section ${tone ? `craft-section-${escapeHtml(tone)}` : ""}">
      ${title ? `<div class="craft-section-head"><span class="craft-section-title">${escapeHtml(title)}</span></div>` : ""}
      <div class="craft-section-body">
        ${crafts.length ? crafts.map((craft) => renderCraft(craft)).join("") : emptyMarkup}
        ${footerMarkup}
      </div>
    </section>
  `;
}

function renderHeaderActionRows({ smokeBusy = false, trainingBusy = false } = {}) {
  return HEADER_TEST_ROW_DEFS.map((row) => {
    const buttons = row.tests.map((testId) => renderHeaderTestButton(testId, { smokeBusy, trainingBusy })).join("");
    if (!buttons && !row.includeMenu) return "";
    return `
      <div class="panel-header-action-row panel-header-action-row-${escapeHtml(row.id)}">
        <span class="panel-header-action-label">${escapeHtml(row.label)}</span>
        ${buttons}
        ${row.includeMenu ? renderHeaderMenuButton() : ""}
      </div>
    `;
  }).join("");
}

function renderHeaderTestButton(testId, { smokeBusy = false, trainingBusy = false } = {}) {
  const def = HEADER_TEST_DEFS[testId];
  if (!def) return "";

  const testState = getHeaderTestState(testId);
  if (testState.status === "hidden") return "";

  const isRunning = testState.status === "running";
  const isSuccess = testState.status === "success";
  const disabled = isRunning || isSuccess || smokeBusy || trainingBusy;
  const label = isRunning ? `${def.label}...` : def.label;
  const progress = Math.max(0.04, Math.min(1, Number(testState.progress) || 0));
  const classes = [
    "header-smoke",
    def.variant === "quiet" ? "header-smoke-quiet" : "",
    isSuccess ? "header-smoke-success" : "",
  ]
    .filter(Boolean)
    .join(" ");

  return `
    <div class="header-smoke-wrap ${isRunning ? "header-smoke-wrap-running" : ""}">
      <button
        class="${classes}"
        data-action="${escapeHtml(def.action)}"
        data-header-test-id="${escapeHtml(testId)}"
        type="button"
        ${disabled ? "disabled" : ""}
      >
        ${
          isSuccess
            ? '<span class="header-smoke-check" aria-hidden="true">✓</span>'
            : ""
        }
        <span>${escapeHtml(label)}</span>
      </button>
      ${
        isRunning
          ? `
            <span class="header-smoke-progress" aria-hidden="true">
              <span
                class="header-smoke-progress-fill ${testState.indeterminate ? "header-smoke-progress-fill-indeterminate" : ""}"
                ${testState.indeterminate ? "" : `style="width:${Math.round(progress * 100)}%"`}
              ></span>
            </span>
          `
          : ""
      }
    </div>
  `;
}

function renderHeaderMenuButton() {
  return `
    <button class="header-menu" data-action="open-options" type="button" aria-label="Open model settings">
      <span></span>
      <span></span>
      <span></span>
    </button>
  `;
}

function shouldShowDevHeader() {
  return state.uiPreferences?.showDevHeader === true;
}

async function setActiveTheme(themeId, options = {}) {
  const nextThemeId = themeApi?.normalizeThemeId?.(themeId) || state.themeId;
  const persist = options?.persist !== false;
  if (!nextThemeId || nextThemeId === state.themeId) return;
  state.themeId = nextThemeId;
  themeApi?.applyTheme?.(nextThemeId);
  render();
  if (!persist) return;
  try {
    await themeApi?.writeTheme?.(nextThemeId);
  } catch (error) {
    console.warn("[sidepanel] failed to persist theme", error);
  }
}

function renderTrainingDataView(craft) {
  const trainingState = getTrainingDataState(craft.id);
  const meta = getTrainingDataMeta(trainingState.samples);
  const page = getTrainingDataPage(craft.id);
  const totalPages = Math.max(1, Math.ceil(Math.max(1, meta.totalSamples) / TRAINING_DATA_PAGE_SIZE));
  const pageStart = Math.max(0, (page - 1) * TRAINING_DATA_PAGE_SIZE);
  const pageSamples = trainingState.samples.slice(pageStart, pageStart + TRAINING_DATA_PAGE_SIZE);
  const pageEnd = pageStart + pageSamples.length;
  const isEditable = craftStore?.isEditableCraft?.(craft) !== false;

  return `
    <section class="training-data-view">
      <section class="training-data-hero">
        <div class="training-data-hero-copy">
          <div class="detail-eyebrow">Dataset workspace</div>
          <h1 class="training-data-title">${escapeHtml(craft.name)}</h1>
          <p class="training-data-copy">
            Maintain the prompt-to-JSON samples for this craft here. The first grammar and style correction can be a demo case, but the structure should remain generic for other tasks.
          </p>
        </div>
        <div class="training-data-hero-actions">
          <button
            class="detail-action"
            data-action="training-data-add-sample"
            data-craft-id="${escapeHtml(craft.id)}"
            type="button"
            ${isEditable ? "" : "disabled"}
          >
            Add sample
          </button>
          <span class="training-data-hero-note">${escapeHtml(isEditable ? "Saved locally" : "Read only")}</span>
        </div>
      </section>
      <section class="training-data-meta-grid">
        ${renderTrainingDataMetaCard("Samples", formatCount(meta.totalSamples))}
        ${renderTrainingDataMetaCard("Ready", formatCount(meta.readySamples))}
        ${renderTrainingDataMetaCard("Drafts", formatCount(meta.draftSamples + meta.reviewSamples))}
        ${renderTrainingDataMetaCard("Invalid Rows", formatCount(meta.invalidSamples))}
      </section>
      <section class="training-data-toolbar">
        <div class="training-data-toolbar-copy">
          <span>${escapeHtml(meta.lastUpdatedAt ? `Updated ${formatTimeAgo(meta.lastUpdatedAt)}` : "No samples yet")}</span>
          ${
            meta.totalSamples
              ? `<span>${escapeHtml(`Page ${page}/${totalPages} · ${formatCount(pageStart + 1)}-${formatCount(pageEnd)} of ${formatCount(meta.totalSamples)}`)}</span>`
              : ""
          }
        </div>
        ${renderTrainingDataSaveState(trainingState)}
      </section>
      ${renderTrainingDataAgentPanel(craft.id)}
      ${
        trainingState.loading
          ? `
            <section class="training-data-empty">
              <div class="detail-eyebrow">Loading</div>
              <div class="training-data-empty-title">Preparing training samples</div>
              <div class="training-data-empty-copy">Reading the local artifact store for this craft.</div>
            </section>
          `
          : pageSamples.length
            ? `
              ${renderTrainingDataTable(craft, pageSamples, isEditable)}
              ${renderTrainingDataPagination(craft.id, page, totalPages, meta.totalSamples)}
            `
            : `
              <section class="training-data-empty">
                <div class="detail-eyebrow">Dataset</div>
                <div class="training-data-empty-title">No samples for this craft yet</div>
                <div class="training-data-empty-copy">
                  Add the first seed rows here so the agent has reviewed examples to expand from later.
                </div>
              </section>
            `
      }
    </section>
  `;
}

function renderTrainingDataAgentPanel(craftId) {
  const agentRun = state.agentRuns[craftId] || null;
  if (!agentRun) return "";

  return `
    <section class="drawer-log-panel training-data-agent-panel">
      <div class="drawer-log-head">
        <span class="detail-eyebrow">Agent activity</span>
        <span class="drawer-log-status">${escapeHtml(getAgentLogStatusText(agentRun))}</span>
      </div>
      <div
        class="drawer-log-scroll"
        data-preserve-scroll-key="training-data-agent:${escapeHtml(craftId)}"
        data-scroll-anchor="bottom"
        data-scroll-start="bottom"
      >
        ${renderAgentLogs(agentRun)}
      </div>
    </section>
  `;
}

function renderTrainingDataMetaCard(label, value) {
  return `
    <article class="training-data-meta-card">
      <span class="detail-eyebrow">${escapeHtml(label)}</span>
      <strong>${escapeHtml(value)}</strong>
    </article>
  `;
}

function renderTrainingDataSaveState(trainingState) {
  if (trainingState.saveError) {
    return `<span class="training-data-save training-data-save-error">${escapeHtml(trainingState.saveError)}</span>`;
  }
  if (trainingState.saving) {
    return `<span class="training-data-save">Saving draft...</span>`;
  }
  if (trainingState.lastSavedAt) {
    return `<span class="training-data-save">Saved ${escapeHtml(formatTimeAgo(trainingState.lastSavedAt))}</span>`;
  }
  return `<span class="training-data-save">Local draft</span>`;
}

function renderToolScriptsSaveState(toolState) {
  if (toolState.saveError) {
    return `<span class="training-data-save training-data-save-error">${escapeHtml(toolState.saveError)}</span>`;
  }
  if (toolState.saving) {
    return `<span class="training-data-save">Saving bundle...</span>`;
  }
  if (toolState.lastSavedAt) {
    return `<span class="training-data-save">Saved ${escapeHtml(formatTimeAgo(toolState.lastSavedAt))}</span>`;
  }
  return `<span class="training-data-save">Local bundle draft</span>`;
}

function renderTrainingDataTable(craft, samples, isEditable) {
  return `
    <div class="training-data-table-wrap">
      <table class="training-data-table">
        <thead>
          <tr>
            <th>Split</th>
            <th>Status</th>
            <th>Prompt</th>
            <th>Expected JSON</th>
            <th>Source</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          ${samples.map((sample) => renderTrainingDataRow(craft, sample, isEditable)).join("")}
        </tbody>
      </table>
    </div>
  `;
}

function renderTrainingDataRow(craft, sample, isEditable) {
  const structured = hasStructuredTrainingTrace(sample);
  const structuredInspection = structured ? buildStructuredTrainingInspection(sample) : null;
  const validation = validateTrainingSample(sample);
  const validationOk = validation.jsonState?.ok === true;
  const promptValue = structuredInspection?.promptText || (structured ? summarizeStructuredTrainingPrompt(sample) : sample.promptText);
  const expectedValue =
    structuredInspection?.targetText || (structured ? summarizeStructuredTrainingTarget(sample) : sample.expectedJsonText);
  const promptRows = structured ? countTrainingTextareaRows(promptValue, 10, 28) : 6;
  const expectedRows = structured ? countTrainingTextareaRows(expectedValue, 9, 20) : 7;
  const rawRows = structuredInspection?.rawText ? countTrainingTextareaRows(structuredInspection.rawText, 12, 28) : 0;
  const promptReadonly = structured ? "readonly" : "";
  const expectedReadonly = structured ? "readonly" : "";
  const promptPlaceholder = structured ? "Stored as multi-turn transcript" : "Prompt text";
  const expectedPlaceholder = structured ? "Stored as supervised assistant turn" : "{\"result\": \"\"}";
  return `
    <tr>
      <td>
        <select
          class="training-data-select"
          data-training-sample-field="split"
          data-craft-id="${escapeHtml(craft.id)}"
          data-sample-id="${escapeHtml(sample.id)}"
          ${isEditable ? "" : "disabled"}
        >
          ${["train", "validation", "test"]
            .map((value) => `<option value="${value}" ${sample.split === value ? "selected" : ""}>${escapeHtml(value)}</option>`)
            .join("")}
        </select>
      </td>
      <td>
        <select
          class="training-data-select"
          data-training-sample-field="status"
          data-craft-id="${escapeHtml(craft.id)}"
          data-sample-id="${escapeHtml(sample.id)}"
          ${isEditable ? "" : "disabled"}
        >
          ${["draft", "review", "ready", "blocked"]
            .map((value) => `<option value="${value}" ${sample.status === value ? "selected" : ""}>${escapeHtml(value)}</option>`)
            .join("")}
        </select>
        <div class="training-data-chip-row">
          <span class="training-data-chip ${validationOk ? "training-data-chip-valid" : "training-data-chip-invalid"}">
            ${escapeHtml(validationOk ? "valid" : "invalid")}
          </span>
          ${structured
            ? `<span class="training-data-chip training-data-chip-valid">multi-turn</span>`
            : ""}
        </div>
      </td>
      <td>
        <textarea
          class="training-data-textarea training-data-textarea-prompt"
          data-training-sample-field="promptText"
          data-craft-id="${escapeHtml(craft.id)}"
          data-sample-id="${escapeHtml(sample.id)}"
          rows="${promptRows}"
          placeholder="${escapeHtml(promptPlaceholder)}"
          ${promptReadonly}
          ${isEditable ? "" : "disabled"}
        >${escapeHtml(promptValue)}</textarea>
      </td>
      <td>
        <textarea
          class="training-data-textarea training-data-textarea-json"
          data-training-sample-field="expectedJsonText"
          data-craft-id="${escapeHtml(craft.id)}"
          data-sample-id="${escapeHtml(sample.id)}"
          rows="${expectedRows}"
          placeholder="${escapeHtml(expectedPlaceholder)}"
          ${expectedReadonly}
          ${isEditable ? "" : "disabled"}
        >${escapeHtml(expectedValue)}</textarea>
        <div class="training-data-validation ${validationOk ? "training-data-validation-ok" : "training-data-validation-error"}">
          ${escapeHtml(validation.detail)}
        </div>
        ${
          structuredInspection?.rawText
            ? `
              <details class="training-data-structured-details" open>
                <summary class="training-data-structured-summary">Raw multi-turn row JSON</summary>
                <textarea
                  class="training-data-textarea training-data-textarea-json training-data-textarea-structured"
                  rows="${rawRows}"
                  readonly
                  spellcheck="false"
                >${escapeHtml(structuredInspection.rawText)}</textarea>
              </details>
            `
            : ""
        }
      </td>
      <td>
        <input
          class="training-data-input"
          data-training-sample-field="source"
          data-craft-id="${escapeHtml(craft.id)}"
          data-sample-id="${escapeHtml(sample.id)}"
          value="${escapeHtml(sample.source)}"
          placeholder="forum, manual, import..."
          ${isEditable ? "" : "disabled"}
        >
        <div class="training-data-row-meta">${escapeHtml(formatTimeAgo(sample.updatedAt || sample.createdAt))}</div>
      </td>
      <td>
        <button
          class="training-data-delete"
          data-action="training-data-delete-sample"
          data-craft-id="${escapeHtml(craft.id)}"
          data-sample-id="${escapeHtml(sample.id)}"
          type="button"
          ${isEditable ? "" : "disabled"}
        >
          Delete
        </button>
      </td>
    </tr>
  `;
}

function renderTrainingDataPagination(craftId, page, totalPages, totalSamples) {
  if (!totalSamples) return "";
  return `
    <div class="training-data-pagination">
      <button
        class="detail-secondary"
        data-action="training-data-prev-page"
        data-craft-id="${escapeHtml(craftId)}"
        type="button"
        ${page <= 1 ? "disabled" : ""}
      >
        Previous
      </button>
      <span class="training-data-pagination-label">${escapeHtml(`Page ${page} of ${totalPages}`)}</span>
      <button
        class="detail-secondary"
        data-action="training-data-next-page"
        data-craft-id="${escapeHtml(craftId)}"
        type="button"
        ${page >= totalPages ? "disabled" : ""}
      >
        Next
      </button>
    </div>
  `;
}

function renderSetupLoading() {
  return h("section", { className: "setup-gate" }, [
    h("div", { className: "setup-gate-eyebrow", key: "eyebrow" }, "Startup"),
    h("h1", { className: "setup-gate-title", key: "title" }, "Loading model setup"),
    h("p", { className: "setup-gate-copy", key: "copy" }, "Checking required startup roles."),
  ]);
}

function renderStartupError() {
  return h("section", { className: "setup-gate" }, [
    h("div", { className: "setup-gate-eyebrow", key: "eyebrow" }, "Startup"),
    h("h1", { className: "setup-gate-title", key: "title" }, "Startup failed"),
    h("p", { className: "setup-gate-copy", key: "copy" }, "Reload the extension once and inspect the error if it still fails."),
    h("div", { className: "setup-inline-error", key: "error" }, state.startupError || "Unknown startup error."),
  ]);
}

function renderSetupGate() {
  const selectedProviderId = getInlineSetupProviderId();
  const provider = state.providers?.openai || {};
  return h("section", { className: "setup-gate" }, [
    h("div", { className: "setup-gate-eyebrow", key: "eyebrow" }, "First Start"),
    h("h1", { className: "setup-gate-title", key: "title" }, "Finish setup"),
    h("section", { className: "inline-setup-card", key: "provider-card" }, [
      h("div", { className: "inline-setup-head", key: "head" }, [
        h("div", { className: "detail-eyebrow", key: "label" }, "Provider"),
      ]),
      h("div", { className: "inline-setup-row", key: "row" }, [
        h("label", { className: "inline-setup-field", key: "provider-field" }, [
          h("span", { className: "inline-setup-label", key: "label" }, "Provider"),
          h(
            "select",
            {
              "data-setup-provider": "true",
              key: "select",
              value: selectedProviderId,
              onChange: (event) => setInlineSetupProvider(event.target.value),
            },
            [
              h("option", { value: "openai", key: "openai" }, "OpenAI"),
              h("option", { value: "local_qwen", key: "local_qwen" }, "Local Qwen WebGPU"),
            ],
          ),
        ]),
        h("label", { className: "inline-setup-field inline-setup-field-wide", key: "api-key-field" }, [
          h("span", { className: "inline-setup-label", key: "label" }, "API Key"),
          h("input", {
            "data-setup-api-key": "true",
            key: "input",
            type: "password",
            value: String(provider.apiKey || ""),
            placeholder: selectedProviderId === "local_qwen" ? "not required for local WebGPU" : "sk-...",
            spellCheck: false,
            disabled: selectedProviderId === "local_qwen",
            onChange: (event) => setInlineSetupApiKey(event.target.value),
          }),
        ]),
      ]),
    ]),
    h("section", { className: "inline-setup-card", key: "models-card" }, [
      h("div", { className: "inline-setup-head", key: "head" }, [
        h("div", { className: "detail-eyebrow", key: "label" }, "Starter models"),
      ]),
      h("div", { className: "inline-setup-grid", key: "grid" }, [
        renderInlineSetupSlot("agent", true),
        renderInlineSetupSlot("batch", true),
        renderInlineSetupSlot("vision", false),
      ]),
    ]),
    h("div", { className: "setup-aux-grid", key: "aux-grid" }, [
      h("section", { className: "inline-setup-card", key: "theme-card" }, [
        h("div", { className: "inline-setup-head", key: "head" }, [
          h("div", { className: "detail-eyebrow", key: "label" }, "Theme"),
        ]),
        h("label", { className: "inline-setup-field", key: "field" }, [
          h("span", { className: "inline-setup-label", key: "label" }, "Theme"),
          h(
            "select",
            {
              "data-setup-theme": "true",
              key: "select",
              value: state.themeId,
              onChange: (event) => setInlineSetupTheme(event.target.value),
            },
            (themeApi?.THEMES || []).map((theme) =>
              h("option", { value: theme.id, key: theme.id }, theme.title),
            ),
          ),
        ]),
      ]),
      h("section", { className: "inline-setup-card", key: "sharing-card" }, [
        h("div", { className: "inline-setup-head", key: "head" }, [
          h("div", { className: "detail-eyebrow", key: "label" }, "Sharing"),
        ]),
        h("label", { className: "inline-setup-field", key: "server-field" }, [
          h("span", { className: "inline-setup-label", key: "label" }, "Server"),
          h("input", {
            "data-setup-sync-field": "signalingUrls",
            key: "input",
            type: "text",
            value: String(state.syncSettings.signalingUrls || ""),
            placeholder: craftSync?.DEFAULT_SIGNALING_URL || "wss://api.metricspace.org/signal",
            spellCheck: false,
            onChange: (event) => setInlineSetupSyncField("signalingUrls", event.target.value),
          }),
        ]),
        h("label", { className: "inline-setup-field", key: "token-field" }, [
          h("span", { className: "inline-setup-label", key: "label" }, "Session code"),
          h("div", { className: "inline-setup-inline", key: "inline" }, [
            h("input", {
              "data-setup-sync-field": "token",
              key: "input",
              type: "password",
              value: String(state.syncSettings.token || ""),
              placeholder: "private by default",
              spellCheck: false,
              onChange: (event) => setInlineSetupSyncField("token", event.target.value),
            }),
            h(
              "button",
              {
                className: "inline-mini-button",
                key: "button",
                type: "button",
                "data-action": "randomize-setup-sync-token",
              },
              "New code",
            ),
          ]),
        ]),
      ]),
    ]),
    state.setupError
      ? h("div", { className: "setup-inline-error", key: "error" }, state.setupError)
      : null,
    h("div", { className: "setup-gate-actions", key: "actions" }, [
      h(
        "button",
        {
          className: "workspace-primary",
          key: "save",
          type: "button",
          disabled: state.setupSaving,
          "data-action": "save-inline-setup",
        },
        state.setupSaving ? "Saving..." : "Save setup",
      ),
    ]),
  ]);
}

function getInlineSetupProviderId() {
  for (const slotId of getRequiredSetupSlotIds()) {
    const providerId = String(state.slots?.[slotId]?.providerId || "").trim();
    if (providerId) return providerId;
  }
  return "openai";
}

function setInlineSetupProvider(providerId) {
  const nextProviderId = String(providerId || "openai").trim() || "openai";
  for (const slotId of getRequiredSetupSlotIds()) {
    state.slots = configApi.patchModelSlot(
      state.slots,
      slotId,
      {
        providerId: nextProviderId,
        modelName: "",
      },
      state.providers,
    );
  }
  state.setupError = "";
  render();
}

function setInlineSetupApiKey(apiKey) {
  state.providers = configApi.patchProviderRecord(state.providers, "openai", {
    enabled: true,
    apiKey,
  });
  state.setupError = "";
  render();
}

function setInlineSetupSyncField(field, value) {
  const key = String(field || "").trim();
  if (!key) return;
  state.syncSettings = {
    ...state.syncSettings,
    [key]: value,
    ...(key === "token" ? { tokenAutoGenerated: false } : {}),
  };
  state.setupError = "";
  render();
}

function randomizeInlineSetupSyncToken() {
  state.syncSettings = {
    ...state.syncSettings,
    token: craftSync?.generateRandomToken?.() || "",
    tokenAutoGenerated: true,
  };
  state.setupError = "";
  render();
}

function setInlineSetupSlotModel(slotId, modelName) {
  const key = String(slotId || "").trim();
  if (!key) return;
  const providerId = String(state.slots?.[key]?.providerId || getInlineSetupProviderId()).trim() || "openai";
  state.slots = configApi.patchModelSlot(
    state.slots,
    key,
    {
      providerId,
      modelName,
    },
    state.providers,
  );
  state.setupError = "";
  render();
}

function setInlineSetupSlotReasoning(slotId, reasoningEffort) {
  const key = String(slotId || "").trim();
  if (!key) return;
  const providerId = String(state.slots?.[key]?.providerId || getInlineSetupProviderId()).trim() || "openai";
  state.slots = configApi.patchModelSlot(
    state.slots,
    key,
    {
      providerId,
      reasoningEffort,
    },
    state.providers,
  );
  state.setupError = "";
  render();
}

function setInlineSetupTheme(themeId) {
  state.setupError = "";
  const nextThemeId = String(themeId || "").trim();
  if (!nextThemeId || nextThemeId === state.themeId) {
    render();
    return;
  }
  void setActiveTheme(nextThemeId);
}

function renderInlineSetupSlot(slotId, showReasoning) {
  const slotDef = configApi.getSlotDefinition(slotId) || { label: slotId };
  const slot = state.slots?.[slotId] || {};
  const options = getSetupSlotModelOptions(slotId, slot.providerId, slot.modelName);
  const reasoningOptions = (configApi.REASONING_EFFORT_VALUES || []).filter(Boolean);
  const setupStatus = configApi.getSlotSetupStatus(slotId, state.slots, state.providers);

  return h(
    "article",
    {
      className: `inline-setup-slot ${setupStatus.ok ? "" : "inline-setup-slot-pending"}`.trim(),
      key: slotId,
    },
    [
      h("div", { className: "inline-setup-slot-head", key: "head" }, [
        h("strong", { key: "label" }, slotDef.label),
      ]),
      h("label", { className: "inline-setup-field", key: "model-field" }, [
        h("span", { className: "inline-setup-label", key: "label" }, "Model"),
        h(
          "select",
          {
            "data-setup-slot-model": slotId,
            key: "select",
            value: String(slot.modelName || ""),
            disabled: !options.length,
            onChange: (event) => setInlineSetupSlotModel(slotId, event.target.value),
          },
          options.length
            ? options.map((option) =>
                h("option", { value: option.modelName, key: option.modelName }, option.label),
              )
            : [h("option", { value: "", key: "empty" }, "No models configured for this provider")],
        ),
      ]),
      showReasoning
        ? h("label", { className: "inline-setup-field", key: "reasoning-field" }, [
            h("span", { className: "inline-setup-label", key: "label" }, "Reasoning"),
            h(
              "select",
              {
                "data-setup-slot-reasoning": slotId,
                key: "select",
                value: String(slot.reasoningEffort || ""),
                onChange: (event) => setInlineSetupSlotReasoning(slotId, event.target.value),
              },
              reasoningOptions.map((value) =>
                h("option", { value, key: value }, capitalizeReasoning(value)),
              ),
            ),
          ])
        : null,
    ],
  );
}

function getSetupSlotModelOptions(slotId, providerId, currentModelName) {
  const providerOptions = configApi
    .collectModelOptions(state.providers, slotId)
    .filter((option) => option.providerId === providerId);
  const currentModel = String(currentModelName || "").trim();

  if (!currentModel || providerOptions.some((option) => option.modelName === currentModel)) {
    return providerOptions.map((option) => ({
      modelName: option.modelName,
      label: option.modelName,
    }));
  }

  return [
    {
      modelName: currentModel,
      label: `${currentModel} (Current)`,
    },
    ...providerOptions.map((option) => ({
      modelName: option.modelName,
      label: option.modelName,
    })),
  ];
}

function renderSmokeTestStatus() {
  const headerTrainingRun = getHeaderTrainingRun();
  if (headerTrainingRun && isTrainingRunActive(headerTrainingRun)) {
    const progress = Math.max(0, Math.min(1, Number(headerTrainingRun.progress || 0)));
    const totalSamples = Number(headerTrainingRun.totalSamples || 0);
    const completedSamples = Number(headerTrainingRun.completedSamples || 0);
    const phaseTotalSamples = Number(headerTrainingRun.phaseTotalSamples || 0);
    const phaseCompletedSamples = Number(headerTrainingRun.phaseCompletedSamples || 0);
    const phaseUnitLabel = String(headerTrainingRun.phaseUnitLabel || "").trim() || "samples";
    const statParts = [];
    if (phaseTotalSamples > 0) {
      statParts.push(
        `${formatCompactCount(phaseCompletedSamples)} / ${formatCompactCount(phaseTotalSamples)} ${phaseUnitLabel}`,
      );
      statParts.push(formatItemsPerSecond(headerTrainingRun.samplesPerSecond || 0, phaseUnitLabel));
    } else if (Number(headerTrainingRun.samplesPerSecond || 0) > 0) {
      statParts.push(formatItemsPerSecond(headerTrainingRun.samplesPerSecond || 0, phaseUnitLabel));
    }
    const comparisonStat = getTrainingComparisonStat(headerTrainingRun);
    if (comparisonStat) {
      statParts.push(comparisonStat);
    }
    statParts.push(
      totalSamples > 0
        ? `job ${formatCompactCount(completedSamples)} / ${formatCompactCount(totalSamples)} steps`
        : `job ${formatCompactCount(completedSamples)} steps`,
    );
    if (Number(headerTrainingRun.currentEpoch || 0) > 0 && Number(headerTrainingRun.epochsTotal || 0) > 0) {
      statParts.push(`epoch ${headerTrainingRun.currentEpoch}/${headerTrainingRun.epochsTotal}`);
    }
    return `
      <section class="smoke-status smoke-status-${escapeHtml(getTrainingBannerTone(headerTrainingRun))}">
        <div class="smoke-status-line">${escapeHtml(headerTrainingRun.message || headerTrainingRun.phaseLabel || "Training run")}</div>
        <div class="smoke-status-progress" aria-hidden="true">
          <span class="smoke-status-progress-fill" style="width:${Math.round(progress * 100)}%"></span>
        </div>
        <div class="smoke-status-meta">${escapeHtml(statParts.join(" · "))}</div>
        ${
          headerTrainingRun.error
            ? `<div class="smoke-status-meta">${escapeHtml(headerTrainingRun.error)}</div>`
            : ""
        }
      </section>
    `;
  }

  if (!["error", "success"].includes(String(state.smokeTest?.status || "")) || !state.smokeTest?.message) return "";

  return `
    <section class="smoke-status smoke-status-${escapeHtml(state.smokeTest.status || "idle")}">
      <div class="smoke-status-line">${escapeHtml(state.smokeTest.message)}</div>
      ${
        state.smokeTest.failureKind && state.smokeTest.status === "error"
          ? `<div class="smoke-status-meta smoke-status-kind">Type: ${escapeHtml(formatFailureKindLabel(state.smokeTest.failureKind))}</div>`
          : ""
      }
      ${
        state.smokeTest.detail
          ? `<div class="smoke-status-meta">${escapeHtml(state.smokeTest.detail)}</div>`
          : ""
      }
      ${
        state.smokeTest.helpText
          ? `<div class="smoke-status-meta smoke-status-help">Next: ${escapeHtml(state.smokeTest.helpText)}</div>`
          : ""
      }
      ${
        state.smokeTest.reportText
          ? `
            <div class="smoke-status-actions">
              <button class="inline-mini-button smoke-status-copy" data-action="copy-test-error-report" type="button">
                ${escapeHtml(state.smokeTest.status === "success" ? "Copy report" : "Copy error report")}
              </button>
              ${
                state.smokeTest.copyStatus === "copied"
                  ? '<span class="smoke-status-copy-note">Copied.</span>'
                  : ""
              }
            </div>
          `
          : ""
      }
    </section>
  `;
}

function renderSyncStatus() {
  const snapshot = state.syncSnapshot?.sync || null;
  if (!snapshot) return "";

  const tone =
    snapshot.lastError
      ? "error"
      : snapshot.connection === "connected"
      ? "success"
      : snapshot.running
        ? "running"
        : "idle";

  const modeLabel =
    snapshot.requestedMode === "continuous"
      ? "Live sync"
      : snapshot.requestedMode === "manual"
        ? "One-shot sync"
        : "Sync off";

  const isQuietLocalState =
    snapshot.requestedMode !== "continuous" &&
    !snapshot.remoteCraftCount &&
    !snapshot.transportPeerCount &&
    !snapshot.remotePeerCount &&
    !snapshot.lastError;
  if (isQuietLocalState) return "";
  const statusLabel = isQuietLocalState ? "Local only" : modeLabel;
  const meta = [];

  if (!isQuietLocalState) {
    meta.push(
      `${formatCount(snapshot.remoteCraftCount || 0)} shared`,
      `${formatCount(snapshot.transportPeerCount || 0)} live links`,
      `${formatCount(snapshot.remotePeerCount || 0)} visible peers`,
    );
  }

  if (snapshot.lastError) {
    meta.push(trimText(snapshot.lastError, 64));
  } else if (snapshot.lastSyncedAt) {
    meta.push(`updated ${formatTimeAgo(snapshot.lastSyncedAt)}`);
  }

  return `
    <section class="sync-rail sync-rail-${escapeHtml(tone)}">
      <div class="sync-rail-main">
        <span class="sync-rail-status">${escapeHtml(statusLabel)}</span>
        ${meta.length ? `<span class="sync-rail-meta">${escapeHtml(meta.join(" · "))}</span>` : ""}
      </div>
      <div class="sync-rail-actions">
          ${
            snapshot.requestedMode !== "continuous"
              ? `<button class="inline-mini-button" data-action="sync-now" type="button">Sync now</button>`
              : ""
          }
          <button class="inline-mini-button" data-action="open-share-settings" type="button">Share</button>
      </div>
    </section>
  `;
}

function renderPanelFooter() {
  return `
    <footer class="panel-footer">
      <span class="panel-footer-version">v${escapeHtml(EXTENSION_VERSION)}</span>
      <span class="panel-footer-sync">${escapeHtml(buildFooterPeerStatus())}</span>
    </footer>
  `;
}

function getHeaderTrainingRun() {
  return state.debugTrainingRun || null;
}

function getTrainingBannerTone(run) {
  const status = String(run?.status || "");
  if (status === "completed") return "success";
  if (status === "failed") return "error";
  if (["queued", "starting", "running"].includes(status)) return "running";
  return "idle";
}

function renderCraft(craft) {
  const isActive = craft.id === state.activeCraftId;
  const agentRun = state.agentRuns[craft.id] || null;
  const liveSignal = isRemoteSharedCraft(craft) ? getCraftLiveSignal(craft) : null;
  const cardNote = getCraftCardSecondaryText(craft);
  const maturityText = getCraftMaturityText(craft, agentRun);

  return `
    <article
      class="craft-card ${isActive ? "craft-card-active" : ""}"
      data-craft-card-id="${escapeHtml(craft.id)}"
    >
      <button
        class="craft-row"
        data-action="toggle-craft"
        data-craft-id="${escapeHtml(craft.id)}"
        type="button"
        aria-expanded="${isActive ? "true" : "false"}"
      >
        <span class="craft-main">
          <span class="craft-name-row">
            <span class="craft-name">${escapeHtml(craft.name)}</span>
            <span
              class="${escapeHtml(getCraftMaturityBadgeClassName(craft, agentRun))}"
              title="${escapeHtml(getCraftMaturityTitle(craft, agentRun))}"
              style="--craft-maturity-progress:${escapeHtml(getCraftMaturityBadgeStyle(craft, agentRun)["--craft-maturity-progress"])};"
            ><span class="craft-maturity-badge-label">${escapeHtml(maturityText)}</span></span>
          </span>
          ${cardNote ? `<span class="craft-note">${escapeHtml(cardNote)}</span>` : ""}
        </span>
      </button>
      ${liveSignal ? renderCraftLiveStrip(liveSignal) : ""}
      ${isActive ? renderWorkspace(craft) : ""}
    </article>
  `;
}

function renderCraftLiveStrip(signal) {
  return `
    <div class="craft-live-strip">
      <span class="craft-live-dot" aria-hidden="true"></span>
      <span class="craft-live-text">${escapeHtml(signal.message)}</span>
      ${signal.meta ? `<span class="craft-live-meta">${escapeHtml(signal.meta)}</span>` : ""}
    </div>
  `;
}

function renderWorkspace(craft) {
  const isRemoteShared = isRemoteSharedCraft(craft);

  return `
    <section class="craft-workspace">
      ${isRemoteShared ? renderRemoteCraftDrawer(craft) : renderCraftingDrawer(craft)}
    </section>
  `;
}

function renderCraftStateChips(craft, useState) {
  const chips = [renderCraftPrimaryStateChip(craft, useState)];
  const shareChip = getCraftShareStateChip(craft);
  const roleChip = getCraftRoleStateChip(craft);

  if (shareChip) {
    chips.push(renderSimpleCraftStateChip(shareChip));
  }
  if (roleChip) {
    chips.push(renderSimpleCraftStateChip(roleChip));
  }

  return `<span class="craft-state-row">${chips.join("")}</span>`;
}

function renderCraftPrimaryStateChip(craft, useState) {
  const label = escapeHtml(useState.label);
  const tone = escapeHtml(useState.tone);
  if (isRemoteSharedCraft(craft)) {
    const owner = escapeHtml(getSharedOwnerName(craft));
    return `
      <span class="craft-state-tooltip">
        <span class="craft-state craft-state-${tone}" title="Shared by ${owner}">${label}</span>
        <span class="craft-state-popover" role="tooltip">
          Shared by <strong>${owner}</strong>. This copy stays synced until you fork it locally.
        </span>
      </span>
    `;
  }
  if (!isVanillaStarterCraft(craft)) {
    return `<span class="craft-state craft-state-${tone}">${label}</span>`;
  }

  const tooltip = escapeHtml(
    `Runs on ${getVanillaTargetModelLabel(craft)}. Open Craft to configure instructions, tools, data, and tuning.`,
  );
  return `
    <span class="craft-state-tooltip">
      <span class="craft-state craft-state-${tone}" title="${tooltip}">${label}</span>
      <span class="craft-state-popover" role="tooltip">${tooltip}</span>
    </span>
  `;
}

function renderSimpleCraftStateChip(chip) {
  return `
    <span class="craft-state craft-state-${escapeHtml(chip.tone)}">
      ${chip.icon ? `<span class="craft-state-icon craft-state-icon-${escapeHtml(chip.icon)}" aria-hidden="true"></span>` : ""}
      ${escapeHtml(chip.label)}
    </span>
  `;
}

function renderCraftWorkspaceButton(craftId) {
  return `
    <button
      class="drawer-icon-button"
      data-action="open-bundle"
      data-craft-id="${escapeHtml(craftId)}"
      type="button"
      aria-label="${escapeHtml(CRAFT_WORKSPACE_BUTTON_LABEL)}"
      title="${escapeHtml(CRAFT_WORKSPACE_BUTTON_LABEL)}"
    >
      <span class="drawer-icon-glyph" aria-hidden="true">&#9881;</span>
    </button>
  `;
}

function renderCraftOfficialDescription(craft, statusLabel = "") {
  const description = buildCraftOfficialDescriptionPreview(craft, 220);
  return `
    <section class="drawer-log-panel">
      <div class="drawer-log-head">
        <span class="detail-eyebrow">Official description</span>
        ${statusLabel ? `<span class="drawer-log-status">${escapeHtml(statusLabel)}</span>` : ""}
      </div>
      ${
        description.full
          ? description.truncated
            ? `
              <details class="shared-craft-summary-details">
                <summary class="shared-craft-summary-preview">
                  <span class="shared-craft-summary">${escapeHtml(description.preview)}</span>
                  <span class="shared-craft-summary-toggle">More</span>
                </summary>
                <div class="shared-craft-summary shared-craft-summary-full">${escapeHtml(description.full)}</div>
              </details>
            `
            : `<div class="shared-craft-summary">${escapeHtml(description.full)}</div>`
          : `<div class="shared-craft-note">No verified description yet. Use Refine to let the agent define this capability.</div>`
      }
    </section>
  `;
}

function renderCraftPromptEditor(craft, currentPrompt, agentRun = null) {
  const promptChanged = didCraftPromptChange(craft, currentPrompt, {
    countEmptyAsChange: hasExplicitAgentPromptDraft(craft?.id),
  });
  const promptNote = getCraftPromptEditorNote(agentRun, promptChanged);
  const promptNoteIsPendingChange = isCraftPromptEditorNotePendingChange(agentRun, promptChanged);
  return `
    <label class="crafting-request crafting-definition" aria-label="Refine command">
      <div class="detail-eyebrow">Refine command</div>
      <textarea
        data-agent-prompt-field="true"
        data-craft-id="${escapeHtml(craft.id)}"
        rows="3"
        placeholder="Describe the correction, missing behavior, or the error the agent should address next."
      >${escapeHtml(currentPrompt)}</textarea>
      ${
        promptNote
          ? `<div class="crafting-request-note${promptNoteIsPendingChange ? " crafting-request-note-pending" : ""}">${escapeHtml(promptNote)}</div>`
          : ""
      }
    </label>
  `;
}

function renderCraftDrawerToolbar(craft, currentPrompt, agentRun = null, isEditable = true) {
  const refineDisabled =
    !isEditable ||
    !String(currentPrompt || "").trim() ||
    isAgentRunActive(agentRun) ||
    agentRun?.stopRequested === true;
  const showCopyDebugAction = shouldShowAgentRunDebugButton(agentRun);
  const showStopAction = isEditable && isAgentRunActive(agentRun) && Boolean(String(agentRun?.runId || "").trim());
  return `
    <div class="crafting-toolbar">
      <div class="crafting-actions">
        <button
          class="detail-action"
          data-action="start-agent-run"
          data-craft-id="${escapeHtml(craft.id)}"
          type="button"
          ${refineDisabled ? "disabled" : ""}
        >
          ${escapeHtml(getCraftPrimaryActionLabel(craft, currentPrompt, agentRun, isEditable))}
        </button>
        ${
          showCopyDebugAction
            ? `
              <button
                class="detail-secondary"
                data-action="copy-agent-run-debug"
                data-craft-id="${escapeHtml(craft.id)}"
                type="button"
              >
                Copy debug JSON
              </button>
            `
            : ""
        }
        ${
          showStopAction
            ? `
              <button
                class="detail-secondary"
                data-action="stop-agent-run"
                data-craft-id="${escapeHtml(craft.id)}"
                type="button"
                ${agentRun?.stopRequested === true ? "disabled" : ""}
              >
                ${escapeHtml(agentRun?.stopRequested === true ? "Stopping..." : "Stop")}
              </button>
            `
            : ""
        }
      </div>
      <div class="crafting-links crafting-links-utility">
        ${renderCraftWorkspaceButton(craft.id)}
        ${
          isEditable
            ? `<button class="drawer-link drawer-link-danger" data-action="delete-craft" data-craft-id="${escapeHtml(craft.id)}" type="button">Delete</button>`
            : ""
        }
      </div>
    </div>
  `;
}

function renderRemoteCraftDrawer(craft) {
  const owner = getSharedOwnerName(craft);
  const syncMeta = craft?.sync || {};
  const metaItems = [owner, syncMeta.syncedAt ? `synced ${formatTimeAgo(syncMeta.syncedAt)}` : ""].filter(Boolean);

  return `
    <section class="craft-drawer">
      <div class="crafting-status-line">
        ${renderMetaTokens(metaItems)}
      </div>
      <div class="shared-craft-note">Read only until you fork.</div>
      <div class="crafting-toolbar">
        <div class="crafting-actions">
          <button
            class="detail-action"
            data-action="fork-craft"
            data-craft-id="${escapeHtml(craft.id)}"
            type="button"
          >
            Fork
          </button>
        </div>
        <div class="crafting-links crafting-links-utility">
          ${renderCraftWorkspaceButton(craft.id)}
        </div>
      </div>
      ${renderCraftOfficialDescription(craft, owner)}
    </section>
  `;
}

function renderLocalCraftSharePanel(craft) {
  const isShared = craft?.sharing?.enabled === true;
  const canShare = canEnableCraftSharing();
  const shareButtonAction = isShared || canShare ? "toggle-share" : "open-share-settings";
  const shareButtonLabel = isShared ? "Unshare craft" : canShare ? "Share craft" : "Set code to share";
  const shareButtonClass = isShared ? "detail-secondary" : "detail-action";
  const note = isShared && !canShare
    ? "This craft is still marked shared, but the current auto-generated private code keeps this install local only until you switch back to your own share code."
    : isShared
      ? "Shared with peers that use the same signaling server and the same share code."
    : canShare
      ? "Local only right now. Share this craft when another person uses the same signaling server and share code."
      : "Local only. This install still uses an auto-generated private code. Replace it in Settings before sharing a craft.";

  return `
    <section class="use-context">
      <div class="row-between">
        <span class="detail-eyebrow">Share status</span>
        <span class="drawer-log-status">${escapeHtml(isAutoGeneratedSyncToken() ? "Auto private code" : "Custom share code")}</span>
      </div>
      <div class="shared-craft-note">${escapeHtml(note)}</div>
      <div class="shared-craft-note">
        ${escapeHtml(`Session: ${buildPeerSessionLabel()}. Crafts stay local only until you explicitly share them.`)}
      </div>
      <div class="crafting-links">
        <button
          class="${escapeHtml(shareButtonClass)}"
          data-action="${escapeHtml(shareButtonAction)}"
          data-craft-id="${escapeHtml(craft.id)}"
          type="button"
        >
          ${escapeHtml(shareButtonLabel)}
        </button>
        <button class="drawer-link" data-action="open-share-settings" type="button">Share settings</button>
      </div>
    </section>
  `;
}

function renderCraftingDrawer(craft) {
  const agentRun = state.agentRuns[craft.id] || null;
  const isEditable = craftStore?.isEditableCraft?.(craft) !== false;
  const currentPrompt = getCurrentAgentPromptValue(craft.id, craft);
  const showAgentSurface = Boolean(agentRun);

  return `
    <section class="craft-drawer">
      ${renderCraftOfficialDescription(craft)}
      ${renderCraftPromptEditor(craft, currentPrompt, agentRun)}
      ${renderCraftDrawerToolbar(craft, currentPrompt, agentRun, isEditable)}
      ${showAgentSurface ? renderAgentInspector(craft.id, agentRun, { isEditable }) : ""}
    </section>
  `;
}

function renderTrainingRunLine(craft) {
  const run = state.trainingRuns[craft.id] || null;
  const statusText = isTrainingRunActive(run)
    ? run?.phaseLabel || "Local run active"
    : run?.status === "completed"
      ? "Last local run completed"
      : run?.status === "failed"
        ? "Local run failed"
        : "Local run idle";
  const items = [statusText];

  if (Number(run?.adaptTestAcc) > 0) {
    items.push(`adapted ${formatAccuracy(run.adaptTestAcc)}`);
  }

  if (run?.error) {
    items.push(trimText(run.error, 72));
  }

  return `
    <div class="crafting-run-line">
      ${renderMetaTokens(items)}
    </div>
  `;
}

function renderAgentStepList(entries, emptyText = "") {
  const normalizedEntries = (Array.isArray(entries) ? entries : []).filter((entry) =>
    String(entry?.message || "").trim(),
  );
  if (!normalizedEntries.length) {
    return emptyText ? `<div class="agent-log-empty">${escapeHtml(emptyText)}</div>` : "";
  }

  return `
    <div class="agent-log-list agent-progress-step-list">
      ${normalizedEntries
        .map(
          (entry) => `
            <div class="agent-log-entry agent-log-entry-${escapeHtml(entry.level || "info")} ${entry.status ? `agent-log-entry-${escapeHtml(entry.status)}` : ""} ${entry.kind ? `agent-log-entry-${escapeHtml(entry.kind)}` : ""}">
              <span class="agent-log-time">${escapeHtml(entry.time || "Step")}</span>
              <span class="agent-log-content">
                <span class="agent-log-title">${escapeHtml(entry.title || entry.message)}</span>
                ${entry.detail && entry.detail !== entry.title ? `<span class="agent-log-message agent-log-detail">${escapeHtml(entry.detail)}</span>` : ""}
              </span>
            </div>
          `,
        )
        .join("")}
    </div>
  `;
}

function renderAgentProgressOverview(agentRun, progressEntries = []) {
  const overview = getAgentProgressOverview(agentRun, progressEntries);
  if (!overview.stageItems.length) return "";
  return `
    <div class="agent-progress-overview">
      <div class="agent-progress-meter-stack">
        <div class="agent-progress-meter-row">
          <span class="agent-progress-meter-label">Run flow</span>
          <span class="agent-progress-meter-track">
            <span class="agent-progress-meter-fill ${["starting", "running"].includes(getAgentUiState(agentRun)) && overview.stagePercent > 0 ? "agent-progress-meter-fill-running" : ""}" style="width:${escapeHtml(String(overview.stagePercent))}%"></span>
          </span>
          <span class="agent-progress-meter-value">${escapeHtml(overview.flowLabel)}</span>
        </div>
        <div class="agent-progress-meter-row">
          <span class="agent-progress-meter-label">Turn budget</span>
          <span class="agent-progress-meter-track">
            <span class="agent-progress-meter-fill agent-progress-meter-fill-subtle" style="width:${escapeHtml(String(overview.turnPercent))}%"></span>
          </span>
          <span class="agent-progress-meter-value">${escapeHtml(overview.etaLabel || overview.turnLabel)}</span>
        </div>
      </div>
    </div>
  `;
}

function renderAgentLogs(agentRun) {
  return renderAgentStepList(agentRun?.logs, "No agent run started for this craft yet.");
}

function renderAgentQuestions(craftId, agentRun) {
  const questions = normalizeAgentQuestions(agentRun?.questions);
  const craft = findCraft(craftId);
  const currentPrompt = getCurrentAgentPromptValue(craftId, craft);
  const continueLabel = getCraftPrimaryActionLabel(craft, currentPrompt, agentRun, true);
  if (!questions.length) {
    return `<div class="agent-log-empty">No open questions for this craft.</div>`;
  }

  return `
    <div class="agent-question-list">
      ${questions
        .map(
          (entry, index) => `
            <label class="agent-question-card">
              <div class="agent-question-head">
                <span class="detail-eyebrow">Question ${index + 1}</span>
                ${entry.reason ? `<span class="drawer-log-status">${escapeHtml(entry.reason)}</span>` : ""}
              </div>
              <div class="agent-question-text">${escapeHtml(entry.question)}</div>
              <textarea
                class="agent-question-input"
                data-agent-question-answer="true"
                data-craft-id="${escapeHtml(craftId)}"
                data-question-id="${escapeHtml(entry.id)}"
                rows="3"
                placeholder="Add the answer for the next agent run..."
                ${isAgentRunActive(agentRun) ? "disabled" : ""}
              >${escapeHtml(entry.answer || "")}</textarea>
            </label>
          `,
        )
        .join("")}
      ${
        agentRun?.status === "needs_input"
          ? `
            <div class="agent-panel-actions">
              <button
                class="detail-action"
                data-action="start-agent-run"
                data-craft-id="${escapeHtml(craftId)}"
                type="button"
              >
                ${escapeHtml(continueLabel)}
              </button>
            </div>
          `
          : ""
      }
    </div>
  `;
}

function getAgentProgressHeadline(agentRun) {
  const uiState = getAgentUiState(agentRun);
  if (uiState === "starting") return "Starting agent";
  if (uiState === "running") return "Agent running";
  if (uiState === "needs_input") return "Needs input";
  if (uiState === "blocked") return "Agent blocked";
  if (uiState === "failed") return "Agent failed";
  if (uiState === "stopped_limited") return "Agent stopped with limitation";
  if (uiState === "stopped") return "Agent stopped";
  if (!agentRun) return "Idle";
  return "Agent";
}

function getAgentProgressPrimaryText(agentRun, report) {
  if (agentRun?.status === "blocked") {
    return String(agentRun?.error || report?.currentState || "The agent run is blocked.");
  }
  if (agentRun?.status === "failed") {
    return String(agentRun?.error || report?.currentState || "The agent run failed.");
  }
  const currentState = String(report?.currentState || "").trim();
  if (currentState) return currentState;
  const responseText = String(agentRun?.responseText || "").trim();
  if (responseText) return responseText;
  const lastLog = String(getLastAgentLogMessage(agentRun) || "").trim();
  if (lastLog) return lastLog;
  if (agentRun?.status === "starting") return "Waiting for the background agent to acknowledge the run.";
  if (agentRun?.status === "blocked") return String(agentRun?.error || "The agent run is blocked.");
  if (agentRun?.status === "failed") return String(agentRun?.error || "The agent run failed.");
  if (agentRun?.status === "needs_input") return "The agent needs additional answers before it can continue.";
  return "No progress has been recorded yet.";
}

function getAgentProgressSecondaryText(agentRun, report, primaryText) {
  const nextAction = String(report?.nextAction || "").trim();
  if (nextAction && nextAction !== primaryText) return nextAction;
  if (agentRun?.status === "needs_input") return "Answer the open questions to continue the same run.";
  if (getAgentUiState(agentRun) === "blocked") return "Fix the blocker or narrow the capability, then restart the run.";
  if (agentRun?.status === "failed") return "Restart the agent after adjusting the capability or adding more context.";
  if (getAgentUiState(agentRun) === "stopped_limited") {
    return "The run stopped because part of the requested behavior is not covered by the approved tools.";
  }
  if (getAgentUiState(agentRun) === "stopped") {
    return "Start another run if you want the agent to keep refining from the current craft state.";
  }
  return "";
}

function buildAgentProgressEntries(agentRun, report) {
  const logs = (Array.isArray(agentRun?.logs) ? agentRun.logs : []).filter((entry) =>
    String(entry?.message || "").trim(),
  );
  if (logs.length) {
    return buildAgentProgressEntriesFromLogs(logs, agentRun);
  }

  const fallbackEntries = [];
  const currentState = String(report?.currentState || agentRun?.responseText || agentRun?.error || "").trim();
  if (currentState) {
    fallbackEntries.push({
      level: agentRun?.status === "failed" ? "error" : "info",
      status: agentRun?.status === "failed" ? "error" : "done",
      kind: "note",
      time: "State",
      title: "Current state",
      detail: currentState,
      message: currentState,
    });
  }
  const nextAction = String(report?.nextAction || "").trim();
  if (nextAction && nextAction !== currentState) {
    fallbackEntries.push({
      level: agentRun?.status === "failed" ? "warn" : "info",
      status: agentRun?.status === "failed" ? "warn" : "done",
      kind: "note",
      time: "Next",
      title: "Next step",
      detail: nextAction,
      message: nextAction,
    });
  }
  return fallbackEntries;
}

function getAgentProgressMetaText(agentRun, progressEntries = [], craft = null) {
  const parts = [];
  const maturity = getCraftMaturity(craft, agentRun);
  const stepCount = Math.max(0, Number(progressEntries.length) || 0);
  const turnsUsed = Math.max(0, Number(agentRun?.turnsUsed || 0));
  const maxTurns = Math.max(0, Number(agentRun?.maxTurns || 0));
  parts.push(
    maturity.isExplicit
      ? `Agent-reported maturity ${formatCraftMaturityPercent(maturity.percent)} (${formatCraftMaturityPhase(maturity.phase)})`
      : "Agent-reported maturity not set yet",
  );
  if (stepCount > 0) {
    parts.push(`${formatCount(stepCount)} steps`);
  }
  if (maxTurns > 0) {
    parts.push(`${formatCount(turnsUsed)} / ${formatCount(maxTurns)} turns`);
  }
  return parts.join(" · ");
}

function isAgentProvenanceNoise(entry) {
  const haystack = `${String(entry?.title || "")} ${String(entry?.detail || "")}`.toLowerCase();
  if (!haystack.trim()) return true;
  if (String(entry?.kind || "").toLowerCase() === "constraint") return true;
  if (/pinned supervisor tools|supervisor-tools fixiert|lokale trainingsdaten gelesen/.test(haystack)) {
    return true;
  }
  if (/browser tool /.test(haystack) || /artifact store/.test(haystack)) {
    return true;
  }
  if (/user clarification/.test(haystack)) {
    return true;
  }
  return false;
}

function isAgentProvenanceMilestone(entry) {
  const haystack = `${String(entry?.title || "")} ${String(entry?.detail || "")}`.toLowerCase();
  if (!haystack.trim()) return false;
  if (isAgentProvenanceNoise(entry)) return false;
  if (
    /reviewed starter row|starter row|freigegebener selection-flow erkannt|reviewed pattern|capability priority|leerer ausgangsdatensatz|empty dataset|runtime limitation|priority order|currentsamples is an empty array|brief includes a complete agentic sample|reviewed contract covers|active-text contract|read_active_text_target|replace_active_text_target/.test(
      haystack,
    )
  ) {
    return false;
  }
  const kind = String(entry?.kind || "").toLowerCase();
  if (kind === "match") {
    return /(umbenannt|benannt|festgelegt|ausgewaehlt|verifiziert|bestaetigt|browser|tab|seite|eingabefeld|fokussiert|markiert|hinzugefuegt|aktualisiert|entfernt)/.test(
      haystack,
    );
  }
  if (kind === "sample") {
    return /(browser|tab|seite|sichtbar|verifiziert|belegt|eingabefeld|fokussiert|markiert|ausgewaehlt)/.test(
      haystack,
    );
  }
  return true;
}

function buildAgentProvenanceFingerprint(entry) {
  return `${String(entry?.kind || "").toLowerCase()}\n${String(entry?.title || "").trim().toLowerCase()}\n${String(entry?.detail || "").trim().toLowerCase()}`;
}

function buildAgentOperationHighlight(agentRun) {
  const operations = Array.isArray(agentRun?.operations) ? agentRun.operations : [];
  if (!operations.length) return [];
  return [
    {
      id: "ops-summary",
      title: describeAgentOperationMilestone(operations),
      detail: describeAgentOperationMilestoneDetail(operations),
      kind: "operation",
      sampleId: "",
      operationType: operations.length === 1 ? String(operations[0]?.type || "") : "",
    },
  ];
}

function buildVisibleAgentProvenance(agentRun) {
  const normalizedEntries = normalizeAgentProvenance(agentRun?.provenance).filter(isAgentProvenanceMilestone);
  const hasOperationMilestone = normalizedEntries.some((entry) => String(entry?.kind || "").toLowerCase() === "operation");
  const entries = hasOperationMilestone ? [] : [...buildAgentOperationHighlight(agentRun)];
  const seen = new Set(entries.map((entry) => buildAgentProvenanceFingerprint(entry)));
  let seenOperationMilestone = entries.some((entry) => String(entry?.kind || "").toLowerCase() === "operation");

  for (const entry of normalizedEntries) {
    const isOperationMilestone = String(entry?.kind || "").toLowerCase() === "operation";
    if (seenOperationMilestone && isOperationMilestone) continue;
    const key = buildAgentProvenanceFingerprint(entry);
    if (seen.has(key)) continue;
    seen.add(key);
    entries.push(entry);
    if (isOperationMilestone) {
      seenOperationMilestone = true;
    }
    if (entries.length >= 8) break;
  }

  return entries.slice(0, 8);
}

function renderAgentProvenance(agentRun) {
  const provenance = buildVisibleAgentProvenance(agentRun);
  if (!provenance.length) {
    return `<div class="agent-log-empty">No milestone highlights have been recorded for this run yet.</div>`;
  }

  return `
    <div class="agent-provenance-list">
      ${provenance
        .map(
          (entry) => `
            <article class="agent-provenance-card">
              <div class="agent-provenance-head">
                <span class="detail-eyebrow">${escapeHtml(entry.kind || "match")}</span>
                ${entry.operationType ? `<span class="drawer-log-status">${escapeHtml(entry.operationType)}</span>` : ""}
              </div>
              <div class="agent-provenance-title">${escapeHtml(entry.title)}</div>
              <div class="agent-provenance-detail">${escapeHtml(entry.detail || "No detail recorded.")}</div>
              ${entry.sampleId ? `<div class="agent-provenance-meta">Sample: ${escapeHtml(entry.sampleId)}</div>` : ""}
            </article>
          `,
        )
        .join("")}
    </div>
  `;
}

function renderAgentProgress(craftId, agentRun, { isEditable = true } = {}) {
  const craft = findCraft(craftId);
  const report = normalizeAgentReport(agentRun?.report, {
    currentState: agentRun?.responseText || "No progress report yet.",
  });
  const headline = getAgentProgressHeadline(agentRun);
  const primaryText = getAgentProgressPrimaryText(agentRun, report);
  const secondaryText = getAgentProgressSecondaryText(agentRun, report, primaryText);
  const progressEntries = buildAgentProgressEntries(agentRun, report);
  const metaText = getAgentProgressMetaText(agentRun, progressEntries, craft);
  const clipboardStatus = String(agentRun?.clipboardStatus || "idle");
  const clipboardError = String(agentRun?.clipboardError || "");
  const clipboardMessage =
    clipboardStatus === "copied"
      ? "Debug JSON copied."
      : clipboardStatus === "copy_failed"
        ? (clipboardError || "Clipboard write failed.")
        : "";

  return `
    <div class="agent-progress-view">
      <div class="agent-progress-status">${escapeHtml(headline)}</div>
      ${metaText ? `<div class="agent-progress-meta">${escapeHtml(metaText)}</div>` : ""}
      ${renderAgentProgressOverview(agentRun, progressEntries)}
      <div class="agent-progress-copy">${escapeHtml(primaryText)}</div>
      ${secondaryText ? `<div class="agent-progress-next">${escapeHtml(secondaryText)}</div>` : ""}
      ${renderAgentStepList(progressEntries)}
      ${clipboardMessage ? `<div class="agent-progress-next${clipboardStatus === "copy_failed" ? " agent-progress-next-error" : ""}">${escapeHtml(clipboardMessage)}</div>` : ""}
    </div>
  `;
}

function countAgentProgressEntries(agentRun) {
  const report = normalizeAgentReport(agentRun?.report, {
    currentState: agentRun?.responseText || "",
  });
  return buildAgentProgressEntries(agentRun, report).length;
}

function renderAgentTabCount(count) {
  const safeCount = Math.max(0, Number(count) || 0);
  return `<span class="agent-tab-count">(${escapeHtml(String(safeCount))})</span>`;
}

function renderAgentInspector(craftId, agentRun, { isEditable = true } = {}) {
  const activeTab = ["progress", "questions", "provenance"].includes(String(agentRun?.activeTab || ""))
    ? String(agentRun.activeTab)
    : "progress";
  const progressCount = countAgentProgressEntries(agentRun);
  const questionCount = countOpenAgentQuestions(agentRun);
  const provenanceCount = buildVisibleAgentProvenance(agentRun).length;

  return `
    <section class="drawer-log-panel agent-inspector-panel">
      <div class="agent-tab-row" role="tablist" aria-label="Agent views">
        <button
          class="agent-tab-button ${activeTab === "progress" ? "agent-tab-button-active" : ""}"
          data-action="set-agent-tab"
          data-craft-id="${escapeHtml(craftId)}"
          data-agent-tab="progress"
          type="button"
        >
          Progress
          ${renderAgentTabCount(progressCount)}
        </button>
        <button
          class="agent-tab-button ${activeTab === "questions" ? "agent-tab-button-active" : ""}"
          data-action="set-agent-tab"
          data-craft-id="${escapeHtml(craftId)}"
          data-agent-tab="questions"
          type="button"
        >
          Questions
          ${renderAgentTabCount(questionCount)}
        </button>
        <button
          class="agent-tab-button ${activeTab === "provenance" ? "agent-tab-button-active" : ""}"
          data-action="set-agent-tab"
          data-craft-id="${escapeHtml(craftId)}"
          data-agent-tab="provenance"
          type="button"
        >
          Provenance
          ${renderAgentTabCount(provenanceCount)}
        </button>
      </div>
      <div
        class="drawer-log-scroll"
        data-preserve-scroll-key="agent-inspector:${escapeHtml(craftId)}:${escapeHtml(activeTab)}"
        ${activeTab === "progress" ? 'data-scroll-anchor="bottom" data-scroll-start="bottom"' : ""}
      >
        ${
          activeTab === "questions"
            ? renderAgentQuestions(craftId, agentRun)
            : activeTab === "provenance"
              ? renderAgentProvenance(agentRun)
              : renderAgentProgress(craftId, agentRun, { isEditable })
        }
      </div>
    </section>
  `;
}

function renderHoverExamplePopover(examples) {
  const normalizedExamples = normalizePromptExampleTexts(examples);
  if (!normalizedExamples.length) return "";
  const exampleCount = Math.max(1, normalizedExamples.length);
  return `
    <div class="prompt-hover-popover" role="note">
      ${
        normalizedExamples.length > 1
          ? `
            <div class="prompt-hover-rotator" style="--example-count:${exampleCount};">
              ${normalizedExamples
                .map(
                  (example, index) => `
                    <div class="prompt-hover-example prompt-hover-example-rotating" style="--example-index:${index};">
                      ${escapeHtml(example)}
                    </div>
                  `,
                )
                .join("")}
            </div>
          `
          : `<div class="prompt-hover-example">${escapeHtml(normalizedExamples[0] || "No example available.")}</div>`
      }
    </div>
  `;
}

function renderStarterTutorialExampleCards() {
  const activePrompt = String(state.draftGoal || "").trim();
  return STARTER_TUTORIAL_EXAMPLES.map((example, index) => {
    const active = activePrompt === example.prompt;
    return `
      <article class="tutorial-example-card ${active ? "tutorial-example-card-active" : ""}">
        <div class="tutorial-example-head">
          <div>
            <div class="tutorial-example-title">${escapeHtml(example.title)}</div>
            <div class="tutorial-example-meta">${active ? "Currently inserted into the input below." : "Starter prompt"}</div>
          </div>
          <button
            class="workspace-secondary tutorial-example-button"
            data-action="tutorial-use-create-example"
            data-example-index="${index}"
            type="button"
          >
            Use example
          </button>
        </div>
        <div class="tutorial-example-copy">${escapeHtml(example.prompt)}</div>
      </article>
    `;
  }).join("");
}

function renderAddRow() {
  return `
    <button
      class="add-row"
      data-action="toggle-create"
      type="button"
      aria-label="Create a new capability"
      title="Create a new capability"
      data-hover-label="Create a new capability"
    >
      <span class="add-plus">+</span>
    </button>
  `;
}

function renderComposerRow() {
  const description = String(state.draftGoal || "");
  return `
    <section class="composer-row" aria-label="New capability">
      <label class="composer-definition">
        <span class="detail-eyebrow">Describe the capability</span>
        <textarea
          class="composer-definition-input"
          data-draft-field="goal"
          rows="3"
          placeholder="Describe what the capability should do in the browser."
        >${escapeHtml(description)}</textarea>
      </label>
      <div class="composer-actions">
        <button class="composer-action" data-action="create-craft" type="button" ${String(description).trim() ? "" : "disabled"}>
          Create
        </button>
        <button class="composer-cancel" data-action="cancel-create" type="button" aria-label="Close">×</button>
      </div>
    </section>
  `;
}

async function createCraftFromDraft() {
  const description = String(state.draftGoal || "").trim();
  if (!description) return;
  const explicitName = normalizeCraftNameCandidate(state.draftName);
  const name = explicitName || createDefaultAbilityName();
  const starterModelName = getDefaultVanillaModelName();
  const goal = description;

  const now = new Date().toISOString();
  const craft = craftStore?.normalizeCraft?.(
    {
      id: ensureUniqueCraftId(name),
      name,
      nameSource: explicitName ? "user" : "placeholder",
      summary: "",
      accuracy: null,
      tools: [],
      tooling: { ready: 0, total: 0 },
      useStatus: "vanilla",
      inputMode: "free_text",
      inputHint: `Uses ${starterModelName} until you open Craft.`,
      inputExamples: [
        "Summarize the selected paragraph in three bullets.",
        "Rewrite this message to be shorter and clearer.",
        "Extract the key facts as JSON.",
      ],
      actionLabel: "Craft",
      tokenSpend: 0,
      costUsd: 0,
      inputPlaceholder: "Describe the capability.",
      stage: "Vanilla",
      targetSlot: "target",
      accuracyFloor: "",
      augmentationMode: "",
      seedRows: 0,
      datasetRows: 0,
      openGaps: null,
      agentPrompt: goal,
      metricsReady: false,
      starterMode: "vanilla_target",
      starterModelName,
      coverageGaps: [],
      navigatorRecords: [],
      training: createDefaultCraftTrainingConfig({
        targetSlot: "target",
        starterModelName,
      }),
      createdAt: now,
      updatedAt: now,
    },
    state.crafts.length,
  ) || null;
  if (!craft) return;

  const previousCrafts = [...state.crafts];
  const previousActiveCraftId = state.activeCraftId;
  const previousCraftingCraftId = state.craftingCraftId;
  const previousTutorialOverlay = state.tutorialOverlay;

  state.crafts = [...state.crafts, craft];
  state.activeCraftId = craft.id;
  state.craftingCraftId = craft.id;
  state.agentRuns[craft.id] = createPendingCraftingRun(goal, {}, craft);
  scheduleAgentRunPersist(craft.id);
  state.tutorialOverlay = null;
  resetDraft();
  render();

  try {
    state.crafts = await persistCrafts([...previousCrafts, craft]);
    state.activeCraftId = craft.id;
    state.craftingCraftId = craft.id;
    render();
    await startAgentRun(craft.id);
  } catch (error) {
    state.crafts = previousCrafts;
    state.activeCraftId = previousActiveCraftId;
    state.craftingCraftId = previousCraftingCraftId;
    delete state.agentRuns[craft.id];
    void deleteAgentRunState(craft.id).catch(() => {});
    state.tutorialOverlay = previousTutorialOverlay;
    state.smokeTest = {
      ...createEmptySmokeTestState(),
      status: "error",
      message: "The starter craft could not be created.",
      detail: error instanceof Error ? error.message : String(error || "Unknown create error."),
      updatedAt: formatClock(new Date()),
    };
    render();
  }
}

function resetDraft() {
  state.createOpen = false;
  state.draftName = "";
  state.draftGoal = "";
}

function normalizeCraftNameCandidate(value, max = 80) {
  const text = String(value == null ? "" : value)
    .replace(/\s+/g, " ")
    .replace(/^["'`]+|["'`]+$/g, "")
    .replace(/[.?!,:;]+$/g, "")
    .trim();
  if (!text) return "";
  if (text.length <= max) return text;
  return text.slice(0, max).trimEnd();
}

function normalizeNameComparison(value) {
  return String(value == null ? "" : value)
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
  const value = String(name || "").trim();
  return /^new capability(?: \d+)?$/i.test(value) || /^craft \d+$/i.test(value);
}

function ensureUniqueCraftName(name, excludedCraftId = "") {
  const baseName = normalizeCraftNameCandidate(name) || "New Capability";
  const existingNames = new Set(
    state.crafts
      .filter((craft) => String(craft?.id || "").trim() !== String(excludedCraftId || "").trim())
      .map((craft) => String(craft?.name || "").trim())
      .filter(Boolean),
  );
  if (!existingNames.has(baseName)) return baseName;

  let counter = 2;
  while (existingNames.has(`${baseName} ${counter}`)) {
    counter += 1;
  }
  return `${baseName} ${counter}`;
}

function createDefaultAbilityName() {
  return ensureUniqueCraftName("New Capability");
}

function createDefaultCraftTrainingConfig(craft = {}) {
  const modelName =
    String(craft?.starterModelName || getDefaultVanillaModelName()).trim() || getDefaultVanillaModelName();
  return {
    defaultShardId: "default-local-run",
    shards: [
      {
        id: "default-local-run",
        label: "Configured target model",
        modelName,
        slotId: String(craft?.targetSlot || "target"),
      },
    ],
  };
}

function getTrainingConfig(craft) {
  if (craft?.training && typeof craft.training === "object") return craft.training;
  if (craftStore?.isEditableCraft?.(craft) === false) return null;
  return createDefaultCraftTrainingConfig(craft);
}

function getSelectedTrainingShard(craft) {
  const training = getTrainingConfig(craft);
  const shards = Array.isArray(training?.shards) ? training.shards : [];
  if (!shards.length) return null;
  const selectedId = String(training?.defaultShardId || shards[0].id);
  return shards.find((shard) => shard.id === selectedId) || shards[0];
}

function findCraft(craftId) {
  return state.crafts.find((craft) => craft.id === craftId) || null;
}

async function listLocalArtifactsForCraft(craftId) {
  const key = String(craftId || "").trim();
  if (!key) return [];
  const synced = await callCraftSyncWithTimeout(
    "listLocalArtifacts",
    [{ craftId: key }],
    null,
    `Listing local artifacts for ${key} timed out.`,
    LOCAL_STORAGE_TIMEOUT_MS,
    { throwOnFailure: true },
  );
  if (Array.isArray(synced)) {
    return synced;
  }
  return [];
}

async function deleteLocalArtifactById(artifactId) {
  const key = String(artifactId || "").trim();
  if (!key) return;
  if (craftSync?.deleteLocalArtifact) {
    await callCraftSyncWithTimeout(
      "deleteLocalArtifact",
      [key],
      null,
      `Deleting local artifact ${key} timed out.`,
      LOCAL_STORAGE_DELETE_TIMEOUT_MS,
      { throwOnFailure: true, retryTimeoutMs: LOCAL_STORAGE_DELETE_TIMEOUT_MS },
    );
  }
}

async function deleteCraftArtifacts(craftId) {
  const key = String(craftId || "").trim();
  if (!key) return;

  const artifactIds = new Set(
    (await listLocalArtifactsForCraft(key))
      .map((record) => String(record?.id || "").trim())
      .filter(Boolean),
  );

  [
    getBundleTrainingDataArtifactId(key),
    getToolScriptsArtifactId(key),
    getBrowserCapabilityBundleArtifactId(key),
    getWeightsArtifactId(key),
    getPolicyBundleArtifactId(key),
  ].forEach((artifactId) => {
    if (artifactId) artifactIds.add(artifactId);
  });

  for (const artifactId of artifactIds) {
    await deleteLocalArtifactById(artifactId);
  }
}

async function deleteCraft(craftId) {
  const key = String(craftId || "").trim();
  if (!key) return;
  const craft = findCraft(key);
  if (!craft || craftStore?.isEditableCraft?.(craft) === false) return;

  const craftName = String(craft.name || "this craft").trim() || "this craft";
  const confirmed = globalThis.confirm?.(`Delete "${craftName}"?`) !== false;
  if (!confirmed) return;

  stopAgentRunPoll(key);
  agentRunPersistTimers.forEach((timer, timerCraftId) => {
    if (timerCraftId === key) {
      globalThis.clearTimeout(timer);
      agentRunPersistTimers.delete(timerCraftId);
    }
  });

  delete state.agentRuns[key];
  delete state.trainingRuns[key];
  delete state.promptDrafts[key];
  delete state.agentPromptDrafts[key];
  delete state.useMessages[key];
  delete state.craftResponses[key];
  delete state.trainingDataStates[key];
  delete state.toolScriptStates[key];
  delete state.capabilityWeightsStates[key];

  if (state.trainingDataCraftId === key) {
    state.trainingDataCraftId = null;
  }
  if (state.activeCraftId === key) {
    state.activeCraftId = null;
  }
  if (state.craftingCraftId === key) {
    state.craftingCraftId = null;
  }

  try {
    await deleteCraftArtifacts(key);
    await deleteAgentRunState(key);
    state.crafts = await persistCrafts(state.crafts.filter((entry) => String(entry?.id || "").trim() !== key));
    reconcileSidepanelStateWithCrafts();
  } catch (error) {
    state.smokeTest = {
      ...createEmptySmokeTestState(),
      status: "error",
      message: "The craft could not be deleted.",
      detail: error instanceof Error ? error.message : String(error || "Unknown delete error."),
      updatedAt: formatClock(new Date()),
    };
  }

  render();
}

async function refreshCraftsState() {
  state.crafts = (await craftStore?.readCrafts?.()) || [];
  reconcileSidepanelStateWithCrafts();
  for (const craft of state.crafts) {
    void loadCapabilityWeightsState(craft?.id);
  }
}

async function refreshUiPreferences() {
  state.uiPreferences = await readUiPreferences(craftSync);
}

async function persistCrafts(crafts) {
  return (await craftStore?.writeCrafts?.(crafts)) || [];
}

function buildAgentRunBrief(craftId, prompt) {
  const basePrompt = String(prompt || "").trim();
  const previousRun = state.agentRuns[craftId] || null;
  const answeredQuestions = getAnsweredAgentQuestions(previousRun);
  if (!previousRun || !answeredQuestions.length) {
    return basePrompt;
  }

  const answeredSection = answeredQuestions
    .map((entry) => `- Question: ${entry.question}\n  Answer: ${String(entry.answer || "").trim()}`)
    .join("\n");

  return [
    basePrompt,
    previousRun?.report?.nextAction ? `Last agent step: ${previousRun.report.nextAction}` : "",
    "Use these answered questions in the next plan:",
    answeredSection,
  ]
    .filter(Boolean)
    .join("\n\n");
}

function createPendingCraftingRun(prompt, overrides = {}, craft = null) {
  const run = {
    runId: "",
    status: "starting",
    phase: "starting",
    mode: "remote",
    modelRef: "",
    responseText: "",
    finalStatus: "",
    tokens: 0,
    costUsd: 0,
    error: "",
    report: normalizeAgentReport(null, {
      objective: trimText(prompt, 180),
      currentState: "Loading local training context.",
      nextAction: "",
    }),
    questions: [],
    provenance: [],
    suggestedName: "",
    officialDescription: null,
    activeTab: "progress",
    logs: [],
    operations: [],
    useSurface: null,
    turnsUsed: 0,
    maxTurns: 0,
    errorDetail: null,
    toolTrace: [],
    lastToolFailure: null,
    workspaceCodeDive: null,
    clipboardStatus: "idle",
    clipboardError: "",
    activityRecorded: false,
    useSurfaceApplied: false,
    officialDescriptionApplied: false,
    completedAt: "",
    ...overrides,
  };
  run.maturity = getCraftMaturity(craft, run);
  return run;
}

function normalizeCraftInputMode(value, fallback = "free_text") {
  const candidate = String(value || "").trim().toLowerCase();
  if (!candidate) return fallback;
  if (["action_only", "execution_only", "none", "no_input"].includes(candidate)) return "context_only";
  return CRAFT_INPUT_MODES.includes(candidate) ? candidate : fallback;
}

function craftUsesTextInputFromMode(mode) {
  return mode === "free_text" || mode === "mixed";
}

function craftUsesTextInput(craft) {
  return craftUsesTextInputFromMode(getCraftInputMode(craft));
}

function detectPromptIntent(prompt = "") {
  const text = String(prompt || "").trim().toLowerCase();
  return {
    translation: /(uebersetz|übersetz|translate|translation)/i.test(text),
    correction: /(rechtschreib|grammatik|stil|korrigier|korrektur)/i.test(text),
    extraction: /(extrah|json|struktur|felder|schema)/i.test(text),
    browserSelection: /(markiert|markierten|ausgew[aä]hlt|selection|selected text|eingabefeld|zwischenablage|clipboard|inplace)/i.test(text),
    currentTab: /(current tab|aktuell(?:e|en)? (?:tab|seite)|browserseite|geoeffnete seite|geöffnete seite)/i.test(text),
    noExtraInput: /(ohne (?:weitere|zus[aä]tzliche) (?:texteingabe|eingabe)|nur ausf(?:u|ü)hren|direkt im browser)/i.test(text),
  };
}

function collectLoadedTrainingPromptExamples(craftId, limit = 6) {
  const key = String(craftId || "").trim();
  const trainingState = key ? state.trainingDataStates[key] : null;
  const samples = Array.isArray(trainingState?.samples) ? trainingState.samples : [];
  const prioritized = [
    ...samples.filter((sample) => sample.status === "ready"),
    ...samples.filter((sample) => sample.status === "review"),
    ...samples.filter((sample) => sample.status === "draft"),
    ...samples.filter((sample) => sample.status === "blocked"),
  ];
  return normalizePromptExampleTexts(prioritized.map((sample) => sample.promptText), limit);
}

function deriveDefaultCraftInputExamples(craft, mode = getCraftInputMode(craft), prompt = "") {
  const intent = detectPromptIntent(prompt || craft?.agentPrompt || craft?.summary || craft?.name || "");
  if (!craftUsesTextInputFromMode(mode)) {
    if (intent.translation) {
      return [
        "Translate the selected text directly in the browser.",
        "Convert the selected paragraph in place to English.",
        "Replace the active input field with the translated version.",
      ];
    }
    if (intent.correction) {
      return [
        "Correct the selected text directly and replace it.",
        "Smooth the style of the selected paragraph.",
        "Overwrite the active input field with the improved version.",
      ];
    }
    return [
      "Revise the selected passage directly in the browser.",
      "Apply the current context without extra text input.",
      "Run the capability directly against the selection or focus target.",
    ];
  }

  if (intent.translation) {
    return [
      "Translate this paragraph into clear English.",
      "Rewrite this message in German.",
      "Rewrite the selected text in plain language.",
    ];
  }
  if (intent.correction) {
    return [
      "Correct spelling and grammar in this paragraph.",
      "Make this email shorter and friendlier.",
      "Improve style and clarity while preserving the meaning.",
    ];
  }
  if (intent.extraction) {
    return [
      "Extrahiere die wichtigsten Fakten als JSON.",
      "Ordne diesen Text in die erwarteten Felder ein.",
      "Gib nur die relevanten Daten in stabiler JSON-Struktur aus.",
    ];
  }
  return [
    "Summarize the content in three bullet points.",
    "Rewrite the text more clearly and concisely.",
    "Map the input into the defined structure.",
  ];
}

function getCraftInputExampleBundle(craft) {
  const fromCraft = normalizePromptExampleTexts(craft?.inputExamples);
  if (fromCraft.length) {
    return {
      examples: fromCraft,
      sourceLabel: "from capability definition",
    };
  }

  const fromTraining = collectLoadedTrainingPromptExamples(craft?.id);
  if (fromTraining.length) {
    return {
      examples: fromTraining,
      sourceLabel: "from training data",
    };
  }

  return {
    examples: normalizePromptExampleTexts(deriveDefaultCraftInputExamples(craft)),
    sourceLabel: "starter examples until training data exists",
  };
}

function getCraftingAgentPromptExamples(craft) {
  return normalizePromptExampleTexts([
    String(craft?.agentPrompt || "").trim(),
    "Replace the selected browser text, focused input, or clipboard text with a version that fixes spelling and grammar.",
    "Build a capability that translates selected browser text directly into English without extra text input.",
    "Derive 12 seed-grounded prompt-to-JSON samples for training from this task.",
    "Switch the capability to run on browser selection so it no longer needs free text input.",
  ]);
}

function deriveUseSurfaceInputMode(rawUseSurface, craft, prompt) {
  const currentMode = getCraftInputMode(craft);
  const explicit = normalizeCraftInputMode(rawUseSurface?.inputMode, "");
  if (explicit) return explicit;
  const intent = detectPromptIntent(prompt);
  if (intent.browserSelection || intent.noExtraInput) return "selection";
  if (intent.currentTab) return "current_tab";
  return currentMode;
}

function deriveUseSurfaceActionLabel(rawUseSurface, craft, prompt, inputMode) {
  const explicitLabel = String(rawUseSurface?.actionLabel || "").trim();
  if (explicitLabel) return explicitLabel;
  const intent = detectPromptIntent(prompt);
  if (intent.translation) return "Translate";
  if (intent.correction) return "Correct";
  if (intent.extraction) return "Extract";
  if (!craftUsesTextInputFromMode(inputMode)) return "Run";
  return getCraftActionLabel(craft);
}

function deriveUseSurfaceHint(rawUseSurface, craft, inputMode) {
  const explicitHint = String(rawUseSurface?.inputHint || "").trim();
  if (explicitHint) return explicitHint;
  if (inputMode === "selection") return "Uses the current selection or the focused field. No extra prompt field is needed.";
  if (inputMode === "current_tab") return "Uses the current tab. No extra prompt field is needed.";
  if (inputMode === "context_only") return "Runs on the current craft context. No extra prompt field is needed.";
  if (inputMode === "mixed") return "Uses the current context and optional text input.";
  return "Uses text input.";
}

function deriveUseSurfacePlaceholder(rawUseSurface, craft, prompt, inputMode) {
  if (!craftUsesTextInputFromMode(inputMode)) return "";
  const explicit = String(rawUseSurface?.inputPlaceholder || "").trim();
  if (explicit) return explicit;
  const existing = String(craft?.inputPlaceholder || "").trim();
  if (existing) return existing;
  const intent = detectPromptIntent(prompt);
  if (intent.translation) return "What should be translated or rewritten?";
  if (intent.correction) return "What text should the capability improve?";
  if (intent.extraction) return "What content should be turned into the target structure?";
  return "Describe what the capability should do.";
}

function deriveUseSurfaceExamples(rawUseSurface, craft, prompt, inputMode) {
  const explicit = normalizePromptExampleTexts(rawUseSurface?.inputExamples);
  if (explicit.length) return explicit;
  const existing = normalizePromptExampleTexts(craft?.inputExamples);
  if (existing.length) return existing;
  const fromTraining = collectLoadedTrainingPromptExamples(craft?.id);
  if (fromTraining.length) return fromTraining;
  return normalizePromptExampleTexts(deriveDefaultCraftInputExamples(craft, inputMode, prompt));
}

function normalizeAgentUseSurfaceSuggestion(rawUseSurface, craft, prompt) {
  const inputMode = deriveUseSurfaceInputMode(rawUseSurface, craft, prompt);
  return {
    inputMode,
    inputHint: deriveUseSurfaceHint(rawUseSurface, craft, inputMode),
    inputPlaceholder: deriveUseSurfacePlaceholder(rawUseSurface, craft, prompt, inputMode),
    actionLabel: deriveUseSurfaceActionLabel(rawUseSurface, craft, prompt, inputMode),
    inputExamples: deriveUseSurfaceExamples(rawUseSurface, craft, prompt, inputMode),
  };
}

function hasExplicitAgentPromptDraft(craftId) {
  const key = String(craftId || "").trim();
  if (!key) return false;
  return Object.prototype.hasOwnProperty.call(state.agentPromptDrafts, key);
}

function getPersistedCraftPrompt(craft) {
  return String(craft?.agentPrompt || "").trim();
}

function getCurrentAgentPromptValue(craftId, craft) {
  const key = String(craftId || craft?.id || "").trim();
  if (key && hasExplicitAgentPromptDraft(key)) {
    return String(state.agentPromptDrafts[key] || "");
  }
  return String(craft?.agentPrompt || "");
}

function classifyAgentPromptRevision(previousPrompt, nextPrompt) {
  const previousText = String(previousPrompt || "").trim();
  const nextText = String(nextPrompt || "").trim();
  if (!previousText && !nextText) return "unchanged";
  if (!nextText) return "cleared";
  if (!previousText) return "new";
  if (previousText === nextText) return "unchanged";
  const previousLower = previousText.toLowerCase();
  const nextLower = nextText.toLowerCase();
  if (nextLower.includes(previousLower)) return "extended";
  if (previousLower.includes(nextLower)) return "trimmed";
  return "rewritten";
}

function buildAgentPromptRevisionContext(craft, prompt, { hasExplicitDraft = false } = {}) {
  const previousBrief = getPersistedCraftPrompt(craft);
  const latestBrief = String(prompt || "").trim();
  const changeKind = classifyAgentPromptRevision(previousBrief, latestBrief);
  return {
    hasExplicitDraft,
    previousBrief,
    latestBrief,
    changeKind,
    userMayBeUnsatisfied: hasExplicitDraft && changeKind !== "unchanged",
    latestBriefIsAuthoritative: hasExplicitDraft,
  };
}

function getCraftOfficialDescription(craft) {
  return String(craft?.summary || "").trim();
}

function buildCraftOfficialDescriptionPreview(craft, maxLength = 180) {
  const full = getCraftOfficialDescription(craft);
  if (!full) {
    return {
      full: "",
      preview: "",
      truncated: false,
    };
  }
  const preview = trimText(full, maxLength);
  return {
    full,
    preview,
    truncated: preview !== full,
  };
}

function didCraftPromptChange(craft, prompt, { countEmptyAsChange = false } = {}) {
  const nextPrompt = String(prompt || "").trim();
  if (!nextPrompt) {
    return countEmptyAsChange && Boolean(getPersistedCraftPrompt(craft));
  }
  const persistedPrompt = getPersistedCraftPrompt(craft);
  if (!persistedPrompt) return true;
  return nextPrompt !== persistedPrompt;
}

function getCraftPrimaryActionLabel(craft, prompt, agentRun, isEditable = true) {
  if (!isEditable) return "Read only";
  if (
    agentRun?.status === "needs_input" &&
    !didCraftPromptChange(craft, prompt, {
      countEmptyAsChange: hasExplicitAgentPromptDraft(craft?.id),
    })
  ) {
    return "Continue";
  }
  return "Refine";
}

function isCraftPromptEditorNotePendingChange(agentRun, promptChanged = false) {
  return Boolean(promptChanged) && agentRun?.stopRequested !== true && isAgentRunActive(agentRun);
}

function getCraftPromptEditorNote(agentRun, promptChanged = false) {
  if (agentRun?.stopRequested === true) {
    return "Stopping the current run. You can refine again as soon as the stop is confirmed.";
  }
  if (isCraftPromptEditorNotePendingChange(agentRun, promptChanged)) {
    return "You changed this refine request while refine is running. These edits apply only to the next refine run.";
  }
  if (agentRun?.status === "needs_input") {
    return promptChanged
      ? "This refine request changed. Refine starts a fresh run instead of continuing the open questions."
      : "Answer the open questions to continue this run, or change the refine request to start a fresh refine.";
  }
  return "";
}

async function startAgentRun(craftId) {
  const craft = findCraft(craftId);
  if (!craft) return;
  if (craftStore?.isEditableCraft?.(craft) === false) {
    state.useMessages[craftId] = {
      time: formatClock(new Date()),
      text: "This shared craft is read only. Fork it before letting the agent edit samples.",
    };
    render();
    return;
  }

  const currentPrompt = getCurrentAgentPromptValue(craftId, craft);
  const prompt = String(currentPrompt || "").trim();
  const hasExplicitDraft = hasExplicitAgentPromptDraft(craftId);
  if (!prompt) {
    state.useMessages[craftId] = {
      time: formatClock(new Date()),
      text: "Enter the refine request before starting refine.",
    };
    render();
    return;
  }
  const promptChanged = didCraftPromptChange(craft, prompt, {
    countEmptyAsChange: hasExplicitDraft,
  });
  const promptRevision = buildAgentPromptRevisionContext(craft, prompt, {
    hasExplicitDraft,
  });

  if (prompt !== String(craft.agentPrompt || "").trim()) {
    craft.agentPrompt = prompt;
    touchCraft(craft);
    state.crafts = await persistCrafts(state.crafts);
  }

  const previousRun = state.agentRuns[craftId];
  if (isAgentRunActive(previousRun) && previousRun?.runId) return;

  const resumeSameRun = previousRun?.status === "needs_input" && previousRun?.runId && !promptChanged;
  const openQuestionCount = countOpenAgentQuestions(previousRun);

  if (resumeSameRun && openQuestionCount > 0) {
    state.useMessages[craftId] = {
      time: formatClock(new Date()),
      text: `Answer ${formatCount(openQuestionCount)} open questions before resuming the same agent run.`,
    };
    render();
    return;
  }

  state.activeCraftId = craftId;
  state.craftingCraftId = craftId;
  if (resumeSameRun) {
    state.agentRuns[craftId] = {
      ...(previousRun && typeof previousRun === "object" ? previousRun : {}),
      runId: String(previousRun?.runId || ""),
      status: "running",
      phase: "resuming",
      finalStatus: "",
      error: "",
      completedAt: "",
      activeTab: "progress",
      logs: [
        ...(Array.isArray(previousRun?.logs) ? previousRun.logs : []),
        createAgentLog("info", "Applying the answered questions and resuming the same run."),
      ].slice(-120),
      activityRecorded: false,
    };
    scheduleAgentRunPersist(craftId);
    refreshCraftCard(craftId);
  } else {
    state.agentRuns[craftId] = createPendingCraftingRun(prompt, {}, craft);
    scheduleAgentRunPersist(craftId);
    refreshCraftCard(craftId);
  }
  try {
    await withTimeout(
      (async () => {
        await loadTrainingDataState(craftId);
        await flushTrainingDataSave(craftId);
      })(),
      CRAFT_AGENT_START_TIMEOUT_MS,
      "Loading the local training context took too long.",
    );
    const trainingState = getTrainingDataState(craftId);
    const sampleCount = Array.isArray(trainingState.samples) ? trainingState.samples.length : 0;
    const requestBrief = buildAgentRunBrief(craftId, prompt);
    const toolingLabel = getCraftingAgentToolingLabel(craft);
    const previousQuestions = normalizeAgentQuestions(previousRun?.questions);

    if (!resumeSameRun) {
      state.agentRuns[craftId] = {
          ...createPendingCraftingRun(prompt, {}, craft),
          report: normalizeAgentReport(null, {
            objective: trimText(prompt, 180),
            currentState: sampleCount
              ? `${formatCount(sampleCount)} local training rows are loaded as context.`
              : "No local training rows are available for this craft yet.",
            nextAction: toolingLabel
              ? `Preparing the run with ${toolingLabel}.`
              : "Preparing the run.",
          }),
          logs: [
            createAgentLog("info", "Loading local training data."),
            createAgentLog(
              "info",
              sampleCount
                ? `${formatCount(sampleCount)} existing training rows are being checked as reference.`
                : "No seed rows exist yet. The agent has to secure the target structure first.",
            ),
            toolingLabel ? createAgentLog("info", `Supervisor tools pinned: ${toolingLabel}.`) : null,
            createAgentLog("info", "Matching the brief against the current prompt-to-JSON structure."),
            createAgentLog("info", "Starting the full SDK agent run in the background."),
          ].filter(Boolean),
        };
      refreshCraftCard(craftId);
    }

    const pendingRun = state.agentRuns[craftId];
    if (pendingRun) {
      pendingRun.report = normalizeAgentReport(pendingRun.report, {
        currentState: "Waiting for the background agent to acknowledge the run.",
        nextAction: "",
      });
      scheduleAgentRunPersist(craftId);
      refreshCraftCard(craftId);
    }
    const response = await withTimeout(
      sendRuntimeMessage({
        type: "agent:start-crafting-run",
        slotId: "agent",
        craft: {
          id: craft.id,
          name: craft.name,
          nameSource: String(craft.nameSource || ""),
          summary: craft.summary,
          agentPrompt: craft.agentPrompt,
          stage: craft.stage,
          inputMode: getCraftInputMode(craft),
          inputHint: String(craft.inputHint || ""),
          actionLabel: String(craft.actionLabel || ""),
          inputExamples: normalizePromptExampleTexts(craft.inputExamples),
          accuracy: describeCraftAccuracyForAgent(craft),
          seedRows: describeCraftSeedRows(craft),
          datasetRows: describeCraftDatasetRows(craft),
          coverageGaps: describeCraftCoverageGaps(craft),
          tools: Array.isArray(craft.tools) ? craft.tools : [],
          refineContext: promptRevision,
          agentTooling: getCraftingAgentTooling(craft),
          targetSlot: craft.targetSlot || "agent",
        },
        brief: requestBrief,
        currentSamples: summarizeTrainingSamplesForAgent(trainingState.samples),
        previousQuestions,
        questionAnswers: previousQuestions,
        resumeJobId: resumeSameRun ? previousRun.runId : "",
      }),
      CRAFT_AGENT_START_TIMEOUT_MS,
      "The background agent did not acknowledge the run in time.",
    );
    if (!response?.ok || !response.run) {
      throw new Error(response?.error || "Crafting agent run could not be started.");
    }
    applyCraftingRunSnapshot(craftId, response.run);
    scheduleAgentRunPoll(craftId);
  } catch (error) {
    const run = state.agentRuns[craftId];
    const message = error instanceof Error ? error.message : String(error || "Revision failed.");
    if (!run) return;
    run.logs.push(createAgentLog("error", message));
    run.status = "failed";
    run.phase = "failed";
    run.finalStatus = "";
    run.error = message;
    run.completedAt = new Date().toISOString();
    run.report = normalizeAgentReport(run.report, {
      currentState: message,
      nextAction: "Inspect the failure and restart the agent run with additional context.",
    });
    scheduleAgentRunPersist(craftId);
    await finalizeCraftingRun(craftId);
    refreshCraftCard(craftId);
  }
}

async function stopAgentRun(craftId) {
  const key = String(craftId || "").trim();
  if (!key) return;
  const run = state.agentRuns[key];
  if (!run?.runId || !isAgentRunActive(run) || run.stopRequested === true) return;

  run.stopRequested = true;
  run.logs = [
    ...(Array.isArray(run.logs) ? run.logs : []),
    createAgentLog("warn", "Stop requested. Waiting for the background runtime to abort the run."),
  ].slice(-120);
  scheduleAgentRunPersist(key);
  stopAgentRunPoll(key);
  refreshCraftCard(key);

  try {
    const response = await sendRuntimeMessage({
      type: "agent:stop-crafting-run",
      jobId: run.runId,
      reason: "The crafting agent was stopped by the user.",
    });
    if (!response?.ok) {
      throw new Error(response?.error || "Stopping the crafting run failed.");
    }
    const nextRun = response?.run ? applyCraftingRunSnapshot(key, response.run) : null;
    if (nextRun && isFinalAgentRunStatus(nextRun.status)) {
      await finalizeCraftingRun(key);
      return;
    }
    refreshCraftCard(key);
  } catch (error) {
    const currentRun = state.agentRuns[key];
    if (!currentRun) return;
    currentRun.stopRequested = false;
    currentRun.status = "failed";
    currentRun.phase = "failed";
    currentRun.error = error instanceof Error ? error.message : String(error || "Stopping the crafting run failed.");
    currentRun.logs = [
      ...(Array.isArray(currentRun.logs) ? currentRun.logs : []),
      createAgentLog("error", currentRun.error),
    ].slice(-120);
    scheduleAgentRunPersist(key);
    await finalizeCraftingRun(key);
    refreshCraftCard(key);
  }
}

async function startTrainingRun(craftId) {
  return await startTrainingRunWithOptions(craftId, {});
}

function isKnownHeaderTestId(testId) {
  return Object.prototype.hasOwnProperty.call(HEADER_TEST_DEFS, String(testId || ""));
}

function normalizeHeaderTrainingTestId(testId) {
  const resolvedTestId = String(testId || "").trim();
  if (["finetuning", "proofreadFinetuning", "visionFinetuning"].includes(resolvedTestId)) {
    return resolvedTestId;
  }
  return "finetuning";
}

function getPackagedExtensionAssetUrl(relativePath = "") {
  const assetPath = String(relativePath || "").trim().replace(/^\/+/, "");
  if (!assetPath) {
    throw new Error("Missing packaged asset path.");
  }
  const runtimeUrl = globalThis.chrome?.runtime?.getURL?.(assetPath);
  if (!runtimeUrl) {
    throw new Error(`Packaged asset ${assetPath} is unavailable in this runtime.`);
  }
  return runtimeUrl;
}

async function loadPackagedJsonAsset(relativePath) {
  const assetUrl = getPackagedExtensionAssetUrl(relativePath);
  const response = await fetch(assetUrl, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Failed to load packaged asset ${relativePath}: HTTP ${response.status}`);
  }
  try {
    return await response.json();
  } catch (error) {
    throw new Error(
      `Packaged asset ${relativePath} does not contain valid JSON: ${
        error instanceof Error ? error.message : String(error || "Unknown parse error.")
      }`,
    );
  }
}

function takeTailDatasetRows(rows = [], count = 0) {
  const pool = Array.isArray(rows) ? rows : [];
  const maxRemovable = Math.max(0, pool.length - 1);
  const safeCount = Math.max(0, Math.min(Number(count || 0), maxRemovable));
  if (!safeCount) return [];
  return pool.splice(pool.length - safeCount, safeCount);
}

function slicePackagedTrainingDataset(dataset = {}, {
  maxTrainPairs = 0,
  maxValidationPairs = 0,
  maxTestPairs = 0,
} = {}) {
  const trainLimit = Math.max(0, Number(maxTrainPairs || 0));
  const validationLimit = Math.max(0, Number(maxValidationPairs || 0));
  const testLimit = Math.max(0, Number(maxTestPairs || 0));
  const flatRows = Array.isArray(dataset?.rows) ? dataset.rows.slice() : [];

  let trainRows = Array.isArray(dataset?.train)
    ? dataset.train.slice(0, trainLimit > 0 ? trainLimit : undefined)
    : [];
  let validationRows = Array.isArray(dataset?.validation)
    ? dataset.validation.slice(0, validationLimit > 0 ? validationLimit : undefined)
    : [];
  let testRows = Array.isArray(dataset?.test)
    ? dataset.test.slice(0, testLimit > 0 ? testLimit : undefined)
    : [];

  if (!trainRows.length && flatRows.length) {
    const pool = flatRows.slice();
    const defaultTestCount = Math.max(1, Math.min(Math.floor(pool.length * 0.2), Math.max(1, pool.length - 1)));
    const defaultValidationCount = Math.max(
      1,
      Math.min(Math.floor(pool.length * 0.15), Math.max(1, pool.length - defaultTestCount - 1)),
    );
    if (!testRows.length) {
      testRows = takeTailDatasetRows(pool, testLimit > 0 ? Math.min(testLimit, defaultTestCount) : defaultTestCount);
    }
    if (!validationRows.length) {
      validationRows = takeTailDatasetRows(
        pool,
        validationLimit > 0 ? Math.min(validationLimit, defaultValidationCount) : defaultValidationCount,
      );
    }
    trainRows = trainLimit > 0 ? pool.slice(0, trainLimit) : pool;
  }

  if (!validationRows.length && validationLimit > 0) {
    validationRows = takeTailDatasetRows(
      trainRows,
      Math.max(1, Math.min(validationLimit, Math.floor(trainRows.length * 0.2) || 1)),
    );
  }
  if (!testRows.length && testLimit > 0) {
    testRows = takeTailDatasetRows(
      trainRows,
      Math.max(1, Math.min(testLimit, Math.floor(trainRows.length * 0.2) || 1)),
    );
  }
  if (!trainRows.length && flatRows.length) {
    const maxCount = trainLimit > 0 ? trainLimit : flatRows.length;
    trainRows = flatRows.slice(0, Math.max(1, Math.min(maxCount, flatRows.length)));
  }
  if (!validationRows.length && trainRows.length) {
    validationRows = trainRows.slice(-1);
  }
  if (!testRows.length && trainRows.length) {
    testRows = validationRows.length ? validationRows.slice(0, 1) : trainRows.slice(-1);
  }

  return {
    train: trainRows,
    validation: validationRows,
    test: testRows,
  };
}

async function loadProofreadTrainingDatasetFromPackage() {
  const dataset = await loadPackagedJsonAsset(DEBUG_PROOFREAD_TRAINING.relativeDatasetPath);
  if (!dataset || typeof dataset !== "object") {
    throw new Error("The packaged proofread dataset is invalid.");
  }
  const sliced = slicePackagedTrainingDataset(dataset, {
    maxTrainPairs: DEBUG_PROOFREAD_TRAINING.maxTrainPairs,
    maxValidationPairs: DEBUG_PROOFREAD_TRAINING.maxValidationPairs,
    maxTestPairs: DEBUG_PROOFREAD_TRAINING.maxTestPairs,
  });
  return {
    ...sliced,
    meta: {
      ...(dataset?.meta && typeof dataset.meta === "object" ? dataset.meta : {}),
      source: "packaged_extension_asset",
      artifactPath: DEBUG_PROOFREAD_TRAINING.relativeDatasetPath,
      splitCounts: {
        train: sliced.train.length,
        validation: sliced.validation.length,
        test: sliced.test.length,
      },
    },
  };
}

async function getHeaderTrainingSmokePreset(testId) {
  const resolvedTestId = normalizeHeaderTrainingTestId(testId);
  if (resolvedTestId === "proofreadFinetuning") {
    return {
      headerTestId: resolvedTestId,
      shardId: `${DEBUG_FIXED_TRAINING.shardId}-${DEBUG_PROOFREAD_TRAINING.shardIdSuffix}`,
      modelName: DEBUG_FIXED_TRAINING.modelName,
      datasetPayload: await loadProofreadTrainingDatasetFromPackage(),
      smokeMode: "",
      configOverrides: {
        profile: "proofread_webgpu_smoke",
        comparisonPurpose: "browser_text_training_benchmark",
        compareBy: "forward_tokens_per_second",
        resourceProfile: "memory_conservative",
        maxTrainPairs: DEBUG_PROOFREAD_TRAINING.maxTrainPairs,
        maxValidationPairs: DEBUG_PROOFREAD_TRAINING.maxValidationPairs,
        maxTestPairs: DEBUG_PROOFREAD_TRAINING.maxTestPairs,
        epochs: DEBUG_PROOFREAD_TRAINING.epochs,
        maxSeqLen: DEBUG_PROOFREAD_TRAINING.maxSeqLen,
        modelBatchSize: DEBUG_PROOFREAD_TRAINING.modelBatchSize,
        batchTokens: DEBUG_PROOFREAD_TRAINING.batchTokens,
      },
    };
  }
  if (resolvedTestId === "visionFinetuning") {
    const smokeImageDataUrl = createLabeledVisionSmokeImageDataUrl("red", "#ff0000");
    return {
      headerTestId: resolvedTestId,
      shardId: `${DEBUG_FIXED_TRAINING.shardId}-vision`,
      modelName: getLocalQwenSmokeModelName("vision"),
      datasetPayload: createLocalQwenMultimodalSmokeDatasetPayload({
        imageDataUrl: smokeImageDataUrl,
      }),
      smokeMode: "multimodal_smoke",
      configOverrides: {
        profile: LOCAL_QWEN_MULTIMODAL_SMOKE_CONFIG.profile,
        maxTrainPairs: LOCAL_QWEN_MULTIMODAL_SMOKE_CONFIG.maxTrainPairs,
        maxTestPairs: LOCAL_QWEN_MULTIMODAL_SMOKE_CONFIG.maxTestPairs,
        epochs: LOCAL_QWEN_MULTIMODAL_SMOKE_CONFIG.epochs,
        maxSeqLen: LOCAL_QWEN_MULTIMODAL_SMOKE_CONFIG.maxSeqLen,
        modelBatchSize: LOCAL_QWEN_MULTIMODAL_SMOKE_CONFIG.modelBatchSize,
        batchTokens: LOCAL_QWEN_MULTIMODAL_SMOKE_CONFIG.batchTokens,
        evaluationMode: LOCAL_QWEN_MULTIMODAL_SMOKE_CONFIG.evaluationMode,
      },
    };
  }
  return {
    headerTestId: "finetuning",
    shardId: DEBUG_FIXED_TRAINING.shardId,
    modelName: DEBUG_FIXED_TRAINING.modelName,
    datasetPayload: createLocalQwenPipelineSmokeDatasetPayload(),
    smokeMode: "pipeline_e2e",
    configOverrides: {
      profile: LOCAL_QWEN_PIPELINE_SMOKE_CONFIG.profile,
      comparisonPurpose: "pipeline_correctness_smoke",
      compareBy: "forward_tokens_per_second",
      resourceProfile: "memory_conservative",
      maxTrainPairs: LOCAL_QWEN_PIPELINE_SMOKE_CONFIG.maxTrainPairs,
      maxTestPairs: LOCAL_QWEN_PIPELINE_SMOKE_CONFIG.maxTestPairs,
      epochs: LOCAL_QWEN_PIPELINE_SMOKE_CONFIG.epochs,
      maxSeqLen: LOCAL_QWEN_PIPELINE_SMOKE_CONFIG.maxSeqLen,
      modelBatchSize: LOCAL_QWEN_PIPELINE_SMOKE_CONFIG.modelBatchSize,
      batchTokens: LOCAL_QWEN_PIPELINE_SMOKE_CONFIG.batchTokens,
    },
  };
}

function getHeaderTrainingTestId(run) {
  const headerTestId = String(run?.headerTestId || "").trim();
  return isKnownHeaderTestId(headerTestId) ? headerTestId : "finetuning";
}

function getHeaderTrainingTestLabel(testId) {
  return String(HEADER_TEST_DEFS[testId]?.label || "Training");
}

async function startHeaderTrainingRun(testId = "finetuning") {
  const activeRun = state.debugTrainingRun;
  if (isTrainingRunActive(activeRun)) return;
  const requestedTestId = normalizeHeaderTrainingTestId(testId);

  startHeaderTestRun(requestedTestId);
  updateHeaderTestRun(requestedTestId, { progress: 0.08, indeterminate: true });
  render();

  try {
    if (requestedTestId === "proofreadFinetuning") {
      updateHeaderTestRun(requestedTestId, { progress: 0.16, indeterminate: true });
      const fixture = await provisionProofreadWorkspaceTrainingFixture();
      const fallbackShard = Array.isArray(fixture.craft?.training?.shards)
        ? fixture.craft.training.shards.find((entry) => String(entry?.id || "").trim() === "proofread-smoke-shard") || null
        : null;
      const shard = getSelectedTrainingShard(fixture.craft) || fallbackShard;
      const datasetPayload = await buildTrainingDatasetPayload(fixture.craft.id);
      const configOverrides = {
        profile: "proofread_webgpu_smoke",
        maxTrainPairs: DEBUG_PROOFREAD_TRAINING.maxTrainPairs,
        maxValidationPairs: DEBUG_PROOFREAD_TRAINING.maxValidationPairs,
        maxTestPairs: DEBUG_PROOFREAD_TRAINING.maxTestPairs,
        epochs: DEBUG_PROOFREAD_TRAINING.epochs,
        maxSeqLen: DEBUG_PROOFREAD_TRAINING.maxSeqLen,
        modelBatchSize: DEBUG_PROOFREAD_TRAINING.modelBatchSize,
        batchTokens: DEBUG_PROOFREAD_TRAINING.batchTokens,
      };
      const initialRun = createInitialTrainingRun({
        craftId: fixture.craft.id,
        headerTestId: requestedTestId,
        shardId: String(shard?.id || "proofread-smoke-shard"),
        modelName: String(shard?.modelName || "Qwen/Qwen3.5-0.8B"),
        smokeMode: "",
        autoCopyOnComplete: false,
      });
      state.trainingRuns[fixture.craft.id] = { ...initialRun };
      state.debugTrainingRun = { ...initialRun };
      updateHeaderTestRun(requestedTestId, { progress: 0.32, indeterminate: false });
      render();

      const response = await sendRuntimeMessage({
        type: "training:start-fixed-run",
        craftId: fixture.craft.id,
        shardId: initialRun.shardId,
        modelName: initialRun.modelName,
        datasetPayload,
        smokeMode: "",
        configOverrides,
        persistBundle: true,
      });
      if (!response?.ok || !response.run) {
        throw new Error(response?.error || "Training could not be started.");
      }
      const startedRun = {
        ...response.run,
        craftId: fixture.craft.id,
        headerTestId: requestedTestId,
        shardId: initialRun.shardId,
        modelName: initialRun.modelName,
      };
      applyTrainingRun(fixture.craft.id, startedRun);
      const nextRun = applyDebugTrainingRun(startedRun);
      syncHeaderTrainingTestProgress(nextRun);
      void maybeFinalizeTrainingRun(fixture.craft.id);
      void maybeFinalizeDebugTrainingRun();
      scheduleDebugTrainingPoll();
      return;
    }

    const preset = await getHeaderTrainingSmokePreset(requestedTestId);
    state.debugTrainingRun = createInitialTrainingRun({
      craftId: DEBUG_FIXED_TRAINING.craftId,
      headerTestId: preset.headerTestId,
      shardId: preset.shardId,
      modelName: preset.modelName,
      smokeMode: preset.smokeMode,
      autoCopyOnComplete: false,
    });
    render();

    const response = await sendRuntimeMessage({
      type: "training:start-fixed-run",
      craftId: DEBUG_FIXED_TRAINING.craftId,
      shardId: preset.shardId,
      modelName: preset.modelName,
      datasetPayload: preset.datasetPayload,
      smokeMode: preset.smokeMode,
      configOverrides: preset.configOverrides,
      persistBundle: false,
    });
    if (!response?.ok || !response.run) {
      throw new Error(response?.error || "Training could not be started.");
    }
    const nextRun = applyDebugTrainingRun({
      ...response.run,
      headerTestId: preset.headerTestId,
    });
    syncHeaderTrainingTestProgress(nextRun);
    void maybeFinalizeDebugTrainingRun();
    scheduleDebugTrainingPoll();
  } catch (error) {
    const failedCraftId =
      requestedTestId === "proofreadFinetuning"
        ? String(state.debugTrainingRun?.craftId || DEV_PROOFREAD_WORKSPACE_CRAFT_ID).trim()
        : "";
    state.debugTrainingRun = {
      ...(state.debugTrainingRun || createInitialTrainingRun({ craftId: failedCraftId, headerTestId: requestedTestId })),
      status: "failed",
      phaseLabel: "Training unavailable",
      message: "Training could not be started.",
      error: error instanceof Error ? error.message : String(error || "Training unavailable."),
    };
    if (failedCraftId) {
      state.trainingRuns[failedCraftId] = mergeTrainingRun(state.trainingRuns[failedCraftId] || {}, {
        craftId: failedCraftId,
        headerTestId: requestedTestId,
        status: "failed",
        phaseLabel: "Training unavailable",
        message: "Training could not be started.",
        error: error instanceof Error ? error.message : String(error || "Training unavailable."),
      });
      void maybeFinalizeTrainingRun(failedCraftId);
    }
    void maybeFinalizeDebugTrainingRun();
    render();
  }
}

async function startTrainingRunWithOptions(craftId, options = {}) {
  const craft = findCraft(craftId);
  const training = craft ? getTrainingConfig(craft) : null;
  if (!craft || !training) return;

  const activeRun = state.trainingRuns[craftId];
  if (isTrainingRunActive(activeRun)) return;

  if (options.openCraft !== false) {
    state.activeCraftId = craftId;
    state.craftingCraftId = craftId;
  }

  const shard =
    Array.isArray(training.shards) && options.overrideShardId
      ? training.shards.find((entry) => entry.id === options.overrideShardId) || getSelectedTrainingShard(craft)
      : getSelectedTrainingShard(craft);
  state.trainingRuns[craftId] = createInitialTrainingRun({
    craftId,
    shardId: shard?.id || "",
    modelName: shard?.modelName || "",
    autoCopyOnComplete: options.autoCopyOnComplete === true,
  });
  render();

  try {
    const datasetPayload = await buildTrainingDatasetPayload(craftId);
    const response = await sendRuntimeMessage({
      type: "training:start-fixed-run",
      craftId,
      shardId: shard?.id || "",
      modelName: shard?.modelName || "",
      datasetPayload,
      persistBundle: options.persistBundle ?? true,
    });
    if (!response?.ok || !response.run) {
      throw new Error(response?.error || "Training could not be started.");
    }
    applyTrainingRun(craftId, response.run);
    void maybeFinalizeTrainingRun(craftId);
    scheduleTrainingPoll(craftId);
  } catch (error) {
    state.trainingRuns[craftId] = {
      ...(state.trainingRuns[craftId] || {}),
      status: "failed",
      phaseLabel: "Training unavailable",
      message: "Training could not be started.",
      error:
        error instanceof Error
          ? error.message
          : String(error || "Training unavailable."),
    };
    void maybeFinalizeTrainingRun(craftId);
    render();
  }
}

function createInitialTrainingRun({
  craftId = "",
  headerTestId = "",
  shardId = "",
  modelName = "",
  smokeMode = "",
  autoCopyOnComplete = false,
} = {}) {
  return {
    status: "starting",
    craftId: String(craftId || "").trim(),
    headerTestId,
    shardId,
    modelName,
    smokeMode,
    phaseLabel: "Starting training",
    progress: 0,
    totalSamples: 0,
    completedSamples: 0,
    phaseTotalSamples: 0,
    phaseCompletedSamples: 0,
    phaseUnitLabel: "job steps",
    samplesPerSecond: 0,
    estimatedRemainingMs: 0,
    currentEpoch: 0,
    epochsTotal: 0,
    baseValidationAcc: 0,
    baseTestAcc: 0,
    adaptValidationAcc: 0,
    adaptTestAcc: 0,
    adapterSizeMb: 0,
    metrics: {},
    smoke: null,
    dataset: null,
    runtime: null,
    history: [],
    message: "Preparing training run.",
    error: "",
    workdir: "",
    manifestPath: "",
    startedAt: "",
    endedAt: "",
    autoCopyOnComplete,
    clipboardStatus: "idle",
    clipboardError: "",
    lastReport: null,
    lastReportUpdatedAt: "",
    activityRecorded: false,
  };
}

function mergeTrainingRun(current = {}, payload = {}) {
  const metrics =
    payload?.metrics && typeof payload.metrics === "object"
      ? { ...payload.metrics }
      : current.metrics && typeof current.metrics === "object"
        ? { ...current.metrics }
        : {};
  const nextRun = {
    ...current,
    jobId: String(payload?.jobId ?? current.jobId ?? ""),
    craftId: String(payload?.craftId ?? current.craftId ?? ""),
    headerTestId: String(payload?.headerTestId ?? current.headerTestId ?? ""),
    shardId: String(payload?.shardId ?? current.shardId ?? ""),
    modelName: String(payload?.modelName ?? current.modelName ?? ""),
    smokeMode: String(payload?.smokeMode ?? current.smokeMode ?? ""),
    status: String(payload?.status ?? current.status ?? "idle"),
    phaseLabel: String(payload?.phaseLabel ?? payload?.phase ?? current.phaseLabel ?? "Idle"),
    progress: Math.max(0, Math.min(1, Number(payload?.progress ?? current.progress ?? 0))),
    totalSamples: Number(payload?.totalSamples ?? current.totalSamples ?? 0),
    completedSamples: Number(payload?.completedSamples ?? current.completedSamples ?? 0),
    phaseTotalSamples: Number(payload?.phaseTotalSamples ?? current.phaseTotalSamples ?? 0),
    phaseCompletedSamples: Number(payload?.phaseCompletedSamples ?? current.phaseCompletedSamples ?? 0),
    phaseUnitLabel: String(payload?.phaseUnitLabel ?? current.phaseUnitLabel ?? ""),
    samplesPerSecond: Number(payload?.samplesPerSecond ?? current.samplesPerSecond ?? 0),
    estimatedRemainingMs: Number(payload?.estimatedRemainingMs ?? current.estimatedRemainingMs ?? 0),
    currentEpoch: Number(payload?.currentEpoch ?? current.currentEpoch ?? 0),
    epochsTotal: Number(payload?.epochsTotal ?? current.epochsTotal ?? 0),
    baseValidationAcc: Number(metrics.baseValidationAcc ?? current.baseValidationAcc ?? 0),
    baseTestAcc: Number(metrics.baseTestAcc ?? current.baseTestAcc ?? 0),
    adaptValidationAcc: Number(metrics.adaptValidationAcc ?? metrics.validationEvalAcc ?? current.adaptValidationAcc ?? 0),
    adaptTestAcc: Number(metrics.adaptTestAcc ?? current.adaptTestAcc ?? metrics.baseTestAcc ?? 0),
    adapterSizeMb: Number(payload?.adapterSizeMb ?? current.adapterSizeMb ?? 0),
    metrics,
    smoke:
      payload?.smoke && typeof payload.smoke === "object"
        ? cloneJson(payload.smoke, null)
        : current.smoke && typeof current.smoke === "object"
          ? cloneJson(current.smoke, null)
          : null,
    dataset:
      payload?.dataset && typeof payload.dataset === "object"
        ? cloneJson(payload.dataset, null)
        : current.dataset && typeof current.dataset === "object"
          ? cloneJson(current.dataset, null)
          : null,
    runtime:
      payload?.runtime && typeof payload.runtime === "object"
        ? cloneJson(payload.runtime, null)
        : current.runtime && typeof current.runtime === "object"
          ? cloneJson(current.runtime, null)
          : null,
    history: Array.isArray(payload?.history)
      ? cloneJson(payload.history, [])
      : Array.isArray(current.history)
        ? cloneJson(current.history, [])
        : [],
    message: String(payload?.message ?? current.message ?? ""),
    error: String(payload?.error ?? current.error ?? ""),
    workdir: String(payload?.workdir ?? current.workdir ?? ""),
    manifestPath: String(payload?.manifestPath ?? current.manifestPath ?? ""),
    startedAt: String(payload?.startedAt ?? current.startedAt ?? ""),
    endedAt: String(payload?.endedAt ?? current.endedAt ?? ""),
    autoCopyOnComplete: current.autoCopyOnComplete === true,
    clipboardStatus: String(current.clipboardStatus || "idle"),
    clipboardError: String(current.clipboardError || ""),
    lastReport:
      payload?.lastReport && typeof payload.lastReport === "object"
        ? cloneJson(payload.lastReport, null)
        : current.lastReport && typeof current.lastReport === "object"
          ? cloneJson(current.lastReport, null)
          : null,
    lastReportUpdatedAt: String(payload?.lastReportUpdatedAt ?? current.lastReportUpdatedAt ?? ""),
    activityRecorded: current.activityRecorded === true,
  };
  return nextRun;
}

function applyTrainingRun(craftId, payload) {
  const current = state.trainingRuns[craftId] || {};
  const nextRun = mergeTrainingRun(current, payload);

  state.trainingRuns[craftId] = nextRun;
  if (String(nextRun.status || "") === "completed") {
    void loadCapabilityWeightsState(craftId, { force: true });
  }

  const craft = findCraft(craftId);
  if (craft && Number.isFinite(nextRun.adaptTestAcc) && nextRun.adaptTestAcc > 0) {
    craft.accuracy = nextRun.adaptTestAcc;
    craft.metricsReady = true;
    touchCraft(craft, nextRun.endedAt || nextRun.startedAt || new Date().toISOString());
  }

  render();
}

function applyDebugTrainingRun(payload) {
  state.debugTrainingRun = mergeTrainingRun(state.debugTrainingRun || {}, payload);
  render();
  return state.debugTrainingRun;
}

function createTrainingClipboardReport(craftId, run) {
  const craft = findCraft(craftId);
  return {
    reportVersion: 2,
    type: "local_qwen_webgpu_training_report",
    timestamp: new Date().toISOString(),
    page: {
      href: globalThis.location.href,
      userAgent: globalThis.navigator?.userAgent || "",
      language: globalThis.navigator?.language || "",
      platform: globalThis.navigator?.platform || "",
    },
    extension: {
      id: globalThis.chrome?.runtime?.id || "",
      version: globalThis.chrome?.runtime?.getManifest?.().version || "",
    },
    craft: craft
      ? {
          id: craft.id,
          name: craft.name,
        }
      : null,
    run: {
      jobId: String(run?.jobId || ""),
      headerTestId: String(run?.headerTestId || ""),
      shardId: String(run?.shardId || ""),
      modelName: String(run?.modelName || ""),
      status: String(run?.status || ""),
      phaseLabel: String(run?.phaseLabel || ""),
      message: String(run?.message || ""),
      error: String(run?.error || ""),
      progress: Number(run?.progress || 0),
      totalSamples: Number(run?.totalSamples || 0),
      completedSamples: Number(run?.completedSamples || 0),
      phaseTotalSamples: Number(run?.phaseTotalSamples || 0),
      phaseCompletedSamples: Number(run?.phaseCompletedSamples || 0),
      phaseUnitLabel: String(run?.phaseUnitLabel || ""),
      samplesPerSecond: Number(run?.samplesPerSecond || 0),
      currentEpoch: Number(run?.currentEpoch || 0),
      epochsTotal: Number(run?.epochsTotal || 0),
      adapterSizeMb: Number(run?.adapterSizeMb || 0),
      metrics: run?.metrics && typeof run.metrics === "object" ? { ...run.metrics } : {},
      comparison:
        run?.runtime?.trainingComparison && typeof run.runtime.trainingComparison === "object"
          ? { ...run.runtime.trainingComparison }
          : null,
      throughput: {
        overallForwardTokensPerSecond: Number(run?.metrics?.overallForwardTokensPerSecond || 0),
        overallSupervisedTokensPerSecond: Number(run?.metrics?.overallSupervisedTokensPerSecond || 0),
        totalForwardTokens: Number(run?.metrics?.totalForwardTokens || 0),
        totalSupervisedTokens: Number(run?.metrics?.totalSupervisedTokens || 0),
        trainStepForwardTokensPerSecond: Number(run?.metrics?.trainStepForwardTokensPerSecond || 0),
        evalForwardTokensPerSecond: Number(run?.metrics?.evalForwardTokensPerSecond || 0),
        measuredForwardTokensPerSecond: Number(run?.metrics?.measuredForwardTokensPerSecond || 0),
        trainStepForwardTokens: Number(run?.metrics?.trainStepForwardTokens || 0),
        evalForwardTokens: Number(run?.metrics?.evalForwardTokens || 0),
        totalMeasuredForwardMs: Number(run?.metrics?.totalMeasuredForwardMs || 0),
        trainStepForwardMs: Number(run?.metrics?.trainStepForwardMs || 0),
        evalForwardMs: Number(run?.metrics?.evalForwardMs || 0),
        nonForwardOverheadMs: Number(run?.metrics?.nonForwardOverheadMs || 0),
        forwardWorkloadShare: Number(run?.metrics?.forwardWorkloadShare || 0),
        nonForwardOverheadShare: Number(run?.metrics?.nonForwardOverheadShare || 0),
      },
      baseTestAcc: Number(run?.baseTestAcc || 0),
      adaptTestAcc: Number(run?.adaptTestAcc || 0),
      smokeMode: String(run?.smokeMode || ""),
      smoke: run?.smoke && typeof run.smoke === "object" ? { ...run.smoke } : null,
      dataset: run?.dataset && typeof run.dataset === "object" ? { ...run.dataset } : null,
      runtime: run?.runtime && typeof run.runtime === "object" ? { ...run.runtime } : null,
      history: Array.isArray(run?.history) ? run.history : [],
      startedAt: String(run?.startedAt || ""),
      endedAt: String(run?.endedAt || ""),
    },
    diagnosis: diagnoseTrainingRunReport(run),
  };
}

function storeTrainingRunReport(craftId, run, { copiedAt = "" } = {}) {
  if (!run || typeof run !== "object") return null;
  const payload = createTrainingClipboardReport(craftId, run);
  const normalizedCopiedAt = String(copiedAt || "").trim();
  if (normalizedCopiedAt) {
    payload.copiedAt = normalizedCopiedAt;
  }
  run.lastReport = payload;
  run.lastReportUpdatedAt = String(payload.copiedAt || payload.timestamp || new Date().toISOString());
  return payload;
}

async function copyTrainingRunReport(craftId) {
  const key = String(craftId || "").trim();
  if (!key) return;
  const run = state.trainingRuns[key];
  if (!run || typeof run !== "object") return;

  run.clipboardStatus = "copying";
  run.clipboardError = "";
  render();

  try {
    const copiedAt = new Date().toISOString();
    const payload = storeTrainingRunReport(key, run, { copiedAt });
    await copyTextToClipboard(JSON.stringify(payload, null, 2));
    run.clipboardStatus = "copied";
    run.clipboardError = "";
  } catch (error) {
    run.clipboardStatus = "copy_failed";
    run.clipboardError = error instanceof Error ? error.message : String(error || "Clipboard write failed.");
  }

  render();
}

function buildAgentLogTail(run, limit = 12) {
  const lastToolFailure = buildAgentLastToolFailure(run);
  return (Array.isArray(run?.logs) ? run.logs : [])
    .map((entry) => normalizeAgentLogEntry(entry, run, lastToolFailure))
    .filter((entry) => entry.message || entry.title || entry.detail)
    .slice(-limit);
}

function compactAgentToolTraceArgs(value, depth = 0) {
  if (typeof value === "string") {
    return trimText(value, depth === 0 ? 400 : 220);
  }
  if (value == null || typeof value !== "object") {
    return value;
  }
  if (Array.isArray(value)) {
    return value.slice(0, depth === 0 ? 8 : 5).map((entry) => compactAgentToolTraceArgs(entry, depth + 1));
  }
  const source = asPlainObject(value);
  if (depth >= 2) {
    return { keys: Object.keys(source).slice(0, 8) };
  }
  const out = {};
  if (Array.isArray(source.scripts)) {
    out.scriptCount = source.scripts.length;
    out.scripts = source.scripts.slice(0, 4).map((script) => {
      const scriptSource = String(script?.source || "");
      return {
        id: String(script?.id || ""),
        name: String(script?.name || ""),
        entrypoint: String(script?.entrypoint || ""),
        language: String(script?.language || ""),
        description: trimText(String(script?.description || ""), 200),
        sourceLength: scriptSource.length,
        sourcePreview: trimText(scriptSource, 160),
      };
    });
  }
  if (Array.isArray(source.capabilities)) {
    out.capabilityCount = source.capabilities.length;
    out.capabilities = source.capabilities.slice(0, 6).map((capability) => ({
      id: String(capability?.id || ""),
      name: String(capability?.name || ""),
      toolName: String(capability?.toolName || ""),
      version: String(capability?.version || ""),
      description: trimText(String(capability?.description || ""), 200),
    }));
  }
  for (const [key, rawValue] of Object.entries(source)) {
    if (key === "scripts" || key === "capabilities") continue;
    if (key === "bundleOverride" && rawValue && typeof rawValue === "object") {
      out.bundleOverride = { keys: Object.keys(asPlainObject(rawValue)).slice(0, 8) };
      continue;
    }
    out[key] = compactAgentToolTraceArgs(rawValue, depth + 1);
  }
  return out;
}

function compactAgentToolTraceEntry(entry, run = null, lastToolFailure = null) {
  const source = asPlainObject(entry);
  const semanticToolFailure =
    ["blocked", "failed"].includes(String(run?.status || "").trim().toLowerCase()) &&
    String(source.action || "").trim() &&
    String(lastToolFailure?.action || "").trim() === String(source.action || "").trim() &&
    source.ok !== false;
  return {
    ...cloneJson(source, {}),
    thought: trimText(String(source.thought || ""), 280),
    ok: semanticToolFailure ? false : source.ok !== false,
    summary: trimText(
      String(
        semanticToolFailure
          ? lastToolFailure?.summary || lastToolFailure?.error || source.summary || source.error || ""
          : source.summary || source.error || ""
      ),
      260,
    ),
    error: trimText(
      String(
        semanticToolFailure
          ? lastToolFailure?.error || lastToolFailure?.summary || source.error || ""
          : source.error || ""
      ),
      600,
    ),
    errorDetail:
      semanticToolFailure && lastToolFailure?.errorDetail && typeof lastToolFailure.errorDetail === "object"
        ? cloneJson(lastToolFailure.errorDetail, null)
        : source.errorDetail && typeof source.errorDetail === "object"
          ? cloneJson(source.errorDetail, null)
          : null,
    errorStack: trimText(
      String(
        semanticToolFailure
          ? lastToolFailure?.errorStack || source.errorStack || ""
          : source.errorStack || ""
      ),
      4_000,
    ),
    args: compactAgentToolTraceArgs(source.args),
  };
}

function buildAgentToolTraceTail(run, limit = 8) {
  const lastToolFailure = buildAgentLastToolFailure(run);
  return (Array.isArray(run?.toolTrace) ? run.toolTrace : [])
    .slice(-limit)
    .map((entry) => compactAgentToolTraceEntry(entry, run, lastToolFailure));
}

function extractAgentWorkspaceDiagnosis(run) {
  return run?.workspaceCodeDive && typeof run.workspaceCodeDive === "object"
    ? run.workspaceCodeDive
    : null;
}

function parseAgentUnderlyingFailingToolHint(value = "") {
  const match = /underlyingfailingtool\s*[:=]\s*([a-z0-9_./-]+)/i.exec(String(value || ""));
  return match?.[1] ? String(match[1]).trim() : "";
}

function buildWorkspaceDiagnosisFailureDetail(run) {
  const diagnosis = extractAgentWorkspaceDiagnosis(run);
  if (!diagnosis) return null;
  const query = asPlainObject(diagnosis.query);
  const codingHandoff = asPlainObject(diagnosis.codingHandoff);
  const errorText = trimText(
    String(query.errorText || codingHandoff.errorText || run?.error || ""),
    1_600,
  );
  const underlyingFailingTool =
    parseAgentUnderlyingFailingToolHint(errorText) ||
    String(codingHandoff.underlyingFailingTool || "").trim();
  const problemText = `${String(query.problem || "")}\n${errorText}`.toLowerCase();
  const detail = {
    reason: "workspace_diagnosis_blocker",
    diagnosisType: "workspace_code_dive",
    failingTool: String(query.failingTool || codingHandoff.failingTool || "").trim(),
    underlyingFailingTool,
    errorText,
    bestHypothesis:
      diagnosis.bestHypothesis && typeof diagnosis.bestHypothesis === "object"
        ? cloneJson(diagnosis.bestHypothesis, null)
        : null,
    patchTargets: Array.isArray(diagnosis.patchTargets)
      ? diagnosis.patchTargets.slice(0, 4).map((entry) => ({
          path: String(entry?.path || ""),
          confidence: String(entry?.confidence || ""),
          why: trimText(String(entry?.why || ""), 220),
        }))
      : [],
  };
  if (/modeltransform|model transform|pre\/execute\/post null|empty scripts|entrypoint|tool script wiring|capability scripts/.test(problemText)) {
    detail.classification = "reviewed_runtime_wiring";
    detail.modelTransformFallbackUsed = /modeltransform|model transform/.test(problemText);
    detail.capabilityScriptsEmpty = /empty scripts|pre:\s*''|execute:\s*''|post:\s*''/.test(problemText);
  }
  return detail;
}

function buildSyntheticAgentLastToolFailure(run) {
  if (run?.lastToolFailure && typeof run.lastToolFailure === "object") return null;
  if (!["blocked", "failed"].includes(String(run?.status || "").trim().toLowerCase())) return null;
  const diagnosis = extractAgentWorkspaceDiagnosis(run);
  if (!diagnosis) return null;
  const query = asPlainObject(diagnosis.query);
  const codingHandoff = asPlainObject(diagnosis.codingHandoff);
  const errorDetail = buildWorkspaceDiagnosisFailureDetail(run);
  const action = String(query.failingTool || codingHandoff.failingTool || "").trim();
  const underlyingAction = String(
    errorDetail?.underlyingFailingTool ||
      codingHandoff.underlyingFailingTool ||
      ""
  ).trim();
  const error = trimText(
    String(query.errorText || codingHandoff.errorText || run?.error || diagnosis?.bestHypothesis?.summary || ""),
    600,
  );
  const summary = trimText(
    String(run?.report?.currentState || run?.error || error || diagnosis?.bestHypothesis?.summary || ""),
    260,
  );
  if (!action && !error && !summary) return null;
  return {
    action,
    underlyingAction,
    args: {
      problem: trimText(String(query.problem || ""), 600),
      errorText: trimText(String(query.errorText || ""), 800),
      focusFiles: Array.isArray(query.focusFiles) ? cloneJson(query.focusFiles.slice(0, 8), []) : [],
    },
    summary,
    error,
    errorDetail,
    errorStack: "",
    report: run?.report && typeof run.report === "object" ? cloneJson(run.report, null) : null,
    recordedAt: String(run?.completedAt || new Date().toISOString()),
  };
}

function resolveAgentRunErrorDetail(run) {
  const syntheticFailure = buildSyntheticAgentLastToolFailure(run);
  return run?.errorDetail && typeof run.errorDetail === "object"
    ? cloneJson(run.errorDetail, null)
    : run?.lastToolFailure?.errorDetail && typeof run.lastToolFailure.errorDetail === "object"
      ? cloneJson(run.lastToolFailure.errorDetail, null)
      : syntheticFailure?.errorDetail && typeof syntheticFailure.errorDetail === "object"
        ? cloneJson(syntheticFailure.errorDetail, null)
        : buildWorkspaceDiagnosisFailureDetail(run);
}

function buildAgentLastToolFailure(run) {
  const failureSource =
    run?.lastToolFailure && typeof run.lastToolFailure === "object"
      ? run.lastToolFailure
      : buildSyntheticAgentLastToolFailure(run);
  const failure = failureSource ? cloneJson(failureSource, null) : null;
  if (!failure) return null;
  failure.args = compactAgentToolTraceArgs(failure.args);
  failure.errorDetail =
    failure.errorDetail && typeof failure.errorDetail === "object"
      ? cloneJson(failure.errorDetail, null)
      : resolveAgentRunErrorDetail(run);
  failure.errorStack = trimText(String(failure.errorStack || ""), 4_000);
  failure.summary = trimText(String(failure.summary || failure.error || ""), 260);
  failure.error = trimText(String(failure.error || ""), 600);
  return failure;
}

function buildFallbackPatchTargets(run) {
  const lastToolFailure = buildAgentLastToolFailure(run);
  const action = String(lastToolFailure?.action || "").trim().toLowerCase();
  const haystack = [
    String(run?.error || ""),
    String(lastToolFailure?.error || ""),
    String(run?.report?.currentState || ""),
  ].join(" ").toLowerCase();
  if (/run_agentic_smoke/.test(action) || /local_qwen|qwen|onnx|webgpu|ortrun/.test(haystack)) {
    return [
      { path: "fuck-api-train-local-ai/bg/ml-inference.js", why: "Local Qwen WebGPU/ONNX runtime path.", confidence: "high" },
      { path: "fuck-api-train-local-ai/bg/llm.js", why: "local_qwen handoff and error propagation.", confidence: "medium" },
      { path: "fuck-api-train-local-ai/shared/local_qwen_runtime.mjs", why: "Runtime plan and ONNX alias resolution.", confidence: "medium" },
    ];
  }
  if (/run_agentic_smoke|run_capability_eval|active_text|clipboard|focused editable/.test(`${action} ${haystack}`)) {
    return [
      { path: "fuck-api-train-local-ai/bg/craft-use-runner.js", why: "Reviewed capability runtime and active-text flow.", confidence: "high" },
      { path: "fuck-api-train-local-ai/shared/browser-capability-bundle.mjs", why: "Reviewed capability contract and defaults.", confidence: "medium" },
      { path: "fuck-api-train-local-ai/bg/crafting-agent-runner.js", why: "Crafting-agent save/smoke orchestration.", confidence: "medium" },
    ];
  }
  return [
    { path: "fuck-api-train-local-ai/bg/crafting-agent-runner.js", why: "Crafting-agent orchestration and blocked-state handling.", confidence: "medium" },
    { path: "fuck-api-train-local-ai/sidepanel.js", why: "Debug export and agent progress surface.", confidence: "low" },
  ];
}

function buildFallbackValidationTargets(run) {
  const lastToolFailure = buildAgentLastToolFailure(run);
  const action = String(lastToolFailure?.action || "").trim().toLowerCase();
  const haystack = [
    String(run?.error || ""),
    String(lastToolFailure?.error || ""),
  ].join(" ").toLowerCase();
  const tests = new Set(["tests/crafting_agent_ui_regressions.test.mjs"]);
  if (/run_agentic_smoke|local_qwen|qwen|onnx|webgpu|ortrun/.test(`${action} ${haystack}`)) {
    tests.add("tests/local_qwen_error_regressions.test.mjs");
    tests.add("tests/local_qwen_runtime.test.mjs");
  }
  if (/run_agentic_smoke|run_capability_eval|active_text|clipboard|focused editable/.test(`${action} ${haystack}`)) {
    tests.add("tests/craft_use_runtime_regressions.test.mjs");
    tests.add("tests/qwen_agent_adapter.test.mjs");
  }
  return [...tests];
}

function buildFallbackExecutionPath(run) {
  const lastToolFailure = buildAgentLastToolFailure(run);
  const action = String(lastToolFailure?.action || "").trim().toLowerCase();
  const haystack = [
    String(run?.error || ""),
    String(lastToolFailure?.error || ""),
  ].join(" ").toLowerCase();
  if (/run_agentic_smoke|local_qwen|qwen|onnx|webgpu|ortrun/.test(`${action} ${haystack}`)) {
    return [
      "fuck-api-train-local-ai/sidepanel.js",
      "fuck-api-train-local-ai/bg/service_worker.js",
      "fuck-api-train-local-ai/bg/crafting-agent-runner.js",
      "fuck-api-train-local-ai/bg/llm.js",
      "fuck-api-train-local-ai/bg/ml-inference.js",
      "fuck-api-train-local-ai/shared/local_qwen_runtime.mjs",
    ];
  }
  if (/run_agentic_smoke|run_capability_eval|active_text|clipboard|focused editable/.test(`${action} ${haystack}`)) {
    return [
      "fuck-api-train-local-ai/sidepanel.js",
      "fuck-api-train-local-ai/bg/service_worker.js",
      "fuck-api-train-local-ai/bg/crafting-agent-runner.js",
      "fuck-api-train-local-ai/bg/craft-use-runner.js",
      "fuck-api-train-local-ai/shared/browser-capability-bundle.mjs",
    ];
  }
  return [
    "fuck-api-train-local-ai/sidepanel.js",
    "fuck-api-train-local-ai/bg/service_worker.js",
    "fuck-api-train-local-ai/bg/crafting-agent-runner.js",
  ];
}

function buildCodingPromptFromRun(craft, run, report, executionPath, patchTargets, validationTargets, lastToolFailure = null) {
  return [
    "Patch this Chrome extension workspace to resolve the blocked crafting-agent path.",
    `Capability objective: ${String(report?.objective || craft?.summary || run?.responseText || "").trim()}`,
    lastToolFailure?.action ? `Failing tool: ${String(lastToolFailure.action).trim()}` : "",
    lastToolFailure?.underlyingAction ? `Underlying failing tool: ${String(lastToolFailure.underlyingAction).trim()}` : "",
    lastToolFailure?.error ? `Observed error: ${String(lastToolFailure.error).trim()}` : String(run?.error || "").trim() ? `Observed error: ${String(run.error).trim()}` : "",
    executionPath.length ? `Execution path to inspect first: ${executionPath.join(" -> ")}` : "",
    patchTargets.length ? `Start with these patch targets: ${patchTargets.map((entry) => entry.path).join(", ")}` : "",
    validationTargets.length ? `Validate the fix with: ${validationTargets.join(", ")}` : "",
    "Keep the fix inside this workspace. Do not bypass the reviewed tool path with a workaround.",
    "Preserve concise German user-facing copy where the runtime surfaces status in the UI.",
  ].filter(Boolean).join("\n");
}

function buildAgentCodingHandoff(craft, run, report) {
  const lastToolFailure = buildAgentLastToolFailure(run);
  const workspaceDiagnosis =
    run?.workspaceCodeDive && typeof run.workspaceCodeDive === "object"
      ? cloneJson(run.workspaceCodeDive, null)
      : null;
  const patchTargets = Array.isArray(workspaceDiagnosis?.patchTargets) && workspaceDiagnosis.patchTargets.length
    ? workspaceDiagnosis.patchTargets
    : buildFallbackPatchTargets(run);
  const validationTargets = Array.isArray(workspaceDiagnosis?.validationTargets) && workspaceDiagnosis.validationTargets.length
    ? workspaceDiagnosis.validationTargets
    : buildFallbackValidationTargets(run);
  const executionPath = Array.isArray(workspaceDiagnosis?.executionPath) && workspaceDiagnosis.executionPath.length
    ? workspaceDiagnosis.executionPath
    : buildFallbackExecutionPath(run);
  const prompt =
    String(workspaceDiagnosis?.codingHandoff?.prompt || "").trim() ||
    buildCodingPromptFromRun(craft, run, report, executionPath, patchTargets, validationTargets, lastToolFailure);
  const resolvedErrorDetail = resolveAgentRunErrorDetail(run);

  return {
    format: "patch_handoff_v1",
    repo: {
      kind: "chrome_extension_mv3_sidepanel",
      appPath: "fuck-api-train-local-ai",
    },
    objective: String(report?.objective || craft?.summary || "").trim(),
    failure: {
      runStatus: String(run?.status || ""),
      uiState: getAgentUiState(run),
      failingTool: String(lastToolFailure?.action || workspaceDiagnosis?.query?.failingTool || ""),
      underlyingFailingTool: String(lastToolFailure?.underlyingAction || resolvedErrorDetail?.underlyingFailingTool || ""),
      error: String(lastToolFailure?.error || run?.error || ""),
      errorDetail: resolvedErrorDetail,
    },
    architectureSlice: {
      executionPath,
      files: Array.isArray(workspaceDiagnosis?.files)
        ? workspaceDiagnosis.files.map((entry) => ({
            path: String(entry?.path || ""),
            subsystem: String(entry?.subsystem || ""),
            rationale: String(entry?.rationale || ""),
          }))
        : [],
    },
    patchTargets,
    validationTargets,
    prompt,
  };
}

function createAgentRunClipboardReport(craftId, run) {
  const key = String(craftId || "").trim();
  const craft = findCraft(key);
  const maturity = getCraftMaturity(craft, run);
  const report = normalizeAgentReport(run?.report, {
    currentState: String(run?.responseText || "").trim(),
  });
  const codingHandoff = buildAgentCodingHandoff(craft, run, report);
  const resolvedErrorDetail = resolveAgentRunErrorDetail(run);
  const lastToolFailure = buildAgentLastToolFailure(run);
  return {
    reportVersion: 2,
    type: "crafting_agent_run_debug",
    intent: "coding_handoff",
    timestamp: new Date().toISOString(),
    page: {
      href: globalThis.location.href,
      userAgent: globalThis.navigator?.userAgent || "",
      language: globalThis.navigator?.language || "",
      platform: globalThis.navigator?.platform || "",
    },
    extension: {
      id: globalThis.chrome?.runtime?.id || "",
      version: globalThis.chrome?.runtime?.getManifest?.().version || "",
    },
    craft: craft
      ? {
          id: String(craft.id || ""),
          name: String(craft.name || ""),
          stage: String(craft.stage || ""),
          summary: String(craft.summary || ""),
          agentPrompt: String(craft.agentPrompt || ""),
          inputMode: String(craft.inputMode || ""),
          actionLabel: String(craft.actionLabel || ""),
          inputHint: String(craft.inputHint || ""),
          inputExamples: Array.isArray(craft.inputExamples) ? cloneJson(craft.inputExamples, []) : [],
        }
      : null,
    localUi: {
      craftId: key,
      activeCraftId: String(state.activeCraftId || ""),
      craftingCraftId: String(state.craftingCraftId || ""),
      activeAgentTab: String(run?.activeTab || "progress"),
      promptDraft: String(state.agentPromptDrafts[key] || ""),
    },
    run: run && typeof run === "object"
      ? {
          runId: String(run.runId || run.jobId || ""),
          status: String(run.status || ""),
          phase: String(run.phase || ""),
          mode: String(run.mode || ""),
          modelRef: String(run.modelRef || ""),
          responseText: String(run.responseText || ""),
          finalStatus: normalizeAgentFinalStatus(run.finalStatus),
          uiState: getAgentUiState(run),
          error: String(run.error || ""),
          errorDetail: resolvedErrorDetail,
          tokens: Math.max(0, Number(run.tokens || 0)),
          costUsd: Math.max(0, Number(run.costUsd || 0)),
          turnsUsed: Math.max(0, Number(run.turnsUsed || 0)),
          maxTurns: Math.max(0, Number(run.maxTurns || 0)),
          completedAt: String(run.completedAt || ""),
          activeTab: String(run.activeTab || "progress"),
          report,
          maturity,
          logs: buildAgentLogTail(run, 24),
          toolTrace: buildAgentToolTraceTail(run, 12),
          lastToolFailure,
          workspaceCodeDive: run?.workspaceCodeDive && typeof run.workspaceCodeDive === "object" ? cloneJson(run.workspaceCodeDive, null) : null,
          questions: normalizeAgentQuestions(run.questions),
          provenance: normalizeAgentProvenance(run.provenance),
          suggestedName: normalizeCraftNameCandidate(run.suggestedName),
          officialDescription: String(run.officialDescription || "").trim(),
          operations: Array.isArray(run.operations) ? cloneJson(run.operations, []) : [],
          useSurface: run.useSurface && typeof run.useSurface === "object" ? cloneJson(run.useSurface, null) : null,
          activityRecorded: run.activityRecorded === true,
          useSurfaceApplied: run.useSurfaceApplied === true,
          officialDescriptionApplied: run.officialDescriptionApplied === true,
        }
      : null,
    codingHandoff,
    diagnostics: {
      uiState: getAgentUiState(run),
      progressHeadline: getAgentProgressHeadline(run),
      progressPrimaryText: getAgentProgressPrimaryText(run, report),
      progressSecondaryText: getAgentProgressSecondaryText(run, report, getAgentProgressPrimaryText(run, report)),
      maturity,
      progressEntries: buildAgentProgressEntries(run, report),
      visibleProvenance: buildVisibleAgentProvenance(run),
      workspaceDiagnosis: run?.workspaceCodeDive && typeof run.workspaceCodeDive === "object" ? cloneJson(run.workspaceCodeDive, null) : null,
    },
  };
}

async function copyAgentRunDebugReport(craftId) {
  const key = String(craftId || "").trim();
  if (!key) return;
  const run = state.agentRuns[key];
  if (!run || typeof run !== "object") return;

  run.clipboardStatus = "copying";
  run.clipboardError = "";
  scheduleAgentRunPersist(key);
  refreshCraftCard(key);

  try {
    const payload = createAgentRunClipboardReport(key, run);
    payload.copiedAt = new Date().toISOString();
    await copyTextToClipboard(JSON.stringify(payload, null, 2));
    run.clipboardStatus = "copied";
    run.clipboardError = "";
  } catch (error) {
    run.clipboardStatus = "copy_failed";
    run.clipboardError = error instanceof Error ? error.message : String(error || "Clipboard write failed.");
  }

  scheduleAgentRunPersist(key);
  refreshCraftCard(key);
}

async function maybeFinalizeTrainingRun(craftId) {
  const run = state.trainingRuns[craftId];
  if (!run) return;
  if (!["completed", "failed"].includes(String(run.status || ""))) return;

  storeTrainingRunReport(craftId, run);
  render();

  if (run.activityRecorded !== true) {
    const craft = findCraft(craftId);
    if (craft) {
      recordCraftActivity(craft, {
        label: "training",
        status: String(run.status || "completed"),
        source: run.modelName || "local qwen training",
        input:
          run.totalSamples > 0
            ? `${formatCompactCount(run.completedSamples)} / ${formatCompactCount(run.totalSamples)} samples`
            : "training run",
        output: JSON.stringify(
          {
            baseTestAcc: Number(run.baseTestAcc || 0),
            adaptTestAcc: Number(run.adaptTestAcc || 0),
            adapterSizeMb: Number(run.adapterSizeMb || 0),
            startedAt: String(run.startedAt || ""),
            endedAt: String(run.endedAt || ""),
          },
          null,
          2,
        ),
        meta: String(run.jobId || ""),
        updatedAt: run.endedAt || new Date().toISOString(),
      });
      state.crafts = await persistCrafts(state.crafts);
    }
    run.activityRecorded = true;
  }

  if (run.autoCopyOnComplete !== true) return;
  if (run.clipboardStatus === "copied" || run.clipboardStatus === "copy_failed") return;

  run.clipboardStatus = "copying";
  render();

  try {
    const payload = storeTrainingRunReport(craftId, run, { copiedAt: new Date().toISOString() });
    await copyTextToClipboard(JSON.stringify(payload, null, 2));
    run.clipboardStatus = "copied";
    run.clipboardError = "";
    run.message =
      run.status === "completed"
        ? "Native browser training completed and report copied to clipboard."
        : "Native browser training failed and the failure report was copied to clipboard.";
  } catch (error) {
    run.clipboardStatus = "copy_failed";
    run.clipboardError = error instanceof Error ? error.message : String(error || "Clipboard write failed.");
  }

  render();
}

async function maybeFinalizeDebugTrainingRun() {
  const run = state.debugTrainingRun;
  if (!run) return;
  const status = String(run.status || "");
  const testId = getHeaderTrainingTestId(run);
  const testLabel = getHeaderTrainingTestLabel(testId);
  if (!["completed", "failed"].includes(status)) return;

  if (status === "completed") {
    if (!["success", "hidden"].includes(getHeaderTestState(testId).status)) {
      markHeaderTestSuccess(testId);
    }
    return;
  }

  if (state.smokeTest?.testId === testId && state.smokeTest.status === "error") return;

  const report = createTrainingClipboardReport(
    String(run?.craftId || DEBUG_FIXED_TRAINING.craftId).trim() || DEBUG_FIXED_TRAINING.craftId,
    run,
  );
  markHeaderTestFailure(
    testId,
    `${testLabel} failed.`,
    run.error || run.message || `The ${testLabel.toLowerCase()} test failed.`,
    report,
  );
  render();
}

function scheduleTrainingPoll(craftId) {
  const run = state.trainingRuns[craftId];
  if (!run?.jobId) return;
  if (!isTrainingRunActive(run)) return;

  globalThis.setTimeout(async () => {
    const currentRun = state.trainingRuns[craftId];
    if (!currentRun?.jobId || currentRun.jobId !== run.jobId) return;
    try {
      const response = await sendRuntimeMessage({
        type: "training:get-run-status",
        jobId: run.jobId,
      });
      if (!response?.ok || !response.run) {
        throw new Error(response?.error || "No training status returned.");
      }
      applyTrainingRun(craftId, response.run);
      void maybeFinalizeTrainingRun(craftId);
      scheduleTrainingPoll(craftId);
    } catch (error) {
      const failedRun = state.trainingRuns[craftId];
      if (!failedRun || failedRun.jobId !== run.jobId) return;
      failedRun.status = "failed";
      failedRun.phaseLabel = "Status polling failed";
      failedRun.message = "Native browser training status polling failed.";
      failedRun.error = error instanceof Error ? error.message : String(error || "Status polling failed.");
      void maybeFinalizeTrainingRun(craftId);
      render();
    }
  }, TRAINING_STATUS_POLL_MS);
}

function scheduleDebugTrainingPoll() {
  const run = state.debugTrainingRun;
  if (!run?.jobId) return;
  if (!isTrainingRunActive(run)) return;

  globalThis.setTimeout(async () => {
    const currentRun = state.debugTrainingRun;
    if (!currentRun?.jobId || currentRun.jobId !== run.jobId) return;
    try {
      const response = await sendRuntimeMessage({
        type: "training:get-run-status",
        jobId: run.jobId,
      });
      if (!response?.ok || !response.run) {
        throw new Error(response?.error || "No training status returned.");
      }
      const craftId = String(currentRun?.craftId || "").trim();
      const runUpdate = craftId ? { ...response.run, craftId } : response.run;
      if (craftId) {
        applyTrainingRun(craftId, runUpdate);
        void maybeFinalizeTrainingRun(craftId);
      }
      const nextRun = applyDebugTrainingRun(runUpdate);
      syncHeaderTrainingTestProgress(nextRun);
      void maybeFinalizeDebugTrainingRun();
      scheduleDebugTrainingPoll();
    } catch (error) {
      const failedRun = state.debugTrainingRun;
      if (!failedRun || failedRun.jobId !== run.jobId) return;
      failedRun.status = "failed";
      failedRun.phaseLabel = "Status polling failed";
      failedRun.message = "Native browser training status polling failed.";
      failedRun.error = error instanceof Error ? error.message : String(error || "Status polling failed.");
      const failedCraftId = String(failedRun?.craftId || "").trim();
      if (failedCraftId) {
        state.trainingRuns[failedCraftId] = mergeTrainingRun(state.trainingRuns[failedCraftId] || {}, {
          craftId: failedCraftId,
          headerTestId: failedRun.headerTestId,
          shardId: failedRun.shardId,
          modelName: failedRun.modelName,
          smokeMode: failedRun.smokeMode,
          status: "failed",
          phaseLabel: "Status polling failed",
          message: "Native browser training status polling failed.",
          error: failedRun.error,
        });
        void maybeFinalizeTrainingRun(failedCraftId);
      }
      void maybeFinalizeDebugTrainingRun();
      render();
    }
  }, TRAINING_STATUS_POLL_MS);
}

function stopAgentRunPoll(craftId) {
  const key = String(craftId || "").trim();
  if (!key) return;
  const currentTimer = agentRunPollTimers.get(key);
  if (currentTimer) {
    globalThis.clearTimeout(currentTimer);
    agentRunPollTimers.delete(key);
  }
}

function isFinalAgentRunStatus(status) {
  return ["done", "needs_input", "blocked", "failed", "stopped"].includes(String(status || ""));
}

function mergeAgentQuestionsWithLocalAnswers(nextQuestions, previousQuestions = []) {
  const answersById = new Map(
    normalizeAgentQuestions(previousQuestions).map((entry) => [entry.id, String(entry.answer || "")]),
  );
  return normalizeAgentQuestions(nextQuestions).map((entry) => ({
    ...entry,
    answer: answersById.get(entry.id) || entry.answer || "",
  }));
}

function applyCraftingRunSnapshot(craftId, snapshot) {
  const key = String(craftId || "").trim();
  const craft = findCraft(key);
  const previousRun = state.agentRuns[key] || null;
  const snapshotRun = snapshot && typeof snapshot === "object" ? snapshot : {};
  const nextRunId = String(snapshotRun.jobId || snapshotRun.runId || "");
  const nextQuestions = mergeAgentQuestionsWithLocalAnswers(snapshotRun.questions, previousRun?.questions);
  const rawActiveTab = String(snapshotRun.activeTab || "");
  const activeTab =
    rawActiveTab === "log"
      ? "progress"
      : ["progress", "questions", "provenance"].includes(rawActiveTab)
        ? rawActiveTab
        : "progress";
  const preserveUiState = previousRun && previousRun.runId === nextRunId;

  state.agentRuns[key] = {
    runId: nextRunId,
    status: String(snapshotRun.status || previousRun?.status || "idle"),
    phase: String(snapshotRun.phase || previousRun?.phase || ""),
    mode: String(snapshotRun.mode || previousRun?.mode || ""),
    modelRef: String(snapshotRun.modelRef || previousRun?.modelRef || ""),
    responseText: String(snapshotRun.responseText || previousRun?.responseText || ""),
    finalStatus: normalizeAgentFinalStatus(snapshotRun.finalStatus || previousRun?.finalStatus || ""),
    tokens: Math.max(0, Number(snapshotRun.tokens || 0)),
    costUsd: Math.max(0, Number(snapshotRun.costUsd || 0)),
    turnsUsed: Math.max(0, Number(snapshotRun.turnsUsed || 0)),
    maxTurns: Math.max(0, Number(snapshotRun.maxTurns || 0)),
    error: String(snapshotRun.error || previousRun?.error || ""),
    errorDetail:
      snapshotRun.errorDetail && typeof snapshotRun.errorDetail === "object"
        ? cloneJson(snapshotRun.errorDetail, null)
        : previousRun?.errorDetail && typeof previousRun.errorDetail === "object"
          ? cloneJson(previousRun.errorDetail, null)
          : null,
    logs: Array.isArray(snapshotRun.logs)
      ? snapshotRun.logs
          .map((entry) => normalizeAgentLogEntry(entry))
          .filter((entry) => entry.message || entry.title || entry.detail)
      : Array.isArray(previousRun?.logs)
        ? previousRun.logs
        : [],
    toolTrace: Array.isArray(snapshotRun.toolTrace)
      ? cloneJson(snapshotRun.toolTrace, [])
      : Array.isArray(previousRun?.toolTrace)
        ? previousRun.toolTrace
        : [],
    lastToolFailure:
      snapshotRun.lastToolFailure && typeof snapshotRun.lastToolFailure === "object"
        ? cloneJson(snapshotRun.lastToolFailure, null)
        : previousRun?.lastToolFailure && typeof previousRun.lastToolFailure === "object"
          ? cloneJson(previousRun.lastToolFailure, null)
          : null,
    workspaceCodeDive:
      snapshotRun.workspaceCodeDive && typeof snapshotRun.workspaceCodeDive === "object"
        ? cloneJson(snapshotRun.workspaceCodeDive, null)
        : previousRun?.workspaceCodeDive && typeof previousRun.workspaceCodeDive === "object"
          ? cloneJson(previousRun.workspaceCodeDive, null)
          : null,
    report: normalizeAgentReport(snapshotRun.report, previousRun?.report || {
      currentState: String(snapshotRun.responseText || "").trim(),
    }),
    questions: nextQuestions,
    provenance: normalizeAgentProvenance(snapshotRun.provenance),
    maturity: normalizeCraftMaturity(getCraftMaturity(craft, snapshotRun), snapshotRun.maturity),
    suggestedName: normalizeCraftNameCandidate(snapshotRun.suggestedName || previousRun?.suggestedName || ""),
    officialDescription: String(snapshotRun.officialDescription || previousRun?.officialDescription || "").trim() || null,
    activeTab:
      preserveUiState && !isFinalAgentRunStatus(snapshotRun.status) && previousRun?.activeTab
        ? previousRun.activeTab
        : activeTab,
    operations: Array.isArray(snapshotRun.operations) ? cloneJson(snapshotRun.operations, []) : [],
    useSurface:
      snapshotRun.useSurface && typeof snapshotRun.useSurface === "object"
        ? cloneJson(snapshotRun.useSurface, null)
        : null,
    clipboardStatus: preserveUiState ? String(previousRun?.clipboardStatus || "idle") : "idle",
    clipboardError: preserveUiState ? String(previousRun?.clipboardError || "") : "",
    activityRecorded:
      preserveUiState && previousRun?.activityRecorded === true && isFinalAgentRunStatus(snapshotRun.status),
    useSurfaceApplied:
      preserveUiState && previousRun?.useSurfaceApplied === true && isFinalAgentRunStatus(snapshotRun.status),
    officialDescriptionApplied:
      preserveUiState && previousRun?.officialDescriptionApplied === true && isFinalAgentRunStatus(snapshotRun.status),
    completedAt: String(snapshotRun.completedAt || previousRun?.completedAt || ""),
  };
  // Poll snapshots come from the background runner; echo-persisting them here creates write races.
  refreshCraftCard(key);
  return state.agentRuns[key];
}

async function finalizeCraftingRun(craftId) {
  const key = String(craftId || "").trim();
  const run = state.agentRuns[key];
  if (!run || !isFinalAgentRunStatus(run.status)) return;
  const allowAutoApply = run.status === "done";

  const craft = findCraft(key);
  if (allowAutoApply && craft && !run.officialDescriptionApplied && run.officialDescription) {
    const descriptionMessage = applyAgentOfficialDescription(craft, run.officialDescription);
    if (descriptionMessage) {
      run.logs.push(createAgentLog("success", descriptionMessage));
    }
    run.officialDescriptionApplied = true;
    scheduleAgentRunPersist(key);
  }

  if (allowAutoApply && craft && run.suggestedName) {
    const renameMessage = applyAgentCraftNameSuggestion(craft, run.suggestedName);
    if (renameMessage) {
      run.logs.push(createAgentLog("success", renameMessage));
    }
  }

  if (allowAutoApply && !run.useSurfaceApplied && run.useSurface) {
    const useSurfaceMessage = applyAgentUseSurfaceSuggestion(craft, run.useSurface);
    if (useSurfaceMessage) {
      run.logs.push(createAgentLog("success", useSurfaceMessage));
    }
    run.useSurfaceApplied = true;
    scheduleAgentRunPersist(key);
  }

  if (run.activityRecorded) {
    refreshCraftCard(key);
    return;
  }

  if (!craft) {
    run.activityRecorded = true;
    scheduleAgentRunPersist(key);
    refreshCraftCard(key);
    return;
  }

  if (Number(run.tokens) > 0 || Number(run.costUsd) > 0) {
    incrementCraftSpend(key, {
      totalTokens: Number(run.tokens || 0),
      costUsd: Number(run.costUsd || 0),
    });
  } else {
    touchCraft(craft);
  }

  recordCraftActivity(craft, {
    label: "revision",
    status:
      run.status === "needs_input"
        ? "needs_input"
        : run.status === "blocked"
          ? "blocked"
        : run.status === "failed"
          ? "failed"
          : run.status === "stopped"
            ? "stopped"
            : "completed",
    source: run.modelRef ? `Agent · ${run.modelRef}` : "Agent",
    input: String(state.agentPromptDrafts[key] || craft.agentPrompt || ""),
    output: run.responseText,
    meta: `${craft.stage || "stage unset"}${run.turnsUsed ? ` · ${formatCount(run.turnsUsed)} turns` : ""}${run.operations?.length ? ` · ${formatCount(run.operations.length)} ops` : ""}${run.questions.length ? ` · ${formatCount(run.questions.length)} questions` : ""}`,
  });
  state.crafts = await persistCrafts(state.crafts);
  run.activityRecorded = true;
  scheduleAgentRunPersist(key);
  refreshCraftCard(key);
}

function scheduleAgentRunPoll(craftId) {
  const key = String(craftId || "").trim();
  if (!key) return;
  stopAgentRunPoll(key);
  const run = state.agentRuns[key];
  if (!run?.runId || isFinalAgentRunStatus(run.status)) return;

  const timer = globalThis.setTimeout(async () => {
    const currentRun = state.agentRuns[key];
    if (!currentRun?.runId || currentRun.runId !== run.runId) return;
    try {
      const response = await sendRuntimeMessage({
        type: "agent:get-crafting-run-status",
        jobId: run.runId,
      });
      if (!response?.ok || !response.run) {
        throw new Error(response?.error || "No crafting agent status returned.");
      }
      applyCraftingRunSnapshot(key, response.run);
      const nextRun = state.agentRuns[key];
      if (nextRun && isFinalAgentRunStatus(nextRun.status)) {
        await finalizeCraftingRun(key);
        stopAgentRunPoll(key);
        return;
      }
      scheduleAgentRunPoll(key);
    } catch (error) {
      const failedRun = state.agentRuns[key];
      if (!failedRun || failedRun.runId !== run.runId) return;
      failedRun.status = "failed";
      failedRun.phase = "failed";
      failedRun.error = error instanceof Error ? error.message : String(error || "Status polling failed.");
      failedRun.logs.push(createAgentLog("error", failedRun.error));
      scheduleAgentRunPersist(key);
      await finalizeCraftingRun(key);
      refreshCraftCard(key);
    }
  }, 700);
  agentRunPollTimers.set(key, timer);
}

async function requestAgentPlan(craft, prompt, samples = []) {
  if (!globalThis.chrome?.runtime?.sendMessage) {
    throw new Error("chrome.runtime.sendMessage is unavailable in the side panel.");
  }

  const response = await globalThis.chrome.runtime.sendMessage({
    type: "agent:plan-training-sample-ops",
    slotId: "agent",
    craft: {
      id: craft.id,
      name: craft.name,
      nameSource: String(craft.nameSource || ""),
      summary: craft.summary,
      stage: craft.stage,
      inputMode: getCraftInputMode(craft),
      inputHint: String(craft.inputHint || ""),
      actionLabel: String(craft.actionLabel || ""),
      inputExamples: normalizePromptExampleTexts(craft.inputExamples),
      accuracy: describeCraftAccuracyForAgent(craft),
      seedRows: describeCraftSeedRows(craft),
      datasetRows: describeCraftDatasetRows(craft),
      coverageGaps: describeCraftCoverageGaps(craft),
      tools: Array.isArray(craft.tools) ? craft.tools : [],
      agentTooling: getCraftingAgentTooling(craft),
      targetSlot: craft.targetSlot || "agent",
    },
    brief: prompt,
    currentSamples: summarizeTrainingSamplesForAgent(samples),
  });

  if (!response?.ok) {
    throw new Error(response?.error || "Agent slot returned no result.");
  }

  const plan = response?.object;
  if (!plan || typeof plan !== "object") {
    throw new Error("Agent slot returned no structured revision.");
  }

  const normalizedPlan = normalizeAgentPlanResult({ ...plan, craft }, prompt, samples);
  const text = normalizedPlan.text;
  if (!text) {
    throw new Error("Agent slot returned an empty revision.");
  }

  return {
    mode: response?.resolved?.provider?.type === "local_qwen" ? "local" : "remote",
    modelRef: response?.resolved?.modelRef || "",
    text,
    report: normalizedPlan.report,
    questions: normalizedPlan.questions,
    provenance: normalizedPlan.provenance,
    useSurface: normalizedPlan.useSurface,
    operations: normalizedPlan.operations,
    usage: extractUsageFromResponse(response),
  };
}

function createAgentLog(level, message) {
  return {
    level,
    message,
    time: formatClock(new Date()),
  };
}

function appendAgentLog(craftId, runId, level, message) {
  const run = state.agentRuns[craftId];
  if (!run || run.runId !== runId) return false;
  run.logs.push(createAgentLog(level, message));
  scheduleAgentRunPersist(craftId);
  refreshCraftCard(craftId);
  return true;
}

function setAgentRunTab(craftId, tab) {
  const run = state.agentRuns[craftId];
  if (!run) return;
  const nextTab = ["progress", "questions", "provenance"].includes(String(tab || "")) ? String(tab) : "progress";
  run.activeTab = nextTab;
  scheduleAgentRunPersist(craftId);
  refreshCraftCard(craftId);
}

function canAutoApplyAgentCraftNameSuggestion(craft) {
  if (!craft || typeof craft !== "object") return false;
  const nameSource = String(craft.nameSource || "").trim().toLowerCase();
  if (nameSource === "placeholder") return true;
  if (nameSource === "user" || nameSource === "agent") return false;
  const currentName = String(craft.name || "").trim();
  if (!currentName || isPlaceholderCraftName(currentName)) return true;
  const legacyName = buildLegacyAbilityNameFromDescription(craft.summary || craft.agentPrompt || "");
  return Boolean(legacyName) && normalizeNameComparison(currentName) === normalizeNameComparison(legacyName);
}

function applyAgentCraftNameSuggestion(craft, suggestedName) {
  if (!canAutoApplyAgentCraftNameSuggestion(craft)) return "";
  const candidate = normalizeCraftNameCandidate(suggestedName);
  if (!candidate || isPlaceholderCraftName(candidate)) return "";
  const nextName = ensureUniqueCraftName(candidate, craft.id);
  if (normalizeNameComparison(nextName) === normalizeNameComparison(craft.name)) return "";
  craft.name = nextName;
  craft.nameSource = "agent";
  touchCraft(craft);
  return `Capability renamed to ${nextName}.`;
}

function applyAgentOfficialDescription(craft, officialDescription) {
  if (!craft || typeof craft !== "object") return "";
  const nextDescription = trimText(String(officialDescription || "").trim(), 600);
  if (!nextDescription) return "";
  if (nextDescription === getCraftOfficialDescription(craft)) return "";
  craft.summary = nextDescription;
  touchCraft(craft);
  return "Official capability description updated.";
}

function applyAgentUseSurfaceSuggestion(craft, useSurface) {
  if (!craft || !useSurface || typeof useSurface !== "object") return "";
  const nextMode = normalizeCraftInputMode(useSurface.inputMode, getCraftInputMode(craft));
  const nextHint = String(useSurface.inputHint || "").trim();
  const nextPlaceholder = String(useSurface.inputPlaceholder || "").trim();
  const nextActionLabel = String(useSurface.actionLabel || "").trim();
  const nextExamples = normalizePromptExampleTexts(useSurface.inputExamples);

  let changed = false;
  if (nextMode && nextMode !== getCraftInputMode(craft)) {
    craft.inputMode = nextMode;
    changed = true;
  }
  if (nextHint && nextHint !== String(craft.inputHint || "")) {
    craft.inputHint = nextHint;
    changed = true;
  }
  if (craftUsesTextInputFromMode(nextMode)) {
    if (nextPlaceholder && nextPlaceholder !== String(craft.inputPlaceholder || "")) {
      craft.inputPlaceholder = nextPlaceholder;
      changed = true;
    }
  } else if (String(craft.inputPlaceholder || "")) {
    craft.inputPlaceholder = "";
    changed = true;
  }
  if (nextActionLabel && nextActionLabel !== String(craft.actionLabel || "")) {
    craft.actionLabel = nextActionLabel;
    changed = true;
  }
  if (nextExamples.length) {
    const currentExamples = normalizePromptExampleTexts(craft.inputExamples);
    if (JSON.stringify(currentExamples) !== JSON.stringify(nextExamples)) {
      craft.inputExamples = nextExamples;
      changed = true;
    }
  }

  if (!changed) return "";
  touchCraft(craft);
  const modeLabel = nextMode === "selection"
    ? "browser selection without free text"
    : nextMode === "current_tab"
      ? "current page without free text"
      : nextMode === "context_only"
        ? "fixed context without free text"
        : nextMode === "mixed"
          ? "context plus text"
          : "free text";
  return `Use surface updated: ${modeLabel}${nextExamples.length ? ` · ${formatCount(nextExamples.length)} examples` : ""}.`;
}

function incrementCraftSpend(craftId, usage) {
  const craft = findCraft(craftId);
  if (!craft || !usage) return;
  craft.tokenSpend = Number(craft.tokenSpend || 0) + Number(usage.totalTokens || 0);
  craft.costUsd = Number(craft.costUsd || 0) + Number(usage.costUsd || 0);
  touchCraft(craft);
}

function extractUsageFromResponse(response) {
  const usage = response?.usage || response?.result?.usage || response?.meta?.usage || null;
  if (!usage || typeof usage !== "object") return null;

  const inputTokens = Number(
    usage.inputTokens ??
      usage.promptTokens ??
      usage.input_tokens ??
      usage.prompt_tokens ??
      usage.requestTokens ??
      0,
  );
  const outputTokens = Number(
    usage.outputTokens ??
      usage.completionTokens ??
      usage.output_tokens ??
      usage.completion_tokens ??
      usage.responseTokens ??
      0,
  );
  const totalTokens = Number(usage.totalTokens ?? usage.total_tokens ?? inputTokens + outputTokens);
  const costUsd = Number(usage.costUsd ?? usage.cost_usd ?? 0);

  if (inputTokens <= 0 && outputTokens <= 0 && totalTokens <= 0) return null;

  return {
    inputTokens: Math.max(0, inputTokens),
    outputTokens: Math.max(0, outputTokens),
    totalTokens: Math.max(0, totalTokens),
    costUsd: Math.max(0, costUsd),
  };
}

function touchCraft(craft, timestamp = new Date().toISOString()) {
  if (!craft || typeof craft !== "object") return;
  craft.updatedAt = String(timestamp || "");
}

function recordCraftActivity(craft, record) {
  if (!craft || typeof craft !== "object") return;
  const now = String(record?.updatedAt || new Date().toISOString());
  const nextRecord = {
    id: `record-${Date.now()}-${Math.random().toString(16).slice(2, 8)}`,
    label: String(record?.label || "activity"),
    status: String(record?.status || "recorded"),
    source: String(record?.source || ""),
    input: String(record?.input || ""),
    output: String(record?.output || ""),
    meta: String(record?.meta || ""),
    updatedAt: now,
  };
  const current = Array.isArray(craft.navigatorRecords) ? craft.navigatorRecords : [];
  craft.navigatorRecords = [nextRecord, ...current].slice(0, 20);
  touchCraft(craft, now);
}

function openBundleTab(craftId) {
  const url = buildCraftPageUrl(craftId, { tab: "model" });
  openExtensionPage(url);
}

function openNavigatorTab(craftId) {
  const url = buildCraftPageUrl(craftId, { tab: "browser-functions" });
  openExtensionPage(url);
}

function openTrainingDataTab(craftId) {
  const url = buildCraftPageUrl(craftId, { tab: "training-data" });
  openExtensionPage(url);
}

function openOptionsPage(section = "") {
  const normalizedSection = String(section || "").trim();
  const url = buildOptionsPageUrl(normalizedSection);
  openExtensionPage(url, { allowRuntimeOptionsFallback: !normalizedSection });
}

function buildOptionsPageUrl(section = "", params = {}) {
  const baseUrl = globalThis.chrome?.runtime?.getURL
    ? globalThis.chrome.runtime.getURL("options.html")
    : new URL("options.html", globalThis.location.href).toString();
  const url = new URL(baseUrl);
  const normalizedSection = String(section || "").trim();

  if (normalizedSection) {
    url.searchParams.set("section", normalizedSection);
    url.hash = normalizedSection;
  }

  Object.entries(params).forEach(([key, value]) => {
    const normalizedValue = String(value || "").trim();
    if (normalizedValue) {
      url.searchParams.set(key, normalizedValue);
    }
  });

  return url.toString();
}

function buildCraftPageUrl(craftId = "", params = {}) {
  const baseUrl = globalThis.chrome?.runtime?.getURL
    ? globalThis.chrome.runtime.getURL("craft.html")
    : new URL("craft.html", globalThis.location.href).toString();
  const url = new URL(baseUrl);
  const normalizedCraftId = String(craftId || params.craft || "").trim();
  const normalizedTab = String(params.tab || "model").trim() || "model";

  if (normalizedCraftId) {
    url.searchParams.set("craft", normalizedCraftId);
  }
  url.searchParams.set("tab", normalizedTab);
  url.hash = normalizedTab;

  return url.toString();
}

function openExtensionPage(url, { allowRuntimeOptionsFallback = false } = {}) {
  if (globalThis.chrome?.tabs?.create) {
    globalThis.chrome.tabs.create({ url });
    return;
  }

  if (allowRuntimeOptionsFallback && globalThis.chrome?.runtime?.openOptionsPage) {
    globalThis.chrome.runtime.openOptionsPage();
    return;
  }

  if (typeof globalThis.open === "function") {
    globalThis.open(url, "_blank", "noopener");
    return;
  }

  globalThis.location.href = url;
}

async function saveInlineSetup() {
  const draftSetup = getCurrentSetupStatus();
  if (!state.setupLoaded || !draftSetup.ready || state.setupSaving) {
    if (!state.setupLoaded) {
      state.setupError = "Setup is still loading. Wait a moment and try again.";
      render();
      return;
    }
    if (!draftSetup.ready) {
      state.setupError = draftSetup.missingItems[0]?.reason || "Complete the required setup fields first.";
      render();
    }
    return;
  }

  state.setupSaving = true;
  state.setupError = "";
  state.requiredSetup = draftSetup;
  state.setupLoaded = true;
  state.setupCompleted = true;
  render();
  try {
    if (themeApi?.writeTheme) {
      await withTimeout(
        themeApi.writeTheme(state.themeId),
        SETUP_SAVE_TIMEOUT_MS,
        "Saving the theme timed out.",
      );
    }

    const nextProviders = configApi.patchProviderRecord(state.providers, "openai", {
      enabled: true,
      apiKey: String(state.providers?.openai?.apiKey || ""),
    });
    state.providers = nextProviders;

    state.providers = await withTimeout(
      configApi.writeProviders(state.providers),
      SETUP_SAVE_TIMEOUT_MS,
      "Saving provider setup timed out.",
    );
    state.slots = await withTimeout(
      configApi.writeModelSlots(state.slots, state.providers),
      SETUP_SAVE_TIMEOUT_MS,
      "Saving model setup timed out.",
    );
    if (craftSync?.updateSettings) {
      const savedSyncSettings = await withTimeout(
        craftSync.updateSettings(state.syncSettings),
        SETUP_SAVE_TIMEOUT_MS,
        "Saving share settings timed out.",
      );
      if (savedSyncSettings && typeof savedSyncSettings === "object") {
        state.syncSettings = savedSyncSettings;
      }
      state.syncSnapshot = craftSync?.getState?.() || state.syncSnapshot;
    }
    state.requiredSetup = getCurrentSetupStatus();
    await withTimeout(
      craftSync?.setValue?.(SIDEPANEL_SETUP_COMPLETED_KEY, {
        done: true,
        completedAt: new Date().toISOString(),
      }),
      SETUP_SAVE_TIMEOUT_MS,
      "Saving setup completion timed out.",
    );
    syncCraftSyncSubscriptionState();
  } catch (error) {
    state.setupError = error instanceof Error ? error.message : String(error || "Failed to save setup.");
    state.smokeTest = {
      ...createEmptySmokeTestState(),
      status: "error",
      message: "Setup could not be persisted completely.",
      detail: state.setupError,
      updatedAt: formatClock(new Date()),
    };
  } finally {
    state.setupSaving = false;
    render();
  }
}

async function refreshSetupState(options = {}) {
  if (!configApi) {
    state.setupLoaded = true;
    state.requiredSetup = { ready: true, missingItems: [] };
    state.setupCompleted = true;
    if (options?.syncSubscription !== false) {
      syncCraftSyncSubscriptionState();
    }
    return;
  }

  state.providers = await configApi.readProviders();
  state.slots = await configApi.readModelSlots(state.providers);
  state.syncSettings = (await craftSync?.readSettings?.()) || state.syncSettings;
  state.requiredSetup = configApi.getRequiredSetupStatus(state.slots, state.providers);
  const completedRecord = await craftSync?.getValue?.(SIDEPANEL_SETUP_COMPLETED_KEY, null);
  state.setupCompleted = completedRecord?.done === true || completedRecord === true;
  state.setupLoaded = true;
  if (options?.syncSubscription !== false) {
    syncCraftSyncSubscriptionState();
  }
}

async function refreshStarterTutorialState() {
  try {
    state.starterTutorialSeen = (await craftSync?.getValue?.(STARTER_TUTORIAL_SEEN_KEY, false)) === true;
  } catch (error) {
    console.warn("[sidepanel] failed to read starter tutorial flag", error);
    state.starterTutorialSeen = false;
  }
}

function resetStarterTutorialSeen() {
  if (!state.starterTutorialSeen) return;
  state.starterTutorialSeen = false;
  Promise.resolve(
    craftSync?.deleteValue?.(STARTER_TUTORIAL_SEEN_KEY) ?? craftSync?.setValue?.(STARTER_TUTORIAL_SEEN_KEY, false),
  ).catch((error) => {
    console.warn("[sidepanel] failed to reset starter tutorial flag", error);
  });
}

function rememberStarterTutorialSeen() {
  if (state.starterTutorialSeen) return;
  state.starterTutorialSeen = true;
  void craftSync?.setValue?.(STARTER_TUTORIAL_SEEN_KEY, true).catch((error) => {
    console.warn("[sidepanel] failed to persist starter tutorial flag", error);
  });
}

function getCurrentSetupStatus() {
  if (!configApi) return state.requiredSetup || { ready: true, items: [], missingItems: [] };
  return configApi.getRequiredSetupStatus(state.slots, state.providers);
}

function getRequiredSetupSlotIds() {
  return Array.isArray(configApi?.FIRST_RUN_REQUIRED_SLOT_IDS)
    ? configApi.FIRST_RUN_REQUIRED_SLOT_IDS
    : ["agent", "batch", "vision"];
}

function capitalizeReasoning(value) {
  const text = String(value || "").trim();
  if (!text) return "Off";
  return `${text.slice(0, 1).toUpperCase()}${text.slice(1)}`;
}

function isAutoGeneratedSyncToken(settings = state.syncSettings) {
  return settings?.tokenAutoGenerated === true;
}

function hasCustomSyncPassword(settings = state.syncSettings) {
  return Boolean(String(settings?.token || "").trim()) && settings?.tokenAutoGenerated !== true;
}

function canEnableCraftSharing() {
  return hasCustomSyncPassword(state.syncSettings);
}

function getHeaderTestState(testId) {
  return state.headerTests?.[testId] || createHeaderTestEntry();
}

function clearHeaderTestHideTimer(testId) {
  const timer = headerTestHideTimers.get(testId);
  if (!timer) return;
  globalThis.clearTimeout(timer);
  headerTestHideTimers.delete(testId);
}

function clearSmokeTestBanner() {
  state.smokeTest = createEmptySmokeTestState();
}

function isSmokeTestRunning() {
  return Object.values(state.headerTests || {}).some((entry) => String(entry?.status || "") === "running");
}

function startHeaderTestRun(testId) {
  clearHeaderTestHideTimer(testId);
  state.headerTests = {
    ...state.headerTests,
    [testId]: createHeaderTestEntry("running", 0.08, false),
  };
  clearSmokeTestBanner();
}

function updateHeaderTestRun(testId, { progress, indeterminate } = {}) {
  const current = getHeaderTestState(testId);
  if (current.status !== "running") return;

  state.headerTests = {
    ...state.headerTests,
    [testId]: {
      ...current,
      status: "running",
      progress:
        progress == null
          ? current.progress
          : Math.max(0, Math.min(1, Number(progress) || 0)),
      indeterminate: indeterminate == null ? current.indeterminate === true : indeterminate === true,
    },
  };
  render();
}

function syncHeaderInferenceBenchmarkProgress(payload = {}) {
  const benchmarkId = String(payload?.benchmarkId || "").trim();
  if (!benchmarkId || benchmarkId !== activeInferenceBenchmarkId) return;
  const current = getHeaderTestState("inference");
  if (current.status !== "running") return;

  const stageProgress = Math.max(0, Math.min(1, Number(payload?.progress || 0)));
  const overallProgress =
    INFERENCE_BENCHMARK_PROGRESS_START +
    (INFERENCE_BENCHMARK_PROGRESS_END - INFERENCE_BENCHMARK_PROGRESS_START) * stageProgress;
  updateHeaderTestRun("inference", {
    progress: Math.max(INFERENCE_BENCHMARK_PROGRESS_START, Math.min(INFERENCE_BENCHMARK_PROGRESS_END, overallProgress)),
    indeterminate: false,
  });
}

function syncHeaderTrainingTestProgress(run) {
  const testId = getHeaderTrainingTestId(run);
  const current = getHeaderTestState(testId);
  if (current.status !== "running") return;

  const rawProgress = Number(run?.progress || 0);
  const phaseTotal = Number(run?.phaseTotalSamples || 0);
  const phaseDone = Number(run?.phaseCompletedSamples || 0);
  const fallbackProgress =
    phaseTotal > 0 ? Math.max(0, Math.min(1, phaseDone / phaseTotal)) : current.progress || 0.08;
  const progress = rawProgress > 0 ? rawProgress : fallbackProgress;
  const indeterminate = progress <= 0;
  updateHeaderTestRun(testId, {
    progress: indeterminate ? 0.08 : Math.max(0.08, Math.min(0.96, progress)),
    indeterminate,
  });
}

function markHeaderTestSuccess(testId) {
  clearHeaderTestHideTimer(testId);
  state.headerTests = {
    ...state.headerTests,
    [testId]: createHeaderTestEntry("success", 1, false),
  };
  clearSmokeTestBanner();
  void writeHeaderTestSessionState();
  render();

  const timer = globalThis.setTimeout(() => {
    if (getHeaderTestState(testId).status !== "success") return;
    state.headerTests = {
      ...state.headerTests,
      [testId]: createHeaderTestEntry("hidden", 1, false),
    };
    void writeHeaderTestSessionState();
    render();
  }, HEADER_TEST_SUCCESS_HIDE_MS);
  headerTestHideTimers.set(testId, timer);
}

function markHeaderTestFailure(testId, message, detail = "", report = null) {
  clearHeaderTestHideTimer(testId);
  state.headerTests = {
    ...state.headerTests,
    [testId]: createHeaderTestEntry(),
  };
  state.smokeTest = {
    status: "error",
    message: String(message || "Test failed."),
    detail: buildSmokeFailureDetail(detail, report),
    helpText: buildSmokeFailureHelpText(report),
    failureKind: buildSmokeFailureKind(report),
    updatedAt: formatClock(new Date()),
    reportText:
      report && typeof report === "object"
        ? JSON.stringify(report, null, 2)
        : String(report || ""),
    copyStatus: "idle",
    testId,
  };
}

async function copyCurrentSmokeErrorReport() {
  const reportText = String(state.smokeTest?.reportText || "");
  if (!reportText) return;

  try {
    await copyTextToClipboard(reportText);
    state.smokeTest = {
      ...state.smokeTest,
      copyStatus: "copied",
    };
  } catch (error) {
    state.smokeTest = {
      ...state.smokeTest,
      detail: error instanceof Error ? error.message : String(error || "Clipboard write failed."),
      copyStatus: "idle",
    };
  }
  render();
}

function createSmokeDiagnosis({ failureKind = "", summary = "", nextAction = "", keyFacts = [] } = {}) {
  return {
    failureKind: String(failureKind || "").trim(),
    summary: String(summary || "").trim(),
    nextAction: String(nextAction || "").trim(),
    keyFacts: Array.isArray(keyFacts)
      ? keyFacts
          .map((entry) => ({
            label: String(entry?.label || "").trim(),
            value: String(entry?.value || "").trim(),
          }))
          .filter((entry) => entry.label && entry.value)
      : [],
  };
}

function formatFailureKindLabel(value) {
  return String(value || "")
    .trim()
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .toLowerCase();
}

function getReportDiagnosis(report) {
  return report && typeof report === "object" && report.diagnosis && typeof report.diagnosis === "object"
    ? report.diagnosis
    : null;
}

function buildSmokeFailureDetail(detail, report = null) {
  const diagnosis = getReportDiagnosis(report);
  if (diagnosis?.summary) return diagnosis.summary;
  return String(detail || "");
}

function buildSmokeFailureHelpText(report = null) {
  return String(getReportDiagnosis(report)?.nextAction || "");
}

function buildSmokeFailureKind(report = null) {
  return String(getReportDiagnosis(report)?.failureKind || "");
}

function attachReportDiagnosis(report, diagnosis) {
  if (!report || typeof report !== "object") return null;
  report.diagnosis = diagnosis && typeof diagnosis === "object" ? diagnosis : null;
  return report.diagnosis;
}

function getRuntimeModelIdFromReport(report) {
  return (
    report?.result?.modelRef ||
    report?.response?.run?.modelRef ||
    report?.run?.modelRef ||
    report?.result?.runtime?.runtimeModelId ||
    report?.response?.runtime?.runtimeModelId ||
    report?.diagnostic?.offscreen?.runtime?.runtimeModelId ||
    report?.resolve?.modelRef ||
    report?.resolve?.modelName ||
    "unknown"
  );
}

function diagnoseInferenceSmokeReport(report) {
  const runtimeModelId = getRuntimeModelIdFromReport(report);
  const reply = String(report?.result?.text || "").trim() || "<empty>";
  const benchmarkRequested = Boolean(report?.request?.benchmark);
  const benchmarkMissing =
    benchmarkRequested &&
    !(
      report?.result?.benchmark &&
      typeof report.result.benchmark === "object" &&
      Number(report?.throughput?.overallForwardTokensPerSecond || 0) > 0
    );

  if (report?.configError || report?.resolveError) {
    return createSmokeDiagnosis({
      failureKind: "config",
      summary: "The local inference smoke test could not resolve the configured target model.",
      nextAction: "Check the target slot and provider setup, then rerun the test.",
      keyFacts: [
        { label: "configError", value: String(report?.configError || "") || "none" },
        { label: "resolveError", value: String(report?.resolveError || "") || "none" },
      ],
    });
  }

  if (benchmarkMissing) {
    return createSmokeDiagnosis({
      failureKind: "runtime",
      summary: trimText(
        String(
          report?.error ||
            "The local inference correctness call succeeded, but the requested forward benchmark did not return throughput data.",
        ),
        180,
      ),
      nextAction: "Reload the extension once so the latest service worker and offscreen runtime are active, then rerun Test Inference.",
      keyFacts: [
        { label: "model", value: runtimeModelId },
        { label: "reply", value: trimText(reply, 80) },
      ],
    });
  }

  if (report?.error) {
    return createSmokeDiagnosis({
      failureKind: "runtime",
      summary: trimText(String(report.error || "The local inference test failed."), 180),
      nextAction: "Reload the extension once. If it still fails, inspect the copied runtime report.",
      keyFacts: [
        { label: "model", value: runtimeModelId },
        { label: "reply", value: trimText(reply, 80) },
      ],
    });
  }

  return createSmokeDiagnosis({
    failureKind: "model_behavior",
    summary: `The model replied "${trimText(reply, 80)}" instead of the required token OK.`,
    nextAction: "Use a stronger local model if this slot must follow short literal prompts reliably.",
    keyFacts: [
      { label: "model", value: runtimeModelId },
      { label: "reply", value: trimText(reply, 80) },
    ],
  });
}

function diagnoseAgentSmokeReport(report) {
  const runtimeModelId = getRuntimeModelIdFromReport(report);
  const run = report?.response?.run && typeof report.response.run === "object"
    ? report.response.run
    : report?.run && typeof report.run === "object"
      ? report.run
      : null;
  const steps = Array.isArray(run?.steps) ? run.steps : [];
  const usedReviewedCapability = steps.some((step) => {
    const capabilityName = String(step?.execution?.capability?.name || "").trim().toLowerCase();
    return capabilityName === "classify_review_sentiment";
  });
  const finalLabel = String(
    run?.result?.label ||
      steps[steps.length - 1]?.execution?.finalOutput?.label ||
      "",
  ).trim().toLowerCase();

  if (report?.resolveError) {
    return createSmokeDiagnosis({
      failureKind: "config",
      summary: "The local agentic smoke test could not resolve the forced local_qwen model.",
      nextAction: "Check the local_qwen provider entry and rerun the test.",
      keyFacts: [{ label: "resolveError", value: String(report.resolveError || "") }],
    });
  }

  if (report?.error) {
    return createSmokeDiagnosis({
      failureKind: "runtime",
      summary: trimText(String(report.error || "The local agentic test failed."), 180),
      nextAction: "Inspect the copied report for the craft run payload and local runtime details, then rerun the test.",
      keyFacts: [
        { label: "model", value: runtimeModelId },
        { label: "durationMs", value: String(Math.round(Number(report?.totalDurationMs || 0))) },
      ],
    });
  }

  if (String(run?.status || "").trim().toLowerCase() === "needs_input") {
    return createSmokeDiagnosis({
      failureKind: "model_behavior",
      summary: "The local agent asked a follow-up question instead of completing the reviewed smoke capability run.",
      nextAction: "Use a stronger local model or simplify the reviewed capability prompt so the run can finish in one pass.",
      keyFacts: [
        { label: "model", value: runtimeModelId },
        { label: "stepCount", value: String(steps.length) },
      ],
    });
  }

  if (!usedReviewedCapability) {
    return createSmokeDiagnosis({
      failureKind: "model_behavior",
      summary: "The local agent returned a run result without invoking the reviewed smoke capability.",
      nextAction: "Use a stronger local agent model or inspect the copied run report to see why no call_tool action was chosen.",
      keyFacts: [
        { label: "model", value: runtimeModelId },
        { label: "stepCount", value: String(steps.length) },
      ],
    });
  }

  if (finalLabel !== "positive") {
    return createSmokeDiagnosis({
      failureKind: "model_quality",
      summary: `The reviewed capability run finished, but the final label was "${finalLabel || "<empty>"}" instead of "positive".`,
      nextAction: "Inspect the copied run report and use a stronger local model if the reviewed capability transform is too weak.",
      keyFacts: [
        { label: "model", value: runtimeModelId },
        { label: "stepCount", value: String(steps.length) },
      ],
    });
  }

  return createSmokeDiagnosis({
    failureKind: "model_behavior",
    summary: "The local agent run returned a response, but it did not satisfy the reviewed capability smoke criteria.",
    nextAction: "Inspect the copied run report and simplify the smoke capability if the local model still struggles.",
    keyFacts: [{ label: "model", value: runtimeModelId }],
  });
}

function diagnoseVisionSmokeReport(report) {
  const responses = Array.isArray(report?.responses) ? report.responses : [];
  const runtimeModelId = getRuntimeModelIdFromReport(report);
  const failedCalls = responses.filter((entry) => !entry?.ok);
  const mismatches = responses.filter((entry) => entry?.ok && entry.normalizedText !== entry.expected);
  const passedCount = responses.length - failedCalls.length - mismatches.length;

  if (report?.resolveError) {
    return createSmokeDiagnosis({
      failureKind: "config",
      summary: "The local vision smoke test could not resolve the forced local_qwen vision model.",
      nextAction: "Check the local_qwen provider entry and rerun the test.",
      keyFacts: [{ label: "resolveError", value: String(report.resolveError || "") }],
    });
  }

  if (failedCalls.length) {
    return createSmokeDiagnosis({
      failureKind: "runtime",
      summary: trimText(String(failedCalls[0]?.error || report?.error || "The local vision runtime failed."), 180),
      nextAction: "Reload the extension once. If it still fails, inspect the copied runtime report.",
      keyFacts: [
        { label: "model", value: runtimeModelId },
        { label: "failedCalls", value: String(failedCalls.length) },
      ],
    });
  }

  const mismatchSummary = mismatches
    .map((entry) => `${entry.expected}->${entry.normalizedText || "<empty>"}`)
    .join(", ");
  const collapsedAnswer = new Set(responses.map((entry) => entry.normalizedText).filter(Boolean)).size <= 1;
  return createSmokeDiagnosis({
    failureKind: "model_quality",
    summary: `${passedCount}/${responses.length || 0} vision cases passed.${mismatchSummary ? ` Mismatches: ${mismatchSummary}.` : ""}`,
    nextAction: collapsedAnswer
      ? "The small 0.8B vision model appears to collapse to one label. Use a stronger local vision model."
      : "Use a stronger local vision model or a more diagnostic image set.",
    keyFacts: [
      { label: "model", value: runtimeModelId },
      { label: "cases", value: mismatchSummary || "none" },
    ],
  });
}

function diagnoseTrainingRunReport(run) {
  const status = String(run?.status || "").trim().toLowerCase();
  const error = String(run?.error || "").trim();
  const phaseLabel = String(run?.phaseLabel || "").trim();
  const smokeMode = String(run?.smokeMode || "").trim();

  if (status === "completed") {
    return createSmokeDiagnosis({
      summary:
        smokeMode === "pipeline_e2e"
          ? "The local finetuning smoke run completed successfully, including the staged pipeline preflight."
          : "The local finetuning smoke run completed successfully.",
    });
  }

  if (smokeMode === "pipeline_e2e" && error.includes("Pipeline smoke validation failed")) {
    return createSmokeDiagnosis({
      failureKind: "pipeline_smoke",
      summary: trimText(error, 180),
      nextAction: "Inspect the copied report for the failing pipeline stage, then rerun the finetuning smoke test.",
      keyFacts: [{ label: "phase", value: phaseLabel || "pipeline smoke" }],
    });
  }

  if (phaseLabel === "Training unavailable" || error.includes("could not be started")) {
    return createSmokeDiagnosis({
      failureKind: "runtime",
      summary: trimText(error || "The local finetuning smoke run could not be started.", 180),
      nextAction: "Check the local training runtime and rerun the test.",
      keyFacts: [{ label: "phase", value: phaseLabel || "unknown" }],
    });
  }

  return createSmokeDiagnosis({
    failureKind: "training_runtime",
    summary: trimText(error || phaseLabel || "The local finetuning smoke run failed.", 180),
    nextAction: "Inspect the copied run report for the failing phase and retry after fixing the local runtime.",
    keyFacts: [
      { label: "phase", value: phaseLabel || "unknown" },
      { label: "status", value: status || "unknown" },
    ],
  });
}

function createBaseSmokeReport(type, slotId, request = {}) {
  return {
    reportVersion: 2,
    type,
    timestamp: new Date().toISOString(),
    page: {
      href: globalThis.location.href,
      userAgent: globalThis.navigator?.userAgent || "",
      language: globalThis.navigator?.language || "",
      platform: globalThis.navigator?.platform || "",
    },
    extension: {
      id: globalThis.chrome?.runtime?.id || "",
      version: globalThis.chrome?.runtime?.getManifest?.().version || "",
    },
    request: {
      slotId,
      ...request,
    },
    timingsMs: {},
    diagnosis: null,
  };
}

function buildWorkspaceRoundtripValuePreview(value) {
  try {
    if (typeof value === "string") {
      return trimText(value, 320);
    }
    return trimText(JSON.stringify(value), 320);
  } catch (_error) {
    return trimText(String(value == null ? "" : value), 320);
  }
}

function normalizeWorkspaceRoundtripArtifactIds(artifactIds) {
  return Array.from(
    new Set(
      (Array.isArray(artifactIds) ? artifactIds : [])
        .map((artifactId) => String(artifactId || "").trim())
        .filter(Boolean),
    ),
  ).sort();
}

function normalizeWorkspaceRoundtripUiText(value) {
  return String(value || "").replace(/\s+/g, " ").trim();
}

function summarizeWorkspaceRoundtripUiSnapshot(snapshot) {
  return {
    tab: String(snapshot?.tab || "").trim(),
    title: String(snapshot?.title || "").trim(),
    matched: snapshot?.matched === true,
    missing: Array.isArray(snapshot?.missing) ? [...snapshot.missing] : [],
    forbidden: Array.isArray(snapshot?.forbidden) ? [...snapshot.forbidden] : [],
    text: buildWorkspaceRoundtripValuePreview(String(snapshot?.text || "").trim()),
  };
}

function collectWorkspaceRoundtripUiFormValues(root) {
  if (!root?.querySelectorAll) return "";
  const values = [];
  root.querySelectorAll("textarea, input, select").forEach((element) => {
    const tagName = String(element?.tagName || "").toLowerCase();
    if (tagName === "select") {
      const selectedText = normalizeWorkspaceRoundtripUiText(
        element.options?.[element.selectedIndex]?.textContent || element.value || "",
      );
      if (selectedText) values.push(selectedText);
      return;
    }
    const value = normalizeWorkspaceRoundtripUiText(element.value || "");
    if (value) values.push(value);
  });
  return values.join("\n");
}

function getWorkspaceRoundtripUiExpectations() {
  return [
    {
      tab: "overview",
      expectedAll: ["Data artifact", "Browser runtime", "Model artifact", "Facts"],
      forbiddenText: [],
    },
    {
      tab: "model",
      expectedAll: ["Training output", "validationEvalAcc", "local_qwen/qwen3.5-0.8b-q4f16"],
      forbiddenText: [],
    },
    {
      tab: "skills",
      expectedAll: ["Linked skills", "relation_graphing", "Bundle skill"],
      forbiddenText: ["No explicit skill refs have been linked to this craft yet."],
    },
    {
      tab: "browser-functions",
      expectedAll: ["Read active text target", "Execute script", "__callReviewedTool"],
      forbiddenText: ["No reviewed browser functions are available for this craft yet."],
    },
    {
      tab: "tool-scripts",
      expectedAll: ["Tool script source", "Declared tools", "readActiveTextTarget"],
      forbiddenText: ["No reviewed tool scripts are available for this craft yet."],
    },
    {
      tab: "contracts",
      expectedAll: ["Parameter schema", "Preconditions", "Examples"],
      forbiddenText: ["No reviewed browser functions are available for this craft yet."],
    },
    {
      tab: "resources",
      expectedAll: ["Craft resources", "references/relation-graph-schema.json"],
      forbiddenText: ["No resources have been attached to this craft yet."],
    },
    {
      tab: "training-data",
      expectedAll: ["workspace-roundtrip-sample-train", "Read the active text, build a relation map", "Raw multi-turn row JSON"],
      forbiddenText: ["No samples yet. Start the agent or add the first seed row manually."],
    },
    {
      tab: "lineage",
      expectedAll: ["Recent activity", "seed review", "policy bundle"],
      forbiddenText: ["No activity has been recorded for this craft yet."],
    },
  ];
}

async function readWorkspaceRoundtripUiSnapshot(craftId, tab, expectation = {}) {
  const iframe = document.createElement("iframe");
  iframe.setAttribute("aria-hidden", "true");
  iframe.tabIndex = -1;
  iframe.style.cssText = [
    "position:fixed",
    "left:-10000px",
    "top:0",
    "width:1440px",
    "height:1800px",
    "opacity:0",
    "pointer-events:none",
    "border:0",
  ].join(";");
  iframe.src = buildCraftPageUrl(craftId, { tab });

  const loaded = new Promise((resolve, reject) => {
    iframe.addEventListener("load", () => resolve(), { once: true });
    iframe.addEventListener("error", () => reject(new Error(`Workspace iframe failed to load for ${tab}.`)), {
      once: true,
    });
  });

  document.body.appendChild(iframe);

  try {
    await loaded;
    const timeoutAt = Date.now() + 12000;
    let lastSnapshot = null;
    while (Date.now() < timeoutAt) {
      const doc = iframe.contentDocument;
      const panel = doc?.querySelector?.(".workspace-panel") || doc?.body || null;
      const text = normalizeWorkspaceRoundtripUiText(
        [
          panel?.textContent || "",
          collectWorkspaceRoundtripUiFormValues(panel),
        ]
          .filter(Boolean)
          .join("\n"),
      );
      const title = normalizeWorkspaceRoundtripUiText(doc?.title || "");

      if (
        !text ||
        text.includes("Loading crafts...") ||
        text.includes("Preparing craft workspace.") ||
        text.includes("Preparing training samples from the local artifact store.") ||
        text.includes("No crafts yet. Create a craft in the side panel to inspect it here.") ||
        text.includes("No crafts yet. Create a craft in the side panel to inspect its training data here.")
      ) {
        await sleep(120);
        continue;
      }

      const expectedAll = Array.isArray(expectation?.expectedAll) ? expectation.expectedAll : [];
      const forbiddenText = Array.isArray(expectation?.forbiddenText) ? expectation.forbiddenText : [];
      const missing = expectedAll.filter((entry) => !text.includes(entry));
      const forbidden = forbiddenText.filter((entry) => text.includes(entry));
      const snapshot = {
        tab,
        title,
        text,
        matched: missing.length === 0 && forbidden.length === 0,
        missing,
        forbidden,
      };
      if (snapshot.matched) {
        return snapshot;
      }
      lastSnapshot = snapshot;
      await sleep(120);
    }
    if (lastSnapshot) {
      return lastSnapshot;
    }
    throw new Error(`Workspace tab ${tab} did not finish rendering before timeout.`);
  } finally {
    try {
      iframe.remove();
    } catch (_error) {}
  }
}

function pushWorkspaceRoundtripMismatch(mismatches, label, actual, expected) {
  if (JSON.stringify(actual) === JSON.stringify(expected)) return;
  mismatches.push({
    label: String(label || "").trim() || "roundtrip_mismatch",
    actual: buildWorkspaceRoundtripValuePreview(actual),
    expected: buildWorkspaceRoundtripValuePreview(expected),
  });
}

function pickWorkspaceRoundtripCraftSnapshot(craft) {
  const normalizedCraft = craftStore?.normalizeCraft?.(craft, 0) || {};
  return {
    id: String(normalizedCraft.id || "").trim(),
    name: String(normalizedCraft.name || "").trim(),
    summary: String(normalizedCraft.summary || "").trim(),
    accuracy: normalizedCraft.accuracy == null ? null : Number(normalizedCraft.accuracy),
    tools: Array.isArray(normalizedCraft.tools) ? [...normalizedCraft.tools] : [],
    tooling:
      normalizedCraft.tooling && typeof normalizedCraft.tooling === "object"
        ? cloneJson(normalizedCraft.tooling, { ready: 0, total: 0 })
        : { ready: 0, total: 0 },
    useStatus: String(normalizedCraft.useStatus || "").trim(),
    inputMode: String(normalizedCraft.inputMode || "").trim(),
    inputHint: String(normalizedCraft.inputHint || "").trim(),
    inputExamples: Array.isArray(normalizedCraft.inputExamples) ? [...normalizedCraft.inputExamples] : [],
    actionLabel: String(normalizedCraft.actionLabel || "").trim(),
    tokenSpend: Math.max(0, Number(normalizedCraft.tokenSpend || 0)),
    costUsd: Math.max(0, Number(normalizedCraft.costUsd || 0)),
    inputPlaceholder: String(normalizedCraft.inputPlaceholder || "").trim(),
    stage: String(normalizedCraft.stage || "").trim(),
    targetSlot: String(normalizedCraft.targetSlot || "").trim(),
    accuracyFloor: String(normalizedCraft.accuracyFloor || "").trim(),
    augmentationMode: String(normalizedCraft.augmentationMode || "").trim(),
    seedRows: Math.max(0, Number(normalizedCraft.seedRows || 0)),
    datasetRows: Math.max(0, Number(normalizedCraft.datasetRows || 0)),
    openGaps:
      normalizedCraft.openGaps == null || normalizedCraft.openGaps === ""
        ? null
        : Math.max(0, Number(normalizedCraft.openGaps || 0)),
    agentPrompt: String(normalizedCraft.agentPrompt || "").trim(),
    metricsReady: normalizedCraft.metricsReady === true,
    starterMode: String(normalizedCraft.starterMode || "").trim(),
    starterModelName: String(normalizedCraft.starterModelName || "").trim(),
    coverageGaps: Array.isArray(normalizedCraft.coverageGaps) ? [...normalizedCraft.coverageGaps] : [],
    navigatorRecords: Array.isArray(normalizedCraft.navigatorRecords)
      ? cloneJson(normalizedCraft.navigatorRecords, [])
      : [],
    training:
      normalizedCraft.training && typeof normalizedCraft.training === "object"
        ? cloneJson(normalizedCraft.training, null)
        : null,
  };
}

function pickWorkspaceRoundtripTrainingSnapshot(samples = []) {
  return (Array.isArray(samples) ? samples : []).map((sample, index) => {
    const normalized = normalizeTrainingSample(sample, index);
    return {
      id: String(normalized.id || "").trim(),
      messages: normalizePortableTrainingMessages(normalized.messages),
      tools: cloneJson(normalized.tools, []),
      targetTurnIndex: Number.isInteger(normalized.targetTurnIndex) ? normalized.targetTurnIndex : null,
      split: String(normalized.split || "").trim(),
      status: String(normalized.status || "").trim(),
      source: String(normalized.source || "").trim(),
    };
  });
}

function summarizeWorkspaceRoundtripToolScripts(payload) {
  const normalized = normalizeToolScriptsPayload(payload);
  return {
    declaredTools: Array.isArray(normalized.declaredTools) ? [...normalized.declaredTools] : [],
    scriptIds: Array.isArray(normalized.scripts) ? normalized.scripts.map((entry) => String(entry?.id || "").trim()) : [],
    entrypoints: Array.isArray(normalized.scripts) ? normalized.scripts.map((entry) => String(entry?.entrypoint || "").trim()) : [],
  };
}

function summarizeWorkspaceRoundtripBrowserCapabilities(payload) {
  const capabilities = Array.isArray(payload?.capabilities) ? payload.capabilities : [];
  return {
    capabilityIds: capabilities.map((entry) => String(entry?.id || "").trim()),
    toolNames: capabilities.map((entry) => String(entry?.toolName || "").trim()),
    resources: Array.isArray(payload?.resources) ? [...payload.resources] : [],
    skills: Array.isArray(payload?.skills) ? [...payload.skills] : [],
  };
}

function summarizeWorkspaceRoundtripPolicy(payload) {
  return {
    status: String(payload?.status || "").trim(),
    trainingMode: String(payload?.trainingMode || "").trim(),
    trainingDataFormat: String(payload?.trainingDataFormat || "").trim(),
    supervisionTarget: String(payload?.supervisionTarget || "").trim(),
    allowedTools: Array.isArray(payload?.policySpec?.allowedTools) ? [...payload.policySpec.allowedTools] : [],
    completionSignals: Array.isArray(payload?.policySpec?.completionSignals)
      ? [...payload.policySpec.completionSignals]
      : [],
  };
}

function summarizeWorkspaceRoundtripWeights(payload) {
  return {
    status: String(payload?.status || "").trim(),
    modelName: String(payload?.modelName || "").trim(),
    targetSlot: String(payload?.targetSlot || "").trim(),
    targetModules: Array.isArray(payload?.adapter?.targetModules) ? [...payload.adapter.targetModules] : [],
    modulePaths: Array.isArray(payload?.adapter?.modules)
      ? payload.adapter.modules.map((entry) => String(entry?.modulePath || "").trim())
      : [],
    runtimeModelId: String(payload?.runtime?.runtimeModelId || "").trim(),
    metricKeys:
      payload?.metrics && typeof payload.metrics === "object"
        ? Object.keys(payload.metrics)
        : [],
    completedAt: String(payload?.run?.completedAt || "").trim(),
  };
}

function createWorkspaceRoundtripToolDefinitions() {
  return [
    {
      type: "function",
      function: {
        name: "read_active_text_target",
        description: "Resolve the active selection, focused editable field, or clipboard text.",
        parameters: {
          type: "object",
          properties: {},
          additionalProperties: false,
        },
      },
    },
    {
      type: "function",
      function: {
        name: "open_relation_map",
        description: "Turn grounded evidence into a node-edge relation graph.",
        parameters: {
          type: "object",
          additionalProperties: false,
          properties: {
            text: { type: "string" },
            focus: { type: "string" },
          },
          required: ["text"],
        },
      },
    },
    {
      type: "function",
      function: {
        name: "replace_active_text_target",
        description: "Write a grounded summary back into the active text target.",
        parameters: {
          type: "object",
          additionalProperties: false,
          properties: {
            text: { type: "string" },
          },
          required: ["text"],
        },
      },
    },
  ];
}

function createProofreadWorkspaceToolDefinitions() {
  return [
    {
      type: "function",
      function: {
        name: "read_active_text_target",
        description: "Resolve the active text target in priority order: selection, focused editable field, then clipboard.",
        parameters: {
          type: "object",
          properties: {},
          additionalProperties: false,
        },
      },
    },
    {
      type: "function",
      function: {
        name: "replace_active_text_target",
        description: "Replace the active text target with the full corrected German text.",
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
  ];
}

function buildProofreadWorkspaceTrainingSamples(datasetPayload, now) {
  const splits = [
    ["train", "train"],
    ["validation", "validation"],
    ["test", "test"],
  ];
  const out = [];
  for (const [sourceSplit, targetSplit] of splits) {
    const rows = Array.isArray(datasetPayload?.[sourceSplit]) ? datasetPayload[sourceSplit] : [];
    rows.forEach((row, index) => {
      const sourceId = String(row?.sample_id || `${sourceSplit}-${index + 1}`).trim() || `${sourceSplit}-${index + 1}`;
      out.push(normalizeTrainingSample({
        id: `proofread-${sourceId}`,
        promptText: String(row?.prompt_text || "").trim(),
        messages: cloneJson(row?.messages, []),
        tools: cloneJson(row?.tools, []),
        targetTurnIndex: row?.target_turn_index,
        split: targetSplit,
        status: "ready",
        source: String(row?.source_name || "proofread_bootstrap_workspace").trim() || "proofread_bootstrap_workspace",
        createdAt: now,
        updatedAt: now,
      }, out.length));
    });
  }
  return out;
}

async function buildProofreadWorkspaceTrainingFixture() {
  const now = new Date().toISOString();
  const datasetPayload = await loadProofreadTrainingDatasetFromPackage();
  const trainingSamples = buildProofreadWorkspaceTrainingSamples(datasetPayload, now);
  const splitCounts = {
    train: trainingSamples.filter((sample) => sample.split === "train").length,
    validation: trainingSamples.filter((sample) => sample.split === "validation").length,
    test: trainingSamples.filter((sample) => sample.split === "test").length,
  };
  const rawCraft = {
    id: DEV_PROOFREAD_WORKSPACE_CRAFT_ID,
    name: DEV_PROOFREAD_WORKSPACE_CRAFT_NAME,
    nameSource: "user",
    summary: "Correct selected, focused, or clipboard-sourced German text with minimal edits and write it back directly.",
    accuracy: 0.84,
    tools: ["read_active_text_target", "replace_active_text_target"],
    tooling: { ready: 2, total: 2 },
    useStatus: "ready",
    inputMode: "mixed",
    inputHint: "If a selection exists, it takes priority over the focused input field, then over the clipboard.",
    inputExamples: [
      "Correct the currently selected German text directly in place.",
      "Read the text in the focused input field, correct it, and replace it directly.",
      "Read the copied text from the clipboard, correct it, and write the full version back.",
    ],
    actionLabel: "Correct text",
    tokenSpend: 0,
    costUsd: 0,
    inputPlaceholder:
      "If I selected text, selected an input field, or copied something to the clipboard, the text should be revised for spelling and grammar and replaced in place in that priority order.",
    stage: "Pilot",
    targetSlot: "target",
    accuracyFloor: "0.80",
    augmentationMode: "seed_grounded_sft",
    seedRows: trainingSamples.length,
    datasetRows: trainingSamples.length,
    openGaps: 2,
    agentPrompt:
      "If I selected text, selected an input field, or copied text to the clipboard, read the text in exactly that priority order with read_active_text_target, correct spelling and grammar with minimal changes, and write the full corrected text back in place with replace_active_text_target.",
    metricsReady: true,
    starterMode: "task_craft",
    starterModelName: "Qwen/Qwen3.5-0.8B",
    coverageGaps: [
      "HTML and table fragments should later be enriched with more real browser samples.",
      "The proofread smoke data currently trains only SFT and not a later preference stage.",
    ],
    navigatorRecords: [
      {
        id: "proofread-workspace-record-dataset",
        label: "proofread dataset",
        status: "ready",
        source: "workspace artifact",
        input: DEBUG_PROOFREAD_TRAINING.relativeDatasetPath,
        output: JSON.stringify(splitCounts, null, 2),
        meta: "proofread workspace smoke",
        updatedAt: now,
      },
      {
        id: "proofread-workspace-record-policy",
        label: "proofread policy",
        status: "ready",
        source: "fixture",
        input: "Priority order: selection -> focused_editable -> clipboard",
        output: JSON.stringify({ tools: 2, supervision: "next_assistant_turn" }, null, 2),
        meta: "proofread workspace smoke",
        updatedAt: now,
      },
    ],
    training: {
      defaultShardId: "proofread-smoke-shard",
      shards: [
        {
          id: "proofread-smoke-shard",
          label: "Proofread smoke target",
          modelName: "Qwen/Qwen3.5-0.8B",
          slotId: "target",
        },
      ],
    },
    createdAt: now,
    updatedAt: now,
  };
  const draftCraft = craftStore?.normalizeCraft?.(rawCraft, state.crafts.length) || rawCraft;
  const toolDefinitions = createProofreadWorkspaceToolDefinitions();
  const normalizedSamples = trainingSamples.map((sample, index) =>
    normalizeTrainingSample({
      ...sample,
      tools: Array.isArray(sample?.tools) && sample.tools.length ? sample.tools : toolDefinitions,
    }, index),
  );
  const toolScriptsPayload = normalizeToolScriptsPayload({
    schemaVersion: 1,
    declaredTools: rawCraft.tools,
    scripts: [
      {
        id: "read_active_text_target",
        name: "Read active text target",
        description: "Resolve the highest-priority active text target before proofreading.",
        language: "javascript",
        entrypoint: "readActiveTextTarget",
        source:
          "export async function readActiveTextTarget(runtime) {\n  return await runtime.callBuiltin(\"read_active_text_target\", {});\n}\n",
      },
      {
        id: "replace_active_text_target",
        name: "Replace active text target",
        description: "Write the corrected full text back into the same active target.",
        language: "javascript",
        entrypoint: "replaceActiveTextTarget",
        source:
          "export async function replaceActiveTextTarget(runtime, args) {\n  return await runtime.callBuiltin(\"replace_active_text_target\", args);\n}\n",
      },
    ],
  }, draftCraft);
  const compiledBrowserCapabilitiesPayload = compilePublishedBrowserCapabilityBundlePayload({
    schemaVersion: 1,
    actionProtocolVersion: 1,
    resources: [
      "references/tool-contracts.md",
      "references/proofread-dataset-crafting.md",
    ],
    skills: [
      "proofread_active_text",
      "minimal_in_place_rewrite",
    ],
    capabilities: [
      {
        id: "read_active_text_target",
        name: "Read active text target",
        toolName: "read_active_text_target",
        description: "Read the current selection, focused editable field, or clipboard text in priority order.",
        preconditions: ["A text selection, focused field, or clipboard text must exist."],
        readsFrom: ["selection", "focused_editable", "clipboard"],
        examples: [
          "Read the selected paragraph before proofreading it.",
          "If there is no selection, fall back to the focused text field.",
        ],
        tags: ["active_text", "proofread", "grounding"],
        skillRef: "proofread_active_text",
        resourceRefs: ["references/tool-contracts.md"],
      },
      {
        id: "replace_active_text_target",
        name: "Replace active text target",
        toolName: "replace_active_text_target",
        description: "Write the full corrected text back into the resolved active text target.",
        parameterSchema: {
          type: "object",
          additionalProperties: false,
          properties: {
            text: { type: "string" },
          },
          required: ["text"],
        },
        returnSchema: {
          type: "object",
          additionalProperties: true,
          properties: {
            ok: { type: "boolean" },
          },
          required: ["ok"],
        },
        preconditions: ["The corrected full text is available."],
        writesTo: ["selection", "focused_editable", "clipboard"],
        examples: [
          "Replace the selected text with the corrected full paragraph.",
          "Write the corrected mail draft back into the focused field.",
        ],
        tags: ["active_text", "proofread", "writeback"],
        skillRef: "minimal_in_place_rewrite",
        resourceRefs: ["references/tool-contracts.md"],
      },
    ],
  }, {
    craft: draftCraft,
    toolScriptsPayload,
    publishedBy: "sidepanel_proofread_fixture",
  });
  if (!compiledBrowserCapabilitiesPayload.ok) {
    throw new Error(
      compiledBrowserCapabilitiesPayload.error || "Proofread browser capability package is invalid.",
    );
  }
  const browserCapabilitiesPayload = compiledBrowserCapabilitiesPayload.payload;
  const basePolicyPayload = createBasePolicyPayload(draftCraft);
  const policyPayload = {
    ...basePolicyPayload,
    status: "supervised_only",
    trainingMode: "sft",
    trainingDataFormat: "qwen3_5_native_multiturn_tool_xml_v1",
    supervisionTarget: "next_assistant_turn",
    policySpec: {
      ...basePolicyPayload.policySpec,
      objective:
        "If a selection, focused input field, or clipboard text is present, correct the German text with minimal spelling and grammar edits and write it back in place.",
      bundleSkill:
        "Always use read_active_text_target first so the priority order selection -> focused_editable -> clipboard is respected. Preserve content, order, and format as much as possible. Do not only answer with the text; write it back completely with replace_active_text_target.",
      allowedTools: [...rawCraft.tools],
      completionSignals: [
        "active_text_resolved",
        "proofread_completed",
        "minimal_rewrite_preserved",
        "text_written_back_in_place",
      ],
      stopConditions: [
        "task_completed",
        "missing_required_context",
        "unsafe_or_unapproved_tool_request",
      ],
    },
    rewardSpec: {
      ...basePolicyPayload.rewardSpec,
      mode: "hybrid_validator_plus_judge",
      softJudgeRubric: [
        "tool_choice_quality",
        "priority_order_respected",
        "proofread_quality",
        "format_preservation",
      ],
      scoreWeights: {
        ...(basePolicyPayload.rewardSpec?.scoreWeights || {}),
        jsonMatch: 0.2,
        judgeScore: 0.25,
      },
    },
    judgeSpec: {
      ...basePolicyPayload.judgeSpec,
      promptTemplate:
        "Score only tool correctness, priority-order compliance, proofreading quality, and structure preservation. Penalize unnecessary rewriting and direct free-text answers when the reviewed tool path could apply the correction in place.",
    },
  };
  const baseWeightsPayload = createBaseWeightsPayload(draftCraft);
  const weightsPayload = {
    ...baseWeightsPayload,
    status: "base_model_only",
    targetSlot: "target",
    modelName: "Qwen/Qwen3.5-0.8B",
    runtime: {
      runtimeModelId: "local_qwen/qwen3.5-0.8b-q4f16",
      executionPlan: {
        label: "WebGPU q4f16",
        dtype: {
          embed_tokens: "q4f16",
          vision_encoder: "fp16",
          decoder: "q4f16",
        },
      },
    },
    dataset: {
      sampleCount: normalizedSamples.length,
      trainCount: splitCounts.train,
      validationCount: splitCounts.validation,
      testCount: splitCounts.test,
    },
    run: {
      completedAt: "",
      epochs: 0,
      adapterSizeMb: 0,
    },
  };
  const trainingDataArtifactRecord = {
    id: getBundleTrainingDataArtifactId(draftCraft.id),
    craftId: draftCraft.id,
    kind: TRAINING_DATA_ARTIFACT_KIND,
    payload: {
      samples: serializeTrainingSamples(normalizedSamples),
    },
    meta: {
      sampleCount: normalizedSamples.length,
      updatedAt: now,
    },
  };
  const toolScriptsArtifactRecord = {
    id: getToolScriptsArtifactId(draftCraft.id),
    craftId: draftCraft.id,
    kind: TOOL_SCRIPTS_ARTIFACT_KIND,
    payload: toolScriptsPayload,
    meta: {
      scriptCount: Array.isArray(toolScriptsPayload?.scripts) ? toolScriptsPayload.scripts.length : 0,
      updatedAt: now,
    },
  };
  const browserCapabilitiesArtifactRecord = {
    id: getBrowserCapabilityBundleArtifactId(draftCraft.id),
    craftId: draftCraft.id,
    kind: BROWSER_CAPABILITY_BUNDLE_ARTIFACT_KIND,
    payload: browserCapabilitiesPayload,
    meta: {
      capabilityCount: Array.isArray(browserCapabilitiesPayload?.capabilities)
        ? browserCapabilitiesPayload.capabilities.length
        : 0,
      updatedAt: now,
    },
  };
  const policyArtifactRecord = {
    id: getPolicyBundleArtifactId(draftCraft.id),
    craftId: draftCraft.id,
    kind: POLICY_BUNDLE_ARTIFACT_KIND,
    payload: policyPayload,
    meta: {
      trainingMode: String(policyPayload?.trainingMode || policyPayload?.status || "").trim(),
      updatedAt: now,
    },
  };
  const weightsArtifactRecord = {
    id: getWeightsArtifactId(draftCraft.id),
    craftId: draftCraft.id,
    kind: WEIGHTS_ARTIFACT_KIND,
    payload: weightsPayload,
    meta: {
      status: String(weightsPayload?.status || "").trim(),
      updatedAt: now,
    },
  };
  const bundle = buildCapabilityBundle({
    craft: draftCraft,
    trainingDataRecord: trainingDataArtifactRecord,
    toolScriptsRecord: toolScriptsArtifactRecord,
    browserCapabilitiesRecord: browserCapabilitiesArtifactRecord,
    weightsRecord: weightsArtifactRecord,
    policyRecord: policyArtifactRecord,
    generatedAt: now,
    preserveStoredBrowserCapabilities: true,
  });
  const craft = craftStore?.normalizeCraft?.(
    {
      ...draftCraft,
      bundle,
    },
    state.crafts.length,
  ) || {
    ...draftCraft,
    bundle,
  };

  return {
    craft,
    trainingSamples: normalizedSamples,
    toolScriptsPayload,
    browserCapabilitiesPayload,
    policyPayload,
    weightsPayload,
    datasetMeta: cloneJson(datasetPayload?.meta, {}),
  };
}

async function provisionProofreadWorkspaceTrainingFixture() {
  if (!craftStore?.normalizeCraft || !craftStore?.readLocalCrafts || !craftStore?.writeCrafts || !craftSync) {
    throw new Error("Craft store or craft sync is unavailable in the sidepanel runtime.");
  }
  const fixture = await buildProofreadWorkspaceTrainingFixture();
  await deleteCraftArtifacts(DEV_PROOFREAD_WORKSPACE_CRAFT_ID);
  await deleteAgentRunState(DEV_PROOFREAD_WORKSPACE_CRAFT_ID).catch(() => {});
  resetWorkspaceRoundtripTransientState(DEV_PROOFREAD_WORKSPACE_CRAFT_ID);

  state.crafts = await persistCrafts([
    ...state.crafts.filter((entry) => String(entry?.id || "").trim() !== DEV_PROOFREAD_WORKSPACE_CRAFT_ID),
    fixture.craft,
  ]);
  state.activeCraftId = fixture.craft.id;
  state.craftingCraftId = fixture.craft.id;
  render();

  await persistTrainingDataSnapshot(fixture.craft.id, fixture.trainingSamples, { updateLocalState: true });
  await writeToolScriptsArtifactRecord(fixture.craft.id, fixture.toolScriptsPayload);
  await writeBrowserCapabilityArtifactRecord(fixture.craft.id, fixture.browserCapabilitiesPayload);
  await writePolicyArtifactRecord(fixture.craft.id, fixture.policyPayload);
  await writeCapabilityWeightsArtifactRecord(fixture.craft.id, fixture.weightsPayload);

  await Promise.all([
    loadTrainingDataState(fixture.craft.id, { force: true }),
    loadToolScriptsState(fixture.craft.id, { force: true }),
    loadCapabilityWeightsState(fixture.craft.id, { force: true }),
  ]);

  return fixture;
}

function buildWorkspaceRoundtripSmokeFixture() {
  const now = new Date().toISOString();
  const primaryImage = createLabeledVisionSmokeImageDataUrl("graph", "#1f8a70");
  const secondaryImage = createLabeledVisionSmokeImageDataUrl("evidence", "#d14b8f");
  const rawCraft = {
    id: DEV_WORKSPACE_ROUNDTRIP_CRAFT_ID,
    name: DEV_WORKSPACE_ROUNDTRIP_CRAFT_NAME,
    nameSource: "user",
    summary: "Build grounded relation maps from selected text or screenshots and keep ambiguous evidence visible.",
    accuracy: 0.87,
    tools: ["read_active_text_target", "open_relation_map", "replace_active_text_target"],
    tooling: { ready: 3, total: 3 },
    useStatus: "ready",
    inputMode: "mixed",
    inputHint: "Use the current selection or paste additional evidence before generating the relation map.",
    inputExamples: [
      "Map the company-founder relation and keep the supporting quote.",
      "Turn this screenshot into a relation graph with grounded evidence.",
      "Read the active text, build a graph, and write back a short grounded summary.",
    ],
    actionLabel: "Map relation",
    tokenSpend: 1280,
    costUsd: 0.42,
    inputPlaceholder: "Describe the relation extraction job.",
    stage: "Pilot",
    targetSlot: "target",
    accuracyFloor: "0.82",
    augmentationMode: "retrieval_augmented",
    seedRows: 2,
    datasetRows: 2,
    openGaps: 2,
    agentPrompt: "Use reviewed browser tools to build relation maps and keep explicit evidence for every edge.",
    metricsReady: true,
    starterMode: "task_craft",
    starterModelName: "Qwen/Qwen3.5-0.8B",
    coverageGaps: [
      "Low-confidence OCR from screenshots still needs manual review.",
      "Cross-document entity deduplication is not yet automated.",
    ],
    navigatorRecords: [
      {
        id: "workspace-roundtrip-record-seeds",
        label: "seed review",
        status: "ready",
        source: "dev fixture",
        input: "2 canonical seed rows for grounded relation mapping",
        output: JSON.stringify({ train: 1, validation: 1 }, null, 2),
        meta: "workspace roundtrip",
        updatedAt: now,
      },
      {
        id: "workspace-roundtrip-record-policy",
        label: "policy bundle",
        status: "ready",
        source: "dev fixture",
        input: "Reviewed tool-only execution path",
        output: JSON.stringify({ allowedTools: 3, completionSignals: 4 }, null, 2),
        meta: "policy seeded",
        updatedAt: now,
      },
    ],
    training: {
      defaultShardId: "workspace-smoke-shard",
      shards: [
        {
          id: "workspace-smoke-shard",
          label: "Workspace smoke target",
          modelName: "Qwen/Qwen3.5-0.8B",
          slotId: "target",
        },
        {
          id: "workspace-smoke-vision",
          label: "Workspace smoke vision",
          modelName: "Qwen/Qwen3.5-0.8B",
          slotId: "vision",
        },
      ],
    },
    createdAt: now,
    updatedAt: now,
  };
  const draftCraft = craftStore?.normalizeCraft?.(rawCraft, state.crafts.length) || rawCraft;
  const toolDefinitions = createWorkspaceRoundtripToolDefinitions();
  const trainingSamples = [
    normalizeTrainingSample({
      id: "workspace-roundtrip-sample-train",
      messages: [
        { role: "system", content: "You are a grounded relation-mapping assistant." },
        { role: "user", content: "Read the active text, build a relation map, then summarize the key edge." },
        {
          role: "assistant",
          tool_calls: [
            {
              id: "call-read-active",
              function: {
                name: "read_active_text_target",
                arguments: "{}",
              },
            },
          ],
        },
        {
          role: "tool",
          tool_call_id: "call-read-active",
          name: "read_active_text_target",
          content: JSON.stringify({
            ok: true,
            data: {
              targetType: "selection",
              text: "Ada founded Example Labs in Berlin in 2021.",
            },
          }),
        },
        {
          role: "assistant",
          tool_calls: [
            {
              id: "call-open-map",
              function: {
                name: "open_relation_map",
                arguments: JSON.stringify({
                  text: "Ada founded Example Labs in Berlin in 2021.",
                  focus: "founder relation",
                }),
              },
            },
          ],
        },
        {
          role: "tool",
          tool_call_id: "call-open-map",
          name: "open_relation_map",
          content: JSON.stringify({
            ok: true,
            data: {
              nodes: [
                { id: "ada", label: "Ada" },
                { id: "example-labs", label: "Example Labs" },
              ],
              edges: [
                {
                  source: "ada",
                  relation: "founded",
                  target: "example-labs",
                  evidence: "Ada founded Example Labs in Berlin in 2021.",
                },
              ],
            },
          }),
        },
        {
          role: "assistant",
          content: JSON.stringify({
            summary: "Ada founded Example Labs in Berlin in 2021.",
            relation_graph: {
              nodes: ["Ada", "Example Labs"],
              edges: [
                {
                  source: "Ada",
                  relation: "founded",
                  target: "Example Labs",
                },
              ],
            },
          }, null, 2),
        },
      ],
      tools: toolDefinitions,
      targetTurnIndex: 6,
      split: "train",
      status: "ready",
      source: "dev-workspace-roundtrip",
      createdAt: now,
      updatedAt: now,
    }),
    normalizeTrainingSample({
      id: "workspace-roundtrip-sample-validation",
      messages: [
        { role: "system", content: "You ground every relation in the visible evidence." },
        {
          role: "user",
          content: [
            { type: "text", text: "Inspect the screenshot and return the relation with confidence." },
            { type: "image", image: primaryImage },
            { type: "image", image: secondaryImage },
          ],
        },
        {
          role: "assistant",
          content: JSON.stringify({
            relation: "owner_of",
            confidence: 0.93,
            evidence: "The screenshot labels the purple card as owned by Example Labs.",
          }, null, 2),
        },
      ],
      tools: toolDefinitions,
      targetTurnIndex: 2,
      split: "validation",
      status: "ready",
      source: "vision-fixture",
      createdAt: now,
      updatedAt: now,
    }),
  ];
  const toolScriptsPayload = normalizeToolScriptsPayload({
    schemaVersion: 1,
    declaredTools: rawCraft.tools,
    scripts: [
      {
        id: "read_active_text_target",
        name: "Read active text target",
        description: "Resolve the active selection before any grounded relation mapping step.",
        language: "javascript",
        entrypoint: "readActiveTextTarget",
        source:
          "export async function readActiveTextTarget(runtime) {\n  return await runtime.callBuiltin(\"read_active_text_target\", {});\n}\n",
      },
      {
        id: "open_relation_map",
        name: "Open relation map",
        description: "Convert grounded evidence into a reviewer-friendly node-edge graph.",
        language: "javascript",
        entrypoint: "openRelationMap",
        source:
          "export async function openRelationMap(runtime, args) {\n  return runtime.respond({\n    ok: true,\n    data: {\n      graphId: \"demo-graph\",\n      args,\n    },\n  });\n}\n",
      },
      {
        id: "replace_active_text_target",
        name: "Replace active text target",
        description: "Write the grounded summary back into the active target.",
        language: "javascript",
        entrypoint: "replaceActiveTextTarget",
        source:
          "export async function replaceActiveTextTarget(runtime, args) {\n  return await runtime.callBuiltin(\"replace_active_text_target\", args);\n}\n",
      },
    ],
  }, draftCraft);
  const compiledBrowserCapabilitiesPayload = compilePublishedBrowserCapabilityBundlePayload({
    schemaVersion: 1,
    actionProtocolVersion: 1,
    resources: [
      "references/relation-graph-schema.json",
      "references/evidence-guidelines.md",
    ],
    skills: [
      "relation_graphing",
      "evidence_grounding",
    ],
    capabilities: [
      {
        id: "read_active_text_target",
        name: "Read active text target",
        toolName: "read_active_text_target",
        description: "Resolve the current text target before relation extraction.",
        preconditions: ["A focused field, selection, or clipboard text must exist."],
        readsFrom: ["selection", "focused_editable", "clipboard"],
        examples: ["Read the selected paragraph before mapping entities."],
        tags: ["active_text", "grounding"],
        skillRef: "evidence_grounding",
        resourceRefs: ["references/evidence-guidelines.md"],
      },
      {
        id: "open_relation_map",
        name: "Open relation map",
        toolName: "open_relation_map",
        description: "Create a grounded relation graph from the resolved evidence text.",
        parameterSchema: {
          type: "object",
          additionalProperties: false,
          properties: {
            text: { type: "string" },
            focus: { type: "string" },
          },
          required: ["text"],
        },
        returnSchema: {
          type: "object",
          additionalProperties: false,
          properties: {
            ok: { type: "boolean" },
            data: {
              type: "object",
              additionalProperties: false,
              properties: {
                graphId: { type: "string" },
                nodeCount: { type: "number" },
                edgeCount: { type: "number" },
              },
              required: ["graphId"],
            },
          },
          required: ["ok"],
        },
        preconditions: ["The evidence text has been resolved and normalized."],
        readsFrom: ["selection", "screenshot"],
        writesTo: ["relation_graph"],
        examples: ["Build a graph that keeps the founding edge and the supporting quote."],
        tags: ["relation_graph", "grounded_output"],
        skillRef: "relation_graphing",
        resourceRefs: [
          "references/relation-graph-schema.json",
          "references/evidence-guidelines.md",
        ],
        scripts: {
          pre: "return { ok: true, stage: \"precheck\" };",
          execute:
            "return { ok: true, data: { graphId: \"demo-graph\", nodeCount: 2, edgeCount: 1 } };",
          post: "return { ok: true, stage: \"postcheck\" };",
        },
      },
      {
        id: "replace_active_text_target",
        name: "Replace active text target",
        toolName: "replace_active_text_target",
        description: "Write the grounded summary back into the active target after verification.",
        preconditions: ["The grounded summary must already be prepared."],
        writesTo: ["selection", "focused_editable"],
        examples: ["Replace the selected text with a one-line grounded summary."],
        tags: ["active_text", "writeback"],
        skillRef: "evidence_grounding",
        resourceRefs: ["references/evidence-guidelines.md"],
      },
    ],
  }, {
    craft: draftCraft,
    toolScriptsPayload,
    publishedBy: "sidepanel_roundtrip_fixture",
  });
  if (!compiledBrowserCapabilitiesPayload.ok) {
    throw new Error(
      compiledBrowserCapabilitiesPayload.error || "Workspace roundtrip fixture browser capability package is invalid.",
    );
  }
  const browserCapabilitiesPayload = compiledBrowserCapabilitiesPayload.payload;
  const basePolicyPayload = createBasePolicyPayload(draftCraft);
  const policyPayload = {
    ...basePolicyPayload,
    status: "rl_ready",
    trainingMode: "sft",
    trainingDataFormat: "qwen3_5_native_multiturn_tool_xml_v1",
    supervisionTarget: "next_assistant_turn",
    policySpec: {
      ...basePolicyPayload.policySpec,
      objective: "Build grounded relation graphs from active text or screenshots and keep uncertainty explicit.",
      bundleSkill:
        "Always resolve the evidence with read_active_text_target before creating or editing the relation graph. Use open_relation_map for the graph itself and replace_active_text_target only after the evidence-backed summary is ready.",
      allowedTools: [...rawCraft.tools],
      completionSignals: [
        "structured_json_valid",
        "approved_tool_used_or_explicit_no_tool",
        "task_goal_resolved_or_blocked_with_reason",
        "relation_graph_created",
      ],
      stopConditions: [
        "task_completed",
        "missing_required_context",
        "unsafe_or_unapproved_tool_request",
        "evidence_confidence_too_low",
      ],
    },
    rewardSpec: {
      ...basePolicyPayload.rewardSpec,
      mode: "hybrid_validator_plus_judge",
      softJudgeRubric: [
        "goal_completion",
        "tool_choice_quality",
        "argument_quality",
        "evidence_grounding",
      ],
      scoreWeights: {
        ...(basePolicyPayload.rewardSpec?.scoreWeights || {}),
        jsonMatch: 0.18,
        judgeScore: 0.25,
      },
    },
    judgeSpec: {
      ...basePolicyPayload.judgeSpec,
      promptTemplate:
        "Score only grounded relation quality, tool correctness, evidence fidelity, and brevity. Penalize missing uncertainty markers.",
    },
  };
  const adapterModules = [
    "model.layers.3.self_attn.q_proj",
    "model.layers.3.self_attn.k_proj",
    "model.layers.3.self_attn.v_proj",
    "model.layers.3.self_attn.o_proj",
    "model.layers.3.mlp.gate_proj",
    "model.layers.3.mlp.up_proj",
    "model.layers.3.mlp.down_proj",
  ];
  const baseWeightsPayload = createBaseWeightsPayload(draftCraft);
  const weightsPayload = {
    ...baseWeightsPayload,
    status: "trained_adapter",
    targetSlot: "target",
    modelName: "Qwen/Qwen3.5-0.8B",
    adapter: {
      type: "transformer_lora",
      rank: 16,
      alpha: 32,
      dropout: 0.05,
      targetModules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
      modules: adapterModules.map((modulePath) => ({
        modulePath,
        loraADataUrl: "data:application/octet-stream;base64,QUFBQQ==",
        loraBDataUrl: "data:application/octet-stream;base64,QkJCQg==",
      })),
    },
    metrics: {
      validationEvalAcc: 0.91,
      adaptTestAcc: 0.88,
      edgePrecision: 0.9,
      edgeRecall: 0.86,
    },
    runtime: {
      runtimeModelId: "local_qwen/qwen3.5-0.8b-q4f16",
      executionPlan: {
        label: "WebGPU q4f16",
        dtype: {
          embed_tokens: "q4f16",
          vision_encoder: "fp16",
          decoder: "q4f16",
        },
      },
    },
    dataset: {
      sampleCount: 2,
      trainCount: 1,
      validationCount: 1,
    },
    run: {
      completedAt: now,
      epochs: 2,
      adapterSizeMb: 18.4,
    },
  };
  const trainingDataArtifactRecord = {
    id: getBundleTrainingDataArtifactId(draftCraft.id),
    craftId: draftCraft.id,
    kind: TRAINING_DATA_ARTIFACT_KIND,
    payload: {
      samples: serializeTrainingSamples(trainingSamples),
    },
    meta: {
      sampleCount: trainingSamples.length,
      updatedAt: now,
    },
  };
  const toolScriptsArtifactRecord = {
    id: getToolScriptsArtifactId(draftCraft.id),
    craftId: draftCraft.id,
    kind: TOOL_SCRIPTS_ARTIFACT_KIND,
    payload: toolScriptsPayload,
    meta: {
      scriptCount: Array.isArray(toolScriptsPayload?.scripts) ? toolScriptsPayload.scripts.length : 0,
      updatedAt: now,
    },
  };
  const browserCapabilitiesArtifactRecord = {
    id: getBrowserCapabilityBundleArtifactId(draftCraft.id),
    craftId: draftCraft.id,
    kind: BROWSER_CAPABILITY_BUNDLE_ARTIFACT_KIND,
    payload: browserCapabilitiesPayload,
    meta: {
      capabilityCount: Array.isArray(browserCapabilitiesPayload?.capabilities)
        ? browserCapabilitiesPayload.capabilities.length
        : 0,
      updatedAt: now,
    },
  };
  const policyArtifactRecord = {
    id: getPolicyBundleArtifactId(draftCraft.id),
    craftId: draftCraft.id,
    kind: POLICY_BUNDLE_ARTIFACT_KIND,
    payload: policyPayload,
    meta: {
      trainingMode: String(policyPayload?.trainingMode || policyPayload?.status || "").trim(),
      updatedAt: now,
    },
  };
  const weightsArtifactRecord = {
    id: getWeightsArtifactId(draftCraft.id),
    craftId: draftCraft.id,
    kind: WEIGHTS_ARTIFACT_KIND,
    payload: weightsPayload,
    meta: {
      status: String(weightsPayload?.status || "").trim(),
      updatedAt: now,
    },
  };
  const bundle = buildCapabilityBundle({
    craft: draftCraft,
    trainingDataRecord: trainingDataArtifactRecord,
    toolScriptsRecord: toolScriptsArtifactRecord,
    browserCapabilitiesRecord: browserCapabilitiesArtifactRecord,
    weightsRecord: weightsArtifactRecord,
    policyRecord: policyArtifactRecord,
    generatedAt: now,
    preserveStoredBrowserCapabilities: true,
  });
  const craft = craftStore?.normalizeCraft?.(
    {
      ...draftCraft,
      bundle,
    },
    state.crafts.length,
  ) || {
    ...draftCraft,
    bundle,
  };
  const expectedArtifactIds = normalizeWorkspaceRoundtripArtifactIds([
    trainingDataArtifactRecord.id,
    toolScriptsArtifactRecord.id,
    browserCapabilitiesArtifactRecord.id,
    policyArtifactRecord.id,
    weightsArtifactRecord.id,
  ]);

  return {
    craft,
    trainingSamples,
    toolScriptsPayload,
    browserCapabilitiesPayload,
    policyPayload,
    weightsPayload,
    expectedArtifactIds,
  };
}

function resetWorkspaceRoundtripTransientState(craftId) {
  const key = String(craftId || "").trim();
  if (!key) return;
  delete state.agentRuns[key];
  delete state.trainingRuns[key];
  delete state.promptDrafts[key];
  delete state.agentPromptDrafts[key];
  delete state.useMessages[key];
  delete state.craftResponses[key];
  delete state.trainingDataStates[key];
  delete state.toolScriptStates[key];
  delete state.capabilityWeightsStates[key];
  if (state.trainingDataCraftId === key) {
    state.trainingDataCraftId = null;
  }
}

async function runWorkspaceRoundtripSmokeAndCopy() {
  startHeaderTestRun("workspaceRoundtrip");
  const report = createBaseSmokeReport("craft_workspace_rxdb_roundtrip_smoke", "target", {
    surface: "sidepanel_dev_header",
    craftId: DEV_WORKSPACE_ROUNDTRIP_CRAFT_ID,
    craftName: DEV_WORKSPACE_ROUNDTRIP_CRAFT_NAME,
  });

  try {
    if (!craftStore?.normalizeCraft || !craftStore?.readLocalCrafts || !craftStore?.writeCrafts || !craftSync) {
      throw new Error("Craft store or craft sync is unavailable in the sidepanel runtime.");
    }

    updateHeaderTestRun("workspaceRoundtrip", { progress: 0.12, indeterminate: true });
    const fixture = buildWorkspaceRoundtripSmokeFixture();
    report.fixture = {
      craft: pickWorkspaceRoundtripCraftSnapshot(fixture.craft),
      artifacts: [...fixture.expectedArtifactIds],
      training: pickWorkspaceRoundtripTrainingSnapshot(fixture.trainingSamples),
      toolScripts: summarizeWorkspaceRoundtripToolScripts(fixture.toolScriptsPayload),
      browserCapabilities: summarizeWorkspaceRoundtripBrowserCapabilities(fixture.browserCapabilitiesPayload),
      policy: summarizeWorkspaceRoundtripPolicy(fixture.policyPayload),
      weights: summarizeWorkspaceRoundtripWeights(fixture.weightsPayload),
    };

    const cleanupStartedAt = Date.now();
    await deleteCraftArtifacts(DEV_WORKSPACE_ROUNDTRIP_CRAFT_ID);
    await deleteAgentRunState(DEV_WORKSPACE_ROUNDTRIP_CRAFT_ID).catch(() => {});
    resetWorkspaceRoundtripTransientState(DEV_WORKSPACE_ROUNDTRIP_CRAFT_ID);
    report.timingsMs.cleanup = Date.now() - cleanupStartedAt;

    updateHeaderTestRun("workspaceRoundtrip", { progress: 0.28, indeterminate: false });
    const persistStartedAt = Date.now();
    state.crafts = await persistCrafts([
      ...state.crafts.filter((entry) => String(entry?.id || "").trim() !== DEV_WORKSPACE_ROUNDTRIP_CRAFT_ID),
      fixture.craft,
    ]);
    report.timingsMs.persistCraft = Date.now() - persistStartedAt;
    state.activeCraftId = fixture.craft.id;
    state.craftingCraftId = fixture.craft.id;
    render();

    updateHeaderTestRun("workspaceRoundtrip", { progress: 0.48, indeterminate: false });
    const writeStartedAt = Date.now();
    await writeTrainingDataArtifactRecord(fixture.craft.id, serializeTrainingSamples(fixture.trainingSamples));
    await writeToolScriptsArtifactRecord(fixture.craft.id, fixture.toolScriptsPayload);
    await writeBrowserCapabilityArtifactRecord(fixture.craft.id, fixture.browserCapabilitiesPayload);
    await writePolicyArtifactRecord(fixture.craft.id, fixture.policyPayload);
    await writeCapabilityWeightsArtifactRecord(fixture.craft.id, fixture.weightsPayload);
    report.timingsMs.writeArtifacts = Date.now() - writeStartedAt;

    updateHeaderTestRun("workspaceRoundtrip", { progress: 0.68, indeterminate: true });
    const readBackStartedAt = Date.now();
    const [
      localCrafts,
      listedArtifacts,
      trainingRecord,
      toolScriptsRecord,
      browserCapabilitiesRecord,
      policyRecord,
      weightsRecord,
    ] = await Promise.all([
      craftStore.readLocalCrafts(),
      listLocalArtifactsForCraft(fixture.craft.id),
      readTrainingDataArtifactRecord(fixture.craft.id),
      readToolScriptsArtifactRecord(fixture.craft.id),
      readBrowserCapabilityArtifactRecord(fixture.craft.id),
      readPolicyArtifactRecord(fixture.craft.id),
      readCapabilityWeightsArtifactRecord(fixture.craft.id),
    ]);
    report.timingsMs.readBack = Date.now() - readBackStartedAt;

    const readCraft = (Array.isArray(localCrafts) ? localCrafts : []).find(
      (entry) => String(entry?.id || "").trim() === fixture.craft.id,
    ) || null;
    const listedArtifactIds = Array.from(
      new Set(
        (Array.isArray(listedArtifacts) ? listedArtifacts : [])
          .map((entry) => String(entry?.id || "").trim())
          .filter(Boolean),
      ),
    ).sort();

    await refreshCraftsState();
    state.activeCraftId = fixture.craft.id;
    state.craftingCraftId = fixture.craft.id;
    const [trainingState, toolState, capabilityState] = await Promise.all([
      loadTrainingDataState(fixture.craft.id, { force: true }),
      loadToolScriptsState(fixture.craft.id, { force: true }),
      loadCapabilityWeightsState(fixture.craft.id, { force: true }),
    ]);
    const workspaceUiExpectations = getWorkspaceRoundtripUiExpectations();
    const workspaceUiSnapshots = [];
    for (const expectation of workspaceUiExpectations) {
      workspaceUiSnapshots.push(await readWorkspaceRoundtripUiSnapshot(fixture.craft.id, expectation.tab, expectation));
    }

    const expectedTrainingSnapshot = pickWorkspaceRoundtripTrainingSnapshot(fixture.trainingSamples);
    const actualTrainingSnapshot = pickWorkspaceRoundtripTrainingSnapshot(trainingRecord?.payload?.samples || []);
    const expectedToolScriptsPayload = normalizeToolScriptsPayload(fixture.toolScriptsPayload, fixture.craft);
    const actualToolScriptsPayload = normalizeToolScriptsPayload(toolScriptsRecord?.payload, readCraft || fixture.craft);
    const expectedBrowserCapabilitiesPayload = normalizeBrowserCapabilityBundlePayload(
      fixture.browserCapabilitiesPayload,
      fixture.craft,
      expectedToolScriptsPayload,
    );
    const actualBrowserCapabilitiesPayload = normalizeBrowserCapabilityBundlePayload(
      browserCapabilitiesRecord?.payload,
      readCraft || fixture.craft,
      actualToolScriptsPayload,
    );
    const actualPolicyPayload =
      policyRecord?.payload && typeof policyRecord.payload === "object"
        ? cloneJson(policyRecord.payload, null)
        : null;
    const actualWeightsPayload =
      weightsRecord?.payload && typeof weightsRecord.payload === "object"
        ? cloneJson(weightsRecord.payload, null)
        : null;
    const parsedToolState = parseToolScriptsDraft(String(toolState?.text || ""), readCraft || fixture.craft);
    const mismatches = [];

    if (!readCraft) {
      mismatches.push({
        label: "local_craft_missing",
        actual: "missing",
        expected: fixture.craft.id,
      });
    } else {
      pushWorkspaceRoundtripMismatch(
        mismatches,
        "craft",
        pickWorkspaceRoundtripCraftSnapshot(readCraft),
        pickWorkspaceRoundtripCraftSnapshot(fixture.craft),
      );
    }

    pushWorkspaceRoundtripMismatch(
      mismatches,
      "artifacts_present",
      listedArtifactIds,
      fixture.expectedArtifactIds,
    );
    pushWorkspaceRoundtripMismatch(
      mismatches,
      "training_artifact",
      actualTrainingSnapshot,
      expectedTrainingSnapshot,
    );
    pushWorkspaceRoundtripMismatch(
      mismatches,
      "tool_scripts_artifact",
      actualToolScriptsPayload,
      expectedToolScriptsPayload,
    );
    pushWorkspaceRoundtripMismatch(
      mismatches,
      "browser_capabilities_artifact",
      actualBrowserCapabilitiesPayload,
      expectedBrowserCapabilitiesPayload,
    );
    pushWorkspaceRoundtripMismatch(
      mismatches,
      "policy_artifact",
      actualPolicyPayload,
      fixture.policyPayload,
    );
    pushWorkspaceRoundtripMismatch(
      mismatches,
      "weights_artifact",
      actualWeightsPayload,
      fixture.weightsPayload,
    );
    pushWorkspaceRoundtripMismatch(
      mismatches,
      "sidepanel_training_state",
      pickWorkspaceRoundtripTrainingSnapshot(trainingState?.samples || []),
      expectedTrainingSnapshot,
    );
    pushWorkspaceRoundtripMismatch(
      mismatches,
      "sidepanel_tool_state_parse_ok",
      parsedToolState.ok,
      true,
    );
    if (parsedToolState.ok) {
      pushWorkspaceRoundtripMismatch(
        mismatches,
        "sidepanel_tool_state",
        parsedToolState.payload,
        expectedToolScriptsPayload,
      );
    }
    pushWorkspaceRoundtripMismatch(
      mismatches,
      "sidepanel_weights_state",
      {
        status: String(capabilityState?.status || "").trim(),
        hasTrainedAdapter: capabilityState?.hasTrainedAdapter === true,
      },
      {
        status: "trained_adapter",
        hasTrainedAdapter: true,
      },
    );
    workspaceUiSnapshots.forEach((snapshot) => {
      pushWorkspaceRoundtripMismatch(
        mismatches,
        `workspace_ui_${snapshot.tab}`,
        {
          matched: snapshot.matched,
          missing: snapshot.missing,
          forbidden: snapshot.forbidden,
        },
        {
          matched: true,
          missing: [],
          forbidden: [],
        },
      );
    });

    report.readBack = {
      craft: readCraft ? pickWorkspaceRoundtripCraftSnapshot(readCraft) : null,
      artifactIds: listedArtifactIds,
      training: pickWorkspaceRoundtripTrainingSnapshot(trainingState?.samples || []),
      toolScripts: summarizeWorkspaceRoundtripToolScripts(actualToolScriptsPayload),
      browserCapabilities: summarizeWorkspaceRoundtripBrowserCapabilities(actualBrowserCapabilitiesPayload),
      policy: summarizeWorkspaceRoundtripPolicy(actualPolicyPayload),
      weights: summarizeWorkspaceRoundtripWeights(actualWeightsPayload),
      sidepanel: {
        trainingArtifactId: String(trainingState?.artifactId || "").trim(),
        toolArtifactId: String(toolState?.artifactId || "").trim(),
        weightsArtifactId: String(capabilityState?.artifactId || "").trim(),
      },
      workspaceUi: workspaceUiSnapshots.map((snapshot) => summarizeWorkspaceRoundtripUiSnapshot(snapshot)),
      mismatches,
    };

    updateHeaderTestRun("workspaceRoundtrip", { progress: 0.94, indeterminate: false });
    if (mismatches.length) {
      report.diagnosis = createSmokeDiagnosis({
        failureKind: "workspace_roundtrip",
        summary: `${mismatches.length} workspace roundtrip check${mismatches.length === 1 ? "" : "s"} did not match the RxDB read-back.`,
        nextAction: "Open the copied report, inspect the mismatched artifact snapshot, and compare it with the dedicated craft workspace tabs.",
        keyFacts: [
          { label: "craftId", value: fixture.craft.id },
          { label: "mismatches", value: String(mismatches.length) },
          { label: "artifacts", value: String(listedArtifactIds.length) },
        ],
      });
      markHeaderTestFailure(
        "workspaceRoundtrip",
        "Workspace roundtrip failed.",
        `${mismatches.length} RxDB roundtrip check${mismatches.length === 1 ? "" : "s"} mismatched.`,
        report,
      );
      render();
      return;
    }

    report.diagnosis = createSmokeDiagnosis({
      failureKind: "",
      summary: "The dev workspace smoke test created a craft, persisted every editable workspace artifact, reloaded the same data from RxDB, and rendered the dedicated craft workspace tabs successfully.",
      nextAction: "Open the dedicated craft workspace if you want to inspect the generated fixture visually.",
      keyFacts: [
        { label: "craftId", value: fixture.craft.id },
        { label: "samples", value: String(fixture.trainingSamples.length) },
        { label: "artifacts", value: String(fixture.expectedArtifactIds.length) },
      ],
    });
    markHeaderTestSuccess("workspaceRoundtrip");
  } catch (error) {
    report.error = error instanceof Error ? error.message : String(error || "Unknown workspace roundtrip error");
    report.diagnosis = createSmokeDiagnosis({
      failureKind: "workspace_roundtrip",
      summary: trimText(report.error || "The workspace roundtrip smoke test failed.", 180),
      nextAction: "Inspect the copied report, then rerun the workspace smoke button after fixing the local craft sync path.",
      keyFacts: [
        { label: "craftId", value: DEV_WORKSPACE_ROUNDTRIP_CRAFT_ID },
      ],
    });
    markHeaderTestFailure(
      "workspaceRoundtrip",
      "Workspace roundtrip failed.",
      report.error,
      report,
    );
    render();
  }
}

async function resolveLocalQwenSlotForSmoke(slotId, report) {
  const configuredResolveStartedAt = Date.now();
  const configuredResolveResponse = await sendRuntimeMessage({
    type: "llm:resolve-model",
    slotId,
  });
  report.timingsMs.resolveConfiguredSlot = Date.now() - configuredResolveStartedAt;
  report.configuredResolve = configuredResolveResponse?.ok ? configuredResolveResponse.resolved || null : null;
  report.configuredResolveError = configuredResolveResponse?.ok
    ? ""
    : String(configuredResolveResponse?.error || "llm:resolve-model failed");

  const smokeModelName = getLocalQwenSmokeModelName(slotId);
  report.request.forceLocalQwen = true;
  report.request.providerId = "local_qwen";
  report.request.modelName = smokeModelName;

  const resolveStartedAt = Date.now();
  const resolveResponse = await sendRuntimeMessage({
    type: "llm:resolve-model",
    slotId,
    providerId: "local_qwen",
    modelName: smokeModelName,
  });
  report.timingsMs.resolveModel = Date.now() - resolveStartedAt;
  report.resolve = resolveResponse?.ok ? resolveResponse.resolved || null : null;
  report.resolveError = resolveResponse?.ok
    ? ""
    : String(resolveResponse?.error || "llm:resolve-model failed");

  if (!resolveResponse?.ok || !report.resolve) {
    throw new Error(report.resolveError || `Local Qwen smoke target for slot ${slotId} could not be resolved.`);
  }

  const providerType = String(report.resolve?.provider?.type || "").trim().toLowerCase();
  if (providerType !== "local_qwen") {
    throw new Error(
      `Forced smoke target for slot ${slotId} did not resolve to local_qwen.`,
    );
  }

  return report.resolve;
}

function getLocalQwenSmokeModelName(slotId) {
  const currentSlot = state.slots?.[slotId] || {};
  if (String(currentSlot.providerId || "").trim() === "local_qwen" && String(currentSlot.modelName || "").trim()) {
    return String(currentSlot.modelName || "").trim();
  }

  const localOptions = configApi
    .collectModelOptions(state.providers, slotId)
    .filter((option) => option.providerId === "local_qwen");
  if (localOptions.length) {
    return String(localOptions[0].modelName || "").trim();
  }

  const targetSlot = state.slots?.target || {};
  if (String(targetSlot.providerId || "").trim() === "local_qwen" && String(targetSlot.modelName || "").trim()) {
    return String(targetSlot.modelName || "").trim();
  }

  const providerModelNames = Array.isArray(state.providers?.local_qwen?.modelNames)
    ? state.providers.local_qwen.modelNames.map((item) => String(item || "").trim()).filter(Boolean)
    : [];
  if (providerModelNames.length) {
    return providerModelNames[0];
  }

  throw new Error(`No local_qwen model is available for the ${slotId} smoke check.`);
}

function getSmokeReportModelRef(report) {
  return (
    report?.response?.resolved?.modelRef ||
    report?.result?.resolved?.modelRef ||
    report?.resolve?.modelRef ||
    report?.resolve?.modelName ||
    "unknown"
  );
}

function normalizeSmokeAnswerText(text) {
  return String(text || "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ");
}

function createLabeledVisionSmokeImageDataUrl(label, fillColor) {
  const canvas = globalThis.document?.createElement?.("canvas");
  if (!canvas) {
    throw new Error("Canvas is unavailable in the side panel, so the vision smoke image could not be created.");
  }

  canvas.width = 256;
  canvas.height = 192;
  const context = canvas.getContext("2d");
  if (!context) {
    throw new Error("2D canvas context is unavailable for the vision smoke image.");
  }

  const normalizedLabel = String(label || "").trim().toLowerCase() || "red";
  const normalizedFillColor = String(fillColor || "#ff0000");

  context.fillStyle = "#fffaf2";
  context.fillRect(0, 0, canvas.width, canvas.height);

  context.fillStyle = normalizedFillColor;
  context.fillRect(18, 18, canvas.width - 36, 78);

  context.fillStyle = "#1f1813";
  context.fillRect(18, 114, canvas.width - 36, 6);

  context.textAlign = "center";
  context.textBaseline = "middle";
  context.font = '700 42px "Avenir Next", "Helvetica Neue", Arial, sans-serif';
  context.fillStyle = "#1f1813";
  context.fillText(normalizedLabel.toUpperCase(), canvas.width / 2, 150);

  return canvas.toDataURL("image/png");
}

function normalizeVisionSmokeColor(text) {
  return normalizeSmokeAnswerText(text).split(/\s+/).filter(Boolean)[0] || "";
}

function buildAgentSmokeCraftBundle() {
  return {
    browserCapabilities: {
      artifactId: "smoke-agentic-browser-capabilities",
      payload: {
        schemaVersion: 1,
        actionProtocolVersion: 1,
        skills: [
          "Always invoke the reviewed classify_review_sentiment capability before finishing.",
        ],
        resources: ["sentiment_labels"],
        capabilities: [
          {
            id: "classify_review_sentiment",
            name: "classify_review_sentiment",
            version: "1.0.0",
            description: "Classify the provided product review into JSON { label: positive | negative }.",
            parameterSchema: {
              type: "object",
              additionalProperties: false,
              required: ["reviewText"],
              properties: {
                reviewText: {
                  type: "string",
                  description: "The product review text that should be classified.",
                },
              },
            },
            returnSchema: {
              type: "object",
              additionalProperties: false,
              required: ["label"],
              properties: {
                label: {
                  type: "string",
                  enum: ["positive", "negative"],
                },
              },
            },
            preconditions: ["Use this capability for short product-review sentiment tasks."],
            readsFrom: [],
            writesTo: [],
            examples: [
              "The battery lasts all day and the case feels premium.",
            ],
            tags: ["smoke", "classification"],
            skillRef: "sentiment_classification_smoke",
            resourceRefs: ["sentiment_labels"],
            scripts: {
              pre: "",
              execute: "",
              post: "",
            },
          },
        ],
      },
      meta: {},
    },
    policy: {
      artifactId: "smoke-agentic-policy",
      payload: {
        status: "supervised_only",
        trainingMode: "sft",
        policySpec: {
          bundleSkill: "Always call classify_review_sentiment exactly once before finishing the smoke run.",
          allowedTools: ["classify_review_sentiment"],
          completionSignals: ["classification_ready"],
          stopConditions: ["classification_completed"],
        },
      },
      meta: {},
    },
    summary: {
      browserCapabilityCount: 1,
      hasPolicy: true,
      weightsMode: "base_model_only",
    },
  };
}

function isAgentSmokeRunUsable(run) {
  const steps = Array.isArray(run?.steps) ? run.steps : [];
  const finalLabel = String(
    run?.result?.label ||
      steps[steps.length - 1]?.execution?.finalOutput?.label ||
      "",
  ).trim().toLowerCase();
  return Boolean(
    run &&
      typeof run === "object" &&
      String(run.status || "").trim().toLowerCase() === "done" &&
      steps.some((step) => String(step?.execution?.capability?.name || "").trim().toLowerCase() === "classify_review_sentiment") &&
      finalLabel === "positive",
  );
}

async function runAgentSmokeTestAndCopy() {
  if (isSmokeTestRunning()) return;

  const startedAt = Date.now();
  const prompt = [
    "Use the reviewed capability to classify this short product review.",
    "Review: The battery lasts all day and the case feels premium.",
  ].join(" ");
  const parameters = {
    maxTokens: 700,
    temperature: 0,
    reasoningMode: "no_think",
  };

  startHeaderTestRun("agentic");
  render();

  const report = createBaseSmokeReport("local_qwen_agent_smoke", "agent", {
    prompt,
    capabilityCount: 1,
    parameters,
  });

  try {
    const localResolved = await resolveLocalQwenSlotForSmoke("agent", report);
    updateHeaderTestRun("agentic", { progress: 0.3, indeterminate: false });

    const requestStartedAt = Date.now();
    updateHeaderTestRun("agentic", { progress: 0.62, indeterminate: true });
    const response = await sendRuntimeMessage({
      type: "craft:run",
      craft: {
        id: "smoke-agentic-run",
        name: "Smoke Agentic Run",
        summary: "Use the reviewed capability to classify a short product review into JSON { label: positive | negative }.",
        stage: "Smoke",
        targetSlot: localResolved.slotId || "agent",
        providerId: localResolved.providerId,
        modelName: localResolved.modelName,
        runtimeParameters: {
          ...parameters,
        },
        tools: [],
        bundle: buildAgentSmokeCraftBundle(),
      },
      prompt,
      maxTurns: 3,
    });
    updateHeaderTestRun("agentic", { progress: 0.92, indeterminate: false });
    report.timingsMs.agentRun = Date.now() - requestStartedAt;
    report.response = response?.ok
      ? {
          run: response.run || null,
        }
      : null;
    report.errorDetail = response?.ok ? null : response?.errorDetail || null;
    report.error = response?.ok
      ? ""
      : String(response?.error || "craft:run failed");

    const run = response?.ok ? response.run || null : null;
    const steps = Array.isArray(run?.steps) ? run.steps : [];
    const capabilityNames = steps
      .map((step) => String(step?.execution?.capability?.name || "").trim())
      .filter(Boolean);
    report.result = {
      status: String(run?.status || ""),
      text: String(run?.text || ""),
      result: run?.result || null,
      error: String(run?.error || ""),
      stepCount: steps.length,
      capabilityNames,
      modelRef: String(run?.modelRef || ""),
      usage: run?.usage || null,
    };
    report.run = run;
    report.ok = Boolean(
      response?.ok &&
        isAgentSmokeRunUsable(run) &&
        String(localResolved?.provider?.type || "").trim().toLowerCase() === "local_qwen",
    );
    report.totalDurationMs = Date.now() - startedAt;

    if (report.ok) {
      markHeaderTestSuccess("agentic");
    } else {
      attachReportDiagnosis(report, diagnoseAgentSmokeReport(report));
      markHeaderTestFailure(
        "agentic",
        "Test Agentic failed.",
        report.error || "The local agentic check failed.",
        report,
      );
    }
    render();
  } catch (error) {
    report.ok = false;
    report.error = error instanceof Error ? error.message : String(error || "Unknown agent smoke-test error");
    report.totalDurationMs = Date.now() - startedAt;
    attachReportDiagnosis(report, diagnoseAgentSmokeReport(report));
    markHeaderTestFailure(
      "agentic",
      "Test Agentic failed.",
      trimText(report.error || "Unknown agent smoke-test error", 140),
      report,
    );
    render();
  }
}

async function runVisionSmokeTestAndCopy() {
  if (isSmokeTestRunning()) return;

  const startedAt = Date.now();
  const prompt = "Read the single lowercase color word shown in the image. Return only that one word.";
  const parameters = {
    maxTokens: 12,
    temperature: 0,
    reasoningMode: "no_think",
  };

  startHeaderTestRun("vision");
  render();

  const report = createBaseSmokeReport("local_qwen_vision_smoke", "vision", {
    prompt,
    imageDescription: "Generated labeled PNG cards with a colored banner and one large word: red or blue.",
    parameters,
  });

  try {
    const localResolved = await resolveLocalQwenSlotForSmoke("vision", report);
    updateHeaderTestRun("vision", { progress: 0.2, indeterminate: false });
    const visionCases = [
      {
        expected: "red",
        image: createLabeledVisionSmokeImageDataUrl("red", "#ff0000"),
      },
      {
        expected: "blue",
        image: createLabeledVisionSmokeImageDataUrl("blue", "#0000ff"),
      },
    ];
    const responses = [];
    const requestStartedAt = Date.now();
    for (const [index, testCase] of visionCases.entries()) {
      updateHeaderTestRun("vision", {
        progress: 0.32 + (index / visionCases.length) * 0.44,
        indeterminate: true,
      });
      const response = await sendRuntimeMessage({
        type: "llm:chat",
        slotId: "vision",
        providerId: localResolved.providerId,
        modelName: localResolved.modelName,
        messages: [
          {
            role: "system",
            content: "The image contains one large lowercase color word. Answer with exactly one word: red or blue.",
          },
          {
            role: "user",
            content: [
              { type: "text", text: "Read the word in the image. Reply with red or blue only." },
              { type: "image", image: testCase.image },
            ],
          },
        ],
        parameters,
      });
      responses.push({
        expected: testCase.expected,
        ok: Boolean(response?.ok),
        text: String(response?.text || ""),
        normalizedText: normalizeVisionSmokeColor(response?.text || ""),
        resolved: response?.resolved || null,
        finishReason: response?.finishReason || "",
        usage: response?.usage || null,
        runtime: response?.runtime || null,
        error: response?.ok ? "" : String(response?.error || "llm:chat failed"),
        errorDetail: response?.ok ? null : response?.errorDetail || null,
      });
      updateHeaderTestRun("vision", {
        progress: 0.32 + ((index + 1) / visionCases.length) * 0.52,
        indeterminate: false,
      });
    }
    report.timingsMs.visionChat = Date.now() - requestStartedAt;
    report.responses = responses;
    report.response = responses[0] || null;
    report.error = responses.every((entry) => entry.ok)
      ? ""
      : responses.find((entry) => !entry.ok)?.error || "llm:chat failed";
    report.errorDetail = responses.find((entry) => !entry.ok)?.errorDetail || null;
    report.result = {
      cases: responses.map((entry) => ({
        expected: entry.expected,
        text: entry.text,
        normalizedText: entry.normalizedText,
      })),
      resolved: responses[0]?.resolved || null,
      runtime: responses[0]?.runtime || null,
    };
    report.ok = Boolean(
      responses.length === visionCases.length &&
        responses.every(
          (entry) =>
            entry.ok &&
            entry.normalizedText === entry.expected &&
            String(entry?.resolved?.provider?.type || "").trim().toLowerCase() === "local_qwen",
        ),
    );
    report.totalDurationMs = Date.now() - startedAt;

    if (report.ok) {
      markHeaderTestSuccess("vision");
    } else {
      attachReportDiagnosis(report, diagnoseVisionSmokeReport(report));
      markHeaderTestFailure(
        "vision",
        "Test Vision failed.",
        [
          `vision=${getSmokeReportModelRef(report)}`,
          `cases=${responses
            .map((entry) => `${entry.expected}:${entry.normalizedText || "?"}`)
            .join(",")}`,
        ].join(" | "),
        report,
      );
    }
    render();
  } catch (error) {
    report.ok = false;
    report.error = error instanceof Error ? error.message : String(error || "Unknown vision smoke-test error");
    report.totalDurationMs = Date.now() - startedAt;
    attachReportDiagnosis(report, diagnoseVisionSmokeReport(report));
    markHeaderTestFailure(
      "vision",
      "Test Vision failed.",
      trimText(report.error || "Unknown vision smoke-test error", 140),
      report,
    );
    render();
  }
}

async function runSelfUseSmokeTestAndCopy() {
  if (isSmokeTestRunning()) return;

  const startedAt = Date.now();
  startHeaderTestRun("selfUse");
  render();

  const report = createBaseSmokeReport("local_qwen_self_use_smoke", "vision", {
    expectedTool: "open_relation_map",
    surface: "browser_tool_smoke",
  });

  try {
    const localResolved = await resolveLocalQwenSlotForSmoke("vision", report);
    updateHeaderTestRun("selfUse", { progress: 0.24, indeterminate: false });

    const requestStartedAt = Date.now();
    updateHeaderTestRun("selfUse", { progress: 0.68, indeterminate: true });
    const response = await sendRuntimeMessage({
      type: "tool:test-self-use-smoke",
      providerId: localResolved.providerId,
      modelName: localResolved.modelName,
    });
    updateHeaderTestRun("selfUse", { progress: 0.94, indeterminate: false });

    report.timingsMs.selfUseRun = Date.now() - requestStartedAt;
    report.result =
      response?.ok && response?.report && typeof response.report === "object"
        ? response.report
        : null;
    report.error = response?.ok
      ? String(response?.report?.error || "")
      : String(response?.error || "tool:test-self-use-smoke failed");
    report.ok = Boolean(response?.ok && response?.report?.ok === true);
    report.totalDurationMs = Date.now() - startedAt;

    if (report.ok) {
      markHeaderTestSuccess("selfUse");
    } else {
      markHeaderTestFailure(
        "selfUse",
        "Test Self Use failed.",
        trimText(
          String(
            response?.report?.error ||
              response?.report?.diagnosis?.summary ||
              report.error ||
              "The local self-use smoke test failed.",
          ),
          140,
        ),
        report,
      );
    }
    render();
  } catch (error) {
    report.ok = false;
    report.error = error instanceof Error ? error.message : String(error || "Unknown self-use smoke-test error");
    report.totalDurationMs = Date.now() - startedAt;
    markHeaderTestFailure(
      "selfUse",
      "Test Self Use failed.",
      trimText(report.error || "Unknown self-use smoke-test error", 140),
      report,
    );
    render();
  }
}

function createLocalQwenInferenceBenchmarkPrompt() {
  const paragraph = [
    "This benchmark prompt exists only to measure raw local WebGPU forward throughput.",
    "It repeats a realistic mixed-format passage with names, places, dates, trade-offs, and short factual statements.",
    "Clara compared three notebook chargers during a delayed train ride from Cologne to Berlin and wrote that the lightest unit stayed cool, the cheapest unit whined softly, and the fastest unit reached eighty percent before lunch.",
    "The team later copied her notes into a bug report, attached invoice numbers 1842 and 2077, and added a reminder that reliability matters more than novelty when people work offline for long periods.",
    "After the meeting, the group agreed to keep one conservative setup for broad device support and one optional higher-memory setup for stronger machines with more headroom.",
    "Read every detail carefully and keep the trade-offs, counts, colors, places, and product differences in context.",
  ].join(" ");
  return Array.from({ length: 4 }, () => paragraph).join("\n");
}

function createLocalQwenInferenceBenchmarkId() {
  return `local-qwen-inference-bench-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

async function runSmokeTestAndCopy() {
  if (isSmokeTestRunning()) return;

  startHeaderTestRun("inference");
  render();

  const startedAt = Date.now();
  const prompt = "Return ONLY the token OK.";
  const parameters = {
    maxTokens: 8,
    temperature: 0,
    reasoningMode: "no_think",
  };
  const benchmark = {
    mode: "forward_only",
    iterations: 6,
    warmupIterations: 1,
    promptText: createLocalQwenInferenceBenchmarkPrompt(),
    comparisonPurpose: "browser_raw_inference_benchmark",
    compareBy: "forward_tokens_per_second",
    resourceProfile: "memory_conservative",
  };
  const benchmarkRequested = true;
  const benchmarkId = createLocalQwenInferenceBenchmarkId();
  activeInferenceBenchmarkId = benchmarkId;

  const report = createBaseSmokeReport("local_qwen_sidepanel_smoke", "target", {
    prompt,
    benchmark: {
      mode: benchmark.mode,
      iterations: benchmark.iterations,
      warmupIterations: benchmark.warmupIterations,
      promptChars: benchmark.promptText.length,
      comparisonPurpose: benchmark.comparisonPurpose,
      compareBy: benchmark.compareBy,
      resourceProfile: benchmark.resourceProfile,
    },
    parameters,
  });

  try {
    const configStartedAt = Date.now();
    const configResponse = await sendRuntimeMessage({ type: "llm:get-config" });
    updateHeaderTestRun("inference", { progress: 0.24, indeterminate: false });
    report.timingsMs.getConfig = Date.now() - configStartedAt;
    report.config = configResponse?.ok ? configResponse.config || null : null;
    report.configError = configResponse?.ok ? "" : String(configResponse?.error || "llm:get-config failed");

    const resolveStartedAt = Date.now();
    const resolveResponse = await sendRuntimeMessage({
      type: "llm:resolve-model",
      slotId: "target",
    });
    updateHeaderTestRun("inference", { progress: 0.48, indeterminate: false });
    report.timingsMs.resolveModel = Date.now() - resolveStartedAt;
    report.resolve = resolveResponse?.ok ? resolveResponse.resolved || null : null;
    report.resolveError = resolveResponse?.ok
      ? ""
      : String(resolveResponse?.error || "llm:resolve-model failed");

    const testStartedAt = Date.now();
    updateHeaderTestRun("inference", { progress: 0.56, indeterminate: true });
    const testResponse = await sendRuntimeMessage({
      type: "llm:test-local-qwen-diagnostic",
      slotId: "target",
      prompt,
      parameters,
    });
    report.timingsMs.testModel = Date.now() - testStartedAt;

    const benchmarkStartedAt = Date.now();
    updateHeaderTestRun("inference", { progress: INFERENCE_BENCHMARK_PROGRESS_START, indeterminate: false });
    const benchmarkResponse = await sendRuntimeMessage({
      type: "llm:benchmark-local-qwen-forward",
      slotId: "target",
      parameters,
      benchmarkId,
      promptText: benchmark.promptText,
      iterations: benchmark.iterations,
      warmupIterations: benchmark.warmupIterations,
    });
    updateHeaderTestRun("inference", { progress: 0.94, indeterminate: false });
    report.timingsMs.forwardBenchmark = Date.now() - benchmarkStartedAt;
    const benchmarkResult =
      benchmarkResponse?.ok && benchmarkResponse?.benchmark && typeof benchmarkResponse.benchmark === "object"
        ? benchmarkResponse.benchmark
        : null;
    const benchmarkSummary =
      benchmarkResult?.benchmark?.benchmark && typeof benchmarkResult.benchmark.benchmark === "object"
        ? benchmarkResult.benchmark.benchmark
        : null;
    const benchmarkHasThroughput =
      Number(benchmarkSummary?.forwardTokensPerSecond || 0) > 0 &&
      Number(benchmarkSummary?.totalForwardTokens || 0) > 0;
    report.ok = Boolean(
      testResponse?.ok &&
      testResponse?.diagnostic?.ok &&
      (!benchmarkRequested || (benchmarkResult?.ok === true && benchmarkHasThroughput))
    );
    report.diagnostic = testResponse?.ok ? testResponse.diagnostic || null : null;
    report.result = testResponse?.ok
      ? {
          text: String(testResponse?.diagnostic?.offscreen?.text || ""),
          runtime: testResponse?.diagnostic?.offscreen?.runtime || null,
          benchmark:
            benchmarkSummary
              ? { ...benchmarkSummary }
              : null,
          benchmarkRuntime:
            benchmarkResult?.benchmark?.runtime && typeof benchmarkResult.benchmark.runtime === "object"
              ? { ...benchmarkResult.benchmark.runtime }
              : benchmarkResult?.runtime && typeof benchmarkResult.runtime === "object"
                ? { ...benchmarkResult.runtime }
              : null,
          throughput:
            benchmarkSummary
              ? {
                  overallForwardTokensPerSecond: Number(
                    benchmarkSummary.forwardTokensPerSecond || 0,
                  ),
                  totalForwardTokens: Number(benchmarkSummary.totalForwardTokens || 0),
                  promptTokenCount: Number(benchmarkSummary.promptTokenCount || 0),
                  measuredIterations: Number(benchmarkSummary.iterations || 0),
                  warmupIterations: Number(benchmarkSummary.warmupIterations || 0),
                }
              : null,
        }
      : null;
    report.comparison = {
      purpose: benchmark.comparisonPurpose,
      compareBy: benchmark.compareBy,
      resourceProfile: benchmark.resourceProfile,
    };
    report.throughput =
      report.result?.throughput && typeof report.result.throughput === "object"
        ? { ...report.result.throughput }
        : null;
    if (benchmarkSummary && Number(benchmarkSummary.totalElapsedMs || 0) > 0) {
      report.timingsMs.forwardBenchmark = Number(benchmarkSummary.totalElapsedMs || 0);
    }
    report.error = testResponse?.ok
      ? benchmarkRequested && !benchmarkResult
        ? "The local forward benchmark response was missing. Reload the extension so the latest service worker and offscreen runtime are active, then retry."
        : benchmarkResult && benchmarkResult.ok === false
          ? String(
              benchmarkResult?.benchmark?.error ||
                benchmarkResult.error ||
                "The local forward benchmark failed.",
            )
          : benchmarkRequested && !benchmarkHasThroughput
            ? "The local forward benchmark completed without returning forward throughput."
            : String(testResponse?.diagnostic?.offscreen?.error || "")
      : String(testResponse?.error || "llm:test-local-qwen-diagnostic failed");
    report.totalDurationMs = Date.now() - startedAt;
    if (report.ok) {
      markHeaderTestSuccess("inference");
      state.smokeTest = {
        ...createEmptySmokeTestState(),
        status: "success",
        message: "Test Inference completed with a local forward benchmark.",
        detail: benchmarkSummary
          ? [
              formatItemsPerSecond(benchmarkSummary.forwardTokensPerSecond || 0, "forward tok"),
              `prompt ${formatCompactCount(benchmarkSummary.promptTokenCount || 0)} tok`,
              `${formatCompactCount(benchmarkSummary.iterations || 0)} measured runs`,
            ].join(" · ")
          : "The local inference smoke test completed successfully.",
        updatedAt: formatClock(new Date()),
        reportText: JSON.stringify(report, null, 2),
        copyStatus: "idle",
        testId: "inference",
      };
    } else {
      attachReportDiagnosis(report, diagnoseInferenceSmokeReport(report));
      const runtimeModelId =
        report.result?.runtime?.runtimeModelId ||
        report.diagnostic?.offscreen?.runtime?.runtimeModelId ||
        report.diagnostic?.offscreen?.errorDetail?.runtimePlan?.runtimeModelId ||
        report.resolve?.modelRef ||
        "unknown";
      const shortText = String(report.result?.text || "").trim() || "<empty>";
      markHeaderTestFailure(
        "inference",
        "Test Inference failed.",
        [
          `target=${runtimeModelId}`,
          `reply=${trimText(shortText, 80)}`,
        ].join(" | "),
        report,
      );
    }
    render();
    if (activeInferenceBenchmarkId === benchmarkId) {
      activeInferenceBenchmarkId = "";
    }
  } catch (error) {
    if (activeInferenceBenchmarkId === benchmarkId) {
      activeInferenceBenchmarkId = "";
    }
    report.ok = false;
    report.error = error instanceof Error ? error.message : String(error || "Unknown smoke-test error");
    report.totalDurationMs = Date.now() - startedAt;
    attachReportDiagnosis(report, diagnoseInferenceSmokeReport(report));
    markHeaderTestFailure(
      "inference",
      "Test Inference failed.",
      trimText(report.error || "Unknown smoke-test error", 140),
      report,
    );
    render();
  }
}

function buildToolSmokeFallbackReport(type, error, failureKind, nextAction) {
  const normalizedError = String(error || "Tool smoke test failed.");
  return {
    reportVersion: 1,
    type,
    ok: false,
    error: normalizedError,
    diagnosis: createSmokeDiagnosis({
      failureKind,
      summary: trimText(normalizedError, 180),
      nextAction,
    }),
  };
}

async function runToolSmokeTestAndCopy({
  testId,
  messageType,
  reportType,
  failureTitle,
  failureKind,
  nextAction,
} = {}) {
  if (isSmokeTestRunning()) return;

  startHeaderTestRun(testId);
  render();

  try {
    updateHeaderTestRun(testId, { progress: 0.28, indeterminate: false });
    updateHeaderTestRun(testId, { progress: 0.72, indeterminate: true });
    const response = await sendRuntimeMessage({ type: messageType });

    const report =
      response?.ok && response?.report && typeof response.report === "object"
        ? response.report
        : buildToolSmokeFallbackReport(
            reportType,
            response?.error || `${messageType} failed`,
            failureKind,
            nextAction,
          );

    updateHeaderTestRun(testId, { progress: 0.94, indeterminate: false });

    if (report?.ok) {
      markHeaderTestSuccess(testId);
    } else {
      markHeaderTestFailure(
        testId,
        failureTitle,
        trimText(
          String(report?.error || report?.diagnosis?.summary || "Tool smoke test failed."),
          140,
        ),
        report,
      );
    }
    render();
  } catch (error) {
    const report = buildToolSmokeFallbackReport(
      reportType,
      error instanceof Error ? error.message : String(error || "Unknown tool smoke-test error"),
      failureKind,
      nextAction,
    );
    markHeaderTestFailure(
      testId,
      failureTitle,
      trimText(report.error || "Unknown tool smoke-test error", 140),
      report,
    );
    render();
  }
}

async function runToolTabsSmokeTestAndCopy() {
  await runToolSmokeTestAndCopy({
    testId: "toolTabs",
    messageType: "tool:test-tabs-smoke",
    reportType: "browser_tool_tabs_smoke",
    failureTitle: "Test Tabs failed.",
    failureKind: "tab_management",
    nextAction: "Reload the extension once, then rerun the tab smoke test.",
  });
}

async function runToolVisualSmokeTestAndCopy() {
  await runToolSmokeTestAndCopy({
    testId: "toolVisual",
    messageType: "tool:test-visual-smoke",
    reportType: "browser_tool_visual_smoke",
    failureTitle: "Test Visual failed.",
    failureKind: "visual_runtime",
    nextAction: "Check the configured vision slot, then rerun the visual tool smoke test.",
  });
}

async function runToolCodeSmokeTestAndCopy() {
  await runToolSmokeTestAndCopy({
    testId: "toolCode",
    messageType: "tool:test-code-smoke",
    reportType: "browser_tool_code_smoke",
    failureTitle: "Test Code failed.",
    failureKind: "code_runtime",
    nextAction: "Reload the extension once, then rerun the code-path smoke test.",
  });
}

async function runToolIntegratedSmokeTestAndCopy() {
  await runToolSmokeTestAndCopy({
    testId: "toolIntegrated",
    messageType: "tool:test-integrated-smoke",
    reportType: "browser_agent_integrated_smoke",
    failureTitle: "Test Integrated failed.",
    failureKind: "extension_runtime",
    nextAction: "Reload the extension once and rerun the integrated app test.",
  });
}

async function sendRuntimeMessage(message) {
  if (!globalThis.chrome?.runtime?.sendMessage) {
    throw new Error("chrome.runtime.sendMessage is unavailable in the side panel.");
  }
  return await globalThis.chrome.runtime.sendMessage(message);
}

if (globalThis.chrome?.runtime?.onMessage) {
  globalThis.chrome.runtime.onMessage.addListener((message) => {
    const type = String(message?.type || "").trim();
    if (type === "local_qwen_forward_benchmark_progress") {
      syncHeaderInferenceBenchmarkProgress(message);
    }
  });
}

async function copyTextToClipboard(text) {
  if (globalThis.navigator?.clipboard?.writeText) {
    await globalThis.navigator.clipboard.writeText(String(text || ""));
    return;
  }

  const textarea = globalThis.document.createElement("textarea");
  textarea.value = String(text || "");
  textarea.setAttribute("readonly", "true");
  textarea.style.position = "fixed";
  textarea.style.inset = "-9999px";
  textarea.style.opacity = "0";
  globalThis.document.body.append(textarea);
  textarea.focus();
  textarea.select();

  const copied = globalThis.document.execCommand("copy");
  textarea.remove();

  if (!copied) {
    throw new Error("Clipboard write failed.");
  }
}

function isEditableEventTarget(target) {
  const node = target instanceof Element ? target : null;
  if (!node) return false;
  const tagName = String(node.tagName || "").toLowerCase();
  if (["input", "textarea", "select"].includes(tagName)) return true;
  return node.closest("[contenteditable='true']") != null;
}

function getCraftInputMode(craft) {
  return normalizeCraftInputMode(craft?.inputMode, "free_text");
}

function getCraftInputHint(craft) {
  const hint = String(craft?.inputHint || "").trim();
  if (hint) return hint;

  const mode = getCraftInputMode(craft);
  if (mode === "current_tab") return "Uses the current tab. No extra prompt field is needed.";
  if (mode === "selection") return "Uses the current selection or focused field. No extra prompt field is needed.";
  if (mode === "mixed") return "Uses the current context and optional text.";
  if (mode === "context_only") return "Uses the current craft context. No extra prompt field is needed.";
  return "Uses text input.";
}

function isRemoteSharedCraft(craft) {
  return craftStore?.isRemoteSharedCraft?.(craft) === true;
}

function isForkCraft(craft) {
  return craftStore?.isForkCraft?.(craft) === true;
}

function getCraftForkCount(craft) {
  return Math.max(0, Number(craftStore?.getLineageForkCount?.(craft, state.crafts || []) || 0));
}

function getLiveLinkCount() {
  return Math.max(0, Number(state.syncSnapshot?.sync?.transportPeerCount || 0));
}

function getVisiblePeerCount() {
  return Math.max(0, Number(state.syncSnapshot?.sync?.remotePeerCount || 0));
}

function buildFooterPeerStatus() {
  const snapshot = state.syncSnapshot?.sync || null;
  const liveLinks = getLiveLinkCount();
  const visiblePeers = getVisiblePeerCount();

  if (!hasCustomSyncPassword(state.syncSettings)) {
    return "Local only";
  }

  if (liveLinks > 0) {
    return `${formatPeerCountLabel(liveLinks)} connected`;
  }

  if (visiblePeers > 0) {
    return `${formatPeerCountLabel(visiblePeers)} visible`;
  }

  if (snapshot?.lastError) {
    return "Share error";
  }

  if (snapshot?.running) {
    return "Share ready";
  }

  return "Share ready · sync off";
}

function buildPeerSessionLabel() {
  return `${formatCount(getLiveLinkCount())} live links · ${formatCount(getVisiblePeerCount())} visible peers`;
}

function getSharedOwnerName(craft) {
  return String(craft?.sync?.ownerName || craft?.sync?.ownerDeviceId || "remote peer").trim() || "remote peer";
}

function getCraftCardNote(craft) {
  if (isRemoteSharedCraft(craft)) {
    return `from ${getSharedOwnerName(craft)}`;
  }
  if (isForkCraft(craft)) {
    return "Fork";
  }
  if (craft?.sharing?.enabled === true) {
    return "Shared";
  }
  const status = String(craft?.useStatus || "").toLowerCase();
  if (status === "blocked") return "Blocked";
  return "";
}

function getCraftCardSecondaryText(craft) {
  const description = buildCraftOfficialDescriptionPreview(craft, 96).preview;
  const note = getCraftCardNote(craft);
  return [description, note].filter(Boolean).join(" · ");
}

function getCraftInputDescriptor(craft) {
  if (isVanillaStarterCraft(craft)) return "vanilla qwen3.5";
  const mode = getCraftInputMode(craft);
  if (mode === "current_tab") return "current tab";
  if (mode === "selection") return "browser selection";
  if (mode === "mixed") return "tab + text";
  if (mode === "context_only") return "execute only";
  return "text input";
}

function getCraftActionLabel(craft) {
  return String(craft?.actionLabel || "Run");
}

function isApplePlatform() {
  const platform = String(globalThis.navigator?.platform || "");
  const userAgent = String(globalThis.navigator?.userAgent || "");
  return /Mac|iPhone|iPad|iPod/i.test(platform) || /Mac OS|iPhone|iPad|iPod/i.test(userAgent);
}

function getCraftExecutionShortcutHint(craft) {
  if (!craft || craftUsesTextInput(craft)) return "";
  return isApplePlatform() ? "Cmd + Enter" : "Ctrl + Enter";
}

function getCraftSourceDescriptor(craft) {
  if (isRemoteSharedCraft(craft)) {
    return isForkCraft(craft)
      ? `fork mirror from ${getSharedOwnerName(craft)}`
      : `origin mirror from ${getSharedOwnerName(craft)}`;
  }
  if (isForkCraft(craft)) {
    return `fork of ${trimText(String(craft?.sync?.originName || craft?.name || "origin"), 30)}`;
  }
  if (craft?.sharing?.enabled === true) {
    return "shared from this device";
  }
  return "local only";
}

function getCraftUseState(craft) {
  if (isRemoteSharedCraft(craft)) {
    return {
      label: state.syncSnapshot?.sync?.running ? "linked" : "cached",
      tone: state.syncSnapshot?.sync?.running ? "live" : "muted",
    };
  }

  if (isCraftingActive(craft.id)) {
    return { label: "crafting", tone: "live" };
  }

  const status = String(craft?.useStatus || "").toLowerCase();
  if (status === "blocked") return { label: "blocked", tone: "blocked" };
  if (status === "vanilla") return { label: "vanilla", tone: "baseline" };
  if (status === "lab") return { label: "lab", tone: "muted" };
  return { label: status || "ready", tone: status === "crafting" ? "live" : "ready" };
}

function getCraftShareStateChip(craft) {
  if (isRemoteSharedCraft(craft)) {
    return { label: "mirror", tone: "remote", icon: "remote" };
  }
  if (craft?.sharing?.enabled === true) {
    return { label: "shared", tone: "share", icon: "share" };
  }
  return { label: "local only", tone: "private", icon: "local" };
}

function getCraftRoleStateChip(craft) {
  if (!isForkCraft(craft)) return null;
  return { label: "fork", tone: "fork" };
}

function getCraftForkDescriptor(craft) {
  return `${formatCount(getCraftForkCount(craft))} forks`;
}

function getCraftingStatusLabel(craft) {
  if (isRemoteSharedCraft(craft)) return "read only";
  const agentRun = state.agentRuns[craft.id];
  const agentUiState = getAgentUiState(agentRun);
  if (agentUiState === "starting" || agentUiState === "running") return "agent running";
  if (agentUiState === "needs_input") return "awaiting answers";
  if (agentUiState === "blocked") return "agent blocked";
  if (agentUiState === "failed") return "last run failed";
  if (agentUiState === "stopped_limited") return "agent stopped with limitation";
  if (agentUiState === "stopped") return "agent stopped";

  const trainingRun = state.trainingRuns[craft.id];
  if (isTrainingRunActive(trainingRun)) return "local run active";
  if (trainingRun?.status === "failed") return "local run failed";

  return "idle";
}

function getCraftLiveSignal(craft) {
  if (isRemoteSharedCraft(craft)) {
    return {
      message: `Linked to ${getSharedOwnerName(craft)}`,
      meta:
        String(craft?.sync?.mode || "") === "live"
          ? "updates stream in automatically"
          : "snapshot from the last sync",
    };
  }

  return null;
}

function getCraftMetaPrimary(craft) {
  if (craft?.metricsReady === false && isVanillaStarterCraft(craft)) {
    return "vanilla target";
  }
  if (typeof craft?.accuracy === "number" && Number.isFinite(craft.accuracy) && craft.accuracy > 0) {
    return formatAccuracy(craft.accuracy);
  }
  if (String(craft?.stage || "").trim()) {
    return String(craft.stage).trim();
  }
  return "";
}

function describeCraftAccuracyForAgent(craft) {
  if (typeof craft?.accuracy === "number" && Number.isFinite(craft.accuracy) && craft.accuracy > 0) {
    return formatAccuracy(craft.accuracy);
  }
  return "not evaluated yet";
}

function describeCraftSeedRows(craft) {
  if (Number.isFinite(Number(craft?.seedRows)) && Number(craft.seedRows) > 0) {
    return String(Number(craft.seedRows));
  }
  return "not collected yet";
}

function describeCraftDatasetRows(craft) {
  if (Number.isFinite(Number(craft?.datasetRows)) && Number(craft.datasetRows) > 0) {
    return String(Number(craft.datasetRows));
  }
  return "not released yet";
}

function describeCraftCoverageGaps(craft) {
  if (Number.isFinite(Number(craft?.openGaps)) && Number(craft.openGaps) >= 0) {
    return String(Number(craft.openGaps));
  }
  return "not assessed yet";
}

function getCraftingAgentTooling(craft) {
  return normalizeCraftingAgentToolingPayload(craft?.agentTooling);
}

function getCraftingAgentToolingLabel(craft) {
  return formatCraftingAgentToolLabels(getCraftingAgentTooling(craft));
}

function createAgentToolingProvenanceEntry(craft) {
  const toolingLabel = getCraftingAgentToolingLabel(craft);
  if (!toolingLabel) return null;
  return {
    title: "Pinned supervisor tools",
    detail: `The crafting agent always starts in the sidepanel with ${toolingLabel}. This stays fixed during the run and can be changed later in Craft.`,
    kind: "constraint",
    sampleId: "",
    operationType: "",
  };
}

function getLastAgentLogMessage(agentRun) {
  if (!agentRun || !Array.isArray(agentRun.logs) || !agentRun.logs.length) return "";
  return String(agentRun.logs[agentRun.logs.length - 1]?.message || "");
}

function normalizeAgentFinalStatus(value) {
  const status = String(value || "").trim().toLowerCase();
  return ["done", "blocked", "continue"].includes(status) ? status : "";
}

function hasAgentLimitationNote(agentRun) {
  if (!agentRun || typeof agentRun !== "object") return false;
  const entries = [
    agentRun?.responseText,
    agentRun?.report?.currentState,
    agentRun?.report?.nextAction,
    ...(Array.isArray(agentRun?.logs) ? agentRun.logs.map((entry) => entry?.message) : []),
    ...(Array.isArray(agentRun?.provenance)
      ? agentRun.provenance.flatMap((entry) => [entry?.title, entry?.detail])
      : []),
  ];
  const haystack = entries
    .map((entry) => String(entry || "").trim().toLowerCase())
    .filter(Boolean)
    .join("\n");
  return /(keine freigegebenen werkzeuge|kein passender ablauf|nicht aufgenommen|vorerst ausgelassen|not supported|unsupported|missing tool|cannot be covered|nicht abgedeckt)/.test(
    haystack,
  );
}

function getAgentUiState(agentRun) {
  if (!agentRun) return "idle";
  const status = String(agentRun.status || "").trim().toLowerCase();
  if (status === "starting") return "starting";
  if (status === "running" || status === "reloading") return "running";
  if (status === "needs_input") return "needs_input";
  if (status === "blocked") return "blocked";
  if (status === "failed") {
    return normalizeAgentFinalStatus(agentRun.finalStatus) === "blocked" ? "blocked" : "failed";
  }
  if (status === "done") {
    return hasAgentLimitationNote(agentRun) ? "stopped_limited" : "stopped";
  }
  return status || "idle";
}

function shouldShowAgentRunDebugButton(agentRun) {
  const clipboardStatus = String(agentRun?.clipboardStatus || "idle");
  if (clipboardStatus === "copied" || clipboardStatus === "copying") return false;
  const uiState = getAgentUiState(agentRun);
  return ["stopped", "stopped_limited", "blocked", "failed"].includes(uiState);
}

function getAgentLogStatusText(agentRun) {
  const uiState = getAgentUiState(agentRun);
  if (uiState === "starting") return "starting";
  if (uiState === "running") return "running";
  if (uiState === "needs_input") return "needs input";
  if (uiState === "blocked") return "blocked";
  if (uiState === "failed") return "failed";
  if (uiState === "stopped_limited") return "stopped / limited";
  if (uiState === "stopped") return "stopped";
  return String(agentRun?.status || "idle");
}

function getToolStatusText(craft) {
  const tooling = craft?.tooling;
  if (tooling && typeof tooling === "object") {
    return `tools ${formatCount(tooling.ready || 0)}/${formatCount(tooling.total || 0)}`;
  }

  const total = Array.isArray(craft?.tools) ? craft.tools.length : 0;
  return `tools ${formatCount(total)}/${formatCount(total)}`;
}

function isCraftingActive(craftId) {
  return isAgentRunActive(state.agentRuns[craftId]) || isTrainingRunActive(state.trainingRuns[craftId]);
}

function isAgentRunActive(run) {
  return ["starting", "running", "reloading"].includes(String(run?.status || ""));
}

function isTrainingRunActive(run) {
  return ["queued", "starting", "running"].includes(String(run?.status || ""));
}

function renderMetaTokens(items) {
  const values = items.filter(Boolean);
  return values
    .map(
      (value, index) => `
        ${index ? '<span class="meta-sep" aria-hidden="true">·</span>' : ""}
        <span class="meta-token">${escapeHtml(value)}</span>
      `,
    )
    .join("");
}

function formatClock(value) {
  return value.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function formatCount(value) {
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: 0 }).format(Number(value) || 0);
}

function formatPeerCountLabel(value) {
  const count = Math.max(0, Number(value) || 0);
  return `${formatCount(count)} peer${count === 1 ? "" : "s"}`;
}

function formatTimeAgo(value) {
  const timestamp = new Date(String(value || ""));
  if (Number.isNaN(timestamp.getTime())) return "unknown";

  const diffMs = Math.max(0, Date.now() - timestamp.getTime());
  const diffMinutes = Math.round(diffMs / 60000);
  if (diffMinutes < 1) return "just now";
  if (diffMinutes < 60) return `${diffMinutes}m ago`;
  const diffHours = Math.round(diffMinutes / 60);
  if (diffHours < 24) return `${diffHours}h ago`;
  const diffDays = Math.round(diffHours / 24);
  return `${diffDays}d ago`;
}

function formatCompactCount(value) {
  const number = Number(value);
  if (!Number.isFinite(number) || number <= 0) return "0";
  if (number >= 1000000) return `${(number / 1000000).toFixed(1)}m`;
  if (number >= 1000) return `${(number / 1000).toFixed(1)}k`;
  return `${Math.round(number)}`;
}

function formatItemsPerSecond(value, unitLabel = "samples") {
  const number = Number(value);
  const label = String(unitLabel || "samples").trim() || "samples";
  if (!Number.isFinite(number) || number <= 0) return `0.0 ${label}/s`;
  return `${number.toFixed(1)} ${label}/s`;
}

function getTrainingComparisonStat(run) {
  const metrics = run?.metrics && typeof run.metrics === "object" ? run.metrics : {};
  const compareBy = String(run?.runtime?.trainingComparison?.compareBy || metrics.compareBy || "").trim();
  if (compareBy === "forward_tokens_per_second") {
    const overallForwardTokensPerSecond = Number(metrics.overallForwardTokensPerSecond || 0);
    if (overallForwardTokensPerSecond > 0) {
      return formatItemsPerSecond(overallForwardTokensPerSecond, "forward tok");
    }
    const phaseForwardTokensPerSecond = Number(metrics.phaseForwardTokensPerSecond || 0);
    if (phaseForwardTokensPerSecond > 0) {
      return formatItemsPerSecond(phaseForwardTokensPerSecond, "forward tok");
    }
  }
  return "";
}

function ensureUniqueCraftId(name) {
  const base = slugify(name);
  let candidate = base;
  let index = 2;
  while (findCraft(candidate)) {
    candidate = `${base}-${index}`;
    index += 1;
  }
  return candidate;
}

function isVanillaStarterCraft(craft) {
  return String(craft?.starterMode || "") === "vanilla_target" || String(craft?.useStatus || "") === "vanilla";
}

async function forkCraftIntoLocalDraft(craftId) {
  const craft = findCraft(craftId);
  if (!craft) return;
  const editableCrafts = state.crafts.filter((entry) => craftStore?.isEditableCraft?.(entry) !== false);
  const fork = craftStore?.createForkFromCraft?.(craft, editableCrafts);
  if (!fork) return;

  state.crafts = await persistCrafts([...state.crafts, fork]);
  if (craftStore?.isEditableCraft?.(craft) !== false) {
    await craftSync?.cloneLocalArtifactsForCraft?.(craft.id, fork.id);
  } else {
    await craftSync?.materializeBundleArtifactsForCraft?.(craft, fork.id);
  }
  state.activeCraftId = fork.id;
  state.craftingCraftId = fork.id;
  await loadToolScriptsState(fork.id, { force: true });
  render();
}

async function toggleCraftShareFromSidepanel(craftId) {
  const craft = findCraft(craftId);
  if (!craft || isRemoteSharedCraft(craft)) return;
  const isShared = craft?.sharing?.enabled === true;

  if (!isShared && !canEnableCraftSharing()) {
    state.useMessages[craftId] = {
      time: formatClock(new Date()),
      text: "Set your own share code in Share settings before sharing this craft with someone else.",
    };
    render();
    return;
  }

  state.crafts = await persistCrafts(
    state.crafts.map((entry) => {
      if (entry.id !== craftId) return entry;
      const now = new Date().toISOString();
      return craftStore.normalizeCraft({
        ...entry,
        updatedAt: now,
        sharing: {
          enabled: !isShared,
          publishedAt: isShared ? "" : now,
        },
      });
    }),
  );
  render();
}

function getConfiguredTargetModelName() {
  const targetLabel = configApi?.getSlotDisplayLabel?.("target", state.slots, state.providers);
  const fallback = configApi?.createDefaultModelSlots?.()?.target?.modelName || "unsloth/Qwen3.5-0.8B (Vanilla)";
  return String(targetLabel?.value || fallback).trim() || fallback;
}

function getDefaultVanillaModelName() {
  return String(
    configApi?.createDefaultModelSlots?.()?.target?.modelName || "unsloth/Qwen3.5-0.8B (Vanilla)",
  ).trim() || "unsloth/Qwen3.5-0.8B (Vanilla)";
}

function getVanillaTargetModelLabel(craft) {
  return String(craft?.starterModelName || getDefaultVanillaModelName()).trim() || "unsloth/Qwen3.5-0.8B (Vanilla)";
}

function renderTutorialOverlay() {
  const tutorial = state.tutorialOverlay;
  if (!tutorial) return "";
  if (tutorial.mode === "create") {
    return `
      <section class="tutorial-overlay" aria-label="Capabilities">
        <div class="tutorial-backdrop" data-action="dismiss-tutorial"></div>
        <div class="tutorial-card">
          <div class="tutorial-brand">
            <img
              class="tutorial-brand-mark"
              src="assets/branding/logo-512.png"
              alt="Fuck API, Train Local AI logo"
              width="144"
              height="144"
            >
          </div>
          <div class="tutorial-eyebrow">New capability</div>
          <h2 class="tutorial-title">Start from an example or write your own</h2>
          <p class="tutorial-copy">
            The first example has already been inserted into the input below. Pick another example to replace it, or
            continue and edit the template.
          </p>
          <div class="tutorial-example-list">
            ${renderStarterTutorialExampleCards()}
          </div>
          <div class="tutorial-steps">
            <article class="tutorial-step">
              <div class="tutorial-step-title">1</div>
              <div class="tutorial-step-copy">Review or replace the starter prompt.</div>
            </article>
            <article class="tutorial-step">
              <div class="tutorial-step-title">2</div>
              <div class="tutorial-step-copy">Edit the text in the input if needed.</div>
            </article>
            <article class="tutorial-step">
              <div class="tutorial-step-title">3</div>
              <div class="tutorial-step-copy">Create the capability. The agent will take over and ask follow-up questions if needed.</div>
            </article>
          </div>
          <div class="tutorial-actions">
            <button class="workspace-primary" data-action="tutorial-continue-create" type="button">Continue</button>
          </div>
        </div>
      </section>
    `;
  }

  return `
    <section class="tutorial-overlay" aria-label="Craft tutorial">
      <div class="tutorial-backdrop" data-action="dismiss-tutorial"></div>
      <div class="tutorial-card">
        <div class="tutorial-brand">
          <img
            class="tutorial-brand-mark"
            src="assets/branding/logo-512.png"
            alt="Fuck API, Train Local AI logo"
            width="144"
            height="144"
          >
        </div>
        <div class="tutorial-eyebrow">First Craft</div>
        <h2 class="tutorial-title">${escapeHtml(tutorial.craftName)} is your first capability</h2>
        <p class="tutorial-copy">
          A Craft is one narrow capability you can shape, test, tune locally, and later share as a bundle. The normal next
          step is to open <strong>Craft</strong> and define what this capability should do.
        </p>
        <p class="tutorial-copy">
          This starter already runs on <strong>${escapeHtml(
            tutorial.starterModelName,
          )}</strong>. Describe the capability now and the agent will handle the first build-out steps.
        </p>
        <div class="tutorial-steps">
          <article class="tutorial-step">
            <div class="tutorial-step-title">Craft now</div>
            <div class="tutorial-step-copy">
              Define the task. The agent will derive data, tools, and the first bundle setup from it.
            </div>
          </article>
        </div>
        <div class="tutorial-actions">
          <button class="workspace-primary" data-action="tutorial-open-craft" type="button">Craft now</button>
        </div>
      </div>
    </section>
  `;
}

function formatAccuracy(accuracy) {
  const normalized = Math.max(0, Math.min(1, Number(accuracy) || 0));
  return `${(normalized * 100).toFixed(1)}%`;
}

function formatUsd(value) {
  const number = Number(value);
  if (!Number.isFinite(number) || number <= 0) return "$0.00";
  return new Intl.NumberFormat(undefined, {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: number < 10 ? 2 : 1,
    maximumFractionDigits: number < 10 ? 2 : 1,
  }).format(number);
}

function trimText(value, maxLength) {
  const text = String(value || "").trim();
  if (text.length <= maxLength) return text;
  return `${text.slice(0, Math.max(0, maxLength - 3))}...`;
}

function slugify(value) {
  return (
    String(value || "")
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "") || `craft-${Date.now()}`
  );
}

function escapeHtml(value) {
  return String(value == null ? "" : value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}
