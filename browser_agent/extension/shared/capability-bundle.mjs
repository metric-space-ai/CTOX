export const BUNDLE_SCHEMA_VERSION = 1;
import {
  compilePublishedBrowserCapabilityBundlePayload,
  normalizeBrowserCapabilityBundlePayload,
} from "./browser-capability-bundle.mjs";

export const TRAINING_DATA_ARTIFACT_KIND = "training_samples";
export const TOOL_SCRIPTS_ARTIFACT_KIND = "tool_scripts";
export const BROWSER_CAPABILITY_BUNDLE_ARTIFACT_KIND = "browser_capability_bundle";
export const WEIGHTS_ARTIFACT_KIND = "capability_weights";
export const POLICY_BUNDLE_ARTIFACT_KIND = "policy_bundle";

function asText(value) {
  return String(value == null ? "" : value).trim();
}

function cloneJson(value, fallback = null) {
  try {
    if (typeof globalThis.structuredClone === "function") {
      return globalThis.structuredClone(value);
    }
    return JSON.parse(JSON.stringify(value));
  } catch (_error) {
    return fallback;
  }
}

export function getTrainingDataArtifactId(craftId) {
  return `training-samples:${asText(craftId)}`;
}

export function getToolScriptsArtifactId(craftId) {
  return `tool-scripts:${asText(craftId)}`;
}

export function getWeightsArtifactId(craftId) {
  return `capability-weights:${asText(craftId)}`;
}

export function getBrowserCapabilityBundleArtifactId(craftId) {
  return `browser-capabilities:${asText(craftId)}`;
}

export function getPolicyBundleArtifactId(craftId) {
  return `policy-bundle:${asText(craftId)}`;
}

function normalizeArray(value) {
  return Array.isArray(value) ? value : [];
}

function humanizeToolScriptName(value) {
  const normalized = asText(value)
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .toLowerCase();
  if (!normalized) return "";
  return normalized.charAt(0).toUpperCase() + normalized.slice(1);
}

function inferToolScriptEntrypoint(value) {
  const parts = asText(value)
    .split(/[^A-Za-z0-9]+/)
    .map((entry) => entry.trim())
    .filter(Boolean);
  if (!parts.length) return "";
  return parts[0].toLowerCase() + parts.slice(1).map((entry) => entry.charAt(0).toUpperCase() + entry.slice(1)).join("");
}

function normalizeToolScriptEntrypoint(value) {
  const entrypoint = asText(value);
  if (!entrypoint) return "";
  if (/[\\/]/.test(entrypoint)) return "";
  if (/\.[A-Za-z0-9]+$/.test(entrypoint)) return "";
  return entrypoint;
}

function normalizeToolScriptPayloadOptions(options = null) {
  const source = options && typeof options === "object" ? options : {};
  return {
    inferPlaceholderScripts: source.inferPlaceholderScripts !== false,
    allowToolFallback: source.allowToolFallback !== false,
  };
}

export function normalizeToolScriptEntry(entry, index = 0) {
  const source = asText(entry?.source);
  return {
    id: asText(entry?.id) || `tool-script-${index + 1}`,
    name: asText(entry?.name) || `Tool Script ${index + 1}`,
    description: asText(entry?.description),
    language: asText(entry?.language) || "javascript",
    entrypoint: normalizeToolScriptEntrypoint(entry?.entrypoint),
    source,
  };
}

export function inferToolScriptsPayloadFromCraft(craft = null, options = null) {
  const normalizedOptions = normalizeToolScriptPayloadOptions(options);
  const tools = normalizedOptions.allowToolFallback
    ? normalizeArray(craft?.tools)
        .map((value) => asText(value))
        .filter(Boolean)
    : [];

  return {
    schemaVersion: BUNDLE_SCHEMA_VERSION,
    scripts: normalizedOptions.inferPlaceholderScripts
      ? tools.map((toolName, index) =>
          normalizeToolScriptEntry(
            {
              id: toolName.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "") || `tool-script-${index + 1}`,
              name: humanizeToolScriptName(toolName) || toolName,
              description: "",
              language: "javascript",
              entrypoint: inferToolScriptEntrypoint(toolName),
              source: "",
            },
            index,
          ),
        )
      : [],
    declaredTools: tools,
  };
}

export function normalizeToolScriptsPayload(payload, craft = null, options = null) {
  const fallback = inferToolScriptsPayloadFromCraft(craft, options);
  const declaredTools = normalizeArray(payload?.declaredTools)
    .map((value) => asText(value))
    .filter(Boolean);
  const legacyScriptIds = normalizeArray(payload?.scriptIds)
    .map((value) => asText(value))
    .filter(Boolean);
  const legacyEntrypoints = normalizeArray(payload?.entrypoints);
  const explicitScripts = normalizeArray(payload?.scripts).map((entry, index) => normalizeToolScriptEntry(entry, index));
  const scripts = explicitScripts.length
    ? explicitScripts
    : legacyScriptIds.length
      ? legacyScriptIds.map((scriptId, index) =>
          normalizeToolScriptEntry(
            {
              id: scriptId,
              name: humanizeToolScriptName(declaredTools[index] || scriptId) || declaredTools[index] || scriptId,
              description: "",
              language: "javascript",
              entrypoint: asText(legacyEntrypoints[index] || inferToolScriptEntrypoint(scriptId)),
              source: "",
            },
            index,
          ),
        )
      : fallback.scripts;

  return {
    schemaVersion: BUNDLE_SCHEMA_VERSION,
    scripts,
    declaredTools: declaredTools.length ? declaredTools : fallback.declaredTools,
  };
}

export function createBaseWeightsPayload(craft = null) {
  const trainingConfig = craft?.training && typeof craft.training === "object" ? craft.training : {};
  const shards = Array.isArray(trainingConfig?.shards) ? trainingConfig.shards : [];
  const firstShard = shards[0] && typeof shards[0] === "object" ? shards[0] : {};
  const modelName =
    asText(firstShard?.modelName) ||
    asText(craft?.starterModelName) ||
    "unsloth/Qwen3.5-0.8B (Vanilla)";

  return {
    schemaVersion: BUNDLE_SCHEMA_VERSION,
    status: "base_model_only",
    targetSlot: asText(firstShard?.slotId) || asText(craft?.targetSlot) || "target",
    modelName,
    adapter: null,
    metrics: null,
    runtime: null,
    dataset: null,
    run: null,
  };
}

export function createBasePolicyPayload(craft = null) {
  const tools = normalizeArray(craft?.tools)
    .map((value) => asText(value))
    .filter(Boolean);
  const craftName = asText(craft?.name) || "this craft";
  const hasActiveTextPair = tools.includes("read_active_text_target") && tools.includes("replace_active_text_target");
  return {
    schemaVersion: BUNDLE_SCHEMA_VERSION,
    status: "supervised_only",
    trainingMode: "sft",
    trainingDataFormat: tools.length ? "qwen3_5_native_multiturn_tool_xml_v1" : "supervised_pairs_v1",
    supervisionTarget: tools.length ? "next_assistant_turn" : "schema_bound_output",
    phases: [
      "policy_bundle_release",
      "dataset_release",
      "training",
      "evaluation",
      "rollout_collection",
      "reward_scoring",
      "preference_mining",
      "dpo_training",
      "grpo_training",
    ],
    policySpec: {
      objective: tools.length
        ? `Execute ${craftName} with deterministic browser-safe tool usage and next-assistant-turn supervision over reviewed tools.`
        : `Execute ${craftName} with deterministic structured outputs.`,
      bundleSkill:
        tools.length > 0
          ? hasActiveTextPair
            ? `Allowed tools: ${tools.join(", ")}. Decide the next assistant turn against these reviewed tool signatures. On native Qwen3.5 local runtimes, emit tool use in raw <tool_call><function=...><parameter=...>...</parameter></function></tool_call> format with no suffix. When a reviewed capability can inspect or verify visible browser state, use that multimodal path before acting or answering from prior knowledge. For active-text tasks, first call read_active_text_target to resolve selection, focused_editable, or clipboard in that order, then call replace_active_text_target with the full transformed text. Do not return the transformed text directly when the reviewed tool path can apply it in place.`
            : `Allowed tools: ${tools.join(", ")}. Decide the next assistant turn against these reviewed tool signatures and prefer one reviewed tool call over free-form browser behavior. On native Qwen3.5 local runtimes, emit the reviewed tool call in raw <tool_call><function=...>...</function></tool_call> format rather than inventing a JSON wrapper. When a reviewed capability can inspect or verify visible browser state, use that multimodal path before acting or answering from prior knowledge.`
          : "Prefer structured JSON outputs and deterministic browser-safe behavior.",
      allowedTools: tools,
      completionSignals: [
        "structured_json_valid",
        "approved_tool_used_or_explicit_no_tool",
        "task_goal_resolved_or_blocked_with_reason",
      ],
      stopConditions: [
        "task_completed",
        "missing_required_context",
        "unsafe_or_unapproved_tool_request",
      ],
    },
    rewardSpec: {
      mode: "hybrid_validator_plus_judge",
      hardValidators: [
        "json_schema_valid",
        "allowed_tool_only",
        "completion_signal_present",
      ],
      softJudgeRubric: [
        "goal_completion",
        "tool_choice_quality",
        "argument_quality",
        "brevity_without_omission",
      ],
      scoreWeights: {
        hardSuccess: 0.55,
        jsonMatch: 0.2,
        judgeScore: 0.2,
        costPenalty: 0.05,
      },
      rlSwitchCriteria: {
        requireSftPlateau: true,
        minimumPreferencePairs: 32,
        minimumRolloutGroups: 8,
      },
    },
    judgeSpec: {
      mode: "frozen_agent_judge",
      noSideEffects: true,
      compareMode: "pairwise",
      promptTemplate:
        "Score only task completion quality, tool correctness, JSON correctness, and efficiency. Do not invent hidden requirements.",
    },
  };
}

function base64FromBytes(bytes) {
  if (typeof Buffer !== "undefined") {
    return Buffer.from(bytes).toString("base64");
  }

  let out = "";
  const chunkSize = 0x8000;
  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    const chunk = bytes.subarray(offset, offset + chunkSize);
    out += String.fromCharCode(...chunk);
  }
  return globalThis.btoa(out);
}

export function float32ToDataUrl(value, mime = "application/octet-stream") {
  const array = value instanceof Float32Array ? value : new Float32Array(value || []);
  const view = new Uint8Array(array.buffer.slice(array.byteOffset, array.byteOffset + array.byteLength));
  return `data:${asText(mime) || "application/octet-stream"};base64,${base64FromBytes(view)}`;
}

function countTrainingSamples(payload) {
  return normalizeArray(payload?.samples).length;
}

function countToolScripts(payload) {
  return normalizeArray(payload?.scripts).length;
}

function countBrowserCapabilities(payload) {
  return normalizeArray(payload?.capabilities).length;
}

function describeWeightsMode(payload) {
  const adapter = payload?.adapter && typeof payload.adapter === "object" ? payload.adapter : null;
  const hasTransformerLora = Array.isArray(adapter?.modules) && adapter.modules.some(
    (entry) => asText(entry?.modulePath) && asText(entry?.loraADataUrl) && asText(entry?.loraBDataUrl),
  );
  if (hasTransformerLora) {
    return "trained_adapter";
  }
  return asText(payload?.status) || "base_model_only";
}

export function buildCapabilityBundle({
  craft = null,
  trainingDataRecord = null,
  toolScriptsRecord = null,
  browserCapabilitiesRecord = null,
  weightsRecord = null,
  policyRecord = null,
  generatedAt = new Date().toISOString(),
  preserveStoredBrowserCapabilities = false,
  toolScriptsOptions = null,
  browserCapabilityOptions = null,
} = {}) {
  const normalizedToolScriptsOptions = {
    inferPlaceholderScripts: false,
    ...(toolScriptsOptions && typeof toolScriptsOptions === "object" ? toolScriptsOptions : {}),
  };
  const trainingPayload =
    trainingDataRecord?.payload && typeof trainingDataRecord.payload === "object"
      ? cloneJson(trainingDataRecord.payload, { samples: [] })
      : { samples: [] };
  const toolScriptsPayload = normalizeToolScriptsPayload(
    toolScriptsRecord?.payload,
    craft,
    normalizedToolScriptsOptions,
  );
  const browserCapabilitiesPayload =
    browserCapabilitiesRecord?.payload && typeof browserCapabilitiesRecord.payload === "object"
      ? preserveStoredBrowserCapabilities
        ? cloneJson(
            browserCapabilitiesRecord.payload,
            normalizeBrowserCapabilityBundlePayload(
              browserCapabilitiesRecord.payload,
              craft,
              toolScriptsPayload,
              browserCapabilityOptions,
            ),
          )
        : normalizeBrowserCapabilityBundlePayload(
            browserCapabilitiesRecord.payload,
            craft,
            toolScriptsPayload,
            browserCapabilityOptions,
          )
      : normalizeBrowserCapabilityBundlePayload(null, craft, toolScriptsPayload, browserCapabilityOptions);
  const weightsPayload =
    weightsRecord?.payload && typeof weightsRecord.payload === "object"
      ? cloneJson(weightsRecord.payload, createBaseWeightsPayload(craft))
      : createBaseWeightsPayload(craft);
  const policyPayload =
    policyRecord?.payload && typeof policyRecord.payload === "object"
      ? cloneJson(policyRecord.payload, createBasePolicyPayload(craft))
      : createBasePolicyPayload(craft);

  return {
    schemaVersion: BUNDLE_SCHEMA_VERSION,
    generatedAt: asText(generatedAt) || new Date().toISOString(),
    trainingData: {
      kind: TRAINING_DATA_ARTIFACT_KIND,
      artifactId: asText(trainingDataRecord?.id) || getTrainingDataArtifactId(craft?.id),
      payload: trainingPayload,
      meta: cloneJson(trainingDataRecord?.meta, {}),
    },
    toolScripts: {
      kind: TOOL_SCRIPTS_ARTIFACT_KIND,
      artifactId: asText(toolScriptsRecord?.id) || getToolScriptsArtifactId(craft?.id),
      payload: toolScriptsPayload,
      meta: cloneJson(toolScriptsRecord?.meta, {}),
    },
    browserCapabilities: {
      kind: BROWSER_CAPABILITY_BUNDLE_ARTIFACT_KIND,
      artifactId: asText(browserCapabilitiesRecord?.id) || getBrowserCapabilityBundleArtifactId(craft?.id),
      payload: browserCapabilitiesPayload,
      meta: cloneJson(browserCapabilitiesRecord?.meta, {}),
    },
    weights: {
      kind: WEIGHTS_ARTIFACT_KIND,
      artifactId: asText(weightsRecord?.id) || getWeightsArtifactId(craft?.id),
      payload: weightsPayload,
      meta: cloneJson(weightsRecord?.meta, {}),
    },
    policy: {
      kind: POLICY_BUNDLE_ARTIFACT_KIND,
      artifactId: asText(policyRecord?.id) || getPolicyBundleArtifactId(craft?.id),
      payload: policyPayload,
      meta: cloneJson(policyRecord?.meta, {}),
    },
    summary: {
      sampleCount: countTrainingSamples(trainingPayload),
      toolScriptCount: countToolScripts(toolScriptsPayload),
      browserCapabilityCount: countBrowserCapabilities(browserCapabilitiesPayload),
      weightsMode: describeWeightsMode(weightsPayload),
      policyMode: asText(policyPayload?.trainingMode || policyPayload?.status) || "supervised_only",
      hasPolicy:
        Boolean(policyPayload?.policySpec && typeof policyPayload.policySpec === "object") ||
        Boolean(policyPayload?.rewardSpec && typeof policyPayload.rewardSpec === "object"),
      hasAdapter:
        Array.isArray(weightsPayload?.adapter?.modules) &&
        weightsPayload.adapter.modules.some(
          (entry) => asText(entry?.modulePath) && asText(entry?.loraADataUrl) && asText(entry?.loraBDataUrl),
        ),
    },
  };
}

export function extractBundleArtifactRecords(bundle, craftId) {
  const targetCraftId = asText(craftId);
  if (!targetCraftId || !bundle || typeof bundle !== "object") return [];

  const out = [];
  const normalizedToolScriptsPayload =
    bundle.toolScripts?.payload && typeof bundle.toolScripts.payload === "object"
      ? normalizeToolScriptsPayload(bundle.toolScripts.payload, null, {
          inferPlaceholderScripts: false,
          allowToolFallback: false,
        })
      : null;
  if (bundle.trainingData?.payload && typeof bundle.trainingData.payload === "object") {
    out.push({
      id: getTrainingDataArtifactId(targetCraftId),
      craftId: targetCraftId,
      kind: TRAINING_DATA_ARTIFACT_KIND,
      payload: cloneJson(bundle.trainingData.payload, { samples: [] }),
      meta: {
        importedFromBundle: true,
        importedAt: Date.now(),
        sourceArtifactId: asText(bundle.trainingData.artifactId),
      },
    });
  }
  if (bundle.toolScripts?.payload && typeof bundle.toolScripts.payload === "object") {
    out.push({
      id: getToolScriptsArtifactId(targetCraftId),
      craftId: targetCraftId,
      kind: TOOL_SCRIPTS_ARTIFACT_KIND,
      payload: normalizeToolScriptsPayload(bundle.toolScripts.payload, null, {
        inferPlaceholderScripts: false,
        allowToolFallback: false,
      }),
      meta: {
        importedFromBundle: true,
        importedAt: Date.now(),
        sourceArtifactId: asText(bundle.toolScripts.artifactId),
      },
    });
  }
  if (bundle.browserCapabilities?.payload && typeof bundle.browserCapabilities.payload === "object") {
    const publication =
      bundle.browserCapabilities.payload?.publication && typeof bundle.browserCapabilities.payload.publication === "object"
        ? bundle.browserCapabilities.payload.publication
        : null;
    const compiledBrowserPayload = compilePublishedBrowserCapabilityBundlePayload(bundle.browserCapabilities.payload, {
      toolScriptsPayload: normalizedToolScriptsPayload,
      publishedAt: asText(publication?.publishedAt || publication?.published_at),
      publishedBy: asText(publication?.publishedBy || publication?.published_by) || "bundle_materialize",
    });
    if (compiledBrowserPayload.ok) {
      out.push({
        id: getBrowserCapabilityBundleArtifactId(targetCraftId),
        craftId: targetCraftId,
        kind: BROWSER_CAPABILITY_BUNDLE_ARTIFACT_KIND,
        payload: compiledBrowserPayload.payload,
        meta: {
          importedFromBundle: true,
          importedAt: Date.now(),
          sourceArtifactId: asText(bundle.browserCapabilities.artifactId),
        },
      });
    }
  }
  if (bundle.weights?.payload && typeof bundle.weights.payload === "object") {
    out.push({
      id: getWeightsArtifactId(targetCraftId),
      craftId: targetCraftId,
      kind: WEIGHTS_ARTIFACT_KIND,
      payload: cloneJson(bundle.weights.payload, createBaseWeightsPayload(null)),
      meta: {
        importedFromBundle: true,
        importedAt: Date.now(),
        sourceArtifactId: asText(bundle.weights.artifactId),
      },
    });
  }
  if (bundle.policy?.payload && typeof bundle.policy.payload === "object") {
    out.push({
      id: getPolicyBundleArtifactId(targetCraftId),
      craftId: targetCraftId,
      kind: POLICY_BUNDLE_ARTIFACT_KIND,
      payload: cloneJson(bundle.policy.payload, createBasePolicyPayload(null)),
      meta: {
        importedFromBundle: true,
        importedAt: Date.now(),
        sourceArtifactId: asText(bundle.policy.artifactId),
      },
    });
  }
  return out;
}

export function describeCapabilityBundle(bundle) {
  const summary = bundle?.summary && typeof bundle.summary === "object" ? bundle.summary : {};
  const parts = [];
  const sampleCount = Number(summary.sampleCount || 0);
  const toolScriptCount = Number(summary.toolScriptCount || 0);
  const browserCapabilityCount = Number(summary.browserCapabilityCount || 0);
  const weightsMode = asText(summary.weightsMode);
  const policyMode = asText(summary.policyMode);
  if (sampleCount > 0) parts.push(`${sampleCount} samples`);
  if (toolScriptCount > 0) parts.push(`${toolScriptCount} tool scripts`);
  if (browserCapabilityCount > 0) parts.push(`${browserCapabilityCount} browser capabilities`);
  if (policyMode) {
    parts.push(
      policyMode === "sft"
        ? "policy-guided SFT"
        : policyMode === "dpo"
          ? "policy-guided DPO"
          : policyMode === "grpo"
            ? "policy-guided GRPO"
            : policyMode.replace(/_/g, " "),
    );
  }
  if (weightsMode) {
    parts.push(
      weightsMode === "trained_adapter"
        ? "trained weights"
        : weightsMode === "base_model_only"
          ? "base model bundle"
          : weightsMode.replace(/_/g, " "),
    );
  }
  return parts.join(" · ");
}
