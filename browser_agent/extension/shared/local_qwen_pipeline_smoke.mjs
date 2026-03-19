import { createBasePolicyPayload } from "./capability-bundle.mjs";
import {
  buildCanonicalCallToolAction,
  extractCanonicalToolName,
  normalizeCanonicalToolAction,
} from "./canonical-tool-action.mjs";

export const LOCAL_QWEN_PIPELINE_SMOKE_CONFIG = Object.freeze({
  profile: "pipeline_smoke",
  maxTrainPairs: 4,
  maxTestPairs: 2,
  maxSeqLen: 24,
  rank: 16,
  alpha: 32,
  epochs: 1,
  batchTokens: 64,
  modelBatchSize: 2,
  learningRate: 0.0001,
  optimizer: "full_transformer_lora_spsa",
  spsaEpsilon: 0.001,
  spsaSamples: 1,
  spsaGradientClip: 1,
  reasoningMode: "no_think",
  seed: 42,
  trainMatrixA: true,
  trainMatrixB: true,
});

function asText(value) {
  return String(value == null ? "" : value).trim();
}

function stableJson(value) {
  if (Array.isArray(value)) {
    return `[${value.map((entry) => stableJson(entry)).join(",")}]`;
  }
  if (value && typeof value === "object") {
    return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${stableJson(value[key])}`).join(",")}}`;
  }
  return JSON.stringify(value);
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

function buildPortableToolDefinitions(toolName, description, parameters) {
  return [{
    type: "function",
    function: {
      name: asText(toolName),
      description: asText(description),
      parameters: cloneJson(parameters, {
        type: "object",
        additionalProperties: true,
      }),
    },
  }];
}

function buildPortableToolTrainingRow({
  promptText,
  toolAction,
  description,
  parameters,
}) {
  const normalized = normalizeCanonicalToolAction(toolAction);
  const callId = `call_${asText(normalized.tool_name).replace(/[^a-z0-9]+/gi, "_").toLowerCase()}`;
  return {
    prompt_text: asText(promptText),
    messages: [
      {
        role: "system",
        content: "You are a browser-safe agent. Decide the next assistant action using reviewed tools only.",
      },
      {
        role: "user",
        content: asText(promptText),
      },
      {
        role: "assistant",
        content: "",
        tool_calls: [{
          id: callId,
          type: "function",
          function: {
            name: asText(normalized.tool_name),
            arguments: JSON.stringify(normalized.arguments || {}),
          },
        }],
      },
    ],
    tools: buildPortableToolDefinitions(normalized.tool_name, description, parameters),
    target_turn_index: 2,
    output_mode: "multiturn_tool_agent",
  };
}

function computeJsonMatch(referenceValue, candidateValue) {
  return stableJson(referenceValue) === stableJson(candidateValue) ? 1 : 0.5;
}

function mutateCandidate(referenceAction, rank) {
  const candidate = cloneJson(normalizeCanonicalToolAction(referenceAction), {});
  if (!candidate || typeof candidate !== "object") return referenceAction;
  if (rank === 0) return candidate;
  if (candidate.arguments && typeof candidate.arguments === "object") {
    candidate.arguments = { ...candidate.arguments, draft_rank: rank };
  } else {
    candidate.draft_rank = rank;
  }
  if (rank % 2 === 0) {
    candidate.tool_name = `${asText(candidate.tool_name || "tool")}_draft`;
  }
  return candidate;
}

function extractExpectedJsonFromRow(row) {
  if (row?.expected_json && typeof row.expected_json === "object") {
    return cloneJson(row.expected_json, {});
  }
  const targetTurnIndex = Number.isInteger(row?.target_turn_index) ? row.target_turn_index : -1;
  const messages = Array.isArray(row?.messages) ? row.messages : [];
  const targetMessage = targetTurnIndex >= 0 ? messages[targetTurnIndex] : null;
  const toolCall = Array.isArray(targetMessage?.tool_calls) ? targetMessage.tool_calls[0] : null;
  const functionName = asText(toolCall?.function?.name);
  if (!functionName) return {};
  let argumentsValue = {};
  try {
    argumentsValue = JSON.parse(asText(toolCall?.function?.arguments) || "{}");
  } catch (_error) {
    argumentsValue = {};
  }
  return buildCanonicalCallToolAction({
    toolName: functionName,
    argumentsValue,
  });
}

function normalizeRows(datasetPayload) {
  const train = Array.isArray(datasetPayload?.train) ? datasetPayload.train : [];
  const test = Array.isArray(datasetPayload?.test) ? datasetPayload.test : [];
  return train.concat(test).map((row, index) => ({
    rowId: `smoke-row-${index + 1}`,
    prompt_text: asText(row?.prompt_text || row?.prompt || `Smoke prompt ${index + 1}`),
    expected_json: extractExpectedJsonFromRow(row),
  }));
}

function createStatusEmitter(totalSamples, onStatus) {
  let completedSamples = 0;
  let phaseCompletedSamples = 0;
  let phaseTotalSamples = 0;
  let phase = "";
  let phaseLabel = "";
  let message = "";
  let phaseStartedAt = Date.now();

  const emit = (extra = {}) => {
    if (typeof onStatus !== "function") return;
    const elapsedS = Math.max((Date.now() - phaseStartedAt) / 1000, 1e-6);
    onStatus({
      totalSamples,
      completedSamples,
      phaseCompletedSamples,
      phaseTotalSamples,
      phaseUnitLabel: extra.phaseUnitLabel || "pipeline steps",
      samplesPerSecond: phaseCompletedSamples > 0 ? phaseCompletedSamples / elapsedS : 0,
      phase,
      phaseLabel,
      message,
      ...extra,
    });
  };

  return {
    begin(nextPhase, nextLabel, nextMessage, nextTotal, extra = {}) {
      phase = asText(nextPhase);
      phaseLabel = asText(nextLabel);
      message = asText(nextMessage);
      phaseCompletedSamples = 0;
      phaseTotalSamples = Math.max(0, Number(nextTotal || 0));
      phaseStartedAt = Date.now();
      emit(extra);
    },
    advance(delta = 1, extra = {}) {
      const safeDelta = Math.max(0, Number(delta || 0));
      completedSamples += safeDelta;
      phaseCompletedSamples += safeDelta;
      emit(extra);
    },
    getCompletedSamples() {
      return completedSamples;
    },
  };
}

export function createLocalQwenPipelineSmokeDatasetPayload() {
  const composeEmailParameters = {
    type: "object",
    properties: {
      recipient: { type: "string" },
      subject: { type: "string" },
      body: { type: "string" },
    },
    required: ["recipient", "subject", "body"],
    additionalProperties: false,
  };
  const submitInterestParameters = {
    type: "object",
    properties: {
      listing_url: { type: "string" },
      message: { type: "string" },
    },
    required: ["listing_url", "message"],
    additionalProperties: false,
  };
  return {
    meta: {
      scenario: "pipeline_smoke",
      taskType: "tool_routing",
    },
    train: [
      buildPortableToolTrainingRow({
        promptText: "Bitte verfasse eine kurze Interessensmail fuer die Wohnung in Berlin Mitte.",
        toolAction: buildCanonicalCallToolAction({
          toolName: "compose_email",
          argumentsValue: {
            recipient: "mitte@example.com",
            subject: "Interesse an der Wohnung in Berlin Mitte",
            body: "Guten Tag, ich interessiere mich fuer die Wohnung in Berlin Mitte.",
          },
          bundleRef: "browser-capabilities:smoke",
          capabilityRef: "compose_email",
          skillRef: "reviewed_browser_tool_bundle",
        }),
        description: "Compose a structured email draft.",
        parameters: composeEmailParameters,
      }),
      buildPortableToolTrainingRow({
        promptText: "Schicke fuer das Inserat in Neukolln eine formelle Anfrage.",
        toolAction: buildCanonicalCallToolAction({
          toolName: "compose_email",
          argumentsValue: {
            recipient: "neukoelln@example.com",
            subject: "Anfrage zum Inserat in Neukolln",
            body: "Guten Tag, ich moechte gerne weitere Informationen zum Inserat erhalten.",
          },
          bundleRef: "browser-capabilities:smoke",
          capabilityRef: "compose_email",
          skillRef: "reviewed_browser_tool_bundle",
        }),
        description: "Compose a structured email draft.",
        parameters: composeEmailParameters,
      }),
      buildPortableToolTrainingRow({
        promptText: "Leite fuer das Angebot in Kreuzberg direkt eine Interessenanfrage ein.",
        toolAction: buildCanonicalCallToolAction({
          toolName: "submit_interest_request",
          argumentsValue: {
            listing_url: "https://example.com/kreuzberg",
            message: "Ich habe Interesse am Angebot in Kreuzberg.",
          },
          bundleRef: "browser-capabilities:smoke",
          capabilityRef: "submit_interest_request",
          skillRef: "reviewed_browser_tool_bundle",
        }),
        description: "Submit a structured interest request.",
        parameters: submitInterestParameters,
      }),
      buildPortableToolTrainingRow({
        promptText: "Nutze fuer die Anzeige in Charlottenburg den passenden Browser-Call.",
        toolAction: buildCanonicalCallToolAction({
          toolName: "submit_interest_request",
          argumentsValue: {
            listing_url: "https://example.com/charlottenburg",
            message: "Ich interessiere mich fuer die Anzeige in Charlottenburg.",
          },
          bundleRef: "browser-capabilities:smoke",
          capabilityRef: "submit_interest_request",
          skillRef: "reviewed_browser_tool_bundle",
        }),
        description: "Submit a structured interest request.",
        parameters: submitInterestParameters,
      }),
    ],
    test: [
      buildPortableToolTrainingRow({
        promptText: "Verfasse fuer das Angebot in Prenzlauer Berg eine kurze Mail.",
        toolAction: buildCanonicalCallToolAction({
          toolName: "compose_email",
          argumentsValue: {
            recipient: "prenzlauerberg@example.com",
            subject: "Interesse am Angebot in Prenzlauer Berg",
            body: "Guten Tag, ich interessiere mich fuer das Angebot in Prenzlauer Berg.",
          },
          bundleRef: "browser-capabilities:smoke",
          capabilityRef: "compose_email",
          skillRef: "reviewed_browser_tool_bundle",
        }),
        description: "Compose a structured email draft.",
        parameters: composeEmailParameters,
      }),
      buildPortableToolTrainingRow({
        promptText: "Sende fuer das Listing in Friedrichshain eine strukturierte Interessenanfrage.",
        toolAction: buildCanonicalCallToolAction({
          toolName: "submit_interest_request",
          argumentsValue: {
            listing_url: "https://example.com/friedrichshain",
            message: "Ich moechte mein Interesse fuer das Listing in Friedrichshain bekunden.",
          },
          bundleRef: "browser-capabilities:smoke",
          capabilityRef: "submit_interest_request",
          skillRef: "reviewed_browser_tool_bundle",
        }),
        description: "Submit a structured interest request.",
        parameters: submitInterestParameters,
      }),
    ],
  };
}

export function estimateLocalQwenPipelineSmokeWork(datasetPayload, candidateCount = 3) {
  const rows = normalizeRows(datasetPayload);
  const rolloutCandidates = rows.length * Math.max(2, Number(candidateCount || 0));
  return rows.length + 1 + rolloutCandidates + rolloutCandidates + rows.length + 2;
}

export async function runLocalQwenPipelineSmoke({
  datasetPayload = createLocalQwenPipelineSmokeDatasetPayload(),
  candidateCount = 3,
  onStatus = null,
} = {}) {
  const rows = normalizeRows(datasetPayload);
  const safeCandidateCount = Math.max(2, Number(candidateCount || 0));
  const totalSamples = estimateLocalQwenPipelineSmokeWork(datasetPayload, safeCandidateCount);
  const status = createStatusEmitter(totalSamples, onStatus);

  status.begin(
    "pipeline_dataset_validation",
    "Validating smoke dataset",
    "Checking that the fast finetuning smoke dataset can drive the full staged pipeline.",
    rows.length,
  );
  const allowedTools = new Set();
  for (const row of rows) {
    if (!row.prompt_text || !row.expected_json || typeof row.expected_json !== "object") {
      throw new Error("The finetuning smoke dataset is missing prompt_text or a supervised assistant tool target.");
    }
    allowedTools.add(extractCanonicalToolName(row.expected_json));
    status.advance(1);
  }

  status.begin(
    "pipeline_policy_bundle",
    "Building policy bundle",
    "Creating a frozen bundle skill, reward spec, and judge spec for the smoke pipeline.",
    1,
  );
  const policyBundle = createBasePolicyPayload({
    name: "Header Finetuning Smoke",
    tools: Array.from(allowedTools).filter(Boolean),
  });
  status.advance(1, {
    policyBundle,
  });

  status.begin(
    "pipeline_rollouts",
    "Collecting rollout candidates",
    "Generating grouped rollout candidates against the frozen smoke policy bundle.",
    rows.length * safeCandidateCount,
  );
  const rolloutTrace = [];
  for (const [rowIndex, row] of rows.entries()) {
    const rolloutGroupId = `smoke-group-${rowIndex + 1}`;
    for (let rank = 0; rank < safeCandidateCount; rank += 1) {
      const candidate = mutateCandidate(row.expected_json, rank);
      const toolName = extractCanonicalToolName(candidate) || extractCanonicalToolName(row.expected_json);
      rolloutTrace.push({
        trace_id: `${rolloutGroupId}-candidate-${rank + 1}`,
        rollout_group_id: rolloutGroupId,
        candidate_rank: rank,
        prompt_text: row.prompt_text,
        reference_json: cloneJson(row.expected_json, {}),
        candidate_json: candidate,
        tool_name: toolName,
        schema_ok: Boolean(candidate && typeof candidate === "object"),
        tool_allowed: policyBundle.policySpec.allowedTools.includes(toolName),
        task_completed: rank === 0 || rank === 1,
      });
      status.advance(1);
    }
  }

  status.begin(
    "pipeline_rewards",
    "Scoring rewards",
    "Scoring rollout candidates with hard validators plus the frozen smoke judge rubric.",
    rolloutTrace.length,
  );
  const rewardLabels = rolloutTrace.map((trace) => {
    const hardSuccess = trace.schema_ok && trace.tool_allowed && trace.task_completed ? 1 : 0;
    const jsonMatch = computeJsonMatch(trace.reference_json, trace.candidate_json);
    const judgeScore = Number(((hardSuccess + jsonMatch) / 2).toFixed(4));
    const costPenalty = Number((trace.candidate_rank * 0.05).toFixed(4));
    const totalReward = Number((0.55 * hardSuccess + 0.2 * jsonMatch + 0.2 * judgeScore - 0.05 * costPenalty).toFixed(4));
    status.advance(1);
    return {
      trace_id: trace.trace_id,
      rollout_group_id: trace.rollout_group_id,
      hardSuccess,
      jsonMatch,
      judgeScore,
      costPenalty,
      totalReward,
    };
  });

  status.begin(
    "pipeline_preferences",
    "Mining preferences",
    "Building chosen/rejected preference pairs from the rollout rewards.",
    rows.length,
  );
  const preferenceDataset = [];
  for (const row of rows) {
    const groupRewards = rewardLabels
      .filter((entry) => entry.rollout_group_id === `smoke-group-${rows.indexOf(row) + 1}`)
      .sort((left, right) => right.totalReward - left.totalReward);
    if (groupRewards.length >= 2) {
      const chosen = groupRewards[0];
      const rejected = groupRewards[groupRewards.length - 1];
      const margin = Number((chosen.totalReward - rejected.totalReward).toFixed(4));
      if (margin >= 0.05) {
        preferenceDataset.push({
          preference_id: `${chosen.rollout_group_id}-pref`,
          rollout_group_id: chosen.rollout_group_id,
          prompt_text: row.prompt_text,
          chosen_trace_id: chosen.trace_id,
          rejected_trace_id: rejected.trace_id,
          chosen_reward: chosen.totalReward,
          rejected_reward: rejected.totalReward,
          margin,
        });
      }
    }
    status.advance(1);
  }

  status.begin(
    "pipeline_rl_contracts",
    "Checking RL contracts",
    "Validating that DPO and GRPO smoke prerequisites are satisfied.",
    2,
  );
  const dpoReady = preferenceDataset.length >= 2;
  status.advance(1);
  const grpoReady = new Set(rewardLabels.map((entry) => entry.rollout_group_id)).size >= 2;
  status.advance(1);

  return {
    totalSamples,
    completedSamples: totalSamples,
    policyBundle,
    rolloutTrace,
    rewardLabels,
    preferenceDataset,
    rl: {
      dpoReady,
      grpoReady,
    },
    stats: {
      rows: rows.length,
      rolloutCandidates: rolloutTrace.length,
      rewardLabels: rewardLabels.length,
      preferencePairs: preferenceDataset.length,
    },
  };
}
