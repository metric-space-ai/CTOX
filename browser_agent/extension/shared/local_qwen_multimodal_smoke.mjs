import {
  buildCanonicalCallToolAction,
  normalizeCanonicalToolAction,
} from "./canonical-tool-action.mjs";

const ONE_PIXEL_PNG_DATA_URL =
  "data:image/png;base64," +
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+yF9kAAAAASUVORK5CYII=";

export const LOCAL_QWEN_MULTIMODAL_SMOKE_CONFIG = Object.freeze({
  profile: "multimodal_smoke",
  maxTrainPairs: 1,
  maxTestPairs: 1,
  maxSeqLen: 160,
  rank: 16,
  alpha: 32,
  epochs: 1,
  batchTokens: 32,
  modelBatchSize: 1,
  learningRate: 0.0001,
  optimizer: "full_transformer_lora_spsa",
  spsaEpsilon: 0.001,
  spsaSamples: 1,
  spsaGradientClip: 1,
  reasoningMode: "no_think",
  evaluationMode: "holdout_only",
  seed: 7,
  trainMatrixA: true,
  trainMatrixB: true,
});

function asText(value) {
  return String(value == null ? "" : value).trim();
}

function buildPortableToolDefinitions(toolName, description, parameters) {
  return [{
    type: "function",
    function: {
      name: asText(toolName),
      description: asText(description),
      parameters: parameters && typeof parameters === "object" ? { ...parameters } : { type: "object" },
    },
  }];
}

function buildPortableToolTrainingRow({
  promptText,
  imageDataUrl = "",
  toolAction,
  description,
  parameters,
}) {
  const toolCall = normalizeCanonicalToolAction(toolAction);
  const toolName = asText(toolCall?.tool_name);
  const callId = `call_${toolName.replace(/[^a-z0-9]+/gi, "_").toLowerCase() || "1"}`;
  const content = [{ type: "text", text: asText(promptText) }];
  if (asText(imageDataUrl)) {
    content.push({ type: "image", image: asText(imageDataUrl) });
  }
  return {
    prompt_text: asText(promptText),
    messages: [
      {
        role: "system",
        content: "You are a browser-safe visual agent. Decide the next assistant action using reviewed tools only.",
      },
      {
        role: "user",
        content,
      },
      {
        role: "assistant",
        content: "",
        tool_calls: [{
          id: callId,
          type: "function",
          function: {
            name: toolName,
            arguments: JSON.stringify(toolCall?.arguments || {}),
          },
        }],
      },
    ],
    tools: buildPortableToolDefinitions(toolName, description, parameters),
    target_turn_index: 2,
    output_mode: "multiturn_tool_agent",
  };
}

export function createLocalQwenMultimodalSmokeDatasetPayload({ imageDataUrl = ONE_PIXEL_PNG_DATA_URL } = {}) {
  const resolvedImageDataUrl = String(imageDataUrl || "").trim() || ONE_PIXEL_PNG_DATA_URL;
  const clickPrimaryParameters = {
    type: "object",
    properties: {
      selector_hint: { type: "string" },
    },
    required: ["selector_hint"],
    additionalProperties: false,
  };
  const scrollParameters = {
    type: "object",
    properties: {
      amount: { type: "string" },
    },
    required: ["amount"],
    additionalProperties: false,
  };
  return {
    meta: {
      scenario: "multimodal_smoke",
      taskType: "visual_tool_routing",
    },
    train: [
      buildPortableToolTrainingRow({
        promptText: "Pruefe den Screenshot und gib den Tool-Call fuer den primaeren CTA aus.",
        imageDataUrl: resolvedImageDataUrl,
        toolAction: buildCanonicalCallToolAction({
          toolName: "click_primary_cta",
          argumentsValue: {
            selector_hint: "primary_cta",
          },
          bundleRef: "browser-capabilities:multimodal-smoke",
          capabilityRef: "click_primary_cta",
          skillRef: "reviewed_browser_tool_bundle",
        }),
        description: "Click the reviewed primary CTA from the visible browser state.",
        parameters: clickPrimaryParameters,
      }),
      buildPortableToolTrainingRow({
        promptText: "Wenn der naechste Screen-Abschnitt geladen werden soll, gib den Scroll-Toolcall aus.",
        toolAction: buildCanonicalCallToolAction({
          toolName: "scroll_down",
          argumentsValue: {
            amount: "page",
          },
          bundleRef: "browser-capabilities:multimodal-smoke",
          capabilityRef: "scroll_down",
          skillRef: "reviewed_browser_tool_bundle",
        }),
        description: "Scroll the page by the requested amount.",
        parameters: scrollParameters,
      }),
    ],
    test: [
      buildPortableToolTrainingRow({
        promptText: "Ist ein primaerer CTA sichtbar? Antworte nur mit dem naechsten Tool-Call.",
        imageDataUrl: resolvedImageDataUrl,
        toolAction: buildCanonicalCallToolAction({
          toolName: "click_primary_cta",
          argumentsValue: {
            selector_hint: "primary_cta",
          },
          bundleRef: "browser-capabilities:multimodal-smoke",
          capabilityRef: "click_primary_cta",
          skillRef: "reviewed_browser_tool_bundle",
        }),
        description: "Click the reviewed primary CTA from the visible browser state.",
        parameters: clickPrimaryParameters,
      }),
    ],
  };
}
