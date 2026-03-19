import { buildQwen35TrainingPromptAndTarget } from "./qwen35-chat-template.mjs";

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

function asArray(value) {
  return Array.isArray(value) ? value : [];
}

function normalizeIdSequence(value) {
  if (Array.isArray(value)) return value.flat(Infinity).map((entry) => Number(entry));
  if (ArrayBuffer.isView(value)) return Array.from(value, Number);
  if (value && typeof value.tolist === "function") {
    return normalizeIdSequence(value.tolist());
  }
  if (value?.data && ArrayBuffer.isView(value.data)) {
    return Array.from(value.data, Number);
  }
  return [];
}

function uniqTextList(values) {
  const out = [];
  const seen = new Set();
  for (const value of Array.isArray(values) ? values : [values]) {
    const text = asText(value);
    if (!text || seen.has(text)) continue;
    seen.add(text);
    out.push(text);
  }
  return out;
}

function normalizeImageSourceValue(value) {
  if (typeof value === "string") return asText(value);
  if (value instanceof URL) return value.toString();
  if (value && typeof value === "object") {
    if ("url" in value) return normalizeImageSourceValue(value.url);
    if ("image" in value) return normalizeImageSourceValue(value.image);
    if ("image_url" in value) return normalizeImageSourceValue(value.image_url);
  }
  return "";
}

function normalizeImagePart(part) {
  const image = normalizeImageSourceValue(part?.image ?? part?.image_url);
  if (!image) return null;
  return {
    type: "image",
    image,
  };
}

function normalizeMessageContentParts(content) {
  const source = Array.isArray(content) ? content : [content];
  const out = [];
  for (const part of source) {
    if (typeof part === "string") {
      const text = asText(part);
      if (text) out.push({ type: "text", text });
      continue;
    }
    if (!part || typeof part !== "object") continue;
    const type = asText(part.type).toLowerCase();
    if (type === "text") {
      const text = asText(part.text ?? part.content);
      if (text) out.push({ type: "text", text });
      continue;
    }
    if (type === "image" || type === "image_url") {
      const normalizedImage = normalizeImagePart(part);
      if (normalizedImage) out.push(normalizedImage);
      continue;
    }
    const fallbackText = asText(part.text ?? part.content);
    if (fallbackText) out.push({ type: "text", text: fallbackText });
  }
  return out;
}

function normalizePortableContent(content) {
  if (typeof content === "string") return asText(content);
  const parts = normalizeMessageContentParts(content);
  if (!parts.length) return "";
  return parts.some((part) => part.type === "image")
    ? parts
    : parts
        .map((part) => (part.type === "text" ? asText(part.text) : ""))
        .filter(Boolean)
        .join("\n");
}

function stringifyPortableValue(value) {
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value);
  } catch {
    return String(value == null ? "" : value);
  }
}

function normalizePortableToolCall(value, index = 0) {
  const source = value && typeof value === "object" ? value : {};
  const callId = asText(source.id || source.call_id) || `call_${index + 1}`;
  const rawFunction = source?.function && typeof source.function === "object" ? source.function : source;
  const name = asText(rawFunction.name || source.tool_name || source.toolName);
  const argumentsText =
    typeof rawFunction.arguments === "string"
      ? rawFunction.arguments
      : stringifyPortableValue(rawFunction.arguments ?? source.arguments ?? {});
  if (!name) return null;
  return {
    id: callId,
    type: "function",
    function: {
      name,
      arguments: argumentsText || "{}",
    },
  };
}

export function normalizePortableToolDefinition(value, index = 0) {
  const source = value && typeof value === "object" ? value : {};
  if (asText(source.type).toLowerCase() === "function" && source.function && typeof source.function === "object") {
    const fn = source.function;
    return {
      type: "function",
      function: {
        name: asText(fn.name) || `tool_${index + 1}`,
        description: asText(fn.description),
        parameters:
          fn.parameters && typeof fn.parameters === "object" && !Array.isArray(fn.parameters)
            ? cloneJson(fn.parameters, {})
            : {},
      },
    };
  }
  return {
    type: "function",
    function: {
      name: asText(source.name || source.tool_name || source.toolName) || `tool_${index + 1}`,
      description: asText(source.description),
      parameters:
        source.parameters && typeof source.parameters === "object" && !Array.isArray(source.parameters)
          ? cloneJson(source.parameters, {})
          : source.parameterSchema && typeof source.parameterSchema === "object" && !Array.isArray(source.parameterSchema)
            ? cloneJson(source.parameterSchema, {})
            : {},
    },
  };
}

function normalizePortableTrainingMessage(message, index = 0) {
  const source = message && typeof message === "object" ? message : {};
  const role = asText(source.role).toLowerCase();
  if (!["system", "user", "assistant", "tool"].includes(role)) return null;

  if (role === "tool") {
    const contentValue = source.content;
    const content =
      typeof contentValue === "string"
        ? contentValue
        : stringifyPortableValue(contentValue ?? "");
    return {
      role: "tool",
      content,
      tool_call_id: asText(source.tool_call_id || source.toolCallId) || null,
      name: asText(source.name),
    };
  }

  const normalized = {
    role,
    content: normalizePortableContent(source.content),
  };

  if (role === "assistant") {
    normalized.tool_calls = (Array.isArray(source.tool_calls) ? source.tool_calls : [])
      .map((entry, toolIndex) => normalizePortableToolCall(entry, toolIndex))
      .filter(Boolean);
  }

  if (!normalized.content && role !== "assistant") return null;
  if (role === "assistant" && !normalized.content && !normalized.tool_calls?.length) return null;
  return normalized;
}

export function normalizePortableTrainingMessages(messages = []) {
  return (Array.isArray(messages) ? messages : [])
    .map((message, index) => normalizePortableTrainingMessage(message, index))
    .filter(Boolean);
}

function matchesToolResponseToCall(toolMessage, toolCall) {
  const responseCallId = asText(toolMessage?.tool_call_id);
  const responseName = asText(toolMessage?.name);
  const callId = asText(toolCall?.id);
  const callName = asText(toolCall?.function?.name);
  if (responseCallId && callId && responseCallId === callId) return true;
  if (!responseCallId && responseName && callName && responseName === callName) return true;
  if (responseCallId && !callId && responseName && callName && responseName === callName) return true;
  return false;
}

export function inspectPortableTrainingTrace(messages = [], targetTurnIndex = null) {
  const normalizedMessages = normalizePortableTrainingMessages(messages);
  const result = {
    ok: false,
    normalizedMessages,
    normalizedTargetTurnIndex: Number.isInteger(targetTurnIndex) ? targetTurnIndex : null,
    reason: "",
    hasToolResponses: normalizedMessages.some((message) => message?.role === "tool"),
    missingToolResponse: false,
  };

  if (!normalizedMessages.length) {
    result.reason = "Trace is missing messages.";
    return result;
  }
  if (!Number.isInteger(targetTurnIndex) || targetTurnIndex < 0 || targetTurnIndex >= normalizedMessages.length) {
    result.reason = "Trace is missing a valid target turn index.";
    return result;
  }
  const targetMessage = normalizedMessages[targetTurnIndex];
  if (targetMessage?.role !== "assistant") {
    result.reason = "Target turn must point to an assistant message.";
    return result;
  }

  for (let index = 0; index < targetTurnIndex; index += 1) {
    const message = normalizedMessages[index];
    if (message?.role === "tool") {
      const hasMatchingPreviousCall = normalizedMessages
        .slice(0, index)
        .some((candidate) =>
          candidate?.role === "assistant" &&
          Array.isArray(candidate.tool_calls) &&
          candidate.tool_calls.some((toolCall) => matchesToolResponseToCall(message, toolCall))
        );
      if (!hasMatchingPreviousCall) {
        result.reason = `Tool response at turn ${index + 1} does not match any earlier assistant tool call.`;
        return result;
      }
    }

    if (message?.role !== "assistant") continue;
    const toolCalls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
    if (!toolCalls.length) continue;
    const matchedCallIds = new Set();
    let cursor = index + 1;
    while (cursor < targetTurnIndex) {
      const nextMessage = normalizedMessages[cursor];
      if (nextMessage?.role !== "tool") {
        if (matchedCallIds.size < toolCalls.length) {
          const missingCall = toolCalls.find((toolCall) => !matchedCallIds.has(asText(toolCall?.id)));
          result.reason = `Assistant tool call ${asText(missingCall?.function?.name) || "tool"} before target turn is missing its tool response.`;
          result.missingToolResponse = true;
          return result;
        }
        break;
      }
      for (const toolCall of toolCalls) {
        if (matchesToolResponseToCall(nextMessage, toolCall)) {
          matchedCallIds.add(asText(toolCall?.id));
        }
      }
      cursor += 1;
    }
    if (matchedCallIds.size < toolCalls.length) {
      const missingCall = toolCalls.find((toolCall) => !matchedCallIds.has(asText(toolCall?.id)));
      result.reason = `Assistant tool call ${asText(missingCall?.function?.name) || "tool"} before target turn is missing its tool response.`;
      result.missingToolResponse = true;
      return result;
    }
  }

  result.ok = true;
  return result;
}

function collectPortableImageSources(messages = []) {
  const images = [];
  for (const message of Array.isArray(messages) ? messages : []) {
    const content = message?.content;
    if (!Array.isArray(content)) continue;
    for (const part of content) {
      if (part?.type !== "image") continue;
      const image = asText(part.image);
      if (image) images.push(image);
    }
  }
  return uniqTextList(images);
}

function summarizePortableTrainingMessages(messages = []) {
  const lines = [];
  for (const message of Array.isArray(messages) ? messages : []) {
    if (message?.role === "tool") {
      const toolName = asText(message?.name) || asText(message?.tool_call_id) || "tool";
      lines.push(`[tool ${toolName}]`);
      continue;
    }
    const rendered = normalizePortableContent(message?.content);
    if (rendered) lines.push(rendered);
    for (const toolCall of Array.isArray(message?.tool_calls) ? message.tool_calls : []) {
      const toolName = asText(toolCall?.function?.name) || "tool_call";
      lines.push(`[tool_call ${toolName}]`);
    }
  }
  return lines.join("\n").trim();
}

export function renderQwen35TrainingInspection(messages = [], tools = [], targetTurnIndex = -1) {
  const normalizedMessages = normalizePortableTrainingMessages(messages);
  if (!normalizedMessages.length) {
    return {
      prompt: "",
      target: "",
      promptMessages: [],
      fullMessages: [],
    };
  }
  const resolvedTargetTurnIndex = resolvePortableTargetTurnIndex({
    target_turn_index: targetTurnIndex,
    targetTurnIndex,
  }, normalizedMessages, -1);
  const normalizedTools = (Array.isArray(tools) ? tools : [])
    .map((entry, index) => normalizePortableToolDefinition(entry, index));
  return buildQwen35TrainingPromptAndTarget({
    messages: normalizedMessages,
    tools: normalizedTools,
    targetTurnIndex: resolvedTargetTurnIndex,
  });
}

function resolvePortableTargetTurnIndex(row, messages, index) {
  const candidates = [
    row?.target_turn_index,
    row?.targetTurnIndex,
    row?.supervision?.target_turn_index,
    row?.supervision?.targetTurnIndex,
  ];
  for (const candidate of candidates) {
    if (Number.isInteger(candidate) && candidate >= 0 && candidate < messages.length) {
      return candidate;
    }
  }
  for (let cursor = messages.length - 1; cursor >= 0; cursor -= 1) {
    if (messages[cursor]?.role === "assistant") return cursor;
  }
  throw new Error(`Training row ${index} is missing a valid target_turn_index for multi-turn supervision.`);
}

export function collectTrainingImageSources(row, messages = null) {
  const directSources = uniqTextList([
    row?.image,
    row?.image_url,
    row?.image_data_url,
    ...(Array.isArray(row?.images) ? row.images : []),
  ]);
  const messageSources = messages ? collectPortableImageSources(messages) : [];
  return uniqTextList(directSources.concat(messageSources));
}

export function buildTrainingMessages(row, prompt = "") {
  const portableMessages = normalizePortableTrainingMessages(row?.messages || []);
  const hasPortableSupervision =
    portableMessages.length &&
    (
      Number.isInteger(row?.target_turn_index) ||
      Number.isInteger(row?.targetTurnIndex) ||
      row?.supervision?.target_turn_index != null ||
      row?.supervision?.targetTurnIndex != null ||
      portableMessages.some((message) => message?.role === "assistant" || message?.role === "tool" || asArray(message?.tool_calls).length)
    );
  if (hasPortableSupervision) {
    const targetTurnIndex = resolvePortableTargetTurnIndex(row, portableMessages, -1);
    return portableMessages.slice(0, targetTurnIndex);
  }
  const imageSources = collectTrainingImageSources(row);
  const content = [];
  const promptText = asText(prompt);
  if (promptText) {
    content.push({ type: "text", text: promptText });
  }
  for (const imageSource of imageSources) {
    content.push({ type: "image", image: imageSource });
  }
  return content.length ? [{ role: "user", content }] : [];
}

function summarizeTrainingMessages(messages = []) {
  const portable = normalizePortableTrainingMessages(messages);
  if (portable.length) {
    return summarizePortableTrainingMessages(portable);
  }
  const lines = [];
  for (const message of Array.isArray(messages) ? messages : []) {
    const content = message?.content;
    if (typeof content === "string") {
      const text = asText(content);
      if (text) lines.push(text);
      continue;
    }
    if (!Array.isArray(content)) continue;
    for (const part of content) {
      if (part?.type !== "text") continue;
      const text = asText(part?.text);
      if (text) lines.push(text);
    }
  }
  return lines.join("\n").trim();
}

function normalizePortableTrainingRow(row, index) {
  const fullMessages = normalizePortableTrainingMessages(row?.messages || []);
  if (!fullMessages.length) {
    throw new Error(`Training row ${index} is missing messages for multi-turn supervision.`);
  }
  const targetTurnIndex = resolvePortableTargetTurnIndex(row, fullMessages, index);
  const traceInspection = inspectPortableTrainingTrace(fullMessages, targetTurnIndex);
  if (!traceInspection.ok) {
    throw new Error(`Training row ${index} has an invalid multi-turn trace. ${traceInspection.reason}`);
  }
  const targetMessage = fullMessages[targetTurnIndex];
  if (targetMessage?.role !== "assistant") {
    throw new Error(`Training row ${index} target_turn_index must point to an assistant turn.`);
  }
  const tools = (Array.isArray(row?.tools) ? row.tools : Array.isArray(row?.available_tools) ? row.available_tools : [])
    .map((entry, toolIndex) => normalizePortableToolDefinition(entry, toolIndex));
  const rendered = buildQwen35TrainingPromptAndTarget({
    messages: fullMessages,
    tools,
    targetTurnIndex,
  });
  const promptMessages = rendered.promptMessages;
  const prompt = rendered.prompt;
  const target = rendered.target;
  const imageSources = collectTrainingImageSources(row, promptMessages);
  const promptSummary =
    asText(row?.prompt_text ?? row?.prompt ?? row?.input) ||
    summarizePortableTrainingMessages(promptMessages) ||
    (imageSources.length ? "[vision sample]" : "");
  if (!prompt) {
    throw new Error(`Training row ${index} produced an empty multi-turn prompt.`);
  }
  return {
    prompt,
    target,
    messages: promptMessages,
    fullMessages,
    tools,
    targetTurnIndex,
    promptSummary,
    imageSources,
    usesVision: imageSources.length > 0,
    renderMode: "qwen_native_multiturn",
  };
}

export function normalizeTrainingPairRow(row, index) {
  const portableMessages = normalizePortableTrainingMessages(row?.messages || []);
  const outputMode = asText(row?.output_mode ?? row?.outputMode).toLowerCase();
  if (outputMode === "multiturn_tool_agent" && !portableMessages.length) {
    throw new Error(`Training row ${index} declares output_mode=multiturn_tool_agent but is missing messages.`);
  }
  if (
    portableMessages.length &&
    (
      Number.isInteger(row?.target_turn_index) ||
      Number.isInteger(row?.targetTurnIndex) ||
      row?.supervision?.target_turn_index != null ||
      row?.supervision?.targetTurnIndex != null ||
      portableMessages.some((message) => message?.role === "tool" || asArray(message?.tool_calls).length)
    )
  ) {
    return normalizePortableTrainingRow(row, index);
  }
  throw new Error(
    `Training row ${index} must provide a native Qwen multi-turn transcript with messages, tools, and target_turn_index.`,
  );
}

export function datasetHasVisionPairs(pairs = []) {
  return (Array.isArray(pairs) ? pairs : []).some((pair) => pair?.usesVision === true);
}

export function buildEncodedTrainingExample({
  promptIds,
  promptAttentionMask = null,
  targetIds,
  padId,
  maxLength,
  prompt = "",
  target = "",
  usesVision = false,
  imageSources = [],
  extraInputs = null,
}) {
  if (maxLength <= 0) {
    throw new Error("maxLength must be positive.");
  }
  const safePromptIds = normalizeIdSequence(promptIds);
  const safePromptAttentionMask = normalizeIdSequence(promptAttentionMask);
  const safeTargetIds = normalizeIdSequence(targetIds);
  // Multimodal processors inject image placeholder tokens that must stay aligned
  // with the extracted visual features. Truncating them breaks the forward pass,
  // so keep the full prompt+target sequence for vision examples.
  const effectiveMaxLength = usesVision
    ? Math.max(maxLength, safePromptIds.length + safeTargetIds.length)
    : maxLength;
  const basePromptMask = safePromptAttentionMask.length === safePromptIds.length
    ? safePromptAttentionMask.slice()
    : new Array(safePromptIds.length).fill(1);

  let promptTruncated = false;
  let targetTruncated = false;
  let keptPromptIds = [];
  let keptPromptMask = [];
  let keptTargetIds = [];

  if (safePromptIds.length + safeTargetIds.length <= effectiveMaxLength) {
    keptPromptIds = safePromptIds.slice();
    keptPromptMask = basePromptMask.slice();
    keptTargetIds = safeTargetIds.slice();
  } else if (safeTargetIds.length <= effectiveMaxLength) {
    const promptBudget = effectiveMaxLength - safeTargetIds.length;
    keptPromptIds = promptBudget > 0 ? safePromptIds.slice(-promptBudget) : [];
    keptPromptMask = promptBudget > 0 ? basePromptMask.slice(-promptBudget) : [];
    keptTargetIds = safeTargetIds.slice();
    promptTruncated = keptPromptIds.length !== safePromptIds.length;
  } else {
    let promptBudget = Math.min(safePromptIds.length, Math.max(0, Math.trunc(effectiveMaxLength / 4)));
    if (safeTargetIds.length) {
      promptBudget = Math.min(promptBudget, effectiveMaxLength - 1);
    }
    let targetBudget = effectiveMaxLength - promptBudget;
    if (targetBudget <= 0 && safeTargetIds.length) {
      promptBudget = 0;
      targetBudget = effectiveMaxLength;
    }
    keptPromptIds = promptBudget > 0 ? safePromptIds.slice(-promptBudget) : [];
    keptPromptMask = promptBudget > 0 ? basePromptMask.slice(-promptBudget) : [];
    keptTargetIds = safeTargetIds.slice(0, targetBudget);
    promptTruncated = keptPromptIds.length !== safePromptIds.length;
    targetTruncated = keptTargetIds.length !== safeTargetIds.length;
  }

  const fullIds = keptPromptIds.concat(keptTargetIds);
  const attentionMask = keptPromptMask.concat(new Array(keptTargetIds.length).fill(1));
  const labels = fullIds.slice();
  for (let index = 0; index < keptPromptIds.length; index += 1) {
    labels[index] = -100;
  }

  const padLength = effectiveMaxLength - fullIds.length;
  for (let index = 0; index < padLength; index += 1) {
    fullIds.push(padId);
    attentionMask.push(0);
    labels.push(-100);
  }

  return {
    inputIds: Int32Array.from(fullIds),
    attentionMask: Int32Array.from(attentionMask),
    labels: Int32Array.from(labels),
    prompt,
    target,
    promptTokenCount: safePromptIds.length,
    targetTokenCount: safeTargetIds.length,
    promptTokensKept: keptPromptIds.length,
    targetTokensKept: keptTargetIds.length,
    promptTruncated,
    targetTruncated,
    supervisedTokens: keptTargetIds.length,
    usesVision,
    imageSources: Array.isArray(imageSources) ? imageSources.slice() : [],
    extraInputs: extraInputs && typeof extraInputs === "object" ? { ...extraInputs } : null,
  };
}
