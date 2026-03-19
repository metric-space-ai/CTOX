const QWEN35_TOOL_PREAMBLE = [
  "# Tools",
  "",
  "You have access to the following functions:",
  "",
  "<tools>",
];

const QWEN35_TOOL_INSTRUCTION_BLOCK = [
  "If you choose to call a function ONLY reply in the following format with NO suffix:",
  "",
  "<tool_call>",
  "<function=example_function_name>",
  "<parameter=example_parameter_1>",
  "value_1",
  "</parameter>",
  "<parameter=example_parameter_2>",
  "This is the value for the second parameter",
  "that can span",
  "multiple lines",
  "</parameter>",
  "</function>",
  "</tool_call>",
  "",
  "<IMPORTANT>",
  "Reminder:",
  "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags",
  "- Required parameters MUST be specified",
  "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after",
  "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls",
  "</IMPORTANT>",
].join("\n");

function asText(value) {
  return String(value == null ? "" : value);
}

function trimText(value) {
  return asText(value).trim();
}

function isPlainObject(value) {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function jsonStringify(value) {
  if (Array.isArray(value)) {
    return `[${value.map((entry) => jsonStringify(entry)).join(", ")}]`;
  }
  if (isPlainObject(value)) {
    return `{${Object.entries(value).map(([key, entry]) => `${JSON.stringify(key)}: ${jsonStringify(entry)}`).join(", ")}}`;
  }
  try {
    return JSON.stringify(value);
  } catch {
    return asText(value);
  }
}

function renderContent(content, {
  doVisionCount = false,
  isSystemContent = false,
  addVisionId = false,
  imageCountRef = { value: 0 },
  videoCountRef = { value: 0 },
} = {}) {
  if (typeof content === "string") return content;
  if (content == null) return "";
  if (!Array.isArray(content)) {
    throw new Error("Qwen3.5 chat template received unsupported content type.");
  }

  let out = "";
  for (const item of content) {
    const type = trimText(item?.type);
    const hasImage = Boolean(item && typeof item === "object" && ("image" in item || "image_url" in item || type === "image"));
    const hasVideo = Boolean(item && typeof item === "object" && ("video" in item || type === "video"));
    if (hasImage) {
      if (isSystemContent) {
        throw new Error("System message cannot contain images.");
      }
      if (doVisionCount) imageCountRef.value += 1;
      if (addVisionId) out += `Picture ${imageCountRef.value}: `;
      out += "<|vision_start|><|image_pad|><|vision_end|>";
      continue;
    }
    if (hasVideo) {
      if (isSystemContent) {
        throw new Error("System message cannot contain videos.");
      }
      if (doVisionCount) videoCountRef.value += 1;
      if (addVisionId) out += `Video ${videoCountRef.value}: `;
      out += "<|vision_start|><|video_pad|><|vision_end|>";
      continue;
    }
    if (item && typeof item === "object" && "text" in item) {
      out += asText(item.text);
      continue;
    }
    throw new Error("Unexpected item type in Qwen3.5 chat template content.");
  }

  return out;
}

function normalizeToolCallArgumentsObject(argumentsValue) {
  if (isPlainObject(argumentsValue)) return argumentsValue;
  const text = trimText(argumentsValue);
  if (!text) return {};
  try {
    const parsed = JSON.parse(text);
    return isPlainObject(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

function findLastQueryIndex(messages = [], imageCountRef, videoCountRef, addVisionId) {
  let multiStepTool = true;
  let lastQueryIndex = messages.length - 1;
  for (let offset = messages.length - 1; offset >= 0; offset -= 1) {
    const message = messages[offset];
    if (!multiStepTool || message?.role !== "user") continue;
    const content = renderContent(message.content, {
      doVisionCount: false,
      addVisionId,
      imageCountRef,
      videoCountRef,
    }).trim();
    if (!(content.startsWith("<tool_response>") && content.endsWith("</tool_response>"))) {
      multiStepTool = false;
      lastQueryIndex = offset;
    }
  }
  if (multiStepTool) {
    throw new Error("No user query found in messages.");
  }
  return lastQueryIndex;
}

function renderToolDefinitions(tools = []) {
  if (!Array.isArray(tools) || !tools.length) return "";
  const lines = [...QWEN35_TOOL_PREAMBLE];
  for (const tool of tools) {
    lines.push(jsonStringify(tool));
  }
  lines.push("</tools>");
  return lines.join("\n");
}

function normalizeToolCallName(toolCall = null) {
  if (isPlainObject(toolCall?.function)) {
    return trimText(toolCall.function.name);
  }
  return trimText(toolCall?.name);
}

function normalizeToolCallArguments(toolCall = null) {
  if (isPlainObject(toolCall?.function)) {
    return normalizeToolCallArgumentsObject(toolCall.function.arguments);
  }
  return normalizeToolCallArgumentsObject(toolCall?.arguments);
}

function renderAssistantToolCalls(message) {
  const toolCalls = Array.isArray(message?.tool_calls) ? message.tool_calls : [];
  if (!toolCalls.length) return "";
  const content = trimText(renderContent(message.content));
  const blocks = [];
  toolCalls.forEach((toolCall, index) => {
    const name = normalizeToolCallName(toolCall);
    if (!name) return;
    let block = "";
    if (index === 0) {
      block += content ? "\n\n<tool_call>\n" : "<tool_call>\n";
    } else {
      block += "\n<tool_call>\n";
    }
    block += `<function=${name}>\n`;
    const argsObject = normalizeToolCallArguments(toolCall);
    for (const key of Object.keys(argsObject)) {
      const value = argsObject[key];
      block += `<parameter=${key}>\n`;
      block += isPlainObject(value) || Array.isArray(value) ? jsonStringify(value) : asText(value);
      block += "\n</parameter>\n";
    }
    block += "</function>\n</tool_call>";
    blocks.push(block);
  });
  return blocks.join("");
}

function stripInlineThinkBlock(content = "") {
  const text = asText(content);
  if (!text.includes("</think>")) {
    return {
      reasoningContent: "",
      content: text,
    };
  }
  const [beforeEnd, afterEnd = ""] = text.split("</think>", 2);
  const reasoningContent = beforeEnd.split("<think>").pop() || "";
  return {
    reasoningContent: reasoningContent.replace(/\n+$/g, "").replace(/^\n+/g, ""),
    content: afterEnd.replace(/^\n+/g, ""),
  };
}

export function renderQwen35ChatTemplate(messages = [], {
  tools = [],
  addGenerationPrompt = false,
  enableThinking = false,
  addVisionId = false,
} = {}) {
  if (!Array.isArray(messages) || !messages.length) {
    throw new Error("No messages provided.");
  }

  const imageCountRef = { value: 0 };
  const videoCountRef = { value: 0 };
  const output = [];
  const hasTools = Array.isArray(tools) && tools.length > 0;

  if (hasTools) {
    output.push("<|im_start|>system\n");
    output.push(renderToolDefinitions(tools));
    output.push("\n\n");
    output.push(QWEN35_TOOL_INSTRUCTION_BLOCK);
    if (messages[0]?.role === "system") {
      const content = renderContent(messages[0].content, {
        doVisionCount: false,
        isSystemContent: true,
        addVisionId,
        imageCountRef,
        videoCountRef,
      }).trim();
      if (content) {
        output.push("\n\n");
        output.push(content);
      }
    }
    output.push("<|im_end|>\n");
  } else if (messages[0]?.role === "system") {
    const content = renderContent(messages[0].content, {
      doVisionCount: false,
      isSystemContent: true,
      addVisionId,
      imageCountRef,
      videoCountRef,
    }).trim();
    output.push(`<|im_start|>system\n${content}<|im_end|>\n`);
  }

  const lastQueryIndex = findLastQueryIndex(messages, imageCountRef, videoCountRef, addVisionId);

  messages.forEach((message, index) => {
    const content = renderContent(message?.content, {
      doVisionCount: true,
      addVisionId,
      imageCountRef,
      videoCountRef,
    }).trim();
    const role = trimText(message?.role);
    if (role === "system") {
      if (index !== 0) {
        throw new Error("System message must be at the beginning.");
      }
      return;
    }
    if (role === "user") {
      output.push(`<|im_start|>user\n${content}<|im_end|>\n`);
      return;
    }
    if (role === "assistant") {
      let reasoningContent = typeof message?.reasoning_content === "string" ? message.reasoning_content : "";
      let assistantContent = content;
      if (!reasoningContent) {
        const split = stripInlineThinkBlock(content);
        reasoningContent = split.reasoningContent;
        assistantContent = split.content;
      }
      reasoningContent = trimText(reasoningContent);
      if (index > lastQueryIndex) {
        output.push(`<|im_start|>assistant\n<think>\n${reasoningContent}\n</think>\n\n${assistantContent}`);
      } else {
        output.push(`<|im_start|>assistant\n${assistantContent}`);
      }
      output.push(renderAssistantToolCalls({
        ...message,
        content: assistantContent,
      }));
      output.push("<|im_end|>\n");
      return;
    }
    if (role === "tool") {
      const prevRole = trimText(messages[index - 1]?.role);
      const nextRole = trimText(messages[index + 1]?.role);
      if (prevRole !== "tool") {
        output.push("<|im_start|>user");
      }
      output.push(`\n<tool_response>\n${content}\n</tool_response>`);
      if (index === messages.length - 1 || nextRole !== "tool") {
        output.push("<|im_end|>\n");
      }
      return;
    }
    throw new Error("Unexpected message role.");
  });

  if (addGenerationPrompt) {
    output.push("<|im_start|>assistant\n");
    output.push(enableThinking ? "<think>\n" : "<think>\n\n</think>\n\n");
  }

  return output.join("");
}

function stripQwen35RawWrapper(text = "") {
  return asText(text)
    .replace(/^<\|im_start\|>assistant\s*/i, "")
    .replace(/<\|im_end\|>\s*$/i, "")
    .replace(/<\|endoftext\|>\s*$/i, "")
    .trim();
}

function coerceParameterValue(text = "") {
  const value = asText(text).replace(/^\n+/, "").replace(/\n+$/, "");
  const trimmed = value.trim();
  if (!trimmed) return "";
  if (/^(?:true|false|null|-?\d+(?:\.\d+)?)$/i.test(trimmed)) {
    try {
      return JSON.parse(trimmed);
    } catch {}
  }
  if ((trimmed.startsWith("{") && trimmed.endsWith("}")) || (trimmed.startsWith("[") && trimmed.endsWith("]")) || (trimmed.startsWith("\"") && trimmed.endsWith("\""))) {
    try {
      return JSON.parse(trimmed);
    } catch {}
  }
  return value;
}

export function parseQwen35AssistantRawText(rawText, {
  allowedToolNames = [],
  defaultCallId = "call_1",
} = {}) {
  const cleaned = stripQwen35RawWrapper(rawText);
  if (!cleaned) return null;
  const toolCallMatch = cleaned.match(/<tool_call>\s*<function=([^>\n]+)>\s*([\s\S]*?)<\/function>\s*<\/tool_call>/i);
  if (!toolCallMatch) {
    return {
      role: "assistant",
      content: cleaned,
    };
  }

  const toolName = trimText(toolCallMatch[1]);
  if (Array.isArray(allowedToolNames) && allowedToolNames.length && !allowedToolNames.includes(toolName)) {
    return null;
  }

  const parameterBody = toolCallMatch[2] || "";
  const parameterPattern = /<parameter=([^>\n]+)>\s*([\s\S]*?)<\/parameter>/gi;
  const argumentsObject = {};
  let parameterMatch = parameterPattern.exec(parameterBody);
  while (parameterMatch) {
    argumentsObject[trimText(parameterMatch[1])] = coerceParameterValue(parameterMatch[2] || "");
    parameterMatch = parameterPattern.exec(parameterBody);
  }

  const prefix = cleaned.slice(0, toolCallMatch.index).trim();
  const suffix = cleaned.slice(toolCallMatch.index + toolCallMatch[0].length).trim();
  const content = [prefix, suffix].filter(Boolean).join("\n\n");

  return {
    role: "assistant",
    ...(content ? { content } : { content: "" }),
    tool_calls: [
      {
        id: defaultCallId,
        type: "function",
        function: {
          name: toolName,
          arguments: jsonStringify(argumentsObject),
        },
      },
    ],
  };
}

export function buildQwen35TrainingPromptAndTarget({
  messages = [],
  tools = [],
  targetTurnIndex = -1,
} = {}) {
  if (!Array.isArray(messages) || !messages.length) {
    throw new Error("Qwen3.5 training row is missing messages.");
  }
  if (!Number.isInteger(targetTurnIndex) || targetTurnIndex < 0 || targetTurnIndex >= messages.length) {
    throw new Error("Qwen3.5 training row is missing a valid target turn.");
  }
  const promptMessages = messages.slice(0, targetTurnIndex);
  const fullMessages = messages.slice(0, targetTurnIndex + 1);
  const prompt = renderQwen35ChatTemplate(promptMessages, {
    tools,
    addGenerationPrompt: true,
    enableThinking: false,
  });
  const full = renderQwen35ChatTemplate(fullMessages, {
    tools,
    addGenerationPrompt: false,
    enableThinking: false,
  });
  if (!full.startsWith(prompt)) {
    throw new Error("Qwen3.5 training target does not align with the rendered prompt prefix.");
  }
  return {
    prompt,
    target: full.slice(prompt.length),
    promptMessages,
    fullMessages,
  };
}

export function getQwen35NativeToolInstructionBlock() {
  return QWEN35_TOOL_INSTRUCTION_BLOCK;
}
