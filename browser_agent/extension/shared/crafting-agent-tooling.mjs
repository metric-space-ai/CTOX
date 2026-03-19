const DEFAULT_CRAFTING_AGENT_TOOLS = Object.freeze([
  Object.freeze({
    id: "web_search",
    label: "Web Search",
    purpose: "Search the web for candidate sources, references, and likely URLs before using browser tools.",
  }),
  Object.freeze({
    id: "browser_inspect",
    label: "Browser Inspect",
    purpose: "Use the vision model to inspect the visible browser UI and identify grounded next actions.",
  }),
  Object.freeze({
    id: "browser_action",
    label: "Browser Action",
    purpose: "Execute visible browser interactions like click, type, scroll, keypress, wait, or drag.",
  }),
  Object.freeze({
    id: "browser_tabs",
    label: "Browser Tabs",
    purpose: "List, open, activate, and close browser tabs before running visible or scripted steps.",
  }),
  Object.freeze({
    id: "playwright_ctx",
    label: "Playwright CTX",
    purpose: "Run deterministic DOM or automation code inside the current browser tab context.",
  }),
]);

export const DEFAULT_CRAFTING_AGENT_TOOLING = Object.freeze({
  fixedInSidepanel: true,
  configurableIn: "options",
  tools: DEFAULT_CRAFTING_AGENT_TOOLS,
});

function asText(value) {
  return String(value == null ? "" : value).trim();
}

function normalizeToolId(value) {
  return asText(value)
    .toLowerCase()
    .replace(/[^a-z0-9_]+/g, "_")
    .replace(/^_+|_+$/g, "");
}

function normalizeToolRecord(rawTool) {
  const source = rawTool && typeof rawTool === "object" ? rawTool : {};
  const id = normalizeToolId(source.id || source.toolId || source.name);
  if (!id) return null;
  return {
    id,
    label: asText(source.label || source.name || id),
    purpose: asText(source.purpose || source.description),
  };
}

function mergeToolRecords(rawTools = []) {
  const byId = new Map(
    DEFAULT_CRAFTING_AGENT_TOOLS.map((tool) => [
      tool.id,
      {
        id: tool.id,
        label: tool.label,
        purpose: tool.purpose,
      },
    ]),
  );
  const order = DEFAULT_CRAFTING_AGENT_TOOLS.map((tool) => tool.id);

  for (const rawTool of Array.isArray(rawTools) ? rawTools : []) {
    const normalized = normalizeToolRecord(rawTool);
    if (!normalized) continue;
    const existing = byId.get(normalized.id);
    byId.set(normalized.id, {
      id: normalized.id,
      label: normalized.label || existing?.label || normalized.id,
      purpose: normalized.purpose || existing?.purpose || "",
    });
    if (!order.includes(normalized.id)) {
      order.push(normalized.id);
    }
  }

  return order
    .map((id) => byId.get(id))
    .filter((tool) => tool && tool.id);
}

export function normalizeCraftingAgentToolingPayload(rawValue = null) {
  const source = rawValue && typeof rawValue === "object" ? rawValue : {};
  return {
    fixedInSidepanel: source.fixedInSidepanel !== false,
    configurableIn: asText(source.configurableIn) || DEFAULT_CRAFTING_AGENT_TOOLING.configurableIn,
    tools: mergeToolRecords(source.tools),
  };
}

export function getCraftingAgentToolLabels(rawValue = null) {
  return normalizeCraftingAgentToolingPayload(rawValue).tools
    .map((tool) => asText(tool.label))
    .filter(Boolean);
}

export function formatCraftingAgentToolLabels(rawValue = null, separator = " + ") {
  return getCraftingAgentToolLabels(rawValue).join(separator);
}
