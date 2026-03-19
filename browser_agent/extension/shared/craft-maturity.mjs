const MATURITY_PHASES = new Set(["crafting_progress", "capability_readiness"]);

function asText(value) {
  return String(value == null ? "" : value).trim();
}

function clampPercent(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 0;
  return Math.max(0, Math.min(100, Math.round(numeric)));
}

function normalizePhase(value) {
  const phase = asText(value).toLowerCase();
  return MATURITY_PHASES.has(phase) ? phase : "crafting_progress";
}

export function createEmptyCraftMaturity(overrides = {}) {
  return {
    kind: "unset",
    percent: 0,
    phase: "crafting_progress",
    rationale:
      asText(overrides?.rationale) || "No explicit agent maturity update has been recorded yet.",
    updatedAt: asText(overrides?.updatedAt),
    isExplicit: false,
  };
}

export function normalizeCraftMaturity(rawValue = null, fallbackValue = null) {
  const fallback =
    fallbackValue && typeof fallbackValue === "object"
      ? fallbackValue
      : createEmptyCraftMaturity();
  const source = rawValue && typeof rawValue === "object" ? rawValue : {};
  const hasExplicitAgentValue = asText(source.kind) === "agent_reported";
  const base = hasExplicitAgentValue ? source : fallback;
  const isExplicit = asText(base?.kind) === "agent_reported";

  if (!isExplicit) {
    return createEmptyCraftMaturity(base);
  }

  return {
    kind: "agent_reported",
    percent: clampPercent(base?.percent),
    phase: normalizePhase(base?.phase),
    rationale: asText(base?.rationale),
    updatedAt: asText(base?.updatedAt),
    isExplicit: true,
  };
}

export function createAgentReportedCraftMaturity({
  percent = 0,
  phase = "crafting_progress",
  rationale = "",
  updatedAt = "",
} = {}) {
  return normalizeCraftMaturity({
    kind: "agent_reported",
    percent,
    phase,
    rationale,
    updatedAt,
  });
}

export function gateCraftMaturityForCapability(
  rawValue = null,
  {
    hasTrainedCapability = false,
    reason = "Locked at 0% until a trained capability artifact exists.",
  } = {},
) {
  const normalized = normalizeCraftMaturity(rawValue, createEmptyCraftMaturity());
  if (hasTrainedCapability || !normalized.isExplicit) {
    return normalized;
  }
  const rationale = [asText(normalized.rationale), asText(reason)].filter(Boolean).join(" ").trim();
  return createAgentReportedCraftMaturity({
    percent: 0,
    phase: "crafting_progress",
    rationale,
    updatedAt: normalized.updatedAt,
  });
}

export function formatCraftMaturityPercent(value = 0) {
  return `${clampPercent(value)}%`;
}

export function formatCraftMaturityPhase(phase = "") {
  return normalizePhase(phase) === "capability_readiness"
    ? "capability readiness"
    : "crafting progress";
}
