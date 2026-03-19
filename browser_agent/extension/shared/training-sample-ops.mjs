import { z } from "../vendor/agent-bundle.mjs";

export const TRAINING_SAMPLE_SPLITS = ["train", "validation", "test"];
export const TRAINING_SAMPLE_STATUSES = ["draft", "review", "ready", "blocked"];
export const TRAINING_SAMPLE_INPUT_MODES = ["free_text", "mixed", "selection", "current_tab", "context_only"];

const TRAINING_SAMPLE_REPORT_SCHEMA = z.object({
  objective: z.string().optional(),
  currentState: z.string().optional(),
  nextAction: z.string().optional(),
  matchingSignals: z.array(z.string()).max(8).optional(),
});

const TRAINING_SAMPLE_OPEN_QUESTION_SCHEMA = z.object({
  question: z.string(),
  reason: z.string().optional(),
});

const TRAINING_SAMPLE_PROVENANCE_SCHEMA = z.object({
  title: z.string(),
  detail: z.string(),
  kind: z.enum(["match", "constraint", "operation", "sample"]).optional(),
  sampleId: z.string().optional(),
  operationType: z.enum(["add", "update", "delete"]).optional(),
});

const TRAINING_SAMPLE_USE_SURFACE_SCHEMA = z.object({
  inputMode: z.enum(TRAINING_SAMPLE_INPUT_MODES).optional(),
  inputHint: z.string().optional(),
  inputPlaceholder: z.string().optional(),
  actionLabel: z.string().optional(),
  inputExamples: z.array(z.string()).max(6).optional(),
});

export const TRAINING_SAMPLE_OPS_SCHEMA = z.object({
  summary: z.string(),
  rationale: z.string(),
  report: TRAINING_SAMPLE_REPORT_SCHEMA.optional(),
  openQuestions: z.array(TRAINING_SAMPLE_OPEN_QUESTION_SCHEMA).max(6).optional(),
  provenance: z.array(TRAINING_SAMPLE_PROVENANCE_SCHEMA).max(12).optional(),
  useSurface: TRAINING_SAMPLE_USE_SURFACE_SCHEMA.optional(),
  operations: z.array(
    z.object({
      type: z.enum(["add", "update", "delete"]),
      sampleId: z.string().optional(),
      reason: z.string(),
      fields: z.object({
        promptText: z.string().optional(),
        messages: z.array(z.unknown()).max(64).optional(),
        tools: z.array(z.unknown()).max(32).optional(),
        targetTurnIndex: z.number().int().min(0).max(64).optional(),
        split: z.enum(TRAINING_SAMPLE_SPLITS).optional(),
        status: z.enum(TRAINING_SAMPLE_STATUSES).optional(),
        source: z.string().optional(),
      }).optional(),
    }),
  ).max(8),
});

function asText(value) {
  return String(value == null ? "" : value).trim();
}

function collectJsonTextCandidates(text) {
  const source = asText(text);
  if (!source) return [];

  const candidates = [source];
  const fencedPattern = /```(?:json)?\s*([\s\S]*?)```/gi;
  for (const match of source.matchAll(fencedPattern)) {
    const candidate = asText(match[1]);
    if (candidate) candidates.push(candidate);
  }

  const firstBrace = source.indexOf("{");
  const lastBrace = source.lastIndexOf("}");
  if (firstBrace >= 0 && lastBrace > firstBrace) {
    candidates.push(source.slice(firstBrace, lastBrace + 1).trim());
  }

  return [...new Set(candidates.filter(Boolean))];
}

function unwrapTrainingSampleOpsObject(value, seen = new Set()) {
  if (!value || typeof value !== "object") return null;
  if (seen.has(value)) return null;
  seen.add(value);

  const direct = TRAINING_SAMPLE_OPS_SCHEMA.safeParse(value);
  if (direct.success) return direct.data;

  const wrapperKeys = [
    "plan",
    "result",
    "object",
    "data",
    "payload",
    "response",
    "trainingSampleOps",
  ];

  for (const key of wrapperKeys) {
    const nested = unwrapTrainingSampleOpsObject(value?.[key], seen);
    if (nested) return nested;
  }

  for (const nestedValue of Object.values(value)) {
    if (!nestedValue || typeof nestedValue !== "object" || Array.isArray(nestedValue)) continue;
    const nested = unwrapTrainingSampleOpsObject(nestedValue, seen);
    if (nested) return nested;
  }

  return null;
}

export function parseTrainingSampleOpsText(text) {
  const source = asText(text);
  const candidates = collectJsonTextCandidates(source);

  for (const candidate of candidates) {
    try {
      const parsed = JSON.parse(candidate);
      const normalized = unwrapTrainingSampleOpsObject(parsed);
      if (normalized) return normalized;
    } catch (_error) {
      // Try the next candidate.
    }
  }

  const error = new Error("Local Qwen agent output was not valid JSON for training sample ops.");
  error.detail = {
    rawText: source,
    candidates,
  };
  throw error;
}
