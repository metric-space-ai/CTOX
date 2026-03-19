import { z } from "../vendor/agent-bundle.mjs";
import {
  TRAINING_SAMPLE_SPLITS,
  TRAINING_SAMPLE_STATUSES,
  TRAINING_SAMPLE_INPUT_MODES,
} from "./training-sample-ops.mjs";

const nullableString = (max) => z.string().max(max).nullable();

export const CRAFTING_AGENT_REPORT_CONTRACT = z.object({
  objective: nullableString(400),
  currentState: nullableString(800),
  nextAction: nullableString(800),
  matchingSignals: z.array(z.string().max(240)).max(8),
}).strict();

export const CRAFTING_AGENT_QUESTION_CONTRACT = z.object({
  id: nullableString(80),
  question: z.string().min(3).max(320),
  reason: nullableString(240),
  answer: nullableString(2_000),
}).strict();

export const CRAFTING_AGENT_PROVENANCE_CONTRACT = z.object({
  id: nullableString(80),
  title: z.string().min(1).max(200),
  detail: nullableString(800),
  kind: z.enum(["match", "constraint", "operation", "sample"]).nullable(),
  sampleId: nullableString(120),
  operationType: z.enum(["add", "update", "delete"]).nullable(),
}).strict();

export const CRAFTING_AGENT_MATURITY_CONTRACT = z.object({
  kind: z.literal("agent_reported"),
  percent: z.number().int().min(0).max(100),
  phase: z.enum(["crafting_progress", "capability_readiness"]),
  rationale: nullableString(400),
}).strict();

export const CRAFTING_AGENT_NAME_CONTRACT = z.object({
  name: z.string().min(2).max(80),
  reason: nullableString(240),
}).strict();

export const CRAFTING_AGENT_DESCRIPTION_CONTRACT = z.object({
  text: z.string().min(12).max(600),
  reason: nullableString(240),
}).strict();

export const CRAFTING_AGENT_USE_SURFACE_CONTRACT = z.object({
  inputMode: z.enum(TRAINING_SAMPLE_INPUT_MODES).nullable(),
  inputHint: nullableString(240),
  inputPlaceholder: nullableString(320),
  actionLabel: nullableString(160),
  inputExamples: z.array(z.string().max(240)).max(6),
}).strict();

export const CRAFTING_AGENT_TRAINING_FIELDS_CONTRACT = z.object({
  promptText: z.string().max(8_000).nullable(),
  messages: z.array(z.unknown()).max(64).nullable(),
  tools: z.array(z.unknown()).max(32).nullable(),
  targetTurnIndex: z.number().int().min(0).max(64).nullable(),
  split: z.enum(TRAINING_SAMPLE_SPLITS).nullable(),
  status: z.enum(TRAINING_SAMPLE_STATUSES).nullable(),
  source: nullableString(240),
}).strict();

export const CRAFTING_AGENT_TRAINING_OPERATION_CONTRACT = z.object({
  type: z.enum(["add", "update", "delete"]),
  sampleId: nullableString(120),
  reason: z.string().min(3).max(800),
  fields: CRAFTING_AGENT_TRAINING_FIELDS_CONTRACT.nullable(),
}).strict();

export const CRAFTING_AGENT_TRAINING_DRAFT_CONTRACT = z.object({
  summary: z.string().min(3).max(800),
  rationale: z.string().min(3).max(4_000),
  report: CRAFTING_AGENT_REPORT_CONTRACT,
  maturity: CRAFTING_AGENT_MATURITY_CONTRACT.nullable(),
  openQuestions: z.array(CRAFTING_AGENT_QUESTION_CONTRACT).max(6),
  provenance: z.array(CRAFTING_AGENT_PROVENANCE_CONTRACT).max(12),
  useSurface: CRAFTING_AGENT_USE_SURFACE_CONTRACT.nullable(),
  operations: z.array(CRAFTING_AGENT_TRAINING_OPERATION_CONTRACT).max(8),
}).strict();

export const CRAFTING_AGENT_FINAL_OUTPUT_CONTRACT = z.object({
  status: z.enum(["done", "blocked", "continue"]),
  summary: z.string().min(3).max(800),
  responseText: z.string().min(3).max(8_000),
  report: CRAFTING_AGENT_REPORT_CONTRACT,
  maturity: CRAFTING_AGENT_MATURITY_CONTRACT.nullable(),
  provenance: z.array(CRAFTING_AGENT_PROVENANCE_CONTRACT).max(18),
  suggestedName: CRAFTING_AGENT_NAME_CONTRACT.nullable().optional(),
  officialDescription: CRAFTING_AGENT_DESCRIPTION_CONTRACT.nullable().optional(),
  useSurface: CRAFTING_AGENT_USE_SURFACE_CONTRACT.nullable(),
}).strict();
