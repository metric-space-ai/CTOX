import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

export const APP_AUDIT_SCENARIO_VERSION = 'ctox.business_os.app_audit_scenarios.v1';
const MAX_SCENARIOS = 32;
const MAX_STEPS = 64;
const IDENTIFIER = /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$/;
const ALLOWED_OPS = new Set([
  'click',
  'fill',
  'assert_visible',
  'assert_text',
  'reload',
]);

export function loadAppAuditScenarios(moduleDir) {
  const path = join(moduleDir, 'tests', 'audit-scenarios.json');
  if (!existsSync(path)) return { path, document: null, scenarios: [] };
  const parsed = JSON.parse(readFileSync(path, 'utf8'));
  return { path, document: parsed, scenarios: validateAppAuditScenarioDocument(parsed) };
}

export function validateAppAuditScenarioDocument(document) {
  if (!document || typeof document !== 'object' || Array.isArray(document)) {
    throw new Error('audit scenario document must be an object');
  }
  if (document.version !== APP_AUDIT_SCENARIO_VERSION) {
    throw new Error(`audit scenario version must be ${APP_AUDIT_SCENARIO_VERSION}`);
  }
  const scenarios = document.scenarios;
  if (!Array.isArray(scenarios) || scenarios.length < 1 || scenarios.length > MAX_SCENARIOS) {
    throw new Error(`audit scenario document must contain 1..${MAX_SCENARIOS} scenarios`);
  }
  const seen = new Set();
  return scenarios.map((scenario, index) => validateScenario(scenario, index, seen));
}

export function selectAppAuditScenarios(scenarios, profile, scenarioId = null) {
  if (!['quick', 'release', 'full'].includes(profile)) {
    throw new Error('app audit profile must be quick, release, or full');
  }
  return scenarioId
    ? scenarios.filter((scenario) => scenario.id === scenarioId)
    : scenarios.filter((scenario) => scenario.required_for.includes(profile));
}

function validateScenario(scenario, index, seen) {
  const prefix = `scenarios[${index}]`;
  if (!scenario || typeof scenario !== 'object' || Array.isArray(scenario)) {
    throw new Error(`${prefix} must be an object`);
  }
  const allowedKeys = new Set(['id', 'title', 'required_for', 'steps']);
  rejectUnknownKeys(scenario, allowedKeys, prefix);
  requireIdentifier(scenario.id, `${prefix}.id`);
  if (seen.has(scenario.id)) throw new Error(`${prefix}.id is duplicated`);
  seen.add(scenario.id);
  if (scenario.title != null) requireBoundedText(scenario.title, `${prefix}.title`, 240);
  const requiredFor = scenario.required_for ?? ['release', 'full'];
  if (!Array.isArray(requiredFor) || requiredFor.length < 1
    || requiredFor.some((profile) => !['quick', 'release', 'full'].includes(profile))) {
    throw new Error(`${prefix}.required_for must contain quick, release, or full`);
  }
  if (!Array.isArray(scenario.steps) || scenario.steps.length < 1 || scenario.steps.length > MAX_STEPS) {
    throw new Error(`${prefix}.steps must contain 1..${MAX_STEPS} steps`);
  }
  return {
    id: scenario.id,
    title: scenario.title || scenario.id,
    required_for: Array.from(new Set(requiredFor)),
    steps: scenario.steps.map((step, stepIndex) => validateStep(step, `${prefix}.steps[${stepIndex}]`)),
  };
}

function validateStep(step, prefix) {
  if (!step || typeof step !== 'object' || Array.isArray(step)) {
    throw new Error(`${prefix} must be an object`);
  }
  const allowedKeys = new Set(['op', 'target', 'value', 'contains', 'timeout_ms']);
  rejectUnknownKeys(step, allowedKeys, prefix);
  if (!ALLOWED_OPS.has(step.op)) throw new Error(`${prefix}.op is unsupported`);
  const timeout = step.timeout_ms ?? 15_000;
  if (!Number.isSafeInteger(timeout) || timeout < 250 || timeout > 60_000) {
    throw new Error(`${prefix}.timeout_ms must be an integer from 250 to 60000`);
  }
  const needsTarget = ['click', 'fill', 'assert_visible', 'assert_text'].includes(step.op);
  const target = needsTarget ? validateTarget(step.target, `${prefix}.target`) : null;
  if (!needsTarget && step.target != null) throw new Error(`${prefix}.target is not allowed for ${step.op}`);
  if (step.op === 'fill') validateScenarioValue(step.value, `${prefix}.value`);
  else if (step.value != null) throw new Error(`${prefix}.value is only allowed for fill`);
  if (step.op === 'assert_text') validateScenarioValue(step.contains, `${prefix}.contains`);
  else if (step.contains != null) throw new Error(`${prefix}.contains is only allowed for assert_text`);
  return {
    op: step.op,
    ...(target ? { target } : {}),
    ...(step.value != null ? { value: step.value } : {}),
    ...(step.contains != null ? { contains: step.contains } : {}),
    timeout_ms: timeout,
  };
}

function validateTarget(target, prefix) {
  if (!target || typeof target !== 'object' || Array.isArray(target)) {
    throw new Error(`${prefix} must be an object`);
  }
  const keys = Object.keys(target);
  if (keys.length !== 1 || !['action', 'test_id', 'field'].includes(keys[0])) {
    throw new Error(`${prefix} must contain exactly one of action, test_id, or field`);
  }
  requireIdentifier(target[keys[0]], `${prefix}.${keys[0]}`);
  return { [keys[0]]: target[keys[0]] };
}

function validateScenarioValue(value, prefix) {
  if (value === '$marker' || value === '$marker_email') return;
  requireBoundedText(value, prefix, 500);
}

function requireIdentifier(value, prefix) {
  if (typeof value !== 'string' || !IDENTIFIER.test(value)) {
    throw new Error(`${prefix} must be a bounded identifier`);
  }
}

function requireBoundedText(value, prefix, max) {
  if (typeof value !== 'string' || value.length < 1 || value.length > max
    || /[\u0000-\u0008\u000B\u000C\u000E-\u001F]/.test(value)) {
    throw new Error(`${prefix} must be bounded text`);
  }
}

function rejectUnknownKeys(value, allowed, prefix) {
  const unknown = Object.keys(value).filter((key) => !allowed.has(key));
  if (unknown.length) throw new Error(`${prefix} contains unsupported key ${unknown[0]}`);
}

export function scenarioTargetSelector(target) {
  const [kind, value] = Object.entries(target)[0];
  const attribute = kind === 'action' ? 'data-action' : kind === 'test_id' ? 'data-testid' : 'data-field';
  return `[${attribute}="${cssAttributeEscape(value)}"]`;
}

export function renderScenarioValue(value, marker) {
  if (value === '$marker') return marker;
  if (value === '$marker_email') {
    return `${marker.toLowerCase().replace(/[^a-z0-9]+/g, '-')}@example.test`;
  }
  return value;
}

function cssAttributeEscape(value) {
  return String(value).replace(/["\\]/g, (char) => `\\${char}`);
}
