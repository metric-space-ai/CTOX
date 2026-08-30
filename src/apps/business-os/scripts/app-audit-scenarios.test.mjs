#!/usr/bin/env node
import assert from 'node:assert/strict';
import test from 'node:test';
import {
  APP_AUDIT_SCENARIO_VERSION,
  renderScenarioValue,
  scenarioTargetSelector,
  selectAppAuditScenarios,
  validateAppAuditScenarioDocument,
} from './app-audit-scenarios.mjs';

function validDocument() {
  return {
    version: APP_AUDIT_SCENARIO_VERSION,
    scenarios: [{
      id: 'primary-workflow',
      steps: [
        { op: 'click', target: { action: 'create-record' } },
        { op: 'fill', target: { field: 'name' }, value: '$marker' },
        { op: 'click', target: { action: 'save-record' } },
        { op: 'assert_text', target: { test_id: 'record-list' }, contains: '$marker' },
        { op: 'reload' },
        { op: 'assert_text', target: { test_id: 'record-list' }, contains: '$marker' },
      ],
    }],
  };
}

test('accepts bounded declarative app audit scenarios', () => {
  const scenarios = validateAppAuditScenarioDocument(validDocument());
  assert.equal(scenarios.length, 1);
  assert.deepEqual(scenarios[0].required_for, ['release', 'full']);
  assert.equal(scenarios[0].steps[0].timeout_ms, 15_000);
});

test('rejects executable and navigation-like scenario fields', () => {
  for (const extra of [
    { script: 'window.fetch("https://evil.test")' },
    { url: 'https://evil.test' },
    { command: ['sh', '-c', 'id'] },
  ]) {
    const document = validDocument();
    Object.assign(document.scenarios[0].steps[0], extra);
    assert.throws(() => validateAppAuditScenarioDocument(document), /unsupported key/);
  }
});

test('rejects raw selectors and unbounded scenario shapes', () => {
  const document = validDocument();
  document.scenarios[0].steps[0].target = { selector: 'body *' };
  assert.throws(() => validateAppAuditScenarioDocument(document), /exactly one/);
});

test('renders only scoped semantic selectors and synthetic marker values', () => {
  assert.equal(scenarioTargetSelector({ action: 'save-record' }), '[data-action="save-record"]');
  assert.equal(renderScenarioValue('$marker', 'CTOX_123'), 'CTOX_123');
  assert.equal(renderScenarioValue('$marker_email', 'CTOX_123'), 'ctox-123@example.test');
});

test('selects every scenario required by the requested audit profile', () => {
  const document = validDocument();
  document.scenarios.push({
    id: 'full-only',
    required_for: ['full'],
    steps: [{ op: 'assert_visible', target: { test_id: 'full-result' } }],
  });
  const scenarios = validateAppAuditScenarioDocument(document);
  assert.deepEqual(selectAppAuditScenarios(scenarios, 'release').map(({ id }) => id), ['primary-workflow']);
  assert.deepEqual(selectAppAuditScenarios(scenarios, 'full').map(({ id }) => id), ['primary-workflow', 'full-only']);
  assert.deepEqual(selectAppAuditScenarios(scenarios, 'release', 'full-only').map(({ id }) => id), ['full-only']);
});
