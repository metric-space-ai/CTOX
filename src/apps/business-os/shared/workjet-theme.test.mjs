import assert from 'node:assert/strict';
import test from 'node:test';

import {
  applyWorkjetCategory,
  isPublicWorkjetModule,
  normalizeWorkjetCategory,
  workjetCategoryForModule,
  workjetCategoryForTarget,
  workjetCategoryStyle,
  WORKJET_CATEGORY_IDS,
} from './workjet-theme.js';

test('normalizes the complete Workjet category vocabulary and safe aliases', () => {
  assert.deepEqual(WORKJET_CATEGORY_IDS, [
    'workspace',
    'collaboration',
    'productivity',
    'development',
    'engineering',
    'knowledge',
    'research',
    'sales',
    'recruiting',
    'finance',
    'operations',
    'governance',
    'security',
    'analytics',
    'system',
    'imported',
  ]);
  assert.equal(normalizeWorkjetCategory('Recherche'), 'research');
  assert.equal(normalizeWorkjetCategory('Management'), 'operations');
  assert.equal(normalizeWorkjetCategory('Engineering'), 'engineering');
  assert.equal(normalizeWorkjetCategory('customer-private', 'workspace'), 'workspace');
  assert.equal(normalizeWorkjetCategory('customer-private'), 'imported');
});

test('only public/core module manifests can provide a category accent', () => {
  assert.equal(workjetCategoryForModule({ core: true, category: 'Security' }), 'security');
  assert.equal(workjetCategoryForModule({
    manifest: { source: 'core', category: 'Engineering' },
  }), 'engineering');
  assert.equal(workjetCategoryForModule({
    store: { distribution: 'store', category: 'Research' },
  }), 'research');
  assert.equal(workjetCategoryForModule({
    source: 'runtime',
    store: { distribution: 'runtime' },
    category: 'Sales',
  }), 'imported');
  assert.equal(workjetCategoryForModule({
    core: true,
    category: 'Security',
    customer_id: 'customer-42',
  }), 'imported');
  assert.equal(isPublicWorkjetModule({ source: 'runtime', visibility: 'private' }), false);
});

test('targets and rendered elements use the same canonical category refs', () => {
  assert.equal(workjetCategoryForTarget({
    kind: 'module',
    core: true,
    category: 'Workspace',
  }), 'workspace');
  assert.equal(workjetCategoryForTarget({ kind: 'app', category: 'Development' }), 'development');

  const style = workjetCategoryStyle('Security');
  assert.equal(style.id, 'security');
  assert.equal(style.accent, 'var(--workjet-category-security-accent)');
  assert.equal(style.soft, 'var(--workjet-category-security-accent-soft)');

  const properties = {};
  const element = {
    dataset: {},
    style: { setProperty(name, value) { properties[name] = value; } },
  };
  assert.equal(applyWorkjetCategory(element, 'Security'), 'security');
  assert.equal(element.dataset.workjetCategory, 'security');
  assert.equal(properties['--shell-category-accent'], 'var(--workjet-category-security-accent)');
  assert.equal(properties['--shell-category-border'], 'var(--workjet-category-security-accent-border)');
});
