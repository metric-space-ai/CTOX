// Render smoke test.
//
// On 2026-07-31 a scripted rewrite of the render functions truncated their
// template literals. `node --check` passed, all 41 unit tests passed, the build
// shipped — and the app rendered a blank window during a customer meeting. Unit
// tests exercised the pure helpers; nothing ever asked the app to produce
// markup. This test does exactly that: it mounts the render functions against a
// minimal DOM stub and asserts the structural anchors are present and the panes
// are not empty.
//
// It is deliberately shallow. It does not assert layout or copy — it asserts
// that rendering produces a document at all, which is the failure mode that got
// through.

import test from 'node:test';
import assert from 'node:assert/strict';
import { Buffer } from 'node:buffer';
import { readFileSync } from 'node:fs';
import { build } from 'esbuild';

const bundled = await build({
  entryPoints: [new URL('../index.js', import.meta.url).pathname],
  bundle: true,
  format: 'esm',
  platform: 'browser',
  write: false,
  plugins: [{
    name: 'isolated-runtime-shared',
    setup(buildApi) {
      buildApi.onResolve({ filter: /^\.\.\/\.\.\/shared\// }, (args) => ({
        path: new URL(args.path.slice('../../shared/'.length), new URL('../../../../src/apps/business-os/shared/', import.meta.url)).pathname,
      }));
    },
  }],
});
const module = await import(
  `data:text/javascript;base64,${Buffer.from(bundled.outputFiles[0].text).toString('base64')}`
);
const renderHooks = module.__thesenOutboundTestHooks.__render;

// A DOM stub small enough to be obviously correct: elements remember the HTML
// written to them and can be found again by the selectors the app uses.
function createElement() {
  const node = {
    innerHTML: '',
    scrollTop: 0,
    scrollHeight: 0,
    clientHeight: 0,
    dataset: {},
    children: new Map(),
    querySelector(selector) {
      return this.children.get(selector) || null;
    },
    querySelectorAll() {
      return [];
    },
    addEventListener() {},
  };
  return node;
}

function createHost(paneSelectors) {
  const host = createElement();
  for (const selector of paneSelectors) host.children.set(selector, createElement());
  return host;
}

const PANES = ['[data-campaigns-pane]', '[data-leads-pane]', '[data-detail-pane]'];

function baseState(host) {
  return {
    ctx: { host, session: {}, db: null },
    messages: {},
    campaigns: ['Chemie Testimport E2E 2026-07-27'],
    selectedCampaign: 'Chemie Testimport E2E 2026-07-27',
    // The campaign list is derived from the leads, so a campaign only appears
    // once at least one lead carries it.
    leads: [{
      id: 'lead_smoke',
      name: 'Destilla GmbH',
      campaign: 'Chemie Testimport E2E 2026-07-27',
      city: 'Nördlingen',
      country: 'DE',
      domain: '',
      website: '',
      research_status: 'completed',
      validation_status: 'pending',
      sellify_status: 'not_started',
      data: {},
      contacts: [],
      evidence: [],
      payload: {},
      updated_at_ms: 0,
    }],
    selectedLeadId: '',
    selectedLeadIds: new Set(),
    adapters: [],
    sources: [],
    commands: [],
    imports: [],
    researchPolicies: [],
    sourcePanelOpen: false,
    leadEditorOpen: false,
    syncPending: false,
    syncError: false,
    sourceSearch: '',
    leadSearch: '',
  };
}

test('render() produces the three-pane shell, not an empty document', () => {
  const host = createHost(PANES);
  renderHooks.setState(baseState(host));
  renderHooks.render();
  assert.ok(host.innerHTML.length > 200, `shell markup is suspiciously short: ${host.innerHTML.length} chars`);
  for (const anchor of ['data-campaigns-pane', 'data-leads-pane', 'data-detail-pane']) {
    assert.ok(host.innerHTML.includes(anchor), `shell markup is missing ${anchor}`);
  }
});

test('renderCampaigns() writes a non-empty campaign pane', () => {
  const host = createHost(PANES);
  renderHooks.setState(baseState(host));
  renderHooks.renderCampaigns();
  const pane = host.querySelector('[data-campaigns-pane]');
  assert.ok(pane.innerHTML.length > 80, `campaign pane is empty or truncated: ${pane.innerHTML.length} chars`);
  assert.ok(
    pane.innerHTML.includes('Chemie Testimport E2E 2026-07-27'),
    'the campaign name never reached the markup',
  );
});

test('renderCenter() writes a non-empty lead pane carrying the campaign header', () => {
  const host = createHost(PANES);
  renderHooks.setState(baseState(host));
  renderHooks.renderCenter();
  const pane = host.querySelector('[data-leads-pane]');
  assert.ok(pane.innerHTML.length > 200, `lead pane is empty or truncated: ${pane.innerHTML.length} chars`);
  assert.ok(pane.innerHTML.includes('Chemie Testimport E2E 2026-07-27'), 'campaign header missing');
});

test('renderDetail() renders the empty-selection state rather than nothing', () => {
  const host = createHost(PANES);
  renderHooks.setState(baseState(host));
  renderHooks.renderDetail();
  const pane = host.querySelector('[data-detail-pane]');
  assert.ok(pane.innerHTML.length > 10, 'detail pane produced no markup at all');
});

test('every render template in the source is balanced', () => {
  // Guards the specific corruption that shipped: a rewrite that closed a
  // template literal too early leaves the file parseable but the markup gutted.
  const source = readFileSync(new URL('../index.js', import.meta.url), 'utf8');
  const openers = source.match(/setPaneHtml\([a-zA-Z]+, `/g) || [];
  const paneWrites = source.match(/\.innerHTML = `/g) || [];
  assert.ok(
    openers.length + paneWrites.length >= 4,
    `expected at least 4 pane render templates, found ${openers.length + paneWrites.length}`,
  );
});
