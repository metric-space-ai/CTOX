import assert from 'node:assert/strict';
import test from 'node:test';

import {
  persistLocalBusinessReport,
  resolveBusinessReporterModule,
  saveBusinessReportLocally,
} from './business-reporter.js';

function makeElement(tagName) {
  const children = [];
  return {
    tagName,
    children,
    className: '',
    dataset: {},
    style: {},
    attributes: new Map(),
    append(...nodes) { children.push(...nodes); },
    appendChild(node) { children.push(node); return node; },
    addEventListener() {},
    classList: { add() {}, remove() {} },
    querySelector() { return null; },
    setAttribute(name, value) { this.attributes.set(name, value); },
    getBoundingClientRect() { return { left: 0, top: 0, width: 44, height: 44 }; },
  };
}

test('Business reporter keeps desktop idle free of RAF animation timers', async () => {
  const previousDocument = globalThis.document;
  const previousWindow = globalThis.window;
  const previousDesktopBridge = globalThis.ctoxBusinessOsDesktop;
  const previousSetTimeout = globalThis.setTimeout;
  const previousClearTimeout = globalThis.clearTimeout;
  const previousRequestAnimationFrame = globalThis.requestAnimationFrame;
  try {
    let timeoutCount = 0;
    let rafCount = 0;
    const documentStub = {
      body: makeElement('body'),
      head: makeElement('head'),
      documentElement: { lang: 'de' },
      getElementById() { return null; },
      querySelector() { return null; },
      createElement: makeElement,
    };
    globalThis.document = documentStub;
    globalThis.window = {
      innerWidth: 1440,
      innerHeight: 900,
      addEventListener() {
        throw new Error('desktop idle animation must not install activity listeners');
      },
    };
    globalThis.ctoxBusinessOsDesktop = { openSwitcher() {} };
    globalThis.setTimeout = () => { timeoutCount += 1; return 1; };
    globalThis.clearTimeout = () => {};
    globalThis.requestAnimationFrame = () => { rafCount += 1; return 1; };

    const { initBusinessReporter } = await import(`./business-reporter.js?test=${Date.now()}`);
    initBusinessReporter({
      session: { authenticated: true },
      getActiveModule: () => ({ id: 'ctox', title: 'CTOX' }),
    });

    assert.equal(documentStub.body.children.length, 1);
    assert.equal(timeoutCount, 0);
    assert.equal(rafCount, 0);
  } finally {
    globalThis.document = previousDocument;
    globalThis.window = previousWindow;
    globalThis.ctoxBusinessOsDesktop = previousDesktopBridge;
    globalThis.setTimeout = previousSetTimeout;
    globalThis.clearTimeout = previousClearTimeout;
    globalThis.requestAnimationFrame = previousRequestAnimationFrame;
  }
});

test('Business reporter persists a report locally without waiting for WebRTC readiness', async () => {
  const inserted = new Map();
  const collection = (name) => ({
    async insert(doc) {
      inserted.set(name, structuredClone(doc));
    },
    findOne() {
      return { async exec() { return null; } };
    },
  });
  let syncStarts = 0;
  const neverReady = new Promise(() => {});
  const sync = {
    startCollection() {
      syncStarts += 1;
      return neverReady;
    },
  };

  await Promise.race([
    persistLocalBusinessReport({
      db: {
        raw: {
          business_module_reports: collection('business_module_reports'),
          ctox_bug_reports: collection('ctox_bug_reports'),
        },
      },
      sync,
      session: { user: { id: 'owner-1' } },
      report: {
        result: {
          report_id: 'report-local-first',
          command_id: '',
          task_id: '',
          report_status: 'open',
          delivery_status: 'not_delegated',
        },
        module: { id: 'desktop' },
        kind: 'bug',
        severity: 'high',
        title: 'Reporter remains visible',
        summary: 'The command bridge is offline.',
        expected: 'The report is still listed.',
        clientContext: { source: 'business-os-reporter' },
        now: 1234,
      },
    }),
    new Promise((_, reject) => setTimeout(() => reject(new Error('local report write waited for sync')), 100)),
  ]);

  assert.equal(syncStarts, 2);
  assert.deepEqual(inserted.get('business_module_reports'), {
    id: 'report-local-first',
    report_id: 'report-local-first',
    module_id: 'desktop',
    kind: 'bug',
    severity: 'high',
    title: 'Reporter remains visible',
    summary: 'The command bridge is offline.',
    expected: 'The report is still listed.',
    status: 'open',
    reporter_id: 'owner-1',
    ctox_command_id: '',
    task_id: '',
    inbound_channel: 'desktop',
    client_context: {
      source: 'business-os-reporter',
      report_delivery: {
        status: 'not_delegated',
        command_id: '',
        task_id: '',
      },
    },
    created_at_ms: 1234,
    updated_at_ms: 1234,
  });
  assert.equal(inserted.get('ctox_bug_reports').payload.delivery_status, 'not_delegated');
  assert.equal(inserted.get('ctox_bug_reports').status, 'open');
});

test('Business reporter never claims a cold-start write when report collections are missing', async () => {
  await assert.rejects(
    persistLocalBusinessReport({
      db: { raw: {} },
      report: {
        result: { report_id: 'report-missing' },
        module: { id: 'desktop' },
        title: 'Missing collections',
        now: 1234,
      },
    }),
    /Bugs & Features ist noch nicht bereit/,
  );
});

test('Business reporter saves an app record without creating a CTOX command or task', async () => {
  const inserted = new Map();
  const collection = (name) => ({
    async insert(doc) { inserted.set(name, structuredClone(doc)); },
    findOne() { return { async exec() { return null; } }; },
  });
  const result = await saveBusinessReportLocally({
    db: {
      raw: {
        business_module_reports: collection('business_module_reports'),
        ctox_bug_reports: collection('ctox_bug_reports'),
      },
    },
    session: { user: { id: 'owner-1' } },
    module: { id: 'desktop' },
    kind: 'feature',
    title: 'Stable local identity',
    reportId: 'report-pending',
    now: 1234,
  });

  assert.equal(result.report_id, 'report-pending');
  assert.equal(result.command_id, '');
  assert.equal(result.task_id, '');
  assert.equal(result.task_status, 'not_delegated');
  assert.equal(result.status, 'open');
  assert.equal(result.delivery_status, 'not_delegated');
  assert.equal(inserted.get('business_module_reports').ctox_command_id, '');
  assert.equal(inserted.get('ctox_bug_reports').payload.delivery_status, 'not_delegated');
});

test('Business reporter attributes the report to the focused app window', () => {
  const reports = { id: 'reports', title: 'Bugs & Features' };
  const resolved = resolveBusinessReporterModule({
    activeModule: { id: 'desktop', title: 'Desktop' },
    modules: [reports],
    windowManager: {
      listWindows: () => [
        { id: 'one', ownerId: 'desktop-app:desktop', isFocused: false, state: 'normal' },
        { id: 'two', ownerId: 'desktop-app:reports', isFocused: true, state: 'normal' },
      ],
    },
  });

  assert.equal(resolved, reports);
});

test('Business reporter keeps focused shell apps attributable even before catalog hydration', () => {
  const resolved = resolveBusinessReporterModule({
    activeModule: { id: 'desktop', title: 'Desktop' },
    modules: [],
    windowManager: {
      listWindows: () => [{
        id: 'settings-window',
        ownerId: 'desktop-app:settings',
        title: 'Einstellungen',
        isFocused: true,
        state: 'normal',
      }],
    },
  });

  assert.deepEqual(resolved, { id: 'settings', title: 'Einstellungen' });
});

test('Business reporter falls back to the active module without a focused app window', () => {
  const activeModule = { id: 'desktop', title: 'Desktop' };
  assert.equal(resolveBusinessReporterModule({
    activeModule,
    modules: [],
    windowManager: { listWindows: () => [] },
  }), activeModule);
});
