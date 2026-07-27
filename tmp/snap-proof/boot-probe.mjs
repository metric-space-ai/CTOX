#!/usr/bin/env node
// Minimalprobe: was liefert die Shell-Seite, wo haengt der Boot?
import { createRequire } from 'node:module';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptDir, '../..');
const { chromium } = require(path.join(repoRoot, 'runtime/browser/interactive-reference/node_modules/patchright'));
const executablePath = path.join(
  repoRoot,
  'runtime/browser/interactive-reference/ms-playwright/chromium-1228/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing',
);

const browser = await chromium.launch({ headless: false, executablePath });
const page = await (await browser.newContext({ viewport: { width: 1280, height: 800 } })).newPage();
const failed = [];
page.on('requestfailed', (r) => failed.push(`${r.method()} ${r.url()} ${r.failure()?.errorText || ''}`));
page.on('response', (r) => { if (r.status() >= 400) failed.push(`HTTP ${r.status()} ${r.url()}`); });
const errors = [];
page.on('pageerror', (e) => errors.push(String(e).slice(0, 300)));
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text().slice(0, 300)); });

try {
  const probeUrl = process.env.BOOT_PROBE_URL || 'http://127.0.0.1:8765/?rxdbSmoke=1';
  const resp = await page.goto(probeUrl, { waitUntil: 'domcontentloaded', timeout: 30000 });
  console.log('goto status:', resp?.status());
} catch (e) {
  console.log('goto fehlgeschlagen:', e.message.split('\n')[0]);
}
await page.waitForTimeout(25000);
const state = await page.evaluate(() => ({
  title: document.title,
  authState: document.body?.dataset?.authState || null,
  moduleLoading: document.body?.dataset?.moduleLoading || null,
  hasSmoke: !!globalThis.ctoxBusinessOsSmoke,
  hasApp: !!globalThis.CTOX_BUSINESS_OS_APP,
  modules: (globalThis.ctoxBusinessOsSmoke?.state?.modules || globalThis.CTOX_BUSINESS_OS_APP?.modules || []).length,
  wm: !!(globalThis.ctoxBusinessOsSmoke?.state?.windowManager),
  scripts: [...document.scripts].map((s) => s.src).filter(Boolean).slice(0, 8),
  bodyChildren: document.body ? document.body.children.length : 0,
}));
console.log(JSON.stringify(state, null, 2));
console.log('failed requests:', JSON.stringify(failed.slice(0, 15), null, 2));
console.log('errors:', JSON.stringify(errors.slice(0, 10), null, 2));
await page.screenshot({ path: path.join(scriptDir, 'boot-probe.png') });
fs.writeFileSync(path.join(scriptDir, 'boot-probe.json'), JSON.stringify({ state, failed, errors }, null, 2));
await browser.close();
