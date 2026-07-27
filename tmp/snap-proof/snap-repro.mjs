#!/usr/bin/env node
// Fokussierte Reproduktion: Fenster-Snap links/rechts/oben gegen die laufende
// Business-OS-Instanz (Port 8765). Instrumentiert die Snap-Preview waehrend
// des Drags und testet langsame wie schnelle Zeigerbewegungen.
import { createRequire } from 'node:module';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptDir, '../..');
const outputDir = scriptDir;
const targetUrl = process.env.BUSINESS_OS_INTERACTIVE_URL || 'http://127.0.0.1:8765/?rxdbSmoke=1';

const { chromium } = require(path.join(repoRoot, 'runtime/browser/interactive-reference/node_modules/patchright'));
const executablePath = path.join(
  repoRoot,
  'runtime/browser/interactive-reference/ms-playwright/chromium-1228/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing',
);

const report = { startedAt: new Date().toISOString(), cases: [], console: [] };

const browser = await chromium.launch({
  headless: false,
  executablePath,
  args: ['--disable-features=LocalNetworkAccessChecks,LocalNetworkAccessForNavigations'],
});

try {
  const context = await browser.newContext({ viewport: { width: 1600, height: 1000 }, deviceScaleFactor: 1 });
  const page = await context.newPage();
  page.on('pageerror', (error) => report.console.push({ type: 'pageerror', text: String(error) }));
  page.on('console', (m) => {
    if (m.type() === 'error') report.console.push({ type: 'error', text: m.text() });
  });

  await page.goto(targetUrl, { waitUntil: 'commit', timeout: 240000 });
  await page.waitForFunction(() => {
    const state = globalThis.ctoxBusinessOsSmoke?.state || globalThis.CTOX_BUSINESS_OS_APP;
    return document.body?.dataset?.authState !== 'locked'
      && state?.windowManager
      && Array.isArray(state?.modules)
      && state.modules.length >= 30
      && !document.body.dataset.moduleLoading;
  }, null, { timeout: 300000, polling: 1000 });
  await page.evaluate(() => globalThis.ctoxBusinessOsSmoke?.state?.windowManager?.destroyAll?.());
  await page.waitForTimeout(300);

  // Fenster ueber das Startmenue oeffnen (notes), wie der interaktive QA-Lauf.
  await page.locator('[data-shell-start]').click();
  const panel = page.locator('.shell-start-menu-panel');
  await panel.waitFor({ state: 'visible', timeout: 3000 });
  await panel.locator('.start-menu-search-input').fill('Notizen');
  await page.waitForTimeout(200);
  let item = panel.locator('.start-menu-item[data-target="notes"]:visible').first();
  if (!(await item.count())) {
    await panel.locator('.start-menu-search-input').fill('notes');
    await page.waitForTimeout(200);
    item = panel.locator('.start-menu-item[data-target="notes"]:visible').first();
  }
  await item.click();
  const win = page.locator('.shell-window[data-owner-id="desktop-app:notes"]');
  await win.waitFor({ state: 'visible', timeout: 15000 });
  await page.waitForTimeout(600);

  // Snap-Events mitprotokollieren.
  await page.evaluate(() => {
    globalThis.__snapEvents = [];
    const bus = globalThis.ctoxBusinessOsSmoke?.state?.eventBus;
    bus?.on?.('window:snapped', (d) => globalThis.__snapEvents.push({ t: Date.now(), ...d }));
  });

  async function dragCase(name, target, { steps }) {
    // Fenster vor jedem Fall in die Mitte zurueckholen (unsnap + freie Position).
    await page.evaluate(() => {
      const wm = globalThis.ctoxBusinessOsSmoke?.state?.windowManager;
      const w = wm?.listWindows?.()[0];
      if (!w) return;
      const el = document.getElementById(w.id);
      el.classList.remove('is-snapped');
      el.removeAttribute('data-snap-zone');
      Object.assign(el.style, { left: '500px', top: '280px', width: '520px', height: '360px' });
      w.state = 'normal';
    });
    await page.waitForTimeout(120);

    const headerBox = await win.locator('[data-window-title]').boundingBox()
      || await win.locator('[data-window-header]').boundingBox();
    const start = { x: headerBox.x + Math.min(headerBox.width / 2, 72), y: headerBox.y + headerBox.height / 2 };

    const samples = await page.evaluate(() => { globalThis.__previewSamples = []; return true; });
    void samples;
    await page.mouse.move(start.x, start.y);
    await page.mouse.down();
    // Ziel in Schritten anfahren, Preview-Zustand nach jedem Schritt abtasten.
    for (let i = 1; i <= steps; i += 1) {
      const x = start.x + ((target.x - start.x) * i) / steps;
      const y = start.y + ((target.y - start.y) * i) / steps;
      await page.mouse.move(x, y);
      const sample = await page.evaluate(() => {
        const preview = document.querySelector('.shell-snap-preview');
        return {
          hidden: preview?.hidden ?? null,
          zone: preview?.dataset?.snap || null,
          visible: preview?.classList?.contains('is-visible') || false,
        };
      });
      await page.evaluate((s) => globalThis.__previewSamples.push(s), sample);
    }
    await page.mouse.up();
    await page.waitForTimeout(150);

    const result = await page.evaluate(() => {
      const wm = globalThis.ctoxBusinessOsSmoke?.state?.windowManager;
      const w = wm?.listWindows?.()[0];
      const el = w && document.getElementById(w.id);
      const rect = el?.getBoundingClientRect();
      return {
        snapZone: el?.dataset?.snapZone || '',
        snapped: el?.classList?.contains('is-snapped') || false,
        rect: rect ? { x: Math.round(rect.x), y: Math.round(rect.y), w: Math.round(rect.width), h: Math.round(rect.height) } : null,
        previewSamples: globalThis.__previewSamples,
        snapEvents: globalThis.__snapEvents.splice(0),
      };
    });
    report.cases.push({ name, target, steps, ...result });
    console.log(`${result.snapZone ? 'SNAP' : '----'} ${name}: zone='${result.snapZone}' events=${JSON.stringify(result.snapEvents.map((e) => e.zone))}`);
    await page.screenshot({ path: path.join(outputDir, `case-${name}.png`), scale: 'css' });
  }

  const midY = 420;
  await dragCase('left-slow', { x: 4, y: midY }, { steps: 12 });
  await dragCase('left-fast', { x: 4, y: midY }, { steps: 1 });
  await dragCase('right-slow', { x: 1596, y: midY }, { steps: 12 });
  await dragCase('right-fast', { x: 1596, y: midY }, { steps: 1 });
  await dragCase('top-slow', { x: 800, y: 40 }, { steps: 12 });
  await dragCase('bottom-slow', { x: 800, y: 995 }, { steps: 12 });

  report.ok = report.cases.every((c) => c.snapZone);
} finally {
  await browser.close();
}

report.endedAt = new Date().toISOString();
fs.writeFileSync(path.join(outputDir, 'snap-repro.json'), JSON.stringify(report, null, 2));
console.log(`REPORT ok=${report.ok} cases=${report.cases.length} -> ${path.join(outputDir, 'snap-repro.json')}`);
