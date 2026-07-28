#!/usr/bin/env node
// Minimaltest: wird das injizierte Inline-Script im Browser ausgefuehrt?
import { createRequire } from 'node:module';
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

const context = await chromium.launchPersistentContext(path.join(scriptDir, 'profile'), {
  headless: false,
  executablePath,
  viewport: { width: 1280, height: 800 },
  args: ['--disable-features=LocalNetworkAccessChecks,LocalNetworkAccessForNavigations'],
});
try {
  const page = context.pages()[0] || await context.newPage();
  page.on('pageerror', (e) => console.log('PAGEERROR:', String(e).slice(0, 400)));
  page.on('console', (m) => { if (m.type() === 'error') console.log('CONSOLE:', m.text().slice(0, 250)); });
  await page.goto('http://127.0.0.1:8901/?rxdbSmoke=1', { waitUntil: 'commit', timeout: 60000 });
  await page.waitForTimeout(8000);
  const out = await page.evaluate(() => {
    const scripts = [...document.scripts].map((s, i) => ({
      i,
      src: s.src || null,
      inlineLen: s.src ? 0 : s.textContent.length,
      inlineHead: s.src ? null : s.textContent.slice(0, 60),
    }));
    return {
      ctoxKeys: Object.keys(window).filter((k) => k.startsWith('CTOX')),
      sessionType: typeof window.CTOX_BUSINESS_OS_SESSION,
      scripts: scripts.slice(0, 8),
      totalScripts: scripts.length,
    };
  });
  console.log(JSON.stringify(out, null, 2));
} finally {
  await context.close().catch(() => {});
}
