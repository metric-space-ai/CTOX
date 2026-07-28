#!/usr/bin/env node
// Werden Inline-Scripts ueberhaupt ausgefuehrt? Fruehe Error-Erfassung.
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
  await page.addInitScript(() => {
    globalThis.__errors = [];
    globalThis.__inlineExecuted = [];
    window.addEventListener('error', (e) => {
      globalThis.__errors.push(`${e.message} @ ${e.filename}:${e.lineno}:${e.colno}`);
    }, true);
    window.addEventListener('unhandledrejection', (e) => {
      globalThis.__errors.push(`unhandledrejection: ${String(e.reason).slice(0, 200)}`);
    });
  });
  await page.goto('http://127.0.0.1:8901/?rxdbSmoke=1', { waitUntil: 'commit', timeout: 60000 });
  await page.waitForTimeout(8000);
  const out = await page.evaluate(() => ({
    errors: globalThis.__errors.slice(0, 10),
    shellStyle: document.documentElement.dataset.shellStyle || null,
    theme: document.documentElement.dataset.theme || null,
    sessionType: typeof window.CTOX_BUSINESS_OS_SESSION,
    readyState: document.readyState,
  }));
  console.log(JSON.stringify(out, null, 2));
} finally {
  await context.close().catch(() => {});
}
