#!/usr/bin/env node
// Direktdiagnose: was passiert beim Shell-Load im Harness wirklich?
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
  viewport: { width: 1600, height: 1000 },
  args: ['--disable-features=LocalNetworkAccessChecks,LocalNetworkAccessForNavigations'],
});
const requests = [];
try {
  const page = context.pages()[0] || await context.newPage();
  page.on('response', (r) => requests.push(`${r.status()} ${r.url().slice(0, 140)}`));
  page.on('console', (m) => requests.push(`console.${m.type()}: ${m.text().slice(0, 200)}`));
  page.on('pageerror', (e) => requests.push(`pageerror: ${String(e).slice(0, 300)}`));
  await page.goto('http://127.0.0.1:8901/?rxdbSmoke=1', { waitUntil: 'commit', timeout: 60000 });
  await page.waitForTimeout(20000);
  const diag = await page.evaluate(async () => {
    const appResp = await fetch('app.js?v=diag').catch((e) => ({ error: String(e) }));
    const appText = appResp.ok ? await appResp.text() : `status ${appResp.status || appResp.error}`;
    return {
      href: location.href,
      search: location.search,
      hasSmoke: typeof globalThis.ctoxBusinessOsSmoke,
      hasSession: typeof globalThis.CTOX_BUSINESS_OS_SESSION,
      sessionAuth: globalThis.CTOX_BUSINESS_OS_SESSION?.authenticated,
      hasConfig: typeof globalThis.CTOX_BUSINESS_OS_CONFIG,
      appJsHead: appText.slice(0, 120),
      appJsHasSmokeInstall: appText.includes('ctoxBusinessOsSmoke'),
      moduleScripts: [...document.querySelectorAll('script[type=module]')].map((s) => s.src || s.textContent.slice(0, 60)),
    };
  });
  console.log(JSON.stringify(diag, null, 2));
  console.log('--- requests ---');
  for (const r of requests.slice(0, 40)) console.log(r);
} finally {
  await context.close().catch(() => {});
}
