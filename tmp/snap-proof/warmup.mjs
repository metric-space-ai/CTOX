#!/usr/bin/env node
// Profil-Vorwaermung: einmalig den Cold-Boot (RxDB-Seed/Repair-Reload)
// durchlaufen, damit die eigentliche Repro schnell startet.
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptDir, '../..');
const { chromium } = require(path.join(repoRoot, '../../.local/state/ctox/node_modules/playwright'));
const executablePath = path.join(
  repoRoot,
  'runtime/browser/interactive-reference/ms-playwright/chromium-1228/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing',
);
const targetUrl = process.env.BUSINESS_OS_INTERACTIVE_URL || 'http://127.0.0.1:8901/?rxdbSmoke=1';

const context = await chromium.launchPersistentContext(path.join(scriptDir, 'profile'), {
  headless: false,
  executablePath,
  viewport: { width: 1600, height: 1000 },
  args: ['--disable-features=LocalNetworkAccessChecks,LocalNetworkAccessForNavigations'],
});
try {
  const page = context.pages()[0] || await context.newPage();
  for (let attempt = 1; attempt <= 2; attempt += 1) {
    await page.goto(targetUrl, { waitUntil: 'commit', timeout: 120000 });
    try {
      await page.waitForFunction(() => {
        const state = globalThis.ctoxBusinessOsSmoke?.state;
        return state?.windowManager && Array.isArray(state?.modules) && state.modules.length >= 30;
      }, null, { timeout: attempt === 1 ? 150000 : 90000, polling: 1000 });
      console.log(`warmup ok (attempt ${attempt}): smoke shell bereit`);
      break;
    } catch {
      console.log(`warmup attempt ${attempt}: smoke nicht bereit, reload`);
    }
  }
  const summary = await page.evaluate(() => ({
    hasSmoke: !!globalThis.ctoxBusinessOsSmoke,
    modules: (globalThis.ctoxBusinessOsSmoke?.state?.modules || []).length,
    wm: !!globalThis.ctoxBusinessOsSmoke?.state?.windowManager,
    overlay: (document.body?.innerText || '').replace(/\s+/g, ' ').slice(0, 160),
  }));
  console.log(JSON.stringify(summary));
} finally {
  await context.close().catch(() => {});
}
