import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { createServer } from 'node:http';
import { build } from 'esbuild';
import { chromium } from 'playwright';

// Isolated UI regression, not a substitute for native WebRTC production E2E.
test('Spreadsheet version failures show a retry action, not absence or a saved badge', async () => {
  const [html, css, officeCss, translations, bundle] = await Promise.all([
    readFile(new URL('./index.html', import.meta.url), 'utf8'),
    readFile(new URL('./index.css', import.meta.url), 'utf8'),
    readFile(new URL('../../shared/office-workspace.css', import.meta.url), 'utf8'),
    readFile(new URL('./locales/en.json', import.meta.url), 'utf8'),
    build({ entryPoints: [fileURLToPath(new URL('./index.js', import.meta.url))],
      bundle: true, platform: 'browser', format: 'esm', write: false }),
  ]);
  const fixture = `<!doctype html><html><head><link rel="stylesheet" href="/index.css"></head>
    <body><div id="host">${html}</div><script type="module">
      import { __spreadsheetsTestHooks as hooks } from '/module.js';
      const translations = ${translations};
      const t = (key, fallback) => translations[key] || fallback || key;
      let missing = false;
      const state = {
        selectedId: 'sheet', selectedVersion: null, editorHandle: null,
        spreadsheets: [{ id: 'sheet', title: 'Load regression', filename: 'example.xlsx', current_version_id: 'version' }],
        dirty: false, saving: false, t,
        ctx: { host: document.querySelector('#host'), t,
          db: { collection() { return {
            findOne: () => ({ exec: async () => {
              if (!missing) throw new Error('Timed out waiting for WebRTC peer reopen for spreadsheet_versions');
              return null;
            } }),
            find: () => ({ exec: async () => [] }),
          }; } },
          sync: { startCollection: async () => ({}) },
        },
      };
      await hooks.loadSelectedVersion(state);
      await hooks.renderCenter(state);
      missing = true;
      document.body.dataset.ready = 'true';
    </script></body></html>`;
  const server = createServer((request, response) => {
    const pathname = new URL(request.url, 'http://127.0.0.1').pathname;
    const route = pathname === '/module.js' ? ['text/javascript', bundle.outputFiles[0].text]
      : pathname === '/index.css' ? ['text/css', css]
        : pathname === '/shared/office-workspace.css' ? ['text/css', officeCss]
          : pathname === '/' ? ['text/html', fixture] : null;
    response.writeHead(route ? 200 : 404, { 'content-type': route?.[0] || 'text/plain' });
    response.end(route?.[1] || 'Not found');
  });
  let browser;
  try {
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
    browser = await chromium.launch({ headless: true,
      ...(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH
        ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH } : {}) });
    const page = await browser.newPage();
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    await page.waitForSelector('body[data-ready="true"]');
    const canvas = page.locator('[data-spreadsheets-canvas]');
    assert.match(await canvas.innerText(), /Timed out waiting for WebRTC peer reopen/);
    assert.doesNotMatch(await canvas.innerText(), /No saved version/);
    assert.equal(await page.locator('[data-spreadsheets-dirty-indicator]').isVisible(), false);
    await page.getByRole('button', { name: 'Retry', exact: true }).click();
    await page.waitForFunction(() => document.querySelector('[data-spreadsheets-canvas]')?.textContent.includes('No saved version'));
    assert.equal(await page.locator('[data-spreadsheets-retry-version]').count(), 0);
    assert.equal(await page.locator('[data-spreadsheets-dirty-indicator]').isVisible(), false);
    assert.equal(await page.locator('[data-resizer="right"], [data-spreadsheets-head="runbooks"]').count(), 0);
    assert.deepEqual(errors, []);
  } finally {
    await browser?.close();
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
  }
});
