import test from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';

import { chromium } from 'playwright';

const MODULE_HTML = new URL('./index.html', import.meta.url);
const MODULE_CSS = new URL('./index.css', import.meta.url);
const RESIZER_JS = new URL('../../shared/resizer.js', import.meta.url);

async function startFixtureServer() {
  const [moduleHtml, moduleCss, resizerSource] = await Promise.all([
    readFile(MODULE_HTML, 'utf8'),
    readFile(MODULE_CSS, 'utf8'),
    readFile(RESIZER_JS, 'utf8'),
  ]);
  const pageHtml = `<!doctype html>
    <html>
      <head>
        <meta charset="utf-8">
        <style>
          :root {
            --bg: #f4f7f8;
            --surface: #eef3f4;
            --surface-2: #e5ecee;
            --text: #172126;
            --muted: #66747b;
            --line: #b7c2c7;
            --line-strong: #809097;
            --accent: #087f8c;
            --control-radius: 4px;
            --surface-radius: 6px;
            --shadow: 0 8px 24px rgb(10 25 30 / 18%);
          }
          html, body, #host { width: 100%; height: 100%; margin: 0; overflow: hidden; }
        </style>
        <link rel="stylesheet" href="/index.css">
      </head>
      <body>
        <div id="host">${moduleHtml}</div>
        <script type="module">
          import { CtoxResizer } from '/resizer.js';
          const root = document.querySelector('[data-resize-frame]');
          for (const handle of root.querySelectorAll('[data-resizer-var]')) {
            new CtoxResizer({
              resizerEl: handle,
              containerEl: root,
              cssVar: handle.dataset.resizerVar,
              side: handle.dataset.resizer,
              minWidth: Number(handle.dataset.resizerMin),
              maxWidth: Number(handle.dataset.resizerMax),
            });
          }
          window.setActionsOpen = (open) => {
            root.classList.toggle('is-actions-open', open);
            const drawer = root.querySelector('[data-documents-actions-drawer]');
            const resizer = root.querySelector('[data-documents-actions-resizer]');
            drawer.hidden = !open;
            resizer.hidden = !open;
          };
          window.setActionsOverlay = (overlay) => root.classList.toggle('is-actions-overlay', overlay);
          window.setCompact = (compact) => root.classList.toggle('is-compact', compact);
        </script>
      </body>
    </html>`;
  const server = createServer((request, response) => {
    const path = new URL(request.url, 'http://127.0.0.1').pathname;
    const payload = path === '/index.css'
      ? moduleCss
      : path === '/resizer.js'
        ? resizerSource
        : pageHtml;
    response.writeHead(200, {
      'content-type': path.endsWith('.css')
        ? 'text/css; charset=utf-8'
        : path.endsWith('.js')
          ? 'text/javascript; charset=utf-8'
          : 'text/html; charset=utf-8',
    });
    response.end(payload);
  });
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });
  const address = server.address();
  return {
    url: `http://127.0.0.1:${address.port}/`,
    close: () => new Promise((resolve, reject) => {
      server.closeAllConnections();
      server.close((error) => error ? reject(error) : resolve());
    }),
  };
}

test('Documents columns resize and the optional actions column becomes a compact overlay', async () => {
  const fixture = await startFixtureServer();
  let browser;
  try {
    browser = await chromium.launch({
      headless: true,
      ...(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH
        ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH }
        : {}),
    });
    const page = await browser.newPage({ viewport: { width: 1800, height: 800 } });
    await page.goto(fixture.url, { waitUntil: 'networkidle' });
    const root = page.locator('[data-documents-module]');
    const leftResizer = page.locator('[data-resizer="left"]');
    const rightResizer = page.locator('[data-documents-actions-resizer]');
    const drawer = page.locator('[data-documents-actions-drawer]');

    await expectHidden(rightResizer);
    await leftResizer.focus();
    await leftResizer.press('ArrowRight');
    assert.equal(await root.evaluate((element) => element.style.getPropertyValue('--documents-library-width')), '324px');

    await page.evaluate(() => window.setActionsOpen(true));
    await leftResizer.focus();
    await leftResizer.press('End');
    await rightResizer.focus();
    await rightResizer.press('Home');
    await rightResizer.press('End');
    assert.equal(await root.evaluate((element) => element.style.getPropertyValue('--documents-library-width')), '560px');
    assert.equal(await root.evaluate((element) => element.style.getPropertyValue('--documents-actions-width')), '560px');
    assert.equal(await drawer.evaluate((element) => getComputedStyle(element).position), 'relative');
    assert.equal(Math.round((await drawer.boundingBox()).width), 560);
    assert.ok(
      Math.round((await page.locator('.documents-workbench').boundingBox()).width) >= 480,
      'the editor must retain its minimum width with both side columns maximized',
    );

    await page.setViewportSize({ width: 1180, height: 800 });
    await page.evaluate(() => window.setActionsOverlay(true));
    assert.equal(await drawer.evaluate((element) => getComputedStyle(element).position), 'absolute');
    assert.ok(
      Math.round((await page.locator('.documents-workbench').boundingBox()).width) >= 480,
      'the editor must retain its minimum width while the actions drawer is open',
    );

    await page.evaluate(() => window.setCompact(true));
    assert.equal(await root.evaluate((element) => getComputedStyle(element).gridTemplateColumns.split(' ').length), 1);
    assert.equal(await drawer.evaluate((element) => getComputedStyle(element).position), 'absolute');
    assert.equal(Math.round((await drawer.boundingBox()).width), 560);
  } finally {
    await browser?.close();
    await fixture.close();
  }
});

async function expectHidden(locator) {
  assert.equal(await locator.getAttribute('hidden'), '');
  assert.equal(await locator.evaluate((element) => getComputedStyle(element).display), 'none');
}
