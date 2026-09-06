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

test('Documents library resizes within its two-pane layout and collapses on compact screens', async () => {
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
    assert.equal(await page.locator('[data-documents-actions-resizer]').count(), 0);
    assert.equal(await page.locator('[data-documents-actions-drawer]').count(), 0);
    await leftResizer.focus();
    await leftResizer.press('ArrowRight');
    assert.equal(await root.evaluate((element) => element.style.getPropertyValue('--shell-col-left')), '344px');
    await leftResizer.press('Home');
    assert.equal(await root.evaluate((element) => element.style.getPropertyValue('--shell-col-left')), '300px');
    await leftResizer.press('End');
    assert.equal(await root.evaluate((element) => element.style.getPropertyValue('--shell-col-left')), '560px');
    assert.equal(Math.round((await page.locator('.documents-library-pane').boundingBox()).width), 560);
    await page.setViewportSize({ width: 1180, height: 800 });
    assert.ok(Math.round((await page.locator('.documents-workbench').boundingBox()).width) >= 480,
      'the editor retains its minimum width beside the expanded library');
    await page.setViewportSize({ width: 600, height: 800 });
    await page.evaluate(() => window.setCompact(true));
    assert.equal(await root.evaluate((element) => getComputedStyle(element).gridTemplateColumns.split(' ').length), 1);
    assert.equal(await leftResizer.isVisible(), false);
    assert.equal(Math.round((await page.locator('.documents-workbench').boundingBox()).width), 600);
    assert.ok(await root.evaluate((element) => element.scrollWidth <= element.clientWidth),
      'compact Documents has no horizontal overflow');
  } finally {
    await browser?.close();
    await fixture.close();
  }
});
