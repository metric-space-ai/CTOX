import assert from 'node:assert/strict';
import http from 'node:http';
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const source = readFileSync(new URL('../app.js', import.meta.url), 'utf8');
const cleanup = source.match(/  window\.addEventListener\('(beforeunload|pagehide)',[^\n]*\n(?:[^\n]*\n)*?    if \(state\.ctoxHealthTimer\)[\s\S]*?\n  \}\);/);
assert.ok(cleanup, 'actual shell cleanup registration is required');
const legacy = `window.addEventListener('beforeunload', () => {
  if (state.ctoxHealthTimer) window.clearInterval(state.ctoxHealthTimer);
  state.db?.close?.();
});`;
const server = http.createServer((request, response) => {
  response.writeHead(200, { 'content-type': 'text/html' });
  const old = request.url === '/legacy';
  response.end(`<!doctype html><title>Shell page-exit regression</title>
    <button id="write">Write own fixture</button><output id="result">loading</output>
    <script type="module">
      const request = indexedDB.open('shell-exit-${old ? 'legacy' : 'current'}', 1);
      request.onupgradeneeded = () => request.result.createObjectStore('entries');
      const db = await new Promise((resolve, reject) => {
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
      });
      const state = { db, ctoxHealthTimer: window.setInterval(() => {}, 10000) };
      ${old ? legacy : cleanup[0]}
      window.addEventListener('beforeunload', event => {
        event.preventDefault(); event.returnValue = '';
      });
      const output = document.querySelector('#result');
      document.querySelector('#write').onclick = () => {
        try {
          const tx = db.transaction('entries', 'readwrite');
          tx.objectStore('entries').put('own-fixture', 'marker');
          tx.oncomplete = () => { output.textContent = 'write_ok'; };
          tx.onabort = () => { output.textContent = 'aborted'; };
        } catch (error) { output.textContent = error.name; }
      };
      output.textContent = 'ready';
    </script>`);
});
await new Promise(done => server.listen(0, '127.0.0.1', done));
const playwrightModule = process.env.PLAYWRIGHT_MODULE_PATH
  ? pathToFileURL(resolve(process.env.PLAYWRIGHT_MODULE_PATH, 'index.mjs')).href
  : new URL('../node_modules/playwright/index.mjs', import.meta.url).href;
const { chromium } = await import(playwrightModule);
const systemChrome = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
let browser;
try {
  browser = await chromium.launch({
    headless: true,
    ...(existsSync(systemChrome) ? { executablePath: systemChrome } : {}),
  });
  const base = `http://127.0.0.1:${server.address().port}`;
  for (const [route, expected] of [['legacy', 'InvalidStateError'], ['current', 'write_ok']]) {
    const page = await browser.newPage();
    await page.goto(`${base}/${route}`);
    await page.getByText('ready', { exact: true }).waitFor();
    await page.getByRole('button', { name: 'Write own fixture' }).click();
    await page.getByText('write_ok', { exact: true }).waitFor();
    const dialogReady = page.waitForEvent('dialog', { timeout: 10000 });
    const navigation = page.goto(`${base}/leaving`).catch(error => error);
    const dialog = await dialogReady;
    assert.equal(dialog.type(), 'beforeunload');
    await dialog.dismiss();
    const result = await navigation;
    assert.match(String(result), /ERR_ABORTED/, 'the real beforeunload dialog must cancel navigation');
    assert.equal(page.url(), `${base}/${route}`);
    await page.getByRole('button', { name: 'Write own fixture' }).click();
    await page.getByText(expected, { exact: true }).waitFor();
    console.log(`PASS ${route}: dismissed navigation, subsequent IndexedDB write = ${expected}`);
    await page.close();
  }
} finally {
  await browser?.close();
  await new Promise(done => server.close(done));
}
