#!/usr/bin/env node
// Browser component regression for the actual shell URL/style helpers.
// This exercises HTTP imports, fetches and DOM stylesheet replacement, not Sync.
import assert from 'node:assert/strict';
import http from 'node:http';
import { readFileSync } from 'node:fs';
import { chromium } from 'playwright';

const appSource = readFileSync(new URL('../app.js', import.meta.url), 'utf8');
function sourceFunction(name) {
  const start = appSource.indexOf('\nfunction ' + name + '(') + 1;
  assert(start > 0, 'missing production function: ' + name);
  const end = appSource.indexOf('\n}\n', start);
  assert(end > start, 'unclosed production function: ' + name);
  return appSource.slice(start, end + 2);
}
const helpers = ['instanceModuleAssetPath', 'moduleBasePath', 'moduleIconAssetPath',
  'ensureModuleStylesheet'].map(sourceFunction).join('\n');
const testModule = helpers + `
const APP_BUILD = 'asset-probe';
function moduleRevisionQuery(mod) { return '_' + mod.asset_revision; }
const cases = [];
for (const source of ['modules', 'installed-modules', 'local-modules']) {
  const mod = { id: source + '-fixture', entry: source + '/fixture/index.html', asset_revision: 1 };
  const base = moduleBasePath(mod);
  const entry = await import('./' + base + '/index.js?v=' + APP_BUILD);
  if (entry.source !== source) throw new Error('wrong module source: ' + source);
  const frame = await fetch('./' + base + '/index.html');
  if (!frame.ok || await frame.text() !== source) throw new Error('wrong frame source: ' + source);
  for (const icon of ['icon.svg', source + '/fixture/icon.svg']) {
    mod.icon = icon;
    const image = new Image();
    const ready = new Promise((resolve, reject) => { image.onload = resolve; image.onerror = reject; });
    image.src = './' + moduleIconAssetPath(mod);
    document.body.append(image);
    await ready;
  }
  ensureModuleStylesheet(mod);
  ensureModuleStylesheet(mod);
  const cssPath = new URL('./' + base + '/index.css', document.baseURI).pathname;
  const matching = () => [...document.querySelectorAll('link[rel="stylesheet"]')]
    .filter(link => new URL(link.href).pathname === cssPath);
  if (matching().length !== 1) throw new Error('duplicate stylesheet after warm mount: ' + source);
  mod.asset_revision = 2;
  ensureModuleStylesheet(mod);
  if (matching().length !== 1 || !matching()[0].href.endsWith('_2')) {
    throw new Error('stylesheet revision did not replace old revision: ' + source);
  }
  await Promise.all(matching().map(link => link.sheet ? null : new Promise((resolve, reject) => {
    link.onload = resolve; link.onerror = reject;
  })));
  cases.push({ source, script: entry.url, frame: frame.url, stylesheet: matching()[0].href });
}
document.body.dataset.result = JSON.stringify(cases);
`;
const requests = [];
let activePinned = false;
const server = http.createServer((req, res) => {
  const pathname = new URL(req.url, 'http://localhost').pathname;
  requests.push(pathname);
  if (pathname === '/case') {
    const base = activePinned ? '/business-os/_shell/1.2.3/' : '/business-os/';
    res.writeHead(200, { 'content-type': 'text/html' });
    res.end('<!doctype html><head><base href="' + base + '"></head><body><script type="module" src="app.js"></script>');
    return;
  }
  const expectedBase = activePinned ? '/business-os/_shell/1.2.3/' : '/business-os/';
  if (pathname === expectedBase + 'app.js') {
    res.writeHead(200, { 'content-type': 'text/javascript' }); res.end(testModule); return;
  }
  for (const source of ['modules', 'installed-modules', 'local-modules']) {
    const prefix = (source === 'modules' ? expectedBase : '/business-os/') + source + '/fixture/';
    if (!pathname.startsWith(prefix)) continue;
    const file = pathname.slice(prefix.length);
    if (file === 'index.js') {
      res.writeHead(200, { 'content-type': 'text/javascript' });
      res.end('export const source=' + JSON.stringify(source) + ';export const url=import.meta.url;'); return;
    }
    if (file === 'index.html') { res.writeHead(200, { 'content-type': 'text/html' }); res.end(source); return; }
    if (file === 'index.css') { res.writeHead(200, { 'content-type': 'text/css' }); res.end('body { color: black }'); return; }
    if (file === 'icon.svg') {
      res.writeHead(200, { 'content-type': 'image/svg+xml' });
      res.end('<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"/>'); return;
    }
  }
  res.writeHead(404); res.end('unexpected asset path');
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
let browser;
try {
  browser = await chromium.launch();
  const reports = [];
  for (const pinned of [false, true]) {
    activePinned = pinned;
    const page = await browser.newPage();
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('response', response => { if (response.status() >= 400) errors.push(response.url()); });
    const from = requests.length;
    await page.goto('http://127.0.0.1:' + server.address().port + '/case');
    await page.waitForFunction(() => document.body.dataset.result, null, { timeout: 15000 });
    reports.push({ pinned, cases: await page.evaluate(() => JSON.parse(document.body.dataset.result)),
      requests: requests.slice(from) });
    assert.deepEqual(errors, []);
    await page.close();
  }
  console.log(JSON.stringify({ schema: 'ctox.shell.asset-routing-browser.v1', passed: true, reports }, null, 2));
} finally {
  await browser?.close();
  await new Promise(resolve => server.close(resolve));
}
