#!/usr/bin/env node
import { createServer } from 'node:http';
import { createRequire } from 'node:module';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../../..');
const outputDir = path.join(repoRoot, 'output/playwright', 'ctox-crew-map');
fs.mkdirSync(outputDir, { recursive: true });

const { chromium } = require(resolvePlaywrightModule());
const consoleErrors = [];
const server = createServer((request, response) => serve(request, response));
const port = await new Promise((resolve) => server.listen(0, '127.0.0.1', () => resolve(server.address().port)));
const browser = await chromium.launch({ headless: true, executablePath: existingChromeExecutable(chromium), args: ['--disable-gpu'] });

try {
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await context.newPage();
  page.on('console', (message) => {
    if (message.type() === 'error') consoleErrors.push(message.text());
  });
  page.on('pageerror', (error) => consoleErrors.push(error.message || String(error)));
  page.on('requestfailed', (request) => consoleErrors.push(`${request.method()} ${request.url()} ${request.failure()?.errorText || ''}`));

  const url = `http://127.0.0.1:${port}/`;
  await page.goto(url, { waitUntil: 'networkidle' });
  const first = await collect(page);
  assertResult(first);
  await page.screenshot({ path: path.join(outputDir, 'ctox-crew-map.png'), fullPage: true });

  await page.reload({ waitUntil: 'networkidle' });
  const reloaded = await collect(page);
  assertResult(reloaded);
  if (first.identityColor !== reloaded.identityColor) throw new Error('crew identity changed after a clean reload');
  if (consoleErrors.length) throw new Error(`browser console/network errors: ${consoleErrors.join(' | ')}`);

  const report = { ok: true, url, first, reloaded, consoleErrors };
  fs.writeFileSync(path.join(outputDir, 'ctox-crew-map.json'), JSON.stringify(report, null, 2));
  console.log(JSON.stringify({ ok: true, scenarios: 2, screenshot: path.join(outputDir, 'ctox-crew-map.png') }, null, 2));
} finally {
  await browser.close();
  await new Promise((resolve) => server.close(resolve));
}

async function collect(page) {
  await page.waitForFunction(() => Boolean(window.__ctoxCrewReady));
  await page.waitForTimeout(420);
  return page.evaluate(() => {
    const slots = Array.from(document.querySelectorAll('.ctox-flow-creature-slot'));
    const byTask = Object.fromEntries(slots.map((slot) => [slot.dataset.taskId, {
      node: slot.dataset.creatureNodeId,
      mode: slot.querySelector('.ctox-crew-creature')?.dataset.crewMode || '',
      transform: slot.querySelector('.ctox-crew-creature')?.style.transform || '',
      xEyes: slot.querySelectorAll('.ctox-crew-eyes-x path').length,
    }]));
    document.querySelector('[data-task-id="task-waiting"]')?.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    const focusedAfterClick = window.__focusedTask || '';
    document.querySelector('[data-task-id="task-working"]')?.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
    return {
      count: slots.length,
      byTask,
      focusedAfterClick,
      focusedAfterKeyboard: window.__focusedTask || '',
      visibleTaskId: document.querySelector('.ctox-chat-track code')?.textContent || '',
      fullTaskId: document.querySelector('.ctox-chat-track')?.dataset.taskId || '',
      identityColor: document.querySelector('[data-task-id="task-working"] .ctox-crew-creature')?.style.getPropertyValue('--crew-color') || '',
      chatIdentityColor: document.querySelector('[data-chat-creature] .ctox-crew-creature')?.style.getPropertyValue('--crew-color') || '',
    };
  });
}

function assertResult(result) {
  if (result.count !== 3) throw new Error(`expected 3 map creatures, got ${result.count}`);
  if (result.byTask['task-waiting']?.node !== 'queued' || result.byTask['task-waiting']?.mode !== 'sleeping') throw new Error('waiting creature is not sleeping at queued');
  if (result.byTask['task-working']?.node !== 'running' || result.byTask['task-working']?.mode !== 'working') throw new Error('working creature is not active at running');
  if (!result.byTask['task-working']?.transform) throw new Error('working creature has no procedural transform');
  if (result.byTask['task-waiting']?.transform) throw new Error('waiting creature must remain still');
  if (result.byTask['task-failed']?.node !== 'model-failed' || result.byTask['task-failed']?.xEyes !== 2) throw new Error('failed creature lacks the failed node or X eyes');
  if (result.focusedAfterClick !== 'task-waiting' || result.focusedAfterKeyboard !== 'task-working') throw new Error('map creature selection is not mouse/keyboard reachable');
  if (result.fullTaskId !== 'queue:system::task_1234567890abcdef' || !result.visibleTaskId.startsWith('…')) throw new Error('chat task id deep-link is missing');
  if (!result.identityColor || result.identityColor !== result.chatIdentityColor) throw new Error('chat and map do not share the same creature identity');
}

function serve(request, response) {
  const requestUrl = new URL(request.url || '/', 'http://localhost');
  if (requestUrl.pathname === '/favicon.ico') {
    response.writeHead(204);
    response.end();
    return;
  }
  if (requestUrl.pathname === '/') {
    response.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
    response.end(harnessHtml());
    return;
  }
  const filePath = path.normalize(path.join(repoRoot, decodeURIComponent(requestUrl.pathname)));
  if (!filePath.startsWith(repoRoot) || !fs.existsSync(filePath)) {
    response.writeHead(404);
    response.end('not found');
    return;
  }
  const contentType = filePath.endsWith('.css') ? 'text/css' : filePath.endsWith('.json') ? 'application/json' : 'text/javascript';
  response.writeHead(200, { 'Content-Type': contentType });
  response.end(fs.readFileSync(filePath));
}

function harnessHtml() {
  return `<!doctype html><html lang="de"><head><meta charset="utf-8"><style>
    :root{--background:#080d10;--surface:#10181d;--text:#dce7ea;--muted:#718187;--accent:#1685ee;--success:#34a26f;--danger:#e75c62;--ctox-flow-node-fill:#10181d;--ctox-flow-node-stroke:#314047;--ctox-flow-lane-fill:#0b1216;--ctox-flow-lane-stroke:#213039;--ctox-flow-muted-fill:#718187;--ctox-flow-edge:#314047}
    body{margin:0;background:var(--background);color:var(--text);font-family:system-ui}main{width:1200px;margin:40px auto}svg{width:100%;height:660px}.proof-only{position:absolute;width:1px;height:1px;overflow:hidden;clip-path:inset(50%)}.demo-node{fill:var(--surface);stroke:#314047}.demo-label{fill:var(--text);font:700 18px system-ui;text-anchor:middle}
  </style><link rel="stylesheet" href="/src/apps/business-os/modules/ctox/index.css"></head><body><main><svg viewBox="0 0 1000 620"><g><rect class="demo-node" x="110" y="202" width="140" height="76" rx="12"/><text class="demo-label" x="180" y="246">Queue</text><rect class="demo-node" x="430" y="202" width="140" height="76" rx="12"/><text class="demo-label" x="500" y="246">Working</text><rect class="demo-node" x="750" y="402" width="140" height="76" rx="12"/><text class="demo-label" x="820" y="446">Failed</text></g><g id="crew"></g></svg><div class="proof-only" id="chat"></div><div class="proof-only" data-chat-creature id="chat-creature"></div></main><script type="module">
    import { __ctoxTestHooks } from '/src/apps/business-os/modules/ctox/index.js?v=20260831-ctox-desktopapp-ports-v328';
    import { __businessChatTestInternals, crewCreatureHtml, syncCrewProceduralMotion } from '/src/apps/business-os/shared/business-chat.js?v=20260831-ctox-desktopapp-ports-v328';
    const working={id:'task-working',commandId:'cmd-working',title:'Working task',status:'running',executionPhase:'running'};
    const waiting={id:'task-waiting',commandId:'cmd-waiting',title:'Waiting task',status:'queued',executionPhase:'queued'};
    const failed={id:'task-failed',commandId:'cmd-failed',title:'Failed task',status:'failed',executionPhase:'terminal',terminalStatus:'failed'};
    const model={activeTask:working,activeNodeId:'running',tasks:[working,waiting,failed],nodeMap:new Map([['queued',{id:'queued',x:180,y:240}],['running',{id:'running',x:500,y:240}],['model-failed',{id:'model-failed',x:820,y:440}]])};
    document.querySelector('#crew').innerHTML=__ctoxTestHooks.flowCrewSvg(model,working,{lang:'de'});
    document.querySelectorAll('.ctox-flow-creature-slot').forEach((slot)=>{const select=()=>window.__focusedTask=slot.dataset.taskId;slot.addEventListener('click',select);slot.addEventListener('keydown',(event)=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();select()}})});
    document.querySelector('#chat').innerHTML=__businessChatTestInternals.messageMarkup({id:'m1',role:'ctox',text:'Recherche gestartet.',taskId:'queue:system::task_1234567890abcdef',commandId:'cmd-working',status:'running'});
    document.querySelector('#chat-creature').innerHTML=crewCreatureHtml({id:'chat-random',messages:[{commandId:'cmd-working',taskId:'task-working'}]},'running','map');
    syncCrewProceduralMotion(document.querySelector('main'));
    window.__ctoxCrewReady=true;
  </script></body></html>`;
}

function resolvePlaywrightModule() {
  for (const candidate of [process.env.PLAYWRIGHT_MODULE_PATH, 'playwright', '/tmp/ctox-pw-smoke/node_modules/playwright', '/tmp/ctox-chatbar-pw/node_modules/playwright'].filter(Boolean)) {
    try { return require.resolve(candidate); } catch {}
  }
  throw new Error('No Playwright runtime found');
}

function existingChromeExecutable(chromiumRuntime) {
  return [process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE, chromiumRuntime.executablePath?.(), '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', '/Applications/Chromium.app/Contents/MacOS/Chromium'].filter(Boolean).find(fs.existsSync);
}
