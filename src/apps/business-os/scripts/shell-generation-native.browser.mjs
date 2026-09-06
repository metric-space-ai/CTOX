// Browser/native static-delivery regression gate; deliberately fails on mixed generations.
// This fixture does not certify signed-slot activation, auth, replication or full shell boot.
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createInterface } from 'node:readline';
import { chromium } from 'playwright';

const binary = process.argv[2];
const out = process.argv[3];
assert.ok(binary && path.isAbsolute(binary), 'Pass an absolute native binary path');
assert.ok(out && path.isAbsolute(out), 'Pass an absolute evidence directory');
await mkdir(out, { recursive: true });
const lab = spawn(process.execPath, [fileURLToPath(new URL('./shell-generation-native-lab.mjs', import.meta.url)), binary], { stdio: ['pipe', 'pipe', 'pipe'] });
const lines = createInterface({ input: lab.stdout });
const iterator = lines[Symbol.asyncIterator]();
const labExit = new Promise(resolve => lab.once('exit', (code, signal) => resolve({ code, signal })));
const phases = [];
const samples = [];
const controls = [];
const errors = [];
const requests = [];
let browser;
let context;
let failed;
let stderr = '';
lab.stderr.on('data', chunk => { stderr = (stderr + chunk.toString()).slice(-4000); });
async function nextPhase() {
  let timer;
  try {
    const result = await Promise.race([
      iterator.next(),
      new Promise((_, reject) => { timer = setTimeout(() => reject(new Error('Native lab phase timed out after 60 seconds')), 60000); }),
    ]);
    if (result.done) throw new Error(`Native lab stopped before the next phase: ${stderr}`);
    const phase = JSON.parse(result.value);
    phases.push(phase);
    return phase;
  } finally { clearTimeout(timer); }
}
async function control(url, expected) {
  const isolated = await browser.newContext();
  try {
    const page = await isolated.newPage();
    await page.goto(url, { waitUntil: 'domcontentloaded' });
    await page.getByRole('button', { name: 'Load database runtime', exact: true }).click();
    await page.locator('#result[data-verdict]').waitFor({ timeout: 15000 });
    const observed = await page.locator('#result').evaluate(result => ({
      expected: result.dataset.expected, actual: result.dataset.artifact, verdict: result.dataset.verdict,
    }));
    controls.push(observed);
    assert.deepEqual(observed, { expected, actual: expected, verdict: 'pass' });
  } finally { await isolated.close(); }
}
try {
  const initial = await nextPhase();
  browser = await chromium.launch({ headless: true });
  await control(initial.url, 'A');
  context = await browser.newContext();
  await context.tracing.start({ screenshots: true, snapshots: true, sources: true });
  const pages = [];
  for (let i = 0; i < 10; i += 1) {
    const page = await context.newPage();
    page.on('pageerror', error => errors.push(String(error)));
    page.on('request', request => requests.push({ event: 'request', path: new URL(request.url()).pathname }));
    page.on('response', response => requests.push({ event: 'response', path: new URL(response.url()).pathname, status: response.status() }));
    page.on('requestfailed', request => requests.push({ event: 'failed', path: new URL(request.url()).pathname, error: request.failure()?.errorText }));
    await page.goto(initial.url, { waitUntil: 'domcontentloaded' });
    await page.locator('#load:not([disabled])').waitFor({ timeout: 15000 });
    assert.match(await page.locator('#result').innerText(), /Document A ready/);
    pages.push(page);
  }
  lab.stdin.write('switch\n');
  await nextPhase();
  for (const [i, page] of pages.entries()) {
    await page.getByRole('button', { name: 'Load database runtime', exact: true }).click();
    await page.locator('#result[data-verdict]').waitFor({ timeout: 15000 });
    samples.push(await page.locator('#result').evaluate(result => ({
      expected: result.dataset.expected, actual: result.dataset.artifact,
      verdict: result.dataset.verdict, importMs: Number(result.dataset.elapsedMs), text: result.textContent,
    })));
    if (i === 0) await page.screenshot({ path: path.join(out, 'old-document-after-switch.png') });
  }
  await control(initial.url, 'B');
  assert.equal(errors.length, 0, errors.join('\n'));
  assert.equal(samples.filter(sample => sample.verdict === 'pass').length, 10,
    `Old documents loaded another artifact: ${JSON.stringify(samples)}`);
} catch (error) {
  failed = error;
} finally {
  if (context) await context.tracing.stop({ path: path.join(out, 'trace.zip') });
  if (browser) await browser.close();
  lab.stdin.end('quit\n');
  let cleanupTimer;
  const nativeExit = await Promise.race([
    labExit,
    new Promise(resolve => { cleanupTimer = setTimeout(() => { lab.kill('SIGTERM'); resolve({ error: 'lab cleanup timed out' }); }, 10000); }),
  ]);
  clearTimeout(cleanupTimer);
  if (nativeExit.code !== 0) failed ??= new Error(`Native fixture cleanup failed: ${JSON.stringify(nativeExit)}`);
  const times = samples.map(sample => sample.importMs).filter(Number.isFinite).sort((a,b) => a-b);
  const result = { schema: 'ctox.shell-generation-native-browser.v1', passed: !failed, phases, controls, samples, errors, requests, nativeExit,
    metric: 'one local native server, ten old documents; click-handler dynamic import elapsed time, not full boot or command latency',
    performance: { n: times.length, p50Ms: times[Math.ceil(times.length * .5)-1] ?? null, p95Ms: times[Math.ceil(times.length * .95)-1] ?? null },
    failure: failed ? String(failed) : null };
  await writeFile(path.join(out, 'result.json'), JSON.stringify(result, null, 2));
  console.log(JSON.stringify({ passed: result.passed, controls, samples, performance: result.performance, failure: result.failure, evidence: path.join(out, 'result.json') }));
}
if (failed) process.exitCode = 1;
