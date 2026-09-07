// Native HTTP + real browser diagnostic. No production instance is modified.
// Run with an explicit native binary, then open the printed loopback URL.
// Type "switch" to replace A with B while the browser keeps document A;
// click "Load database runtime" afterwards. Type "quit" to stop and clean up.
import { createHash } from 'node:crypto';
import { spawn } from 'node:child_process';
import { createReadStream } from 'node:fs';
import { copyFile, mkdir, mkdtemp, readFile, rename, rm, writeFile } from 'node:fs/promises';
import net from 'node:net';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { createInterface } from 'node:readline';
import { fileURLToPath } from 'node:url';
import { setTimeout as delay } from 'node:timers/promises';
import { stampShellDocument } from './build-shell-artifact.mjs';

const binary = process.argv[2];
if (!binary || !path.isAbsolute(binary)) {
  throw new Error('Usage: node scripts/shell-generation-native-lab.mjs /absolute/path/to/native-ctox-binary');
}
// Refuse launcher scripts: they may override the isolated root or source credentials.
const signature = await readFile(binary).then(bytes => bytes.subarray(0, 4));
if (signature.subarray(0, 2).toString() === '#!') throw new Error('Pass the native executable, not an installer/launcher script');
const hashFile = async filename => {
  const hash = createHash('sha256');
  for await (const bytes of createReadStream(filename)) hash.update(bytes);
  return hash.digest('hex');
};
const shellRoot = fileURLToPath(new URL('../', import.meta.url));
const fixtureRoot = await mkdtemp(path.join(tmpdir(), 'ctox-shell-generation-native-'));
const appRoot = path.join(fixtureRoot, 'src/apps/business-os');
const report = { schema: 'ctox.shell-generation-native-lab.v1', binary, binarySha256: await hashFile(binary), fixtureRoot,
  scope: 'native static delivery with source-tree fixtures; not signed-slot activation, authenticated boot, replication, or production performance',
  loaderSha256: await hashFile(path.join(shellRoot, 'shared/rxdb-runtime.js')), phases: [] };
let child;
let shuttingDown = false;

async function makeShell(destination, generation) {
  await mkdir(path.join(destination, 'shared'), { recursive: true });
  await mkdir(path.join(destination, 'rxdb/dist'), { recursive: true });
  await copyFile(path.join(shellRoot, 'shared/rxdb-runtime.js'), path.join(destination, 'shared/rxdb-runtime.js'));
  // The real bundle is preserved. A fixture-only export identifies returned bytes.
  const bundle = await readFile(path.join(shellRoot, 'rxdb/dist/ctox-rxdb-js.mjs'));
  await writeFile(path.join(destination, 'rxdb/dist/ctox-rxdb-js.mjs'), Buffer.concat([
    bundle, Buffer.from(`\nexport const shellGenerationLabArtifact = '${generation}';\n`),
  ]));
  const version = generation === 'A' ? '1.0.0' : '1.0.1';
  const html = `<!doctype html><html><head><base href="/business-os/"><title>CTOX native shell generation lab</title></head><body>
<h1>Document ${generation}</h1><p>Load this document, switch the native server to B, then load the database runtime.</p>
<button id="load" disabled>Load database runtime</button><output id="result">Loading entry module</output>
<script type="module" src="app.js?v=20260906-shell-v2-generation-lab"></script></body></html>`;
  await writeFile(path.join(destination, 'index.html'), stampShellDocument(Buffer.from(html), { version, sourceCommit: generation === 'A' ? 'a'.repeat(40) : 'b'.repeat(40) }));
  // Bypass personalized launch-context construction in this deliberately minimal
  // store. The document and all imports still travel through native serve_static.
  await copyFile(path.join(destination, 'index.html'), path.join(destination, 'generation-lab.html'));
  // The same manual APP_BUILD across two artifacts models the observed release alias.
  await writeFile(path.join(destination, 'app.js'), `const APP_BUILD = '20260906-shell-v2-generation-lab';
import { loadRxdbRuntime } from './shared/rxdb-runtime.js';
const button = document.querySelector('#load');
const result = document.querySelector('#result');
button.disabled = false;
result.textContent = 'Document ${generation} ready; database import has not started';
button.addEventListener('click', async () => {
  button.disabled = true;
  const start = performance.now();
  try {
    const runtime = await loadRxdbRuntime();
    const actual = runtime.shellGenerationLabArtifact;
    result.dataset.artifact = actual;
    result.dataset.expected = '${generation}';
    result.dataset.elapsedMs = String(performance.now() - start);
    result.dataset.verdict = actual === '${generation}' ? 'pass' : 'fail';
    result.textContent = 'Document ${generation}; database artifact ' + actual + '; ' + result.dataset.verdict;
  } catch (error) {
    result.dataset.verdict = 'error';
    result.dataset.elapsedMs = String(performance.now() - start);
    result.textContent = String(error);
  }
});\n`);
  await writeFile(path.join(destination, 'app.css'), 'body{font:18px system-ui;padding:32px}button{padding:12px}output{display:block;margin-top:20px}');
  await writeFile(path.join(destination, 'mobile-host.css'), '');
  await writeFile(path.join(destination, 'shared/base.css'), '');
  return { generation, version, bundleSha256: await hashFile(path.join(destination, 'rxdb/dist/ctox-rxdb-js.mjs')) };
}

async function stopNative() {
  if (!child || child.exitCode !== null || child.signalCode !== null) return;
  const running = child;
  const exited = new Promise(resolve => running.once('exit', resolve));
  const signal = name => { try { process.kill(-running.pid, name); } catch (error) { if (error.code !== 'ESRCH') throw error; } };
  signal('SIGTERM');
  const stopped = await Promise.race([exited.then(() => true), delay(5000).then(() => false)]);
  if (!stopped) { signal('SIGKILL'); await exited; }
}

async function startNative(port) {
  const started = performance.now();
  child = spawn(binary, ['business-os', 'serve', '--addr', `127.0.0.1:${port}`], {
    cwd: fixtureRoot, detached: true,
    // No operator credentials, launcher environment, or existing state is inherited.
    env: { PATH: process.env.PATH, CTOX_ROOT: fixtureRoot, CTOX_STATE_ROOT: path.join(fixtureRoot, 'runtime'),
      CTOX_INSTALL_ROOT: path.join(fixtureRoot, 'install'), CTOX_CACHE_ROOT: path.join(fixtureRoot, 'cache'), TMPDIR: fixtureRoot },
    stdio: ['ignore', 'ignore', 'ignore'],
  });
  let spawnError;
  child.once('error', error => { spawnError = error; });
  while (performance.now() - started < 30000) {
    if (spawnError) throw spawnError;
    if (child.exitCode !== null || child.signalCode !== null) throw new Error(`Native fixture terminated: exit=${child.exitCode}, signal=${child.signalCode}`);
    try {
      const response = await fetch(`http://127.0.0.1:${port}/business-os/app.js`, { signal: AbortSignal.timeout(1000) });
      if (response.ok && (await response.text()).includes('shellGenerationLabArtifact')) return performance.now() - started;
    } catch {}
    await delay(100);
  }
  throw new Error('Native fixture did not become ready within 30 seconds');
}

async function shutdown() {
  if (shuttingDown) return;
  shuttingDown = true;
  await stopNative();
  await rm(fixtureRoot, { recursive: true, force: true });
}

try {
  // Minimal recognizable source root; all mutable stores are private to this lab.
  await mkdir(path.join(fixtureRoot, 'contracts/history'), { recursive: true });
  await mkdir(path.join(fixtureRoot, 'src/core'), { recursive: true });
  await writeFile(path.join(fixtureRoot, 'Cargo.toml'), '# isolated native static delivery fixture\n');
  await writeFile(path.join(fixtureRoot, 'contracts/history/creation-ledger.md'), '# Fixture\n');
  await writeFile(path.join(fixtureRoot, 'src/core/main.rs'), '// Fixture root marker\n');
  const first = await makeShell(appRoot, 'A');
  const nextRoot = path.join(fixtureRoot, 'next-shell');
  const second = await makeShell(nextRoot, 'B');
  const socket = net.createServer();
  await new Promise(resolve => socket.listen(0, '127.0.0.1', resolve));
  const port = socket.address().port;
  await new Promise(resolve => socket.close(resolve));
  report.phases.push({ ...first, nativeStartMs: await startNative(port) });
  console.log(JSON.stringify({ ...report, url: `http://127.0.0.1:${port}/business-os/generation-lab.html`, commands: ['switch', 'quit'] }));
  process.once('SIGTERM', () => { shutdown().finally(() => process.exit(0)); });
  process.once('SIGINT', () => { shutdown().finally(() => process.exit(0)); });
  const lines = createInterface({ input: process.stdin, terminal: false });
  let switched = false;
  for await (const line of lines) {
    if (line.trim() === 'quit') break;
    if (line.trim() !== 'switch' || switched) { console.log('Expected switch (once) or quit'); continue; }
    await stopNative();
    await rename(appRoot, path.join(fixtureRoot, 'previous-shell'));
    await rename(nextRoot, appRoot);
    report.phases.push({ ...second, nativeStartMs: await startNative(port) });
    switched = true;
    console.log(JSON.stringify({ ...report, instruction: 'Keep document A open; click Load database runtime now.' }));
  }
} finally {
  await shutdown();
}
