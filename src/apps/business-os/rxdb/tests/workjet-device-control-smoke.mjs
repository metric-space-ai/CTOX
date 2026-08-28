import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const testDir = dirname(fileURLToPath(import.meta.url));
const businessOsRoot = join(testDir, '..', '..');
const replicationSource = readFileSync(
  join(businessOsRoot, 'rxdb', 'src', 'replication-webrtc.mjs'),
  'utf8',
);
const replicationBundle = readFileSync(
  join(businessOsRoot, 'rxdb', 'dist', 'ctox-rxdb-js.mjs'),
  'utf8',
);
const app = readFileSync(join(businessOsRoot, 'app.js'), 'utf8');
const mobileHost = readFileSync(join(businessOsRoot, 'mobile-host.js'), 'utf8');
const nativeIndex = readFileSync(
  join(businessOsRoot, '..', '..', 'core', 'rxdb', 'src', 'plugins', 'replication_webrtc', 'index_mod.rs'),
  'utf8',
);
const nativeConnection = readFileSync(
  join(
    businessOsRoot,
    '..',
    '..',
    'core',
    'rxdb',
    'src',
    'plugins',
    'replication_webrtc',
    'connection_handler_rs.rs',
  ),
  'utf8',
);

assert(
  replicationSource.includes("String(method || '') === 'ctox.workjet.device.v1'"),
  'the source routes device control to the auxiliary WebRTC channel',
);
assert(
  replicationSource.includes('requestAuxiliary('),
  'the source must use the persistent peer auxiliary request path',
);
assert(
  replicationSource.includes("openAuxChannel('', CTOX_WORKJET_DEVICE_CONTROL_CHANNEL"),
  'the browser must negotiate the dedicated device-control DataChannel in its initial offer',
);
assert(
  nativeIndex.includes('"ctox-workjet-device-control-v1"'),
  'the native peer must advertise device-control capability',
);
assert(
  nativeConnection.includes('"ctox-browser-live-v1" | "ctox.workjet.device.v1"'),
  'the native peer must classify the device-control label as an auxiliary channel',
);
assert(
  replicationBundle.includes('ctox.workjet.device.v1') &&
    replicationBundle.includes('requestAuxiliary'),
  'the rebuilt browser bundle must contain the device-control route',
);
assert(
  app.includes('workjetBusinessOsDeviceControl'),
  'the shell exposes the bounded Workjet device-control facade',
);
assert(
  app.includes('await completeWorkjetPairingRedirect()'),
  'the authenticated web shell must complete ctox.dev pairing through its live CTOX guest',
);
assert(
  app.includes("open.textContent = 'In Workjet öffnen'") &&
    app.includes("document.documentElement.dataset.workjetPairingHandoff = 'ready'"),
  'the ctox.dev handoff must require an explicit user-activated return to Workjet',
);
assert(
  !sourceWindow(app, 'async function completeWorkjetPairingRedirect', 2_400).includes(
    'location.replace(`workjet://pair',
  ),
  'the ctox.dev handoff must not rely on an async custom-scheme redirect that browsers block',
);
assert(
  mobileHost.includes("command.type === 'device.control'"),
  'the mobile lifecycle bridge accepts the bounded control message',
);
assert(
  mobileHost.includes("type: 'device.control.result'"),
  'the mobile lifecycle bridge returns a correlated result',
);

const joined = [
  sourceWindow(replicationSource, "String(method || '') === 'ctox.workjet.device.v1'", 900),
  sourceWindow(app, 'globalThis.workjetBusinessOsDeviceControl', 1_400),
  sourceWindow(app, 'async function completeWorkjetPairingRedirect', 1_600),
  sourceWindow(mobileHost, "command.type === 'device.control'", 1_000),
].join('\n');
assert(!/\/api\/workjet\/device/u.test(joined), 'device control must not call an HTTP API');
assert(!/https?:\/\//u.test(joined), 'device control must not contain a network HTTP origin');
assert(!/ManagedRelay|EnvironmentHttp|Clerk|relay\.t3/iu.test(joined), 'legacy remote transports are forbidden');
assert(!/\bfetch\s*\(/u.test(sourceWindow(app, 'async function completeWorkjetPairingRedirect', 1_600)),
  'the ctox.dev handoff must create the invite through WebRTC, never HTTP');

console.log('CTOX Workjet device control WebRTC-only smoke OK');

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function sourceWindow(source, needle, radius) {
  const index = source.indexOf(needle);
  assert(index >= 0, `missing source window: ${needle}`);
  return source.slice(Math.max(0, index - radius), index + needle.length + radius);
}
