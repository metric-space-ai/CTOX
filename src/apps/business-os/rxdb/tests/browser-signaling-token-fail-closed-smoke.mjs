import { createHash, webcrypto } from 'node:crypto';

globalThis.crypto ??= webcrypto;
globalThis.window = { location: { href: 'https://tenant.ctox.dev/' } };

const { __ctoxSyncTestHooks } = await import('../../shared/sync.js');
const { signalingUrlWithBrowserMetadata } = __ctoxSyncTestHooks;

const browserToken = 'browser-role-token';
const config = {
  instance_id: 'biz_test',
  sync_room: 'ctox-business-os:biz_test:room',
  signaling_auth_version: 'ctox-role-bound-v1',
  signaling_browser_token: browserToken,
  signaling_browser_token_hash: sha256(browserToken),
  signaling_native_token_hash: sha256('distinct-native-token'),
};

const signalingUrl = new URL(await signalingUrlWithBrowserMetadata('wss://signaling.ctox.dev/?token=legacy', config));
assert(signalingUrl.pathname === '/v2', 'production signaling path must be /v2');
assert(signalingUrl.searchParams.get('token') === browserToken, 'explicit browser token must replace URL credentials');
assert(signalingUrl.searchParams.get('role') === 'browser', 'signaling role must remain browser');
assert(signalingUrl.searchParams.get('auth_version') === 'ctox-role-bound-v1', 'role-bound auth version is missing');

await assertRejects(
  () => signalingUrlWithBrowserMetadata('wss://signaling.ctox.dev/?token=legacy', {
    ...config,
    signaling_browser_token: '',
    signaling_room_password: 'must-not-be-used',
  }),
  /explicit browser signaling token/,
);
await assertRejects(
  () => signalingUrlWithBrowserMetadata('wss://signaling.ctox.dev/', {
    ...config,
    signaling_browser_token_hash: sha256('different-token'),
  }),
  /does not match its commitment/,
);
await assertRejects(
  () => signalingUrlWithBrowserMetadata('https://signaling.ctox.dev/v2', config),
  /ws\(s\) signaling URL/,
);

console.log('browser signaling token fail-closed smoke OK');

function sha256(value) {
  return createHash('sha256').update(value).digest('hex');
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

async function assertRejects(callback, pattern) {
  try {
    await callback();
  } catch (error) {
    if (pattern.test(String(error?.message || error))) return;
    throw error;
  }
  throw new Error(`expected rejection matching ${pattern}`);
}
