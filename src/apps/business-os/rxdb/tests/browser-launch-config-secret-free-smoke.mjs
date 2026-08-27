import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const appSource = await readFile(new URL('../../app.js', import.meta.url), 'utf8');

assert.doesNotMatch(
  appSource,
  /const roomPassword = String\([\s\S]*?config\.(?:signaling_)?room_password/,
  'browser launch normalization must not read the native room password',
);
assert.doesNotMatch(
  appSource,
  /signaling_room_password:\s*roomPassword/,
  'normalized browser config must not retain the native room password',
);
assert.doesNotMatch(
  appSource,
  /deriveSyncRoomFromPassword/,
  'the browser must not derive its room or signaling authority from the native room password',
);
for (const field of [
  'signaling_auth_version',
  'signaling_browser_token',
  'signaling_browser_token_hash',
  'signaling_native_token_hash',
]) {
  assert.match(appSource, new RegExp(`${field}:`), `browser launch normalization must retain ${field}`);
}

const storedConfigBlock = appSource.slice(
  appSource.indexOf('function readStoredPairingConfig()'),
  appSource.indexOf('function scrubPairingConfigFromUrl()'),
);
assert.match(storedConfigBlock, /sessionStorage\.getItem/, 'browser credential reload state must be session-scoped');
assert.match(storedConfigBlock, /sessionStorage\.setItem/, 'browser credential bootstrap must be session-scoped');
assert.doesNotMatch(
  storedConfigBlock,
  /writeScopedLocalStorage\(PAIRING_CONFIG_KEY/,
  'browser credentials must not be persisted to localStorage',
);
assert.match(
  storedConfigBlock,
  /removeScopedLocalStorage\(PAIRING_CONFIG_KEY/,
  'legacy persistent pairing credentials must be removed fail-closed',
);

console.log('browser launch config secret-free smoke OK');
