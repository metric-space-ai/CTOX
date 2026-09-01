import assert from 'node:assert/strict';

import { REMOTE_MASTER_READY_TIMEOUT_MS } from '../src/replication-webrtc.mjs';

// The native request/answer leg intentionally has a 60-second ceiling. The
// browser must fail after native so large multiplex protocol exchanges cannot
// become a permanent reconnect loop before the token request arrives.
assert.equal(REMOTE_MASTER_READY_TIMEOUT_MS, 65_000);
assert.ok(REMOTE_MASTER_READY_TIMEOUT_MS > 60_000);

console.log('remote master readiness timeout smoke: ok');
