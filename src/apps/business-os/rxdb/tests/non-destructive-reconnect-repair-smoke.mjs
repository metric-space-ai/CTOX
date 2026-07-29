import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const app = readFileSync(new URL('../../app.js', import.meta.url), 'utf8');
const match = app.match(/async function repairRecoveringDataPlane\(\) \{([\s\S]*?)\n\}/);

assert.ok(match, 'repairRecoveringDataPlane must exist');
assert.match(match[1], /state\.sync\.restartCollections\(collections\)/);
assert.doesNotMatch(match[1], /repairBusinessDataPlane|resetBusinessDb/);
assert.doesNotMatch(match[1], /activeCollections/, 'shell recovery must not restart healthy active collections');
assert.match(match[1], /SYNC_RECOVERY_MIN_STALLED_MS/, 'shell recovery must recheck sustained collection failure');

assert.match(app, /collection\.lastError\.retryable === false/, 'non-retryable errors must not enter recovery');
assert.match(app, /collection\.reconnectingSince/, 'recovery must use the per-collection reconnect timestamp');

console.log('Non-destructive reconnect repair smoke OK');

// A packed pairing launch must inherit the instance's ICE servers. Without them
// the WebRTC pool has no STUN/TURN and only connects on the same host, leaving
// every collection in a silent `reconnecting` state across NAT.
const launchNormalizer = app.match(/async function normalizeBusinessOsLaunchConfig\(config, transportFallback = null\) \{([\s\S]*?)\n\}/);
assert.ok(launchNormalizer, 'normalizeBusinessOsLaunchConfig must accept a transport fallback');
assert.match(launchNormalizer[1], /declaredIceServers\.length \? declaredIceServers : fallbackIceServers/);
assert.match(launchNormalizer[1], /transportFallback\?\.ice_servers_refresh_url/);
assert.match(app, /normalizeBusinessOsLaunchConfig\(launch, firstObject\(/, 'launch config must be normalized with the served instance config as transport fallback');

console.log('Pairing ICE inheritance smoke OK');
