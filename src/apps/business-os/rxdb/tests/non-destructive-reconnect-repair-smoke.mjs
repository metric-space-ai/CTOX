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
