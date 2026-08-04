import { createSidecarWithMemoryBackend } from '../dist/ctox-rxdb-js.mjs';

const storage = createSidecarWithMemoryBackend({ databaseName: 'query-meta-concurrency-test' });
const touches = [
  ['doc-a', 1024],
  ['doc-b', 2048],
  ['doc-c', 3072],
  ['doc-d', 4096],
];

// Start every metadata mutation before awaiting any of them. Without the
// QueryMetaStorage mutation queue, each touch can read the same empty stats
// snapshot and the later writes overwrite the other byte deltas.
await Promise.all(
  touches.map(([id, estimatedBytes]) => storage.touchDocuments(
    'business_records',
    [id],
    { estimatedBytes },
  )),
);

const expectedBytes = touches.reduce((sum, [, estimatedBytes]) => sum + estimatedBytes, 0);
const workingSetBytes = await storage.estimateWorkingSetBytes();
const stats = await storage.getCacheStats();

assert(
  workingSetBytes === expectedBytes,
  `test setup must persist ${expectedBytes} document bytes, got ${workingSetBytes}`,
);
assert(
  stats.estimatedBytes === expectedBytes,
  `concurrent metadata deltas must sum to ${expectedBytes} bytes, got ${stats.estimatedBytes}`,
);

console.log('ctox-rxdb query-meta concurrency smoke OK');

function assert(condition, message) {
  if (!condition) throw new Error(message);
}
