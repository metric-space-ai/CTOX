// REGRESSION: after a clean browser collection-version invalidation the replica
// must start its next pull at checkpoint=null, drain until the first empty
// master answer, leave retained pre-invalidation rows gone, and stamp a fresh
// firstPullCompletedAtMs. Uses the same mock-peer harness as
// replication-recovery-smoke.mjs / first-pull-readiness-smoke.mjs.

import { replicateWebRTC } from '../src/replication-webrtc.mjs';

const storage = new Map();
globalThis.localStorage = {
  getItem(key) { return storage.has(key) ? storage.get(key) : null; },
  setItem(key, value) { storage.set(key, String(value)); },
  removeItem(key) { storage.delete(key); },
  key(index) { return [...storage.keys()][index] ?? null; },
  get length() { return storage.size; },
};

const collectionName = 'version_invalidation_full_pull';
const topic = 'room-version-invalidation-full-pull-123456';
const protocol = {
  storageGeneration: 'native-generation-A',
  checkpoint: { epoch: 'native-checkpoint-A' },
  peerSession: { sessionId: 'native-session-A', role: 'ctox_instance' },
  collection: { name: collectionName, schemaHash: `hash-${collectionName}` },
  capabilities: ['ctox-checkpoint-generation-v2'],
};
const token = mintToken({ uid: 'alice', role: 'user', epoch: 1 });
const retainedKey = `ctox.rxdb.checkpoints.v1.${encodeURIComponent(topic)}.${encodeURIComponent(collectionName)}`;

// Seed a retained pull/push/readiness record that would be reused on a normal
// reconnect. Version invalidation must discard it so the next pull starts at null.
storage.set(retainedKey, JSON.stringify({
  validityKey: 'old-pre-invalidation',
  localValidityKey: 'old-local',
  pull: { id: 'retained-old', lwt: 42 },
  push: { id: 'retained-old', lwt: 42 },
  firstPullCompletedAtMs: 1_700_000_000_000,
}));

const primary = new Map([
  ['retained-old', { id: 'retained-old', title: 'pre-invalidation cache row' }],
]);
const collection = {
  name: collectionName,
  schema: { version: 1, hash: async () => `hash-${collectionName}` },
  // Marks that addCollections already ran the destructive cache clear.
  versionInvalidation: { invalidated: true, clearedRows: 1 },
  observe() { return { unsubscribe() {} }; },
  storageCollection: {
    replicationCheckpointStatus: async () => ({ epoch: 'checkpoint-epoch-1', state: 'ready' }),
    getChangedDocumentsSince: async () => ({ documents: [], checkpoint: null }),
    async bulkWrite(docs) {
      for (const doc of docs) primary.set(doc.id, doc);
      return { success: Object.fromEntries(docs.map((doc) => [doc.id, doc])) };
    },
    async hardDeleteByIds(ids) {
      for (const id of ids) primary.delete(id);
    },
    async allDocuments() {
      return [...primary.values()];
    },
  },
};

// Simulate the primary clear that prepareCollectionSchema already performed.
primary.clear();
assert(primary.size === 0, 'setup must start from a cleared primary after invalidation');
assert(storage.get(retainedKey), 'setup must install a retained checkpoint record');

const state = await replicateWebRTC({
  collection,
  topic,
  connectionHandlerCreator: {
    kind: 'ctox-native-webrtc',
    signalingServerUrl: 'wss://signaling.invalid/?token=t&token_iat=1&token_exp=2',
    config: {},
  },
  pull: { batchSize: 2 },
  push: { batchSize: 5 },
  retryTime: 60,
  ctox: { capabilityTokenProvider: async () => token },
});
state.initialReplication?.catch?.(() => {});

assert(state.retainedCheckpoints === null, 'versionInvalidation must force retained checkpoints to null at construction');
assert(state.firstPullCompletedAtMs === 0, 'versionInvalidation must clear firstPullCompletedAtMs at construction');
assert(storage.get(retainedKey) == null, 'versionInvalidation must drop the retained localStorage checkpoint record');

const pullArgs = [];
const remoteDocs = [
  { id: 'remote-1', title: 'after invalidation 1' },
  { id: 'remote-2', title: 'after invalidation 2' },
  { id: 'remote-3', title: 'after invalidation 3' },
];
let pullPass = 0;
state.remoteProtocolForPeer = () => protocol;
state.shared.isPeerOpen = () => true;
state.shared.negotiated = { peerId: 'peer-1', remoteProtocol: protocol };
state.peerStates$.next(new Map([['peer-1', { peerId: 'peer-1', remoteProtocol: protocol }]]));
state.shared.peer = {
  async request(peerId, method, [checkpoint, batchSize], _timeoutMs, collectionArg) {
    assert(peerId === 'peer-1', `unexpected peer ${peerId}`);
    assert(method === 'masterChangesSince', `expected masterChangesSince, got ${method}`);
    assert(collectionArg === collectionName, `unexpected collection ${collectionArg}`);
    pullArgs.push({ checkpoint: checkpoint === null ? null : structuredClone(checkpoint), batchSize });
    pullPass += 1;
    if (pullPass === 1) {
      assert(checkpoint === null, `first pull after invalidation must start at null, got ${JSON.stringify(checkpoint)}`);
      return {
        documents: remoteDocs.slice(0, 2),
        checkpoint: { id: 'remote-2', lwt: 2 },
      };
    }
    if (pullPass === 2) {
      assert(
        checkpoint?.id === 'remote-2',
        `second pull must continue from the previous checkpoint, got ${JSON.stringify(checkpoint)}`,
      );
      return {
        documents: remoteDocs.slice(2),
        checkpoint: { id: 'remote-3', lwt: 3 },
      };
    }
    // Empty answer ends the full pull and stamps firstPullCompletedAtMs.
    assert(
      checkpoint?.id === 'remote-3',
      `final empty pull must use the last checkpoint, got ${JSON.stringify(checkpoint)}`,
    );
    return { documents: [], checkpoint: { id: 'remote-3', lwt: 3 } };
  },
};

await state.resolveReadPermissionDigest?.();
await state.pullFromRemotePeers();

assert(pullPass === 3, `full pull must drain until the empty answer (got ${pullPass} passes)`);
assert(pullArgs[0]?.checkpoint === null, 'recorded first pull checkpoint was not null');
assert(pullArgs.length === 3, 'expected three masterChangesSince round-trips (2 batches + empty)');
assert(primary.size === 3, `full pull must materialize all remote rows, got ${primary.size}`);
assert(!primary.has('retained-old'), 'pre-invalidation retained row must stay gone after the full pull');
assert(primary.has('remote-1') && primary.has('remote-2') && primary.has('remote-3'), 'full pull missed remote rows');
const completedAt = state.firstPullCompletedAtMs;
assert(completedAt > 0, 'empty terminal answer must stamp a fresh firstPullCompletedAtMs');
assert(completedAt !== 1_700_000_000_000, 'firstPullCompletedAtMs must not reuse the pre-invalidation marker');
const retainedAfter = storage.get(retainedKey) ? JSON.parse(storage.get(retainedKey)) : null;
assert(
  retainedAfter?.firstPullCompletedAtMs === completedAt,
  'persisted readiness after the full pull must carry the fresh firstPullCompletedAtMs',
);
assert(retainedAfter?.pull?.id === 'remote-3', 'persisted pull checkpoint must end at the drained cursor');

await state.cancel();
console.log('ctox-rxdb version invalidation full-pull smoke OK assertions=14');
process.exit(0);

function mintToken({ uid, role, epoch }) {
  const header = Buffer.from(JSON.stringify({ alg: 'none', typ: 'JWT' })).toString('base64url');
  const payload = Buffer.from(JSON.stringify({
    sub: uid,
    role,
    capability_epoch: epoch,
  })).toString('base64url');
  return `${header}.${payload}.sig`;
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}
