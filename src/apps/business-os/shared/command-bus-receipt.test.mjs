import test from 'node:test';
import assert from 'node:assert/strict';

import {
  createCommandBus,
  resetBusinessOsCapabilityTokenCacheForTests,
} from './command-bus.js';

test.beforeEach(() => {
  resetBusinessOsCapabilityTokenCacheForTests();
  globalThis.CTOX_BUSINESS_OS_SESSION = {
    capability_token: 'receipt-test-capability',
    capability_expires_at_ms: Date.now() + 60 * 60 * 1000,
  };
});

test.afterEach(() => {
  delete globalThis.CTOX_BUSINESS_OS_SESSION;
  resetBusinessOsCapabilityTokenCacheForTests();
});

test('native observation makes a pending_sync legacy row an accepted receipt', async () => {
  let stored = null;
  const listeners = new Set();
  const collection = {
    async insert(document) {
      stored = { ...document };
    },
    findOne(id) {
      return {
        $: {
          subscribe(listener) {
            listeners.add(listener);
            if (stored?.id === id) listener({ toJSON: () => ({ ...stored }) });
            return { unsubscribe: () => listeners.delete(listener) };
          },
        },
        async exec() {
          return stored?.id === id ? { toJSON: () => ({ ...stored }) } : null;
        },
      };
    },
  };
  const syncState = {
    demandStatus: { peerConnected: true },
    async pushDocumentsToRemotePeers() {
      stored = {
        ...stored,
        // The immutable browser intent can still carry this legacy value after
        // native ownership has begun. The receipt must not expose it as a wait.
        status: 'pending_sync',
        replication_phase: 'native_observed',
        execution_task_id: 'queue-native-observed-1',
      };
      listeners.forEach((listener) => listener({ toJSON: () => ({ ...stored }) }));
    },
    async pullFromRemotePeers() {},
  };
  const bus = createCommandBus({
    db: { raw: { business_commands: collection } },
    sync: { async startCollection() { return { state: syncState }; } },
  });

  const receipt = await bus.dispatch({
    id: 'cmd-native-observed-pending-sync',
    command_type: 'business_os.chat.task',
    sync_queue_tasks: false,
  });

  assert.equal(stored.status, 'pending_sync');
  assert.equal(stored.replication_phase, 'native_observed');
  assert.equal(receipt.status, 'accepted');
  assert.equal(receipt.task_status, 'accepted');
  assert.equal(receipt.execution_task_id, 'queue-native-observed-1');
  assert.equal(receipt.task_id, 'queue-native-observed-1');
});
