// REGRESSION: capability acquisition must distinguish transient control-plane
// outages from terminal authorization rejection while command submission stays
// fail-closed. Transient acquisition gets one awaited refresh retry; terminal
// rejection keeps the negative cache and never reaches local insertion.

import assert from 'node:assert/strict';
import {
  createCommandBus,
  resetBusinessOsCapabilityTokenCacheForTests,
} from '../../shared/command-bus.js';

function clearInjectedCapabilitySessions() {
  delete globalThis.CTOX_BUSINESS_OS_SESSION;
  delete globalThis.ctoxBusinessOsSession;
  delete globalThis.ctoxBusinessOsLaunch;
  delete globalThis.CTOX_DESKTOP_SESSION;
  delete globalThis.ctoxDesktop;
}

function mockDb() {
  const documents = new Map();
  const collection = {
    documents,
    async insert(document) {
      documents.set(document.id, { ...document });
    },
    findOne(id) {
      return {
        async exec() {
          const document = documents.get(id);
          return document ? { toJSON: () => ({ ...document }) } : null;
        },
      };
    },
  };
  return {
    documents,
    db: {
      raw: {
        business_commands: collection,
        ctox_queue_tasks: collection,
      },
    },
  };
}

function capabilityResponse(token) {
  return {
    ok: true,
    status: 200,
    async json() {
      return {
        capability_token: token,
        expires_at_ms: Date.now() + 60 * 60 * 1000,
      };
    },
  };
}

function terminalResponse(status = 403) {
  return {
    ok: false,
    status,
    async json() { return {}; },
  };
}

clearInjectedCapabilitySessions();
resetBusinessOsCapabilityTokenCacheForTests();
const originalFetch = globalThis.fetch;
const nativeSetTimeout = globalThis.setTimeout;

try {
  // 1. A real timeout on the first POST is retried once inside the same submit.
  {
    resetBusinessOsCapabilityTokenCacheForTests();
    const { db, documents } = mockDb();
    let calls = 0;
    globalThis.fetch = (_url, options = {}) => {
      calls += 1;
      if (calls === 1) {
        return new Promise((_, reject) => {
          options.signal?.addEventListener('abort', () => reject(new Error('capability POST aborted')), { once: true });
        });
      }
      return Promise.resolve(capabilityResponse('capability-after-timeout'));
    };
    globalThis.setTimeout = (callback, delay, ...args) => (
      nativeSetTimeout(callback, delay === 120_000 ? 20 : delay, ...args)
    );

    const bus = createCommandBus({ db });
    const receipt = await bus.submit({
      id: 'cmd-auth-timeout-retry',
      command_type: 'business_os.test',
    });

    globalThis.setTimeout = nativeSetTimeout;
    assert.equal(calls, 2, 'submit performs exactly one refresh retry after timeout');
    assert.equal(receipt.ok, true);
    assert.equal(documents.get(receipt.command_id)?.client_context?.capability_token, 'capability-after-timeout');
  }

  // 2. Two transient failures exhaust one submit, but its short anti-storm cache
  // must not impose the terminal 10-second negative-cache window on the next.
  {
    resetBusinessOsCapabilityTokenCacheForTests();
    const { db, documents } = mockDb();
    let calls = 0;
    globalThis.fetch = async () => {
      calls += 1;
      if (calls <= 2) throw new TypeError('temporary network outage');
      return capabilityResponse('capability-after-outage');
    };

    const bus = createCommandBus({ db });
    await assert.rejects(
      bus.submit({ id: 'cmd-auth-transient-exhausted', command_type: 'business_os.test' }),
      (error) => error?.code === 'auth_required'
        && error?.transient === true
        && error?.retryable === true,
    );
    assert.equal(documents.size, 0, 'fail-closed rejects before local insertion');

    const startedAt = Date.now();
    const receipt = await bus.submit({
      id: 'cmd-auth-immediate-follow-up',
      command_type: 'business_os.test',
    });
    const elapsedMs = Date.now() - startedAt;

    assert.equal(receipt.ok, true, 'the immediately following submit can refresh and succeed');
    assert.equal(calls, 3, 'the follow-up reaches the token endpoint instead of the 10-second cache');
    assert.ok(elapsedMs < 2_000, `transient cache recovery is short (observed ${elapsedMs}ms)`);
  }

  // 3. A 403 is terminal: no in-submit retry, and the next submit is rejected
  // from the existing negative cache with transient:false.
  {
    resetBusinessOsCapabilityTokenCacheForTests();
    const { db, documents } = mockDb();
    let calls = 0;
    globalThis.fetch = async () => {
      calls += 1;
      return terminalResponse(403);
    };

    const bus = createCommandBus({ db });
    for (const id of ['cmd-auth-terminal-first', 'cmd-auth-terminal-cached']) {
      await assert.rejects(
        bus.submit({ id, command_type: 'business_os.test' }),
        (error) => error?.code === 'auth_required'
          && error?.transient === false
          && error?.retryable === true,
      );
    }
    assert.equal(calls, 1, 'terminal rejection retains the negative cache');
    assert.equal(documents.size, 0, 'terminal rejection remains fail-closed');
  }
} finally {
  globalThis.fetch = originalFetch;
  globalThis.setTimeout = nativeSetTimeout;
  clearInjectedCapabilitySessions();
  resetBusinessOsCapabilityTokenCacheForTests();
}

console.log('ctox-rxdb command auth resilience smoke OK');
process.exit(0);
