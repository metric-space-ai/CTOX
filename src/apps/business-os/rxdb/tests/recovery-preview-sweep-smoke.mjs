// SYNC-A A0.13: expired recovery-import previews must be swept.
//
// Previews hold DECRYPTED recovery payloads in a module-level map. Before the
// sweep, a preview that was never applied stayed in memory for the lifetime of
// the tab; now every previewImport/applyImport touch point drops entries whose
// expiresAtMs has passed.

import { recoveryJournalTestInternals } from '../src/recovery-journal.mjs';

const { PREVIEW_TTL_MS, previews, sweepExpiredPreviews } = recoveryJournalTestInternals;

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

assert(PREVIEW_TTL_MS === 10 * 60 * 1000, 'preview TTL stays at ten minutes');

const now = Date.now();
previews.clear();
previews.set('fresh', { journal: null, content: { secret: 'still-valid' }, expiresAtMs: now + PREVIEW_TTL_MS });
previews.set('expired-a', { journal: null, content: { secret: 'a' }, expiresAtMs: now - 1 });
previews.set('expired-b', { journal: null, content: { secret: 'b' }, expiresAtMs: now - PREVIEW_TTL_MS });

sweepExpiredPreviews(now);

assert(previews.size === 1, 'sweep removes every expired preview');
assert(previews.has('fresh'), 'sweep keeps unexpired previews');

// Boundary: an entry expiring exactly now is still valid (applyImport uses
// strict less-than), so the sweep must not remove it either.
previews.set('boundary', { journal: null, content: {}, expiresAtMs: now });
sweepExpiredPreviews(now);
assert(previews.has('boundary'), 'sweep uses the same strict-less-than boundary as applyImport');

previews.clear();
console.log('ctox-rxdb recovery preview sweep smoke OK');
