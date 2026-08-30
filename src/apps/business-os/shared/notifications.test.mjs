import assert from 'node:assert/strict';
import test from 'node:test';

import { createNotifications, normalizeSystemNotification } from './notifications.js';

test('normalizes a bounded Decision Hub system notification', () => {
  assert.deepEqual(normalizeSystemNotification({
    kind: 'decision_hub',
    title: '  Architektur   freigeben ',
    message: 'Decision Hub wartet auf deine Entscheidung.',
    tag: 'decision-hub:kpl-e-1',
    recordId: 'kpl-e-1',
    urgency: 'critical',
  }), {
    kind: 'decision_hub',
    title: 'Architektur freigeben',
    body: 'Decision Hub wartet auf deine Entscheidung.',
    tag: 'decision-hub:kpl-e-1',
    urgency: 'critical',
    recordId: 'kpl-e-1',
  });
});

test('rejects empty content and strips unsafe routing tokens', () => {
  assert.equal(normalizeSystemNotification({ title: '', message: '' }), null);
  assert.deepEqual(normalizeSystemNotification({
    title: 'Entscheidung',
    message: 'Bitte prüfen.',
    tag: 'unsafe token',
    recordId: '../../secret',
  }), {
    kind: 'business_os',
    title: 'Entscheidung',
    body: 'Bitte prüfen.',
    urgency: 'normal',
  });
});

test('delivers only the normalized payload through the Workjet mobile bridge', () => {
  let delivered = null;
  globalThis.workjetBusinessOsNotify = (payload) => {
    delivered = payload;
    return true;
  };
  try {
    const notifications = createNotifications({ container: {} });
    assert.equal(notifications.showSystem({
      kind: 'decision_hub',
      title: 'Freigabe',
      message: 'Bitte entscheiden.',
      recordId: 'kpl-e-1',
      context: 'must not cross the bridge',
      action: { callback() { throw new Error('must stay local'); } },
    }), true);
    assert.deepEqual(delivered, {
      kind: 'decision_hub',
      title: 'Freigabe',
      body: 'Bitte entscheiden.',
      urgency: 'normal',
      recordId: 'kpl-e-1',
    });
  } finally {
    delete globalThis.workjetBusinessOsNotify;
  }
});
