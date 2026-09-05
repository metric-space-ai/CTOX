import assert from 'node:assert/strict';
import { test } from 'node:test';
import { pathToFileURL } from 'node:url';
import { createAssistantMessageEventStream } from '@earendil-works/pi-ai';

const bundle = process.env.CTOX_PI_TEST_DIST
  ? pathToFileURL(process.env.CTOX_PI_TEST_DIST).href
  : new URL('../dist/ctox-pi-sidecar.mjs', import.meta.url).href;
const { handleTurnRequest, createVercelPiCodingTextMessage, createVercelPiCodingToolCallMessage } = await import(bundle);

test('successful tool-driven edit still returns its source snapshot', async () => {
  const response = await handleTurnRequest({
    id: 'success', prompt: 'Edit index.js', files: { 'index.js': 'export const v = 1;' },
  }, (_model, context) => {
    const stream = createAssistantMessageEventStream();
    const written = context.messages.some(message => message.role === 'toolResult');
    stream.push({
      type: 'done', reason: written ? 'stop' : 'toolUse',
      message: written ? createVercelPiCodingTextMessage('Done')
        : createVercelPiCodingToolCallMessage('write', { path: 'index.js', content: 'export const v = 2;' }, 'write-1'),
    });
    return stream;
  });
  assert.equal(response.ok, true);
  assert.ok(response.snapshot.some(entry => entry.path.endsWith('index.js')
    && entry.content === 'export const v = 2;'));
});

for (const [stopReason, detail, category] of [
  ['error', 'fetch failed at https://private.invalid/?token=SECRET', 'connection_error'],
  ['error', '401 Unauthorized Authorization: Bearer SECRET', 'authentication_error'],
  ['error', '429 rate limit', 'rate_limited'],
  ['error', 'request timed out', 'timeout'],
  ['error', 'unknown SECRET failure', 'provider_error'],
  ['aborted', 'aborted SECRET request', 'aborted'],
]) {
  test(`terminal ${category} is not a successful coding turn`, async () => {
    const response = await handleTurnRequest({
      id: category, prompt: 'Edit index.js', files: { 'index.js': 'export const v = 1;' },
    }, () => {
      const stream = createAssistantMessageEventStream();
      const message = { ...createVercelPiCodingTextMessage(''), stopReason, errorMessage: detail };
      stream.push({ type: 'error', reason: stopReason, error: message });
      return stream;
    });
    assert.equal(response.ok, false);
    assert.equal(response.id, category);
    assert.equal(response.error, `pi coding turn failed: ${category}`);
    assert.equal(response.snapshot, undefined, 'failed turn must not publish a source snapshot');
    assert.equal(response.messages, undefined, 'raw provider diagnostics must not leak');
    assert.equal(JSON.stringify(response).includes('SECRET'), false);
  });
}
