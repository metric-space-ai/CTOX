import { readFileSync } from 'node:fs';

const source = readFileSync(new URL('./business-chat.js', import.meta.url), 'utf8');
const canonicalRenderCalls = source.match(/chatMessagesMarkup\(chat\.messages\)/g) || [];

if (canonicalRenderCalls.length !== 2) {
  throw new Error(
    `expected both full and in-place chat render paths to use canonical tracking markup, got ${canonicalRenderCalls.length}`,
  );
}
if (source.includes('chat.messages.map(messageMarkup)')) {
  throw new Error('a chat render path still bypasses tracking-control deduplication');
}

console.log('business chat render paths use canonical tracking markup');
