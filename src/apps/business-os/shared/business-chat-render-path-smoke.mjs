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
if (!/function chatMessagesMarkup\(messages = \[\]\) \{\s*return messages\.map\(\(message\) => messageMarkup\(message\)\)\.join\(''\);\s*\}/.test(source)) {
  throw new Error('chatMessagesMarkup is called by the render paths but is not defined');
}
if (!/const handleExternalOpen = async \(event\) => \{\s*const detail = event\.detail \|\| \{\};\s*await hydrateChatsFromRxDb/.test(source)) {
  throw new Error('opening a tracked task must hydrate the authoritative chat before resolving focus');
}

console.log('business chat render paths use canonical tracking markup');
