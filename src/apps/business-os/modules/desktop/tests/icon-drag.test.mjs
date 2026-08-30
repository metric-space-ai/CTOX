import assert from 'node:assert/strict';
import { reorderedIconIds, reorderTargetAtPoint } from '../iconDrag.js';

assert.deepEqual(
  reorderedIconIds(['mail', 'threads', 'files', 'browser'], 'mail', 'files'),
  ['threads', 'files', 'mail', 'browser'],
);
assert.deepEqual(
  reorderedIconIds(['mail', 'threads', 'files'], 'files', 'mail'),
  ['files', 'mail', 'threads'],
);
assert.deepEqual(
  reorderedIconIds(['mail', 'threads', 'files'], 'threads', 'threads'),
  ['mail', 'threads', 'files'],
);
assert.deepEqual(
  reorderedIconIds(['mail', 'threads', 'mail', 'files'], 'mail', 'files'),
  ['threads', 'files', 'mail'],
);

const parent = {};
const icon = (id, left, top, width = 80, height = 96) => ({
  dataset: { iconId: id },
  parentElement: parent,
  getBoundingClientRect: () => ({ left, top, width, height, right: left + width, bottom: top + height }),
});
const dragged = icon('mail', 0, 0);
const threads = icon('threads', 100, 0);
const files = icon('files', 200, 0);
parent.querySelectorAll = () => [dragged, threads, files];

assert.equal(reorderTargetAtPoint(parent, dragged, 140, 40), threads);
assert.equal(reorderTargetAtPoint(parent, dragged, 191, 40), files);
assert.equal(reorderTargetAtPoint(parent, dragged, 500, 500), null);

console.log('ok - touch reorder derives stable order and geometric drop targets');
