import assert from 'node:assert/strict';
import { nearestFreeGridPosition, reorderedIconIds, reorderTargetAtPoint } from '../iconDrag.js';

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

const grid = { offset: 24, cellW: 120, cellH: 140 };
assert.deepEqual(nearestFreeGridPosition({
  rawX: 152,
  rawY: 157,
  grid,
  maxX: 504,
  maxY: 444,
  iconWidth: 104,
  iconHeight: 120,
}), { x: 144, y: 164, distance: Math.hypot(8, 7) });

const occupiedSnap = nearestFreeGridPosition({
  rawX: 144,
  rawY: 164,
  grid,
  maxX: 504,
  maxY: 444,
  iconWidth: 104,
  iconHeight: 120,
  occupied: [{ x: 144, y: 164 }],
  blockedRects: [{ left: 250, top: 0, right: 520, bottom: 170 }],
});
assert.deepEqual(
  { x: occupiedSnap.x, y: occupiedSnap.y },
  { x: 24, y: 164 },
  'drop skips both an occupied icon cell and widget-covered cells',
);

console.log('ok - touch reorder derives stable order and geometric drop targets');
