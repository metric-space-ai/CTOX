import assert from 'node:assert/strict';
import {
  positionIntersectsReservedRects,
  positionOutsideReservedRects,
  rowMajorGridPosition,
} from '../iconLayout.js';

const grid = { cellW: 120, cellH: 140, offset: 24 };
const iconSize = { width: 112, height: 136 };
const widget = { left: 480, top: 12, right: 748, bottom: 332 };

assert.deepEqual(
  rowMajorGridPosition(2, { grid, surfaceWidth: 760, iconSize, reservedRects: [widget] }),
  { x: 264, y: 24 },
);
assert.deepEqual(
  rowMajorGridPosition(3, { grid, surfaceWidth: 760, iconSize, reservedRects: [widget] }),
  { x: 24, y: 164 },
);

const droppedUnderWidget = { x: 560, y: 40 };
const normalizedDrop = positionOutsideReservedRects(droppedUnderWidget, {
  iconSize,
  bounds: { minX: 24, minY: 24, maxX: 640, maxY: 560 },
  reservedRects: [widget],
});
assert.equal(positionIntersectsReservedRects(normalizedDrop, iconSize, [widget]), false);
assert.notDeepEqual(normalizedDrop, droppedUnderWidget);

const clearPosition = { x: 144, y: 304 };
assert.deepEqual(
  positionOutsideReservedRects(clearPosition, {
    iconSize,
    bounds: { minX: 24, minY: 24, maxX: 640, maxY: 560 },
    reservedRects: [widget],
  }),
  clearPosition,
);

console.log('ok - desktop icon layout reserves visible widget rectangles');
