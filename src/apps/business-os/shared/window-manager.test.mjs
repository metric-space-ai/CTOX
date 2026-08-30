import assert from 'node:assert/strict';
import test from 'node:test';

import { clampNormalWindowPosition, detectSnapZone } from './window-manager.js';

const viewport = {
  w: 1200,
  h: 900,
  left: 0,
  right: 0,
  top: 52,
  bottom: 70,
};

test('keeps a normal window completely inside the shell work area', () => {
  assert.deepEqual(clampNormalWindowPosition({
    left: 459,
    top: 240,
    width: 1200,
    height: 778,
  }, viewport), {
    left: 0,
    top: 52,
  });
});

test('clamps free dragging at every work-area edge', () => {
  assert.deepEqual(clampNormalWindowPosition({
    left: -400,
    top: -100,
    width: 640,
    height: 500,
  }, viewport), {
    left: 0,
    top: 52,
  });
  assert.deepEqual(clampNormalWindowPosition({
    left: 900,
    top: 700,
    width: 640,
    height: 500,
  }, viewport), {
    left: 560,
    top: 330,
  });
});

test('respects shell insets such as a visible side panel or chat rail', () => {
  assert.deepEqual(clampNormalWindowPosition({
    left: 900,
    top: 600,
    width: 700,
    height: 500,
  }, {
    ...viewport,
    left: 24,
    right: 280,
    bottom: 90,
  }), {
    left: 220,
    top: 310,
  });
});

test('detects all four snap edges', () => {
  assert.equal(detectSnapZone(15, 300, viewport), 'left');
  assert.equal(detectSnapZone(1185, 300, viewport), 'right');
  assert.equal(detectSnapZone(300, 67, viewport), 'top');
  assert.equal(detectSnapZone(300, 815, viewport), 'bottom');
});

test('detects all four snap corners only inside both edge bands', () => {
  assert.equal(detectSnapZone(15, 67, viewport), 'top-left');
  assert.equal(detectSnapZone(1185, 67, viewport), 'top-right');
  assert.equal(detectSnapZone(15, 815, viewport), 'bottom-left');
  assert.equal(detectSnapZone(1185, 815, viewport), 'bottom-right');

  // SNAP_CORNER is 60, but the effective corner band must not swallow the
  // 30px edge band when only one axis is actually near the corner.
  assert.equal(detectSnapZone(15, 100, viewport), 'left');
  assert.equal(detectSnapZone(45, 67, viewport), 'top');
});

test('left docking does not depend on a horizontal-only drag', () => {
  const dragStart = { x: 215, y: 150 };
  const pointer = { x: 15, y: 300 };
  assert.deepEqual({
    dx: pointer.x - dragStart.x,
    dy: pointer.y - dragStart.y,
  }, {
    dx: -200,
    dy: 150,
  });
  assert.equal(detectSnapZone(pointer.x, pointer.y, viewport), 'left');
});
