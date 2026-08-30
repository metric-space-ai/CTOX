import assert from 'node:assert/strict';
import test from 'node:test';

import {
  clampNormalWindowPosition,
  defaultWindowPosition,
  detectSnapZone,
  shellV2RenderedIconSizeFromAnchor,
  shellV2FramePaletteFromRgba,
  shellV2FrameSampleAt,
  shellV2MorphFrameData,
} from './window-manager.js';

test('derives the shell-v2 frame palette from raster pixels, not a category colour', () => {
  const rgba = new Uint8ClampedArray([
    ...Array(18).fill([240, 183, 116, 255]).flat(),
    ...Array(22).fill([147, 107, 99, 255]).flat(),
    ...Array(24).fill([113, 83, 101, 255]).flat(),
    ...Array(20).fill([15, 42, 76, 255]).flat(),
    ...Array(30).fill([240, 240, 240, 255]).flat(),
  ]);
  const palette = shellV2FramePaletteFromRgba(rgba);
  assert.ok(palette);
  for (const value of Object.values(palette)) assert.match(value, /^#[0-9a-f]{6}$/);
  assert.notEqual(palette.accent, '#75d7c2');
  assert.notEqual(palette.start, '#f0f0f0');
  assert.notEqual(palette.start, palette.end);
});

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

test('centres a fresh shell-v2 pilot in the actual work area', () => {
  assert.deepEqual(defaultWindowPosition({
    shellContract: 'v2',
    width: 1490,
    height: 820,
    cascadeOffset: 44,
  }, {
    w: 1920,
    h: 1080,
    left: 0,
    right: 0,
    top: 48,
    bottom: 48,
  }), {
    left: 215,
    top: 130,
  });
});

test('keeps the established cascade for shell-v1 windows', () => {
  assert.deepEqual(defaultWindowPosition({
    shellContract: 'v1',
    width: 800,
    height: 600,
    cascadeOffset: 44,
  }, viewport), {
    left: 124,
    top: 104,
  });
});

test('uses the rendered launcher glyph, never the 128px source canvas, for shell v2', () => {
  assert.equal(shellV2RenderedIconSizeFromAnchor({ width: 64, height: 64 }), 64);
  assert.equal(shellV2RenderedIconSizeFromAnchor({ width: 56, height: 56 }), 56);
  assert.equal(shellV2RenderedIconSizeFromAnchor({ width: 128, height: 120 }), null);
  assert.equal(shellV2RenderedIconSizeFromAnchor({ width: 0, height: 0 }), null);
});

test('samples local accents from the nearest painted frame edge and its icon joint', () => {
  const iconX = 128 / 1490;
  const iconY = 128 / 820;
  assert.deepEqual(shellV2FrameSampleAt(0, 0, iconX, iconY), { from: 'start', to: 'topJoint', amount: 0 });
  assert.deepEqual(shellV2FrameSampleAt(iconX, 0, iconX, iconY), { from: 'start', to: 'topJoint', amount: 1 });
  assert.deepEqual(shellV2FrameSampleAt(1, 0, iconX, iconY), { from: 'topJoint', to: 'end', amount: 1 });
  assert.deepEqual(shellV2FrameSampleAt(0, iconY, iconX, iconY), { from: 'start', to: 'leftJoint', amount: 1 });
  assert.deepEqual(shellV2FrameSampleAt(0, 0.5, iconX, iconY), { from: 'leftJoint', to: 'leftJoint', amount: 1 });
  assert.deepEqual(shellV2FrameSampleAt(0, 1, iconX, iconY), { from: 'leftJoint', to: 'end', amount: 0 });
  assert.deepEqual(shellV2FrameSampleAt(1, 1, iconX, iconY), { from: 'end', to: 'end', amount: 1 });
});

test('shell-v2 morph is endpoint-exact, spline-like and geometrically fuses brackets into the icon', () => {
  const finalRect = { left: 215, top: 130, width: 1490, height: 820 };
  const anchor = { left: 74, top: 210, width: 128, height: 128, radius: 28 };
  const frames = shellV2MorphFrameData(finalRect, anchor);
  assert.ok(frames.length >= 120, 'spline sampling stays above a 120 Hz display cadence');
  assert.deepEqual(frames[0].point, { left: anchor.left, top: anchor.top });
  assert.deepEqual(frames.at(-1).point, { left: finalRect.left, top: finalRect.top });
  assert.equal(frames[0].scaleX, anchor.width / finalRect.width);
  assert.equal(frames[0].scaleY, anchor.height / finalRect.height);
  assert.equal(frames[0].width, anchor.width);
  assert.equal(frames[0].height, anchor.height);
  assert.equal(frames.at(-1).scaleX, 1);
  assert.equal(frames.at(-1).scaleY, 1);
  assert.equal(frames.at(-1).width, finalRect.width);
  assert.equal(frames.at(-1).height, finalRect.height);
  assert.equal(frames[0].radius, anchor.radius);
  assert.equal(frames.find((frame) => frame.amount >= 0.17).radius, 0);
  assert.equal(frames.at(-1).radius, 0);
  assert.equal(frames[0].contentOpacity, 0);
  assert.equal(frames.find((frame) => frame.amount >= 0.7).contentOpacity, 0);
  assert.equal(frames.at(-1).contentOpacity, 1);
  assert.equal(frames[0].cornerScale, 0);
  assert.equal(frames[0].cornerOpacity, 0);
  assert.equal(frames[0].radius, anchor.radius);
  assert.equal(frames.find((frame) => frame.amount >= 0.17).cornerScale, 1);
  assert.equal(frames.find((frame) => frame.amount >= 0.17).cornerOpacity, 1);
  assert.equal(frames.at(-1).cornerScale, 1);
  assert.equal(frames.at(-1).cornerOpacity, 1);
  for (let index = 1; index < frames.length; index += 1) {
    assert.ok(frames[index].scaleX >= frames[index - 1].scaleX);
    assert.ok(frames[index].scaleY >= frames[index - 1].scaleY);
    assert.ok(frames[index].cornerOpacity >= frames[index - 1].cornerOpacity);
    assert.ok(frames[index].cornerOpacity >= 0 && frames[index].cornerOpacity <= 1);
  }
  // The quadratic control point bends the path: its midpoint must not land
  // on the straight-line midpoint between icon and final window.
  const midpoint = frames.find((frame) => frame.amount >= 0.5).point;
  assert.notEqual(midpoint.top, (anchor.top + finalRect.top) / 2);
  const distance = (point, target) => Math.hypot(point.left - target.left, point.top - target.top);
  assert.ok(distance(midpoint, anchor) < distance(midpoint, finalRect));
});
