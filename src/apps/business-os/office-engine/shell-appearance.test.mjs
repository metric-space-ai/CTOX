import test from 'node:test';
import assert from 'node:assert/strict';
import { applyShellAppearance, readShellAppearance, observeShellAppearance } from './src/shell-appearance.mjs';

function environment() {
  const values = new Map([['--accent', '#a855f7'], ['--surface', '#111111']]);
  const applied = new Map();
  const frames = new Map();
  const observed = [];
  let changed;
  let disconnected = false;
  let mediaListener;
  const media = { matches: true, addEventListener(_, fn) { mediaListener = fn; }, removeEventListener(_, fn) { assert.equal(fn, mediaListener); mediaListener = null; } };
  const view = {
    getComputedStyle: () => ({ fontFamily: 'system-ui', getPropertyValue: name => values.get(name) || '' }),
    matchMedia: () => media,
    CSS: { supports: (_, value) => /^#[0-9a-f]{6}$/i.test(value) },
    requestAnimationFrame(fn) { frames.set(1, fn); return 1; },
    cancelAnimationFrame(id) { frames.delete(id); },
    MutationObserver: class {
      constructor(fn) { changed = fn; }
      observe(node, options) { observed.push({ node, options }); }
      disconnect() { disconnected = true; }
    },
  };
  const root = { dataset: { theme: 'light' }, style: { setProperty(name, value) { applied.set(name, value); }, removeProperty(name) { applied.delete(name); } }, parentElement: null };
  const document = { defaultView: view, documentElement: root, head: {} };
  const host = { ownerDocument: document, parentElement: root };
  return { values, applied, frames, observed, document, host, changed: () => changed(), disconnected: () => disconnected, mediaListener: () => mediaListener };
}

test('Office reads the host palette and explicit shell theme before the system fallback', () => {
  const env = environment();
  const appearance = readShellAppearance(env.host);
  assert.equal(appearance.theme, 'light');
  assert.equal(appearance.tokens.accent, '#a855f7');
  delete env.document.documentElement.dataset.theme;
  assert.equal(readShellAppearance(env.host).theme, 'dark');
  assert.equal(readShellAppearance(env.host, 'light').theme, 'light');
});

test('Office appearance accepts only whitelisted color properties and clears stale values', () => {
  const env = environment();
  applyShellAppearance(env.document, { theme: 'dark', tokens: { accent: '#a855f7', surface: 'url(https://invalid.test)', arbitrary: '#111111' }, fontFamily: 'system-ui' });
  assert.equal(env.document.documentElement.style.colorScheme, 'dark');
  assert.equal(env.applied.get('--ctox-shell-accent'), '#a855f7');
  assert.equal(env.applied.has('--ctox-shell-surface'), false);
  assert.equal(env.applied.has('--ctox-shell-arbitrary'), false);
  applyShellAppearance(env.document, {});
  assert.equal(env.applied.size, 0);
});

test('Office appearance observers coalesce changes, publish changed values only, and clean up', () => {
  const env = environment();
  const published = [];
  const stop = observeShellAppearance(env.host, 'system', value => published.push(value));
  assert.equal(published.length, 1);
  assert.equal(env.observed.length, 3);
  env.changed(); env.changed();
  assert.equal(env.frames.size, 1);
  env.frames.get(1)();
  assert.equal(published.length, 1);
  env.values.set('--accent', '#123456');
  env.changed(); env.frames.get(1)();
  assert.equal(published.at(-1).tokens.accent, '#123456');
  env.changed(); stop();
  assert.equal(env.disconnected(), true);
  assert.equal(env.frames.size, 0);
  assert.equal(env.mediaListener(), null);
});
