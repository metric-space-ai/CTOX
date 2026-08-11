import assert from 'node:assert/strict';
import test from 'node:test';

import {
  detailPaneHasActiveInput,
  preserveScrollDuring,
  renderHtmlIfChanged,
  replaceChildrenIfChanged,
} from './stable-dom.js';

function el(attrs = {}) {
  return {
    innerHTML: attrs.innerHTML || '',
    scrollTop: attrs.scrollTop ?? 0,
    scrollLeft: attrs.scrollLeft ?? 0,
    dataset: { ...(attrs.dataset || {}) },
    children: attrs.children || [],
    contains(node) { return Boolean(attrs.contains?.(node)); },
    replaceChildren(...nodes) {
      this.children = nodes;
      this.innerHTML = nodes.map((node) => node?.outerHTML || String(node ?? '')).join('');
    },
  };
}

test('renderHtmlIfChanged skips an unchanged signature and keeps scroll', () => {
  const well = el({ scrollTop: 420, innerHTML: '<div>a</div>' });
  well.dataset.ctoxRenderSig = 'sig-a';

  const first = renderHtmlIfChanged(well, '<div>a</div>', { signature: 'sig-a' });
  assert.equal(first, false);
  assert.equal(well.scrollTop, 420);
  assert.equal(well.innerHTML, '<div>a</div>');

  const second = renderHtmlIfChanged(well, '<div>b</div>', { signature: 'sig-b' });
  assert.equal(second, true);
  assert.equal(well.innerHTML, '<div>b</div>');
  assert.equal(well.dataset.ctoxRenderSig, 'sig-b');
  assert.equal(well.scrollTop, 420);
});

test('replaceChildrenIfChanged preserves scroll across a real rebuild', () => {
  const well = el({ scrollTop: 880, scrollLeft: 12 });
  const changed = replaceChildrenIfChanged(well, [{ outerHTML: '<li>1</li>' }], {
    signature: 'rows:1',
  });
  assert.equal(changed, true);
  assert.equal(well.scrollTop, 880);
  assert.equal(well.scrollLeft, 12);

  const skipped = replaceChildrenIfChanged(well, [{ outerHTML: '<li>2</li>' }], {
    signature: 'rows:1',
  });
  assert.equal(skipped, false);
  assert.equal(well.children[0].outerHTML, '<li>1</li>');
});

test('preserveScrollDuring restores offsets after a rebuild callback', () => {
  const well = el({ scrollTop: 300 });
  preserveScrollDuring(well, () => {
    well.scrollTop = 0;
    well.innerHTML = '<p>neu</p>';
  });
  assert.equal(well.scrollTop, 300);
  assert.equal(well.innerHTML, '<p>neu</p>');
});

test('detailPaneHasActiveInput reports focused editable controls inside the pane', () => {
  const input = { tagName: 'INPUT', isContentEditable: false };
  const pane = {
    contains(node) { return node === input; },
  };
  const previous = globalThis.document;
  globalThis.document = { activeElement: input };
  try {
    assert.equal(detailPaneHasActiveInput(pane), true);
    globalThis.document.activeElement = { tagName: 'DIV', isContentEditable: false };
    assert.equal(detailPaneHasActiveInput(pane), false);
  } finally {
    if (previous === undefined) delete globalThis.document;
    else globalThis.document = previous;
  }
});
