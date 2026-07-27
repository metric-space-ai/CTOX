import assert from 'node:assert/strict';
import test from 'node:test';

import { renderListOrState } from './list-state.js';

test('renderListOrState always renders existing rows before readiness state', () => {
  const rows = [{ id: 'row-1' }];
  const html = renderListOrState(rows, { ready: false }, {
    renderRows(receivedRows) {
      assert.equal(receivedRows, rows);
      return '<article data-row="row-1">Bestehender Datensatz</article>';
    },
    syncing: 'Noch nicht bereit',
  });

  assert.equal(html, '<article data-row="row-1">Bestehender Datensatz</article>');
});

test('renderListOrState renders the syncing shell for empty unready data', () => {
  const html = renderListOrState([], { ready: false });

  assert.match(html, /class="ctox-syncing"/);
  assert.match(html, /Daten werden synchronisiert\./);
  assert.doesNotMatch(html, /ctox-empty/);
});

test('renderListOrState renders the empty shell once the collection is ready', () => {
  const html = renderListOrState([], { ready: true }, { empty: 'Keine Kontakte.' });

  assert.equal(html, '<div class="ctox-empty">Keine Kontakte.</div>');
});

test('renderListOrState escapes state text but trusts rendered row markup', () => {
  const syncing = renderListOrState([], { ready: false }, {
    syncing: '<img src=x onerror="alert(1)"> & warten',
  });
  const empty = renderListOrState([], { ready: true }, {
    empty: `<script>alert('leer')</script>`,
  });
  const rows = renderListOrState([1], { ready: false }, {
    renderRows: () => '<strong data-safe="caller">Markup</strong>',
  });

  assert.match(syncing, /&lt;img src=x onerror=&quot;alert\(1\)&quot;&gt; &amp; warten/);
  assert.doesNotMatch(syncing, /<img/);
  assert.match(empty, /&lt;script&gt;alert\(&#39;leer&#39;\)&lt;\/script&gt;/);
  assert.doesNotMatch(empty, /<script/);
  assert.equal(rows, '<strong data-safe="caller">Markup</strong>');
});
