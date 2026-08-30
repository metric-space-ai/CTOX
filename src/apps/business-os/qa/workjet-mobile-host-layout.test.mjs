import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

const qaDir = dirname(fileURLToPath(import.meta.url));
const mobileCss = readFileSync(join(qaDir, '..', 'mobile-host.css'), 'utf8');

test('the Workjet mobile host always uses a touch grid instead of desktop coordinates', () => {
  assert.match(
    mobileCss,
    /html\[data-workjet-mobile-host="true"\] \.desktop-icons\s*\{[^}]*display:\s*grid;/s,
  );
  assert.match(
    mobileCss,
    /html\[data-workjet-mobile-host="true"\] \.desktop-icon\s*\{[^}]*position:\s*relative;[^}]*left:\s*auto\s*!important;[^}]*top:\s*auto\s*!important;/s,
  );
  assert.match(mobileCss, /@media \(min-width: 600px\)[\s\S]*repeat\(5,/);
  assert.match(mobileCss, /@media \(min-width: 840px\)[\s\S]*repeat\(6,/);
});

test('the Workjet mobile desktop pans horizontally and vertically without a corner handle', () => {
  const iconLayer = mobileCss.match(
    /html\[data-workjet-mobile-host="true"\] \.desktop-icons\s*\{([^}]*)\}/u,
  )?.[1] ?? '';
  const iconCell = mobileCss.match(
    /html\[data-workjet-mobile-host="true"\] \.desktop-icon\s*\{([^}]*)\}/u,
  )?.[1] ?? '';
  assert.match(iconLayer, /overflow-x:\s*auto/u);
  assert.match(iconLayer, /overflow-y:\s*auto/u);
  assert.match(iconLayer, /touch-action:\s*pan-x pan-y/u);
  assert.match(iconCell, /touch-action:\s*pan-x pan-y/u);
  assert.doesNotMatch(iconLayer, /overflow-x:\s*hidden/u);
  assert.match(mobileCss, /\.desktop-icon:focus-visible[\s\S]*?box-shadow:\s*none !important/u);
  assert.match(mobileCss, /\.desktop-icon\.selected \.desktop-icon-glyph[\s\S]*?outline:\s*0 !important/u);
});

test('the Workjet mobile host keeps icon labels and controls touch-readable', () => {
  assert.match(mobileCss, /--desktop-mobile-icon-size:\s*64px/);
  assert.match(mobileCss, /\.desktop-icon-label\s*\{[^}]*font-size:\s*12px/s);
  assert.match(mobileCss, /:where\(button, select, input, \[role="button"\]\)\s*\{[^}]*min-height:\s*44px/s);
});
