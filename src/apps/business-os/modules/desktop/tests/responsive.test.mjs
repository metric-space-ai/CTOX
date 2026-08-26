import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const css = readFileSync(new URL('../index.css', import.meta.url), 'utf8');
const js = readFileSync(new URL('../index.js', import.meta.url), 'utf8');

assert.match(css, /\.desktop-module\s*\{[\s\S]*container-type:\s*inline-size/);
assert.match(css, /@container \(max-width: 1320px\)\s*\{[\s\S]*\.desktop-widget-container\s*\{[\s\S]*display:\s*none/);
assert.match(css, /\.desktop-hero-widget\s*\{[\s\S]*width:\s*248px/);
assert.doesNotMatch(css, /\.desktop-hero-widget\s*\{[\s\S]*width:\s*334px/);

const selectedIconRule = css.match(/\.desktop-icon\.selected\s*\{([^}]*)\}/u)?.[1] ?? '';
assert.match(selectedIconRule, /background:\s*transparent/u);
assert.match(selectedIconRule, /box-shadow:\s*none/u);
assert.doesNotMatch(selectedIconRule, /border-color/u);

const glyphRule = css.match(/\.desktop-icon \.desktop-icon-glyph\s*\{([^}]*)\}/u)?.[1] ?? '';
assert.match(glyphRule, /background:\s*transparent\s*!important/u);
assert.match(glyphRule, /border:\s*0\s*!important/u);
assert.match(glyphRule, /box-shadow:\s*none\s*!important/u);
assert.doesNotMatch(
  css,
  /:root\[data-shell-style="ctox"\] \.desktop-icon(?::hover)? \.desktop-icon-glyph/u,
  'a shell theme must not reintroduce a second tile around app artwork',
);

const selectedLabelRule = css.match(/\.desktop-icon\.selected \.desktop-icon-label\s*\{([^}]*)\}/u)?.[1] ?? '';
assert.match(selectedLabelRule, /background:\s*transparent/u);
assert.match(css, /--desktop-icon-cell-h:\s*116px/u);
assert.match(css, /\.desktop-icon:focus-visible \.desktop-icon-glyph/u);
assert.match(js, /height:\s*116,/u, 'drag bounds use the rendered desktop icon height');
assert.match(js, /return 'CTOX Backend';/u);
assert.match(js, /el\.setAttribute\('role', 'button'\)/u);
assert.match(js, /el\.setAttribute\('aria-label', label\)/u);

console.log('ok - desktop icons keep one unboxed surface and unclipped compact rows');
