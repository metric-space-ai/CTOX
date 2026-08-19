import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

const qaDir = dirname(fileURLToPath(import.meta.url));
const businessOsDir = join(qaDir, '..');
const css = readFileSync(join(businessOsDir, 'themes', 'ctox-desktop-shell.css'), 'utf8');
const appCss = readFileSync(join(businessOsDir, 'app.css'), 'utf8');
const baseCss = readFileSync(join(businessOsDir, 'shared', 'base.css'), 'utf8');
// The chat dock renders its stylesheet from this module, so its class grammar
// is part of the production surface the theme may restyle.
const chatJs = readFileSync(join(businessOsDir, 'shared', 'business-chat.js'), 'utf8');
// The desktop module owns the wallpaper surface the host theme flattens.
const desktopCss = readFileSync(join(businessOsDir, 'modules', 'desktop', 'index.css'), 'utf8');
const html = readFileSync(join(qaDir, 'ctox-desktop-shell.html'), 'utf8');
const js = readFileSync(join(qaDir, 'ctox-desktop-shell.js'), 'utf8');
const scope = 'html[data-desktop-host="ctox"]';

function styleRuleHeaders(source) {
  const stripped = source.replace(/\/\*[\s\S]*?\*\//g, '');
  const headers = [];

  for (let index = 0; index < stripped.length; index += 1) {
    if (stripped[index] !== '{') continue;
    let start = index - 1;
    while (start >= 0 && !'{};'.includes(stripped[start])) start -= 1;
    const header = stripped.slice(start + 1, index).trim();
    if (header) headers.push(header);
  }

  return headers.filter((header) => !header.startsWith('@'));
}

function splitSelectors(header) {
  const selectors = [];
  let depth = 0;
  let start = 0;

  for (let index = 0; index < header.length; index += 1) {
    if ('(['.includes(header[index])) depth += 1;
    if (')]'.includes(header[index])) depth -= 1;
    if (header[index] === ',' && depth === 0) {
      selectors.push(header.slice(start, index).trim());
      start = index + 1;
    }
  }

  selectors.push(header.slice(start).trim());
  return selectors;
}

test('theme is guarded by the CTOX desktop host scope', () => {
  assert.match(html, /<html[^>]+data-desktop-host="ctox"/);
  const selectors = styleRuleHeaders(css).flatMap(splitSelectors);
  assert.ok(selectors.length > 20, 'fixture theme should contain a substantive scoped layer');
  for (const selector of selectors) {
    assert.ok(selector.startsWith(scope), `unscoped theme selector: ${selector}`);
  }
  assert.doesNotMatch(css, /(^|[\s,{]):root\b/m);
});

test('theme reuses production shell and kit selectors', () => {
  const productionCss = `${appCss}\n${baseCss}\n${chatJs}\n${desktopCss}`;
  const classNames = new Set([...css.replace(/\/\*[\s\S]*?\*\//g, '').matchAll(/\.([a-zA-Z_][\w-]*)/g)].map((match) => match[1]));
  assert.ok(classNames.size > 10, 'theme should exercise the production shell grammar');
  for (const className of classNames) {
    assert.ok(productionCss.includes(`.${className}`), `theme introduces non-production class .${className}`);
  }
});

test('theme declares restrained light and dark primitive tokens', () => {
  assert.match(css, /html\[data-desktop-host="ctox"\]\[data-theme="light"\]\s*\{/);
  assert.match(css, /html\[data-desktop-host="ctox"\]\[data-theme="dark"\]\s*\{/);
  const lightBlock = css.match(/html\[data-desktop-host="ctox"\]\[data-theme="light"\]\s*\{[^}]*\}/)?.[0] ?? '';
  const darkBlock = css.match(/html\[data-desktop-host="ctox"\]\[data-theme="dark"\]\s*\{[^}]*\}/)?.[0] ?? '';
  for (const token of ['--bg', '--surface', '--surface-2', '--line', '--text', '--muted', '--accent', '--accent-soft']) {
    for (const [name, block] of [['light', lightBlock], ['dark', darkBlock]]) {
      assert.equal((block.match(new RegExp(`${token}:`, 'g')) || []).length, 1, `${token} must be defined exactly once in the ${name} token block`);
    }
  }
  assert.match(css, /font-family:\s*var\(--font-sans\)/);
  assert.match(css, /font-size:\s*13px/);
});

test('theme provides accessible focus, reduced motion, and host-width collapse', () => {
  assert.match(css, /:focus-visible/);
  assert.match(css, /outline:\s*2px solid var\(--accent\)/);
  assert.match(css, /@media\s*\(prefers-reduced-motion:\s*reduce\)/);
  assert.match(css, /animation-duration:\s*0\.01ms\s*!important/);
  assert.match(css, /container-name:\s*ctox-desktop-guest/);
  assert.match(css, /@container ctox-desktop-guest \(max-width:\s*840px\)/);
  assert.match(css, /@container ctox-desktop-guest \(max-width:\s*620px\)/);
  assert.match(css, /\.ctox-workspace > \.ctox-pane:last-child/);
  assert.doesNotMatch(css, /\b\d+(?:\.\d+)?(?:d|s|l)?v[wh]\b/i);
  assert.doesNotMatch(css, /position:\s*fixed/i);
});

test('theme and fixture use no gradients or remote assets', () => {
  const source = `${css}\n${html}\n${js}`;
  assert.doesNotMatch(source, /gradient\s*\(/i);
  assert.doesNotMatch(css, /url\s*\(/i);
  assert.doesNotMatch(source, /(?:https?:)?\/\//i);
  assert.doesNotMatch(html, /<(?:img|iframe|video|audio|source)\b/i);
  assert.doesNotMatch(js, /\b(?:fetch|XMLHttpRequest|WebSocket|EventSource)\b/);
  for (const asset of html.matchAll(/(?:href|src)="([^"]+)"/g)) {
    assert.match(asset[1], /^\.\.\/|^\.\//, `asset must be local: ${asset[1]}`);
  }
});

test('fixture excludes duplicate desktop mode and instance controls', () => {
  assert.doesNotMatch(html, /Code\s*(?:\||\/|and)\s*Business OS/i);
  assert.doesNotMatch(html, /data-(?:mode-switch|product-mode|instance-picker|instance-sidebar)\b/i);
  assert.doesNotMatch(html, /class="[^"]*(?:instance-sidebar|desktop-sidebar|mode-switch)[^"]*"/i);
  assert.doesNotMatch(html, /aria-label="[^"]*(?:choose|select|switch) instance[^"]*"/i);
  assert.doesNotMatch(html, /traffic[- ]lights?/i);
});

test('fixture proves the production Business OS shell and workspace grammar', () => {
  for (const className of [
    'app-shell',
    'topbar',
    'module-nav',
    'module-tabs',
    'module-tab',
    'workspace-frame',
    'workspace-pane-center',
    'module-root',
    'module-content',
    'ctox-workspace',
    'ctox-pane',
    'ctox-pane-band',
    'ctox-pane-body',
    'ctox-action-strip',
    'ctox-run-control',
    'ctox-table',
    'ctox-card',
    'ctox-fields',
  ]) {
    assert.match(html, new RegExp(`class="[^"]*\\b${className}\\b`), `fixture must render .${className}`);
  }

  assert.equal((html.match(/<(?:aside|section) class="ctox-pane"/g) || []).length, 3, 'fixture must render three workbench panes');
  assert.match(html, /aria-label="Inspector and activity"/);
  assert.match(html, />Assistant task</);
  assert.match(html, /data-resizer="left"/);
  assert.match(html, /data-resizer="right"/);
});

test('fixture-only script is deterministic and limited to QA toggles', () => {
  assert.match(js, /root\.dataset\.theme/);
  assert.match(js, /shell\.style\.width/);
  assert.match(js, /680px/);
  assert.doesNotMatch(js, /localStorage|sessionStorage|Date\(|Math\.random|setTimeout|setInterval/);
  assert.doesNotMatch(js, /import\s|export\s/);
});

test('theme homogenizes the chat dock into a seamless footer bar', () => {
  const dockRule = css.match(/html\[data-desktop-host="ctox"\] \.ctox-chat-dock,[\s\S]*?\{([\s\S]*?)\}/)?.[1] ?? '';
  assert.ok(dockRule, 'the host theme must restyle the chat dock');
  // Mirror of the topbar: full-width ground, hairline on top, no island.
  assert.match(dockRule, /width:\s*100%/);
  assert.match(dockRule, /border-top:\s*1px solid var\(--line\)/);
  assert.match(dockRule, /border-radius:\s*0/);
  assert.match(dockRule, /background:\s*var\(--bg\)/);
  assert.match(dockRule, /box-shadow:\s*none/);
  assert.match(dockRule, /min-height:\s*44px/);
  // The collapsed dock must not keep the expanded track widths, which is what
  // made the collapsed bar render at full expanded size in the host.
  assert.match(css, /\.ctox-chat-dock\.is-collapsed \{\s*grid-template-columns:/);
  assert.match(css, /\.ctox-chat-root\.is-collapsed/);
});

test('theme keeps the chat carousel and its host palette intact', () => {
  // REGRESSION GUARD: the host layer flattens shadows and transforms in several
  // places. It must never flatten the 3D chat stage, or the carousel of open
  // chat windows collapses into a single flat card under the desktop host.
  assert.doesNotMatch(css, /\.ctox-chat-stage[^{]*\{[^}]*(?:perspective|transform-style)\s*:/);
  assert.doesNotMatch(css, /\.ctox-chat-window(?!\.is-active)[^{,]*\{[^}]*transform:\s*none/);
  assert.doesNotMatch(css, /\.ctox-chat-window[^{,]*\{[^}]*(?:opacity|visibility)\s*:/);
  // The shell's per-module green/blue accents must be replaced on the window
  // too, not only on the chips — otherwise the window keeps a green border and
  // a green send button under the host palette.
  assert.match(css, /\.ctox-chat-window\[data-chat-module\][\s\S]{0,200}--accent:\s*var\(--ctox-host-accent/);
  assert.match(css, /\.ctox-chat-form button \{[^}]*background:\s*var\(--accent\)/);
  // A chip is a flat surface chip; only the active one carries the accent.
  const chipBase = css.match(/html\[data-desktop-host="ctox"\] \.ctox-chat-chip,\n[\s\S]*?\.is-task-idle\) \{([\s\S]*?)\}/)?.[1] ?? '';
  assert.match(chipBase, /background:\s*var\(--surface\)/);
  assert.match(chipBase, /border:\s*1px solid var\(--hairline\)/);
  assert.match(chipBase, /color:\s*var\(--text\)/);
  const chipActive = css.match(/\.ctox-chat-chip\.is-active \{([\s\S]*?)\}/)?.[1] ?? '';
  assert.match(chipActive, /background:\s*var\(--accent-soft\)/);
  assert.match(chipActive, /border-color:\s*var\(--accent\)/);
});
