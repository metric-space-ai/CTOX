import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import test from 'node:test';

const businessOsRoot = join(import.meta.dirname, '..');
const css = readFileSync(join(businessOsRoot, 'app.css'), 'utf8');
const index = readFileSync(join(businessOsRoot, 'index.html'), 'utf8');

test('compact shell keeps start, app navigation, and account actions in one header row', () => {
  const responsiveContract = css.slice(css.lastIndexOf('@media (max-width: 900px)'));
  const moduleNavRule = responsiveContract.match(/\.module-nav\s*\{([^}]*)\}/)?.[1] || '';
  assert.match(responsiveContract, /grid-template-columns:\s*auto minmax\(0, 1fr\) auto/);
  assert.match(responsiveContract, /grid-template-rows:\s*40px/);
  assert.match(responsiveContract, /\.module-nav\s*\{[\s\S]*display:\s*grid[\s\S]*grid-column:\s*2[\s\S]*grid-row:\s*1/);
  assert.match(responsiveContract, /\.topbar-actions\s*\{[\s\S]*grid-column:\s*3[\s\S]*grid-row:\s*1/);
  assert.match(moduleNavRule, /display:\s*grid/);
  assert.doesNotMatch(moduleNavRule, /display:\s*none/);
});

test('responsive shell CSS is cache-bound to the current application build', () => {
  const appBuild = readFileSync(join(businessOsRoot, 'app.js'), 'utf8')
    .match(/const APP_BUILD = '([^']+)'/)?.[1];
  const cssBuild = index.match(/app\.css\?v=([^"']+)/)?.[1];
  assert.ok(appBuild);
  assert.equal(cssBuild, appBuild);
});
