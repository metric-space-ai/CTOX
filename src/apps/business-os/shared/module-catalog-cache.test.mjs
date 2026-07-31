import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const source = await readFile(new URL('../app.js', import.meta.url), 'utf8');
const loader = source.match(/async function loadPackagedModuleCatalog\(\)[\s\S]*?\n\}/)?.[0] || '';
const initialOpen = source.match(/try \{\n\s+const workspaceSession[\s\S]*?flushDeferredCatalogRefresh\(\);\n\s+\}/)?.[0] || '';
const openModule = source.match(/async function openModule\(moduleId, options = \{\}\)[\s\S]*?\n\}/)?.[0] || '';
const openWindowedModule = source.match(/async function openWindowedModule\(mod, options = \{\}\)[\s\S]*?\n\}/)?.[0] || '';

assert.match(loader, /modules\/registry\.json/);
assert.match(loader, /cache: 'no-store'/);
assert.doesNotMatch(
  loader,
  /cache: 'force-cache'/,
  'runtime-installed module releases must not be hidden behind the shell build cache',
);
assert.match(
  loader,
  /const explicitlyAllowedIds = resolveModuleAllowlist\(\)/,
  'the tenant allowlist must make selected packaged apps available without a runtime install',
);
assert.match(
  initialOpen,
  /state\.initialModuleOpened = true;[\s\S]*flushDeferredCatalogRefresh\(\);/,
  'a runtime app that arrives after the first route attempt must not leave catalog refreshes deferred forever',
);
assert.doesNotMatch(
  initialOpen,
  /state\.initialModuleOpened = Boolean\(state\.activeModule\?\.id\)/,
  'catalog refresh readiness describes shell construction, not whether the first requested app already existed',
);
assert.match(
  loader,
  /canonicalSystemIds\.has\(id\) \|\| explicitlyAllowedIds\.has\(id\)/,
  'packaged catalog visibility must stay limited to system apps and explicit tenant selections',
);
assert.match(
  openModule,
  /force: options\.force === true/,
  'a catalog refresh must forward forced remounts into windowed desktop apps',
);
assert.match(
  openWindowedModule,
  /if \(existing && options\.force === true\)[\s\S]*state\.windowManager\.destroy\(existing\.id\)[\s\S]*while \(findDesktopWindow\(mod\.id\)/,
  'a forced windowed-app refresh must close the stale window before importing the new module revision',
);

console.log('runtime module catalog cache contract OK');
