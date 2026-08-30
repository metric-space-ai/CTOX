import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const source = await readFile(new URL('../app.js', import.meta.url), 'utf8');
const registerCustomModuleIcons = source.match(
  /async function registerCustomModuleIcons\(\) \{[\s\S]*?\n\}/,
)?.[0] || '';

assert.match(
  registerCustomModuleIcons,
  /const inlineSvg = inlineModuleIconSvg\(mod\)/,
  'inline module icons should still be registered during startup',
);
assert.match(
  registerCustomModuleIcons,
  /void resolveModuleIconSvg\(mod\)/,
  'external module icons must be loaded in the background',
);
assert.doesNotMatch(
  registerCustomModuleIcons,
  /await resolveModuleIconSvg\(mod\)/,
  'an optional external icon request must not block the workspace bootstrap',
);

console.log('module icon startup contract OK');
