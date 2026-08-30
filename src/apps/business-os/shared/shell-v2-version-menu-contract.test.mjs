import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const appSource = await readFile(new URL('../app.js', import.meta.url), 'utf8');

assert.match(appSource, /commandType:\s*'ctox\.module\.list_versions'[\s\S]*?until:\s*'terminal'/);
assert.match(appSource, /commandType:\s*'ctox\.module\.rollback_version'[\s\S]*?until:\s*'terminal'/);
assert.match(appSource, /BusinessOsPermissions\.AppsModify/);
assert.match(appSource, /BusinessOsPermissions\.AppsRollback/);
assert.match(appSource, /canViewModuleSource\(mod\)/);
assert.match(appSource, /document\.addEventListener\('keydown', keydown, true\)/);
assert.match(appSource, /document\.removeEventListener\('keydown', keydown, true\)/);
assert.match(appSource, /state\.eventBus\?\.on\?\.\('window:closing', closeMenuForWindow\)/);
assert.match(appSource, /state\.eventBus\?\.off\?\.\('window:closing', closingToken\)/);
assert.match(appSource, /event\.key === 'Escape'/);
assert.match(appSource, /\['ArrowDown', 'ArrowUp'\]\.includes\(event\.key\)/);
assert.match(appSource, /if \(busy\) return/);
assert.match(appSource, /workspaceContext:\s*\{ source: 'shell-v2-version-menu'/);
assert.match(appSource, /await openModuleSourceEditor\(mod\.id\)/);
assert.doesNotMatch(appSource, /<div><span>Knowledge<\/span>/);
assert.doesNotMatch(appSource, /<p>Knowledge wirklich auf Version/);

console.log('Business OS shell-v2 version/source/coding menu contract OK');
