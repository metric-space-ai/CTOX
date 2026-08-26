// SPDX-License-Identifier: MIT OR AGPL-3.0-only
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const shellRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const productSurfaces = [
  'index.html',
  'app.js',
  'app.css',
  'mobile-host.js',
  'mobile-host.css',
  'qa/ctox-desktop-shell.html',
];
const forbidden = [
  'CTOX Desktop',
  'CTOX Desktop App',
  'CTOX Mobile',
  'CTOX Business OS App',
  'T3 Code',
  'Workjet Alpha',
];

const findings = [];
for (const relativePath of productSurfaces) {
  const source = await readFile(path.join(shellRoot, relativePath), 'utf8');
  for (const identity of forbidden) {
    if (source.includes(identity)) findings.push(`${relativePath}: ${identity}`);
  }
}

if (findings.length > 0) {
  throw new Error(`Legacy product identity found in Business OS release surfaces:\n${findings.join('\n')}`);
}

console.log('ok - Business OS release surfaces use Workjet product identity');
