// SPDX-License-Identifier: MIT OR AGPL-3.0-only
import { readdir, readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { scanStringLiterals } from './audit-business-os-content.mjs';

const scriptPath = fileURLToPath(import.meta.url);
const shellRoot = path.resolve(path.dirname(scriptPath), '..');
const repoRoot = path.resolve(shellRoot, '../../..');
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

/**
 * These are retired user-facing labels. `shard` remains valid in internal
 * implementation names, so this list is checked only on public copy.
 */
export const FORBIDDEN_VISIBLE_TERMS = Object.freeze([
  'CTOX Business-OS-Shell',
  'Business-OS-Shell',
  'CTOX App Store',
  'Kandidaten-Shards',
  'Shard-Ansicht',
  'Shard view',
  'Listen-Ansicht',
]);

const modulesRoot = path.join(shellRoot, 'modules');
const PUBLIC_COPY_EXTENSIONS = new Set(['.html', '.js', '.mjs', '.json']);
const EXCLUDED_MODULE_DIRECTORIES = new Set([
  'customers',
  'desktop',
  'installed-modules',
  'local-modules',
  'mobile',
  'Mobile',
  'themes',
]);

function normalizePath(value) {
  return String(value).replaceAll('\\', '/');
}

function escapeRegExp(value) {
  return String(value).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function lineNumberAt(source, offset) {
  let line = 1;
  for (let index = 0; index < offset; index += 1) {
    if (source.charCodeAt(index) === 10) line += 1;
  }
  return line;
}

function lineTextAt(source, offset) {
  const start = source.lastIndexOf('\n', offset - 1) + 1;
  const end = source.indexOf('\n', offset);
  return source.slice(start, end === -1 ? source.length : end).trim();
}

function findMatches(value, baseOffset, source, relativePath) {
  const candidates = [];
  for (const term of FORBIDDEN_VISIBLE_TERMS) {
    const pattern = new RegExp(escapeRegExp(term), 'giu');
    for (const match of String(value).matchAll(pattern)) {
      const offset = baseOffset + (match.index ?? 0);
      candidates.push({ offset, end: offset + match[0].length, term });
    }
  }

  // The shorter Business-OS-Shell term overlaps the full CTOX variant. Keep
  // one diagnostic per visible occurrence, preferring the more specific term.
  candidates.sort((left, right) => left.offset - right.offset || right.end - left.end);
  const findings = [];
  for (const candidate of candidates) {
    if (findings.some(({ start, end }) => candidate.offset < end && candidate.end > start)) continue;
    findings.push({
      path: normalizePath(relativePath),
      line: lineNumberAt(source, candidate.offset),
      term: candidate.term,
      text: lineTextAt(source, candidate.offset),
      start: candidate.offset,
      end: candidate.end,
    });
  }
  return findings;
}

function scanHtmlCopy(source, relativePath) {
  const findings = [];
  // Keep offsets stable while removing comments, which are not rendered copy.
  const withoutComments = source.replace(/<!--[\s\S]*?-->/gu, (comment) => comment.replace(/[^\n]/gu, ' '));
  const tagPattern = /<[^>]*>/gu;
  let cursor = 0;
  for (const tag of withoutComments.matchAll(tagPattern)) {
    const tagStart = tag.index ?? 0;
    findings.push(...findMatches(withoutComments.slice(cursor, tagStart), cursor, source, relativePath));
    const tagText = tag[0];
    const attributePattern = /\b(?:aria-(?:label|description|roledescription|placeholder|valuetext|errormessage)|title|alt|placeholder)\s*=\s*(["'])(.*?)\1/giu;
    for (const attribute of tagText.matchAll(attributePattern)) {
      const valueStart = tagStart + (attribute.index ?? 0) + attribute[0].indexOf(attribute[2]);
      findings.push(...findMatches(attribute[2], valueStart, source, relativePath));
    }
    cursor = tagStart + tagText.length;
  }
  findings.push(...findMatches(withoutComments.slice(cursor), cursor, source, relativePath));
  return findings;
}

/** Find retired labels in rendered HTML or product-facing JS/JSON strings. */
export function findForbiddenVisibleTerms(source, relativePath) {
  const normalizedPath = normalizePath(relativePath);
  const extension = path.extname(normalizedPath).toLowerCase();
  if (!PUBLIC_COPY_EXTENSIONS.has(extension)) return [];
  if (extension === '.html') return scanHtmlCopy(String(source), normalizedPath);

  // Product-facing JS and JSON in this scope use quoted fallback/catalog copy.
  // The shared scanner deliberately ignores comments, regexes and CSS.
  const findings = [];
  for (const literal of scanStringLiterals(String(source))) {
    const valueStart = literal.start + 1;
    findings.push(...findMatches(literal.value, valueStart, String(source), normalizedPath));
  }
  return findings;
}

function isPublicModuleFile(relativePath, entryName) {
  const normalized = normalizePath(relativePath);
  const segments = normalized.split('/');
  if (segments.some((segment) => EXCLUDED_MODULE_DIRECTORIES.has(segment))) return false;
  if (entryName === 'registry.json' || entryName.endsWith('.test.js') || entryName.endsWith('.test.mjs')) return false;
  const extension = path.extname(entryName).toLowerCase();
  if (!PUBLIC_COPY_EXTENSIONS.has(extension)) return false;
  if (entryName === 'module.json') return true;
  if (segments.at(-2) === 'locales' && extension === '.json') return true;
  // Public module UI entry points are the only JS source files in scope.
  return segments.length === 6 && /^index\.(?:js|mjs|html)$/u.test(entryName);
}

async function collectPublicModuleFiles() {
  const files = [];
  async function visit(directory, relativeDirectory) {
    const entries = await readdir(directory, { withFileTypes: true });
    for (const entry of entries.sort((left, right) => left.name.localeCompare(right.name))) {
      const relativePath = `${relativeDirectory}/${entry.name}`;
      if (entry.isDirectory()) {
        if (EXCLUDED_MODULE_DIRECTORIES.has(entry.name)) continue;
        await visit(path.join(directory, entry.name), relativePath);
      } else if (entry.isFile() && isPublicModuleFile(relativePath, entry.name)) {
        files.push(relativePath);
      }
    }
  }
  await visit(modulesRoot, 'src/apps/business-os/modules');
  return files.sort();
}

export async function runBrandingGuard() {
  const findings = [];
  for (const relativePath of productSurfaces) {
    const source = await readFile(path.join(shellRoot, relativePath), 'utf8');
    for (const identity of forbidden) {
      if (source.includes(identity)) findings.push(`${relativePath}: ${identity}`);
    }
  }

  const moduleFiles = await collectPublicModuleFiles();
  for (const relativePath of moduleFiles) {
    const source = await readFile(path.resolve(repoRoot, relativePath), 'utf8');
    for (const finding of findForbiddenVisibleTerms(source, relativePath)) {
      findings.push(`${finding.path}:${finding.line} [${finding.term}] ${finding.text}`);
    }
  }

  if (findings.length > 0) {
    throw new Error(`Legacy product identity or visible copy found in Business OS release surfaces:\n${findings.join('\n')}`);
  }
  return { filesAudited: productSurfaces.length + moduleFiles.length, moduleFiles };
}

if (process.argv[1] && path.resolve(process.argv[1]) === scriptPath) {
  try {
    await runBrandingGuard();
    console.log('ok - Business OS release surfaces use Workjet product identity');
  } catch (error) {
    console.error(error.message);
    process.exitCode = 1;
  }
}
