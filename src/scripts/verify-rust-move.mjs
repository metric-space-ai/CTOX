#!/usr/bin/env node

// Verifies that named Rust function bodies survived a move unchanged.
// Signatures, visibility and imports may change; normalized bodies may not.

import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { resolve } from 'node:path';

function parseArgs(argv) {
  const options = { beforeRef: 'HEAD', beforePath: '', afterPath: '', symbols: [], allFunctions: false };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    const next = argv[index + 1];
    if (arg === '--before-ref') options.beforeRef = String(next || 'HEAD');
    else if (arg === '--before-path') options.beforePath = String(next || '');
    else if (arg === '--after-path') options.afterPath = String(next || '');
    else if (arg === '--symbols') {
      options.symbols = String(next || '').split(',').map((value) => value.trim()).filter(Boolean);
    } else if (arg === '--all-functions') {
      options.allFunctions = true;
      continue;
    } else continue;
    index += 1;
  }
  if (!options.beforePath || !options.afterPath || (!options.allFunctions && options.symbols.length === 0)) {
    throw new Error(
      'usage: verify-rust-move.mjs --before-ref <ref> --before-path <path> '
      + '--after-path <path> (--symbols <fn,fn,...> | --all-functions)',
    );
  }
  return options;
}

function rustSourceAt(ref, path) {
  return execFileSync('git', ['show', `${ref}:${path}`], {
    encoding: 'utf8',
    maxBuffer: 32 * 1024 * 1024,
  });
}

function functionBody(source, symbol) {
  const escaped = symbol.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const matcher = new RegExp(`(?:^|\\n)\\s*(?:pub(?:\\([^)]*\\))?\\s+)?(?:async\\s+)?fn\\s+${escaped}\\s*(?:<[^>{}]*>)?\\s*\\(`, 'g');
  const match = matcher.exec(source);
  if (!match) throw new Error(`function ${symbol} not found`);
  const bodyStart = findOpeningBrace(source, matcher.lastIndex);
  const bodyEnd = matchingBrace(source, bodyStart);
  return source.slice(bodyStart + 1, bodyEnd);
}

function findOpeningBrace(source, from) {
  let parens = 1;
  let angleDepth = 0;
  const state = lexerState();
  for (let index = from; index < source.length; index += 1) {
    const consumed = consumeNonCode(source, index, state);
    if (consumed !== index) {
      index = consumed - 1;
      continue;
    }
    const char = source[index];
    if (char === '(') parens += 1;
    else if (char === ')') parens -= 1;
    else if (char === '<') angleDepth += 1;
    else if (char === '>' && angleDepth > 0) angleDepth -= 1;
    else if (char === '{' && parens === 0 && angleDepth === 0) return index;
    else if (char === ';' && parens === 0) break;
  }
  throw new Error('function body opening brace not found');
}

function matchingBrace(source, opening) {
  let depth = 0;
  const state = lexerState();
  for (let index = opening; index < source.length; index += 1) {
    const consumed = consumeNonCode(source, index, state);
    if (consumed !== index) {
      index = consumed - 1;
      continue;
    }
    if (source[index] === '{') depth += 1;
    else if (source[index] === '}') {
      depth -= 1;
      if (depth === 0) return index;
    }
  }
  throw new Error('function body closing brace not found');
}

function lexerState() {
  return { blockCommentDepth: 0 };
}

function consumeNonCode(source, index, state) {
  if (state.blockCommentDepth > 0) {
    if (source.startsWith('/*', index)) state.blockCommentDepth += 1;
    else if (source.startsWith('*/', index)) state.blockCommentDepth -= 1;
    return index + (source.startsWith('/*', index) || source.startsWith('*/', index) ? 2 : 1);
  }
  if (source.startsWith('//', index)) {
    const end = source.indexOf('\n', index + 2);
    return end < 0 ? source.length : end;
  }
  if (source.startsWith('/*', index)) {
    state.blockCommentDepth = 1;
    return index + 2;
  }
  const char = source[index];
  if (char === '"') return consumeQuoted(source, index, '"');
  if (char === "'" && looksLikeCharLiteral(source, index)) return consumeQuoted(source, index, "'");
  if (char === 'r') {
    const rawEnd = consumeRawString(source, index);
    if (rawEnd > index) return rawEnd;
  }
  return index;
}

function consumeQuoted(source, opening, quote) {
  for (let index = opening + 1; index < source.length; index += 1) {
    if (source[index] === '\\') index += 1;
    else if (source[index] === quote) return index + 1;
  }
  return source.length;
}

function looksLikeCharLiteral(source, index) {
  if (source[index + 1] === '\\') return source[index + 3] === "'";
  return source[index + 2] === "'";
}

function consumeRawString(source, index) {
  const prefix = source.slice(index).match(/^r(#+)?"/);
  if (!prefix) return index;
  const hashes = prefix[1] || '';
  const closing = `"${hashes}`;
  const end = source.indexOf(closing, index + prefix[0].length);
  return end < 0 ? source.length : end + closing.length;
}

function normalizedHash(body) {
  const normalized = body.replace(/\s+/g, ' ').trim();
  return createHash('sha256').update(normalized).digest('hex');
}

const options = parseArgs(process.argv.slice(2));
const before = rustSourceAt(options.beforeRef, options.beforePath);
const after = readFileSync(resolve(options.afterPath), 'utf8');
if (options.allFunctions) {
  const matcher = /(?:^|\n)\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:<[^>{}]*>)?\s*\(/g;
  options.symbols = [...after.matchAll(matcher)].map((match) => match[1]);
  if (options.symbols.length === 0) throw new Error('no Rust functions found in after-path');
}
const results = options.symbols.map((symbol) => {
  const beforeHash = normalizedHash(functionBody(before, symbol));
  const afterHash = normalizedHash(functionBody(after, symbol));
  return { symbol, before_sha256: beforeHash, after_sha256: afterHash, equal: beforeHash === afterHash };
});
console.log(JSON.stringify({
  schema: 'ctox.rust_move_verification.v1',
  before_ref: options.beforeRef,
  before_path: options.beforePath,
  after_path: options.afterPath,
  results,
}, null, 2));
if (results.some((result) => !result.equal)) process.exitCode = 1;
