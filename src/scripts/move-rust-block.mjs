#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';

function fail(message) {
  process.stderr.write(`${message}\n`);
  process.exit(1);
}

const [sourcePath, destinationPath, startMarker, endMarker] = process.argv.slice(2);
if (!sourcePath || !destinationPath || !startMarker || !endMarker) {
  fail('usage: move-rust-block.mjs <source> <destination> <start-marker> <end-marker>');
}
if (fs.existsSync(destinationPath)) fail(`destination already exists: ${destinationPath}`);

const source = fs.readFileSync(sourcePath, 'utf8');
const start = source.indexOf(startMarker);
if (start === -1) fail(`start marker not found: ${startMarker}`);
if (source.indexOf(startMarker, start + startMarker.length) !== -1) {
  fail(`start marker is not unique: ${startMarker}`);
}
const end = source.indexOf(endMarker, start + startMarker.length);
if (end === -1) fail(`end marker not found after start: ${endMarker}`);

const block = source.slice(start, end);
const relativeDestination = path.relative(path.dirname(sourcePath), destinationPath);
if (!relativeDestination || relativeDestination.startsWith('..')) {
  fail('destination must be next to or below the source file');
}
const includeLine = `include!(${JSON.stringify(relativeDestination)});\n\n`;
const nextSource = `${source.slice(0, start)}${includeLine}${source.slice(end)}`;

fs.writeFileSync(destinationPath, block);
fs.writeFileSync(sourcePath, nextSource);
process.stdout.write(JSON.stringify({
  source: sourcePath,
  destination: destinationPath,
  moved_bytes: Buffer.byteLength(block),
  moved_lines: block.split('\n').length - 1,
}) + '\n');
