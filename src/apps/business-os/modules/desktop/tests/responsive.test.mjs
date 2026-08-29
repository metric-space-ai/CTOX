import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const css = readFileSync(new URL('../index.css', import.meta.url), 'utf8');

assert.match(css, /\.desktop-module\s*\{[\s\S]*container-type:\s*inline-size/);
assert.match(css, /@container \(max-width: 1320px\)\s*\{[\s\S]*\.desktop-widget-container\s*\{[\s\S]*display:\s*none/);
assert.match(css, /\.desktop-hero-widget\s*\{[\s\S]*width:\s*248px/);
assert.doesNotMatch(css, /\.desktop-hero-widget\s*\{[\s\S]*width:\s*334px/);
assert.match(css, /@container \(max-width: 720px\)\s*\{[\s\S]*grid-template-columns:\s*repeat\(auto-fit, minmax\(80px, 1fr\)\)/);
assert.doesNotMatch(css, /@media \(max-width: 560px\)\s*\{[\s\S]*\.desktop-icons/);

const js = readFileSync(new URL('../index.js', import.meta.url), 'utf8');
assert.match(js, /surfaceWidth > 0 && surfaceWidth <= 720/);

console.log('ok - desktop labels and dock clearance follow the embedded surface width');
