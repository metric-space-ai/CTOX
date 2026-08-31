#!/usr/bin/env node
// Statischer Teil der Shell-V2-Abnahme (siehe docs/business-os-shell-v2-contract.md).
//
// Prueft die beiden Kriterien, die ohne Browser entscheidbar sind:
//   5. Keine `@media (max-width: ...)`-Regeln - ein Fenster ist kein Bildschirm.
//   6. Keine eigenen Spalten-Haltepunkte ausser den shell-eigenen 1024px und 768px.
//
// Die Kriterien 1-4 (Header-Hoehe, Icon- und Steuerungs-Reservierung, Titel-Slot)
// sind Geometrie und gehoeren in die Browser-Abnahme.
//
// Aufruf: node scripts/assert-shell-v2-contract.mjs [--json]
// Exit 0 = konform, 1 = Abweichungen. Bewusst noch NICHT in package.json verdrahtet:
// solange die Migration laeuft, waere ein rotes Pflichtgate nur Laerm.

import { readdirSync, readFileSync, existsSync, statSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const SANCTIONED = new Set([1024, 768]);
// Die Shell-Oberflaeche selbst ist kein Fensterinhalt; Knowledge ist die Referenz.
// kundenpipeline: Kundenapp, von origin nach modules/ verschoben - bis zur
// Betreiber-Entscheidung vom Vertrag ausgenommen (Finger-weg-Regel 31.08.).
const SKIP = new Set(['desktop', 'kundenpipeline']);

function moduleDirs(base) {
  const dir = join(ROOT, base);
  if (!existsSync(dir)) return [];
  return readdirSync(dir)
    .filter((id) => !SKIP.has(id))
    .filter((id) => statSync(join(dir, id)).isDirectory())
    .filter((id) => existsSync(join(dir, id, 'index.css')))
    .map((id) => ({ id, base, css: join(dir, id, 'index.css') }));
}

const findings = [];
// Kundenapps (installed-modules) sind vom Vertrag ausgenommen (31.08.2026).
const apps = [...moduleDirs('modules')];

for (const app of apps) {
  const css = readFileSync(app.css, 'utf8');

  const mediaWidths = [...css.matchAll(/@media\s*\([^)]*?(?:max|min)-width:\s*(\d+)px/g)]
    .map((m) => Number(m[1]));
  if (mediaWidths.length) {
    findings.push({
      app: app.id, base: app.base, rule: 'media-query',
      detail: `${mediaWidths.length}x @media statt @container: ${[...new Set(mediaWidths)].join(', ')}px`,
    });
  }

  // Direktive 31.08.: keine Host-Akzent-Hexwerte und keine eigene --accent-
  // Definition in Modulen - Interaktionsfarben leiten sich aus dem Icon ab.
  const BLUE = /#(346bf1|356bf1|4c7dff|1b4ed8|2563eb|3b82f6|1d4ed8)/i;
  const js = existsSync(join(ROOT, app.base, app.id, 'index.js')) ? readFileSync(join(ROOT, app.base, app.id, 'index.js'), 'utf8') : '';
  if (BLUE.test(css) || BLUE.test(js)) {
    findings.push({ app: app.id, base: app.base, rule: 'hardcoded-accent', detail: 'Host-Blau-Hex in Modulcode; Akzent muss vom Icon abgeleitet sein' });
  }
  if (/^\s*--accent\s*:/m.test(css)) {
    findings.push({ app: app.id, base: app.base, rule: 'accent-override', detail: 'Modul definiert --accent selbst; gehoert der Shell (Icon-Palette)' });
  }

  // Nur STRUKTURELLE Haltepunkte sind auf 1024/768 beschraenkt (Vertrag §6:
  // Spaltenmodell). Inhaltsabgeleitete Feinschwellen (z.B. Beschriftungen
  // ikonisieren) sind zulaessig, solange der Block keine Grid-Struktur
  // veraendert.
  const stray = [];
  for (const m of css.matchAll(/@container\s+business-app-window\s*\([^)]*?max-width:\s*(\d+)px[^)]*\)\s*\{/g)) {
    const w = Number(m[1]);
    if (SANCTIONED.has(w)) continue;
    let i = m.index + m[0].length, depth = 1, start = i;
    while (depth && i < css.length) { if (css[i] === '{') depth++; else if (css[i] === '}') depth--; i++; }
    const body = css.slice(start, i - 1);
    if (/grid-template|grid-column|grid-row|grid-auto/.test(body)) stray.push(w);
  }
  const uniqStray = [...new Set(stray)].sort((a, b) => a - b);
  if (uniqStray.length) {
    findings.push({
      app: app.id, base: app.base, rule: 'breakpoint',
      detail: `strukturelle Haltepunkte statt 1024/768: ${uniqStray.join(', ')}px`,
    });
  }
}

const offenders = new Set(findings.map((f) => f.app));

if (process.argv.includes('--json')) {
  console.log(JSON.stringify({ apps: apps.length, offenders: offenders.size, findings }, null, 2));
} else if (!findings.length) {
  console.log(`Shell-V2-Vertrag (statischer Teil) OK: ${apps.length}/${apps.length} Apps`);
} else {
  console.log(`Shell-V2-Vertrag (statischer Teil): ${apps.length - offenders.size}/${apps.length} Apps konform\n`);
  for (const f of findings.sort((a, b) => a.app.localeCompare(b.app))) {
    console.log(`  ${f.app.padEnd(28)} ${f.rule.padEnd(12)} ${f.detail}`);
  }
}

process.exit(findings.length ? 1 : 0);
