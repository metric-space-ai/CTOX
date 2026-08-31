#!/usr/bin/env node
// Datenzustands-Stufe des Shell-V2-Pruefstands: oeffnet Apps in der lokalen
// Instanz, saet schema-getriebene Beispieldokumente direkt in die laufenden
// RxDB-Collections (window.CTOX_BUSINESS_OS_APP.db) und prueft den
// GEFUELLTEN Zustand: sichtbare Zeilen, keine haengengebliebenen
// Skeleton-/Lade-Overlays, Geometrie weiterhin vertragskonform.
// Aufruf: node scripts/shell-v2-data-probe.mjs [--apps tickets,ctox] [--base URL]
import { chromium } from 'playwright';
import { readFileSync, mkdirSync, existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const BASE = process.argv.includes('--base') ? process.argv[process.argv.indexOf('--base') + 1] : 'http://127.0.0.1:19100';
const APPS = process.argv.includes('--apps') ? process.argv[process.argv.indexOf('--apps') + 1].split(',') : ['tickets', 'ctox', 'documents', 'knowledge', 'threads'];
const OUT = path.resolve(ROOT, '../../..', 'output/playwright/shell-v2-data-probe');
mkdirSync(OUT, { recursive: true });

// Haupt-Collections je App (die Listen-Quelle der linken Spalte).
const FEED = {
  tickets: ['ctox_ticket_items'],
  ctox: ['ctox_queue_tasks'],
  documents: ['documents'],
  knowledge: ['knowledge_items'],
  threads: ['business_chats'],
  support: ['support_conversations'],
};

function sampleValue(name, prop, i) {
  const t = Array.isArray(prop.type) ? prop.type[0] : prop.type;
  if (prop.enum?.length) return prop.enum[i % prop.enum.length];
  if (t === 'number' || t === 'integer') return name.includes('_ms') ? Date.now() - i * 3600000 : i;
  if (t === 'boolean') return i % 2 === 0;
  if (t === 'array') return [];
  if (t === 'object') return {};
  if (name === 'id') return `probe-${i}`;
  if (/title|name|subject/.test(name)) return `Beispiel ${name} ${i} — längerer Text zum Ellipsentest`;
  if (/status|state|phase/.test(name)) return 'open';
  return `probe-${name}-${i}`;
}

function docsFor(app, collection, n = 6) {
  const schemaFile = path.join(ROOT, 'modules', app, 'collections.schema.json');
  if (!existsSync(schemaFile)) return [];
  const cols = JSON.parse(readFileSync(schemaFile, 'utf8')).collections || {};
  const sch = cols[collection]?.schema || cols[collection];
  if (!sch?.properties) return [];
  return Array.from({ length: n }, (_, i) => {
    const doc = {};
    for (const [k, p] of Object.entries(sch.properties)) {
      if (k.startsWith('_')) continue;
      doc[k] = sampleValue(k, p, i);
    }
    doc.id = `probe-${collection}-${i}`;
    if ('updated_at_ms' in sch.properties) doc.updated_at_ms = Date.now() - i * 60000;
    return doc;
  });
}

const browser = await chromium.launch({ headless: true, executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH });
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
await page.goto(`${BASE}/business-os/`, { waitUntil: 'load' });
await page.waitForTimeout(9000);

for (const app of APPS) {
  const feeds = FEED[app] || [];
  const payload = {};
  for (const c of feeds) payload[c] = docsFor(app, c);
  // App ZUERST oeffnen: modul-eigene Collections registrieren sich beim Mount.
  await page.evaluate((mid) => { location.hash = mid; }, app);
  await page.waitForSelector(`section.shell-window[data-owner-id="desktop-app:${app}"]`, { timeout: 20000 }).catch(() => {});
  await page.waitForTimeout(3000);
  const seeded = await page.evaluate(async (data) => {
    const db = window.CTOX_BUSINESS_OS_APP?.db;
    if (!db?.collection) return 'kein-db';
    const out = {};
    for (const [name, docs] of Object.entries(data)) {
      const col = db.collection(name);
      if (!col) { out[name] = 'fehlt'; continue; }
      let ok = 0, err = '';
      for (const d of docs) {
        try { await (col.upsert ? col.upsert(d) : col.insert(d)); ok++; }
        catch (e) { err = String(e?.message || e).slice(0, 90); }
      }
      out[name] = `${ok}/${docs.length}${err ? ' letzter Fehler: ' + err : ''}`;
    }
    return out;
  }, payload);
  const sel = `section.shell-window[data-owner-id="desktop-app:${app}"]`;
  const found = await page.waitForSelector(sel, { timeout: 20000 }).catch(() => null);
  // Auf Replikations-Readiness der Feed-Collections warten (max 30s) - Apps
  // gaten ihre Listen darauf (collectionReadiness), sonst steht dort ewig
  // "wird synchronisiert".
  await page.waitForFunction((names) => {
    const sync = window.CTOX_BUSINESS_OS_APP?.sync;
    if (!sync?.collectionReadiness) return true;
    return names.every((n) => { try { return sync.collectionReadiness(n)?.ready !== false; } catch { return true; } });
  }, feeds, { timeout: 30000 }).catch(() => {});
  await page.waitForTimeout(4000);
  const r = found ? await page.evaluate((s) => {
    const w = document.querySelector(s);
    const wr = w.getBoundingClientRect();
    const rows = [...w.querySelectorAll('.ctox-list-item, [class*="-row"], [class*="shard"], li')]
      .filter((e) => { const r = e.getBoundingClientRect(); return r.width > 40 && r.height > 10 && (e.textContent || '').trim(); });
    const skeletons = [...w.querySelectorAll('[class*="loading"], [class*="skeleton"], [class*="shimmer"], [class*="mls-"]')]
      .filter((e) => { const r = e.getBoundingClientRect(); return r.width > 0 && r.height > 0; })
      .map((e) => String(e.className).split(' ')[0].slice(0, 30));
    // Streifen-Heuristik: Pseudo-Balken ueber Textzeilen (der CTOX-Befund) sind
    // als ::before/::after nicht abzaehlbar - dafuer steht der Screenshot.
    const hscroll = w.querySelector('.shell-window-content').scrollWidth > Math.round(wr.width) + 2;
    return { rows: rows.length, skeletons: [...new Set(skeletons)].slice(0, 4), hscroll };
  }, sel) : { rows: -1 };
  const fails = found
    ? (r.rows === 0 ? ['keine-zeilen'] : []).concat(r.skeletons?.length ? ['skeleton-haengt'] : []).concat(r.hscroll ? ['hscroll'] : [])
    : ['kein-fenster'];
  await page.screenshot({ path: path.join(OUT, `${app}-data.png`) });
  console.log(`${fails.length ? 'FAIL' : 'OK  '} ${app.padEnd(12)} rows=${r.rows} ${fails.join(',')} seed=${JSON.stringify(seeded)}`);
}
await browser.close();
