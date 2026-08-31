#!/usr/bin/env node
// Live-Pruefstand fuer JS-gebaute App-Rahmen: oeffnet Apps in einer echten,
// lokal laufenden Business-OS-Instanz (Arbeitsbaum, leere Daten) per
// #hash-Route und misst die Vertragskriterien am GEMOUNTETEN Zustand -
// das, was der statische Prueflabor-Frame nicht sehen kann.
// Aufruf: node scripts/shell-v2-live-probe.mjs [--base http://127.0.0.1:19100] [--apps a,b]
import { chromium } from 'playwright';
import { mkdirSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const BASE = process.argv.includes('--base')
  ? process.argv[process.argv.indexOf('--base') + 1] : 'http://127.0.0.1:19100';
const APPS = process.argv.includes('--apps')
  ? process.argv[process.argv.indexOf('--apps') + 1].split(',')
  : ['documents', 'importer', 'invoices', 'outbound', 'spreadsheets'];
const OUT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../../..', 'output/playwright/shell-v2-live-probe');
mkdirSync(OUT, { recursive: true });

const browser = await chromium.launch({ headless: true, executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH });
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
page.on('pageerror', (e) => console.log('  [pageerror]', String(e).slice(0, 120)));
await page.goto(`${BASE}/business-os/`, { waitUntil: 'load' });
// Shell-Boot abwarten (Taskbar da)
await page.waitForSelector('nav, [data-shell-start]', { timeout: 30000 }).catch(() => {});
await page.waitForTimeout(8000);

const results = [];
for (const id of APPS) {
  await page.evaluate((mid) => { location.hash = mid; }, id);
  const sel = `section.shell-window[data-owner-id="desktop-app:${id}"]`;
  const found = await page.waitForSelector(sel, { timeout: 20000 }).catch(() => null);
  await page.waitForTimeout(4000); // JS-Mount + Leerzustand rendern lassen
  const r = found ? await page.evaluate((s) => {
    const w = document.querySelector(s);
    const wr = w.getBoundingClientRect();
    const rowPx = Math.round(parseFloat(getComputedStyle(w).getPropertyValue('--shell-v2-header-row-size'))) || 37;
    const heads = [...w.querySelectorAll('.ctox-pane-header')]
      .filter((h) => { const r = h.getBoundingClientRect(); return r.width > 0 && r.height > 0; });
    const topRow = heads.filter((h) => h.getBoundingClientRect().top - wr.top < 46);
    const info = topRow.map((h) => { const cs = getComputedStyle(h), r = h.getBoundingClientRect();
      return { h: Math.round(r.height), left: Math.round(r.left - wr.left), right: Math.round(wr.right - r.right), padL: Math.round(parseFloat(cs.paddingLeft)), padR: Math.round(parseFloat(cs.paddingRight)) }; });
    const first = info[0] || null, last = info[info.length - 1] || null;
    const heights = [...new Set(info.map((x) => x.h))];
    const intruderSet = new Set();
    for (const px of [12, 34, 56, 74]) for (const py of [42, 56, 70]) {
      const top = document.elementsFromPoint(wr.left + px, wr.top + py)
        .find((e) => e.closest('.shell-window-content') && !e.matches('.shell-window-content'));
      if (!top || top.closest('.ctox-pane-header')) continue;
      const cs2 = getComputedStyle(top);
      const painted = cs2.backgroundColor !== 'rgba(0, 0, 0, 0)' || (top.textContent || '').trim().length > 0 || top.matches('img,svg,button,input');
      if (painted && !/module-root|module-content|layout|workspace|-module$/.test(String(top.className)))
        intruderSet.add((String(top.className).split(' ')[0] || top.tagName).slice(0, 28));
    }
    const title = w.querySelector('.ctox-pane-header .ctox-pane-title');
    return {
      headers: info.length, heights, rowPx,
      h1: heights.length === 1 && heights[0] <= rowPx * 2 + 2,
      h2: first ? (first.left + first.padL) >= 80 : false,
      h3: last ? (last.right + last.padR) >= 74 : false,
      h4: !!title, titleText: title?.textContent?.trim().slice(0, 30) || null,
      intruders: [...intruderSet].slice(0, 4),
      hscroll: w.querySelector('.shell-window-content').scrollWidth > Math.round(wr.width) + 2,
    };
  }, sel) : { headers: -1 };
  const fails = found
    ? ['h1', 'h2', 'h3', 'h4'].filter((k) => !r[k]).concat(r.intruders?.length ? ['iconZone'] : []).concat(r.hscroll ? ['hscroll'] : [])
    : ['kein-fenster'];
  await page.screenshot({ path: path.join(OUT, `${id}.png`) });
  results.push({ app: id, fails, detail: r });
  console.log(`${fails.length ? 'FAIL' : 'OK  '} ${id.padEnd(14)} ${fails.join(',')} ${JSON.stringify(r.heights || [])}`);
}
await browser.close();
