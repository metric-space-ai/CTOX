// maps.google.com — solo live probe (plain Playwright, no CTOX stack).
//
// Usage: node scrape-targets/maps.google.com/solo/probe.mjs "<company> <city>"
//
// maps.google.com has NO own scripts/v1.js: it runs through
// _shared/generic-prospect-v1.js (branch `sourceId === "maps.google.com"`).
// This probe mirrors that branch's navigation and selectors 1:1 against the
// live site so a selector drift shows up here before it breaks the adapter.
//
// Prints ONE JSON object:
//   {target, input, fetched_at, fields: {<field_key>: {value, source_url}}}
// Exit 0 only when a real extraction happened (>= 2 fields incl. an address or
// a phone number); otherwise non-zero with a "reason" field.
//
// Google's ordinary cookie-consent dialog IS answered programmatically with
// "Alle ablehnen" — that is normal operation of a public site, and it selects
// the privacy-preserving option. A CAPTCHA or any other bot challenge is NEVER
// solved or evaded: it is reported as a finding and exits non-zero.

import { chromium } from 'playwright';

const TARGET = 'maps.google.com';

const input = (process.argv.slice(2).join(' ') || '').trim();
const country = (process.env.PROBE_COUNTRY || 'DE').trim();

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function emit(payload, code) {
  console.log(JSON.stringify(payload, null, 2));
  process.exit(code);
}

function fail(reason, fields) {
  emit({
    target: TARGET,
    input,
    fetched_at: new Date().toISOString(),
    fields: fields || {},
    reason,
  }, 1);
}

if (!input) {
  emit({
    target: TARGET,
    input: '',
    fetched_at: new Date().toISOString(),
    fields: {},
    reason: 'usage: probe.mjs "<company> <city>"',
  }, 2);
}

// The shared adapter searches `"<company>, <countryName>"`; the caller may
// already supply a city, so the country is appended as a hint only.
const countryName = ({ DE: 'Deutschland', AT: 'Österreich', CH: 'Schweiz' })[country] || country;
const query = [input, countryName].filter(Boolean).join(', ');

const browser = await chromium.launch({ headless: true });
const context = await browser.newContext({ locale: 'de-DE' });
const page = await context.newPage();

try {
  // ------------------------------------------------------------------
  // 1. Maps URL API v1 search (same entry point as the shared adapter).
  // ------------------------------------------------------------------
  const searchUrl = 'https://www.google.com/maps/search/?api=1&query=' + encodeURIComponent(query);
  let response;
  try {
    response = await page.goto(searchUrl, { waitUntil: 'domcontentloaded', timeout: 45000 });
  } catch (err) {
    fail('navigation to maps search failed: ' + String(err.message || err).split('\n')[0]);
  }
  const status = (response && response.status()) || 0;
  if (status === 403 || status === 429 || status >= 500) {
    fail('maps search returned HTTP ' + status);
  }
  await sleep(1500);

  // ------------------------------------------------------------------
  // 2. Consent wall: answer the ordinary cookie dialog with "Alle ablehnen".
  //    Google serves it on consent.google.com in an interstitial page or in
  //    an iframe; both variants are handled. Never a CAPTCHA.
  // ------------------------------------------------------------------
  const rejectPattern = /alle ablehnen|alles ablehnen|reject all|tout refuser|nicht zustimmen/i;
  const consentFrames = [page, ...page.frames()];
  for (const frame of consentFrames) {
    let button;
    try {
      button = frame.getByRole('button', { name: rejectPattern }).first();
      if (!(await button.count())) {
        button = frame.locator('form button, button').filter({ hasText: rejectPattern }).first();
      }
      if (await button.count()) {
        await button.click({ timeout: 5000 }).catch(() => null);
        await page.waitForLoadState('domcontentloaded', { timeout: 15000 }).catch(() => null);
        await sleep(2000);
        break;
      }
    } catch (err) { /* frame detached mid-consent: try the next one */ }
  }

  // ------------------------------------------------------------------
  // 3. Challenge / still-blocked detection. A challenge is a FINDING.
  // ------------------------------------------------------------------
  const bodyText = await page.locator('body').innerText().catch(() => '');
  const currentUrl = page.url();
  if (/\/sorry\/|recaptcha|\/httpservice\/retry/i.test(currentUrl)
    || /ungewöhnlicher datenverkehr|unusual traffic|bestätigen sie, dass sie kein roboter|i'?m not a robot|captcha/i.test(bodyText)) {
    fail('blocked: bot challenge / CAPTCHA served (url=' + currentUrl + ') — not solved by design');
  }
  if (/consent\.google\.com/i.test(currentUrl)
    || /bevor sie zu google maps weitergehen|before you continue to google/i.test(bodyText)) {
    fail('blocked: consent wall not dismissable with "Alle ablehnen" (url=' + currentUrl + ')');
  }

  // ------------------------------------------------------------------
  // 4. Pick the first place result. A single strong hit lands directly on
  //    /maps/place/; a list requires clicking the first entry.
  // ------------------------------------------------------------------
  await page.waitForFunction(
    (value) => document.body?.innerText.toLowerCase().includes(value.toLowerCase()),
    input.split(/\s+/)[0],
    { timeout: 30000 },
  ).catch(() => null);

  if (!/\/maps\/place\//.test(page.url())) {
    // Adapter parity: exact-name hit first, then plain first result.
    let result = page.locator('a[href*="/maps/place/"]').filter({ hasText: input }).first();
    if (!(await result.count())) {
      result = page.locator('a[href*="/maps/place/"]').first();
    }
    if (await result.count()) {
      await result.click({ timeout: 8000 }).catch(() => null);
      await page.waitForTimeout(3000);
    }
  }

  // Detail panel marker: address or phone button.
  await page.waitForSelector('button[data-item-id="address"], button[data-item-id^="phone:tel:"]', { timeout: 20000 })
    .catch(() => null);
  await sleep(1000);

  // ------------------------------------------------------------------
  // 5. Extract from the detail panel — same selectors as the shared adapter,
  //    plus the website authority link.
  // ------------------------------------------------------------------
  const attr = async (selector, attribute) => {
    const locator = page.locator(selector).first();
    if (!(await locator.count())) return '';
    return (await locator.getAttribute(attribute).catch(() => '')) || '';
  };

  const phone = (await attr('button[data-item-id^="phone:tel:"]', 'data-item-id')).replace(/^phone:tel:/, '');
  const address = (await attr('button[data-item-id="address"]', 'aria-label'))
    .replace(/^(?:Adresse|Address):\s*/i, '').trim();
  const website = (await attr('a[data-item-id="authority"]', 'href')).trim();
  const name = (await page.locator('h1').first().innerText().catch(() => '')).replace(/\s+/g, ' ').trim();

  const postal = address.match(/\b(?:D-|A-|CH-)?(\d{4,5})\s+([^,]+)/);

  const sourceUrl = page.url();
  const fields = {};
  const put = (key, value) => {
    const cleaned = String(value || '').replace(/\s+/g, ' ').trim();
    if (cleaned) fields[key] = { value: cleaned, source_url: sourceUrl };
  };
  put('firma_name', name);
  put('firma_telefon', phone);
  put('firma_anschrift', address);
  if (postal) {
    put('firma_plz', postal[1]);
    put('firma_ort', postal[2]);
  }
  put('firma_website', website);

  const count = Object.keys(fields).length;
  if (!address && !phone) {
    fail('no detail panel fields extracted (url=' + sourceUrl
      + ', title=' + JSON.stringify(await page.title().catch(() => '')) + ')', fields);
  }
  if (count < 2) {
    fail('only ' + count + ' field(s) extracted (url=' + sourceUrl + ')', fields);
  }

  emit({
    target: TARGET,
    input,
    query,
    fetched_at: new Date().toISOString(),
    fields,
  }, 0);
} finally {
  await browser.close().catch(() => null);
}
