// handelsregister.de — solo live probe (plain Playwright, no CTOX stack).
//
// Usage: node scrape-targets/handelsregister.de/solo/probe.mjs "<company name>"
//
// Drives the LIVE official German register portal headless via its public
// "Normale Suche" form (no login), extracts the public result-list fields for
// the company and prints ONE JSON object:
//   {target, input, fetched_at, fields: {<field_key>: {value, source_url}}}
// Exit 0 only when at least 2 non-empty fields were extracted (the public
// result row exposes name, seat and register designation); otherwise non-zero
// with a "reason" field. Challenges are never solved or evaded.

import { chromium } from 'playwright';

const TARGET = 'handelsregister.de';
const SEARCH_URL = 'https://www.handelsregister.de/rp_web/normalesuche/welcome.xhtml';

const company = (process.argv[2] || '').trim();
if (!company) {
  console.log(JSON.stringify({
    target: TARGET,
    input: '',
    fetched_at: new Date().toISOString(),
    fields: {},
    reason: 'usage: probe.mjs "<company name>"',
  }));
  process.exit(2);
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

let browser;
function fail(reason, fields) {
  console.log(JSON.stringify({
    target: TARGET,
    input: company,
    fetched_at: new Date().toISOString(),
    fields: fields || {},
    reason,
  }));
  if (browser) browser.close().catch(() => null);
  process.exit(1);
}

// Same normalisation as scripts/v1.js so the probe accepts exactly the entries
// the adapter would accept.
function normalizeCompanyName(value) {
  return String(value || '')
    .toLocaleLowerCase('de-DE')
    .normalize('NFKD')
    .replace(/\p{M}/gu, '')
    .replace(/[^a-z0-9]+/g, ' ')
    .trim()
    .replace(/\s+/g, ' ');
}

browser = await chromium.launch({ headless: true });
const context = await browser.newContext({ locale: 'de-DE' });
const page = await context.newPage();

try {
  // ------------------------------------------------------------------
  // 1. Open the public "Normale Suche" form. The portal is JSF/PrimeFaces
  //    based and slow — generous timeouts throughout.
  // ------------------------------------------------------------------
  let response;
  try {
    response = await page.goto(SEARCH_URL, { waitUntil: 'domcontentloaded', timeout: 90000 });
  } catch (err) {
    fail('navigation to search form failed: ' + String(err.message || err).split('\n')[0]);
  }
  const status = (response && response.status()) || 0;
  if (status === 403 || status === 429 || status >= 500) {
    fail('search form returned HTTP ' + status);
  }
  await sleep(2000); // politeness between navigations

  // Ordinary cookie/consent dialog: click its visible button once.
  const consentPatterns = [/alle akzeptieren/i, /akzeptieren/i, /zustimmen/i, /einverstanden/i, /ok/i];
  for (const pattern of consentPatterns) {
    const button = page.getByRole('button', { name: pattern }).first();
    if (await button.count().catch(() => 0)) {
      await button.click({ timeout: 3000 }).catch(() => null);
      await sleep(1500);
      break;
    }
  }

  const challengeMarkers = async () => page.evaluate(() => {
    const text = document.body ? document.body.innerText : '';
    const challenge = document.querySelector(
      'iframe[src*="recaptcha" i], iframe[src*="captcha" i], iframe[src*="challenge" i], .g-recaptcha, [data-sitekey]',
    );
    const blocked = Boolean(challenge)
      || /captcha|bitte beweisen sie|verify (?:that )?you are human|access denied|request blocked|zugriff verweigert|gesperrt/i.test(text);
    return blocked;
  });

  if (await challengeMarkers()) {
    fail('blocked: access challenge detected on the search form');
  }

  // ------------------------------------------------------------------
  // 2. Fill and submit the public search form (same selectors as v1.js).
  // ------------------------------------------------------------------
  const companyInput = page.locator('[id="form:schlagwoerter"]');
  const exactRadioLabel = page.locator('label[for="form:schlagwortOptionen:2"]');
  const searchButton = page.locator('[id="form:btnSuche"]');

  await companyInput.waitFor({ state: 'visible', timeout: 30000 }).catch(() => null);

  const counts = {
    input: await companyInput.count().catch(() => 0),
    radio: await exactRadioLabel.count().catch(() => 0),
    button: await searchButton.count().catch(() => 0),
  };
  if (counts.input !== 1 || counts.radio !== 1 || counts.button !== 1) {
    fail('portal drift: search form selectors not found (' + JSON.stringify(counts) + ')');
  }

  await companyInput.fill(company);
  await exactRadioLabel.click().catch(() => null); // "genaue Firmenbezeichnung"
  await sleep(500);
  await searchButton.click();

  await page.waitForLoadState('domcontentloaded', { timeout: 90000 }).catch(() => null);
  await page.waitForSelector(
    '[id="ergebnissForm:selectedSuchErgebnisFormTable_data"]',
    { timeout: 60000 },
  ).catch(() => null);
  await sleep(2000);

  if (await challengeMarkers()) {
    fail('blocked: access challenge detected on the result page');
  }

  // ------------------------------------------------------------------
  // 3. Extract the public result rows (name, Sitz, Registerbezeichnung).
  // ------------------------------------------------------------------
  const result = await page.evaluate(() => {
    const clean = (value) => String(value || '').replace(/\s+/g, ' ').trim();
    const resultBody = document.querySelector(
      '[id="ergebnissForm:selectedSuchErgebnisFormTable_data"]',
    );
    const tables = resultBody
      ? [...resultBody.querySelectorAll(':scope > tr > td > table.ui-panelgrid')]
      : [];
    const entries = tables.map((table) => {
      const rows = table.querySelectorAll(':scope > tbody > tr');
      const header = rows[0] ? rows[0].querySelector('.fontTableNameSize') : null;
      const companyRow = rows[1];
      if (!header || !companyRow) return null;
      const nameNode = companyRow.querySelector('td:first-child > .marginLeft20');
      const cityNode = companyRow.querySelector('.sitzSuchErgebnisse');
      const status = clean(companyRow.innerText).match(
        /(aktuell|currently registered|gel[öo]scht|historisch)/i,
      );
      return {
        name: clean(nameNode ? nameNode.innerText : ''),
        city: clean(cityNode ? cityNode.innerText : ''),
        registry: clean(header.innerText),
        status: status ? status[1] : null,
      };
    }).filter((entry) => entry && entry.name);
    return {
      url: location.href,
      title: document.title,
      results_page: Boolean(resultBody) || /Suchergebnis/i.test(document.title),
      entries,
    };
  });

  if (!result.results_page) {
    fail('did not reach a readable result page (title: ' + JSON.stringify(result.title) + ')');
  }
  if (result.entries.length === 0) {
    fail('result page contained no parsable entries (title: ' + JSON.stringify(result.title) + ')');
  }

  const expected = normalizeCompanyName(company);
  const entry = result.entries.find((e) => normalizeCompanyName(e.name) === expected)
    || result.entries[0];

  const sourceUrl = result.url || SEARCH_URL;
  const fields = {};
  const put = (key, value) => {
    if (value) fields[key] = { value, source_url: sourceUrl };
  };
  put('firma_name', entry.name);
  put('firma_ort', entry.city);
  put('register_bezeichnung', entry.registry);
  put('register_status', entry.status);

  const count = Object.keys(fields).length;
  if (count < 2) {
    fail('only ' + count + ' field(s) extracted from ' + result.entries.length
      + ' entr(y|ies) (title: ' + JSON.stringify(result.title) + ')', fields);
  }

  console.log(JSON.stringify({
    target: TARGET,
    input: company,
    fetched_at: new Date().toISOString(),
    exact_match: normalizeCompanyName(entry.name) === expected,
    entries_found: result.entries.length,
    fields,
  }, null, 2));
  process.exit(0);
} finally {
  await browser.close().catch(() => null);
}
