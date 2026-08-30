// moneyhouse.ch — solo live probe (plain Playwright, no CTOX stack).
//
// Usage: node scrape-targets/moneyhouse.ch/solo/probe.mjs "<company name>"
//
// Drives the LIVE site headless and extracts ONLY the publicly visible company
// header fields (Firma / Sitz / Rechtsform). Moneyhouse puts most detail data
// behind a paid login — the probe never logs in and never solves or evades a
// bot challenge. A hard block or a login wall is reported as a finding.
//
// Prints ONE JSON object:
//   {target, input, fetched_at, fields: {<field_key>: {value, source_url}}}
// Exit 0 only when at least 2 non-empty fields were really extracted;
// otherwise non-zero with a "reason" field.

import { chromium } from 'playwright';

const TARGET = 'moneyhouse.ch';
const HOST = 'www.moneyhouse.ch';

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
async function fail(reason, fields) {
  console.log(JSON.stringify({
    target: TARGET,
    input: company,
    fetched_at: new Date().toISOString(),
    fields: fields || {},
    reason,
  }, null, 2));
  await (browser ? browser.close().catch(() => null) : null);
  process.exit(1);
}

const BLOCK_RE = /captcha|cloudflare|verify you are human|access denied|zugriff verweigert|request blocked|too many requests|zu viele anfragen/i;
const LOGIN_RE = /exklusiv für registrierte|jetzt registrieren und|abonnement abschliessen|nur für abonnenten|bitte melden sie sich an/i;

function normalize(value) {
  return String(value || '').normalize('NFKD')
    .replace(/[̀-ͯ]/g, '').toLowerCase().replace(/ß/g, 'ss')
    .replace(/[^a-z0-9]+/g, ' ').trim();
}
const LEGAL_TOKENS = new Set(['ag', 'gmbh', 'kg', 'sa', 'sarl', 'srl', 'se', 'und']);
function identityMatches(name) {
  const tokens = normalize(company).split(/\s+/)
    .filter((t) => t.length >= 3 && !LEGAL_TOKENS.has(t));
  const corpus = normalize(name);
  if (tokens.length === 0 || !corpus) return false;
  return tokens.filter((t) => corpus.includes(t)).length >= Math.max(1, Math.ceil(tokens.length * 0.75));
}
// Mirrors scripts/v1.js legalForm/legalFormMatches: "Novartis AG" must not
// resolve to "Novartis Foundation".
function legalForm(value) {
  const tokens = new Set(normalize(value).split(/\s+/));
  if (tokens.has('gmbh') && tokens.has('kg')) return 'gmbh-kg';
  for (const form of ['kgaa', 'gmbh', 'sarl', 'srl', 'se', 'ag', 'kg', 'og', 'sa']) {
    if (tokens.has(form)) return form;
  }
  return null;
}
function legalFormMatches(value) {
  const expected = legalForm(company);
  return expected === null || legalForm(value) === expected;
}

browser = await chromium.launch({ headless: true });
const context = await browser.newContext({
  locale: 'de-CH',
  userAgent: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
});
const page = await context.newPage();

try {
  // ------------------------------------------------------------------
  // 1. Public portal search.
  // ------------------------------------------------------------------
  const searchUrl = `https://${HOST}/de/search?q=${encodeURIComponent(company)}`;
  let response;
  try {
    response = await page.goto(searchUrl, { waitUntil: 'domcontentloaded', timeout: 45000 });
  } catch (err) {
    await fail('navigation to search failed: ' + String(err.message || err).split('\n')[0]);
  }
  let status = (response && response.status()) || 0;
  if (status === 401 || status === 403 || status === 429 || status >= 500) {
    await fail('search returned HTTP ' + status);
  }
  await sleep(2000); // politeness between navigations

  // Ordinary consent dialog: one click on its visible button.
  for (const pattern of [/alle akzeptieren/i, /akzeptieren/i, /zustimmen/i, /einverstanden/i]) {
    const button = page.getByRole('button', { name: pattern }).first();
    if (await button.count().catch(() => 0)) {
      await button.click({ timeout: 3000 }).catch(() => null);
      await sleep(1500);
      break;
    }
  }

  await page.locator('a[href*="/company/"]').first()
    .waitFor({ state: 'attached', timeout: 15000 }).catch(() => null);

  let bodyText = await page.locator('body').innerText().catch(() => '');
  if (BLOCK_RE.test(bodyText)) {
    await fail('blocked: anti-bot page detected on search');
  }

  // ------------------------------------------------------------------
  // 2. Pick the matching public company profile link.
  // ------------------------------------------------------------------
  const hits = await page.evaluate((host) => Array.from(document.querySelectorAll('a[href*="/company/"]'))
    .map((node) => ({ name: (node.textContent || '').replace(/\s+/g, ' ').trim(), url: node.href }))
    .filter((hit) => {
      try {
        const url = new URL(hit.url);
        // Base profile route only: the search page repeats every company as a
        // "/network" sublink under the generic label "Netzwerk ansehen".
        return url.protocol === 'https:'
          && url.hostname.toLowerCase().replace(/^www\./, '') === host
          && /^\/(de|en|fr|it)\/company\/[^/]+\/?$/i.test(url.pathname);
      } catch (err) { return false; }
    }), 'moneyhouse.ch');

  if (hits.length === 0) {
    await fail('no public company link found on the search page (title: '
      + JSON.stringify(await page.title().catch(() => '')) + ')');
  }
  const slug = (candidate) => decodeURIComponent(new URL(candidate.url).pathname).replace(/-\d+\/?$/, '');
  // Rank before matching: plain identity+legal-form matching alone lets
  // "Pensionskasse Lindt & Sprüngli AG" win over "Lindt & Sprüngli AG".
  const wanted = normalize(company);
  const hit = hits.find((c) => normalize(c.name) === wanted)
    || hits.find((c) => normalize(slug(c)).endsWith(wanted))
    || hits.find((c) => identityMatches(c.name) && legalFormMatches(c.name))
    || hits.find((c) => identityMatches(slug(c)) && legalFormMatches(slug(c)))
    || hits.find((c) => identityMatches(c.name));
  if (!hit) {
    await fail('no identity-matching company hit among ' + hits.length + ' link(s)');
  }

  // ------------------------------------------------------------------
  // 3. Open the public profile page.
  // ------------------------------------------------------------------
  const profileUrl = hit.url.split('#')[0];
  await sleep(2000); // politeness between navigations
  try {
    response = await page.goto(profileUrl, { waitUntil: 'domcontentloaded', timeout: 45000 });
  } catch (err) {
    await fail('navigation to profile failed: ' + String(err.message || err).split('\n')[0]);
  }
  status = (response && response.status()) || 0;
  if (status === 401 || status === 403 || status === 429 || status >= 500) {
    await fail('profile returned HTTP ' + status);
  }
  await sleep(2500);

  bodyText = await page.locator('body').innerText().catch(() => '');
  if (BLOCK_RE.test(bodyText)) {
    await fail('blocked: anti-bot page detected on profile');
  }

  // ------------------------------------------------------------------
  // 4. Extract the publicly visible header fields.
  // ------------------------------------------------------------------
  const extracted = await page.evaluate(() => {
    // Moneyhouse embeds HTML entities inside its JSON-LD strings, so a plain
    // JSON.parse yields "Lindt &amp; Sprüngli AG".
    const clean = (value) => String(value || '')
      .replace(/&amp;/gi, '&').replace(/&nbsp;/gi, ' ')
      .replace(/&#39;|&apos;/gi, "'").replace(/&quot;/gi, '"')
      .replace(/\s+/g, ' ').trim() || null;

    // 4a. JSON-LD organization block (public on the profile head).
    let ld = null;
    for (const node of document.querySelectorAll('script[type="application/ld+json"]')) {
      try {
        const parsed = JSON.parse(node.textContent || 'null');
        const queue = Array.isArray(parsed) ? [...parsed] : [parsed];
        for (const item of queue) {
          if (Array.isArray(item?.['@graph'])) queue.push(...item['@graph']);
          const types = Array.isArray(item?.['@type']) ? item['@type'] : [item?.['@type']];
          if (types.some((type) => /organization|localbusiness|corporation/i.test(String(type)))) {
            ld = {
              name: clean(item.legalName || item.name),
              street: clean(item.address?.streetAddress),
              postalCode: clean(item.address?.postalCode),
              locality: clean(item.address?.addressLocality),
            };
            break;
          }
        }
        if (ld) break;
      } catch (err) { /* malformed third-party JSON-LD tolerated */ }
    }

    // 4b. Label/value lookup over the visible definition rows.
    const rows = Array.from(document.querySelectorAll('tr, li, dl > div, .row, div'));
    const labelValue = (labels) => {
      for (const row of rows) {
        const text = clean(row.innerText || row.textContent);
        if (!text || text.length > 200) continue;
        for (const label of labels) {
          const match = text.match(new RegExp('^' + label + '\\s*[:\\n]?\\s*(.+)$', 'i'));
          if (match && clean(match[1])) return clean(match[1]);
        }
      }
      return null;
    };

    const heading = clean(document.querySelector('h1')?.textContent);
    const name = (ld && ld.name) || labelValue(['Firma', 'Firmenname']) || heading;
    const legalForm = labelValue(['Rechtsform', 'Rechtsform der Firma']);
    let seat = labelValue(['Sitz', 'Domizil', 'Firmensitz']);
    if (!seat && ld) {
      seat = [ld.postalCode, ld.locality].filter(Boolean).join(' ') || null;
    }

    return {
      url: location.href,
      title: document.title,
      name,
      legalForm,
      seat,
      street: ld ? ld.street : null,
      plz: ld ? ld.postalCode : null,
      ort: ld ? ld.locality : null,
      bodySample: clean((document.body?.innerText || '').slice(0, 1500)),
    };
  });

  if (!identityMatches(extracted.name) && !identityMatches(extracted.title)) {
    await fail('profile identity mismatch (title: ' + JSON.stringify(extracted.title) + ')');
  }

  const sourceUrl = extracted.url || profileUrl;
  const fields = {};
  const put = (key, value) => { if (value) fields[key] = { value, source_url: sourceUrl }; };
  put('firma_name', extracted.name);
  put('firma_rechtsform', extracted.legalForm);
  put('firma_sitz', extracted.seat);
  put('firma_anschrift', extracted.street);
  put('firma_plz', extracted.plz);
  put('firma_ort', extracted.ort);

  const count = Object.keys(fields).length;
  if (count < 2) {
    const wall = LOGIN_RE.test(extracted.bodySample || '') || LOGIN_RE.test(bodyText);
    await fail(
      (wall ? 'paywall/login wall: public part exposes no usable fields — ' : '')
        + 'only ' + count + ' field(s) extracted (title: ' + JSON.stringify(extracted.title) + ')',
      fields,
    );
  }

  console.log(JSON.stringify({
    target: TARGET,
    input: company,
    fetched_at: new Date().toISOString(),
    fields,
  }, null, 2));
  process.exit(0);
} finally {
  await browser.close().catch(() => null);
}
