// firmenabc.at — solo live probe (plain Playwright, no CTOX stack).
//
// Usage: node scrape-targets/firmenabc.at/solo/probe.mjs "<company name>"
//
// Drives the LIVE site headless, extracts the prospect.v1 field set for the
// company and prints ONE JSON object:
//   {target, input, fetched_at, fields: {<field_key>: {value, source_url}}}
// Exit 0 only when firma_name plus at least 2 further fields were extracted;
// otherwise non-zero with a "reason" field.
//
// A bot challenge is reported as a hard block and never solved or evaded.
//
// Site mechanics mirrored from ../scripts/v1.js:
//   * The company profile lives at https://www.firmenabc.at/<slug>_<id>.
//   * The profile body carries the anchor line "Informationen zur
//     Firmenstruktur", followed by the street line, a "<plz> <ort>" line and
//     the "T:" / "M:" / "W:" contact lines.
// Search is done through the site's own TYPO3 form: the results route needs a
// server-issued `cHash`, so the form is submitted instead of a URL being built.

import { chromium } from 'playwright';

const TARGET = 'firmenabc.at';
const HOME = 'https://www.firmenabc.at/';
const ALLOWED_HOST = 'firmenabc.at';

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

function fail(reason, fields) {
  console.log(JSON.stringify({
    target: TARGET,
    input: company,
    fetched_at: new Date().toISOString(),
    fields: fields || {},
    reason,
  }));
  process.exit(1);
}

// ---- identity matching (same rules as scripts/v1.js) -----------------------
const normalize = (value) => String(value || '')
  .normalize('NFKD')
  .replace(/[̀-ͯ]/g, '')
  .toLowerCase()
  .replace(/ß/g, 'ss')
  .replace(/[^a-z0-9]+/g, ' ')
  .trim();

const LEGAL_TOKENS = new Set(['ag', 'co', 'gmbh', 'kg', 'mbh', 'og', 'se', 'und', 'company']);

const identityTokens = (value) => normalize(value).split(/\s+/)
  .filter((token) => token.length >= 3 && !LEGAL_TOKENS.has(token));

function identityMatches(corpus) {
  const tokens = identityTokens(company);
  const haystack = normalize(corpus);
  if (tokens.length === 0 || !haystack) return false;
  const hits = tokens.filter((token) => haystack.includes(token)).length;
  return hits >= Math.max(1, Math.ceil(tokens.length * 0.75));
}

const CHALLENGE_RE = /captcha|cloudflare|cf-chl-|challenge-platform|turnstile|verify (?:that )?you are human|zugriff verweigert|access denied|request blocked|too many requests|zu viele anfragen|gesperrt/i;

const browser = await chromium.launch({ headless: true });
const context = await browser.newContext({ locale: 'de-AT' });
const page = await context.newPage();

try {
  // ------------------------------------------------------------------
  // 1. Home page + ordinary cookie consent.
  // ------------------------------------------------------------------
  let response;
  try {
    response = await page.goto(HOME, { waitUntil: 'domcontentloaded', timeout: 45000 });
  } catch (err) {
    fail('navigation to home failed: ' + String(err.message || err).split('\n')[0]);
  }
  await sleep(2500);

  // The edge occasionally answers the very first hit with a 429 interstitial
  // that reloads itself; one polite retry, then it counts as a hard block.
  if ([403, 429].includes((response && response.status()) || 0)) {
    await sleep(5000);
    response = await page.goto(HOME, { waitUntil: 'domcontentloaded', timeout: 45000 })
      .catch(() => null);
    await sleep(2500);
    const retryStatus = (response && response.status()) || 0;
    if ([403, 429].includes(retryStatus)) {
      fail('blocked: home returned HTTP ' + retryStatus + ' twice (rate limit / challenge)');
    }
  }

  for (const pattern of [/alle akzeptieren/i, /akzeptieren/i, /zustimmen/i, /einverstanden/i]) {
    const button = page.getByRole('button', { name: pattern }).first();
    if (await button.count()) {
      await button.click({ timeout: 3000 }).catch(() => null);
      await sleep(1500);
      break;
    }
  }

  const homeText = await page.locator('body').innerText().catch(() => '');
  if (CHALLENGE_RE.test(homeText)) fail('blocked: anti-bot page detected on home');

  // ------------------------------------------------------------------
  // 2. Submit the site's own search form (the results route needs the
  //    server-issued cHash, so a hand-built URL returns zero hits).
  // ------------------------------------------------------------------
  const searchField = page.locator('#whatSearchField').first();
  if (!(await searchField.count())) fail('search field #whatSearchField not found on home page');
  await searchField.fill(company);
  await sleep(1000); // politeness before submitting
  await searchField.press('Enter');
  await page.waitForLoadState('domcontentloaded', { timeout: 30000 }).catch(() => null);
  await sleep(3000);

  const resultsText = await page.locator('body').innerText().catch(() => '');
  if (CHALLENGE_RE.test(resultsText)) fail('blocked: anti-bot page detected on search results');

  // ------------------------------------------------------------------
  // 3. Pick the company profile route from the result list.
  //    Profile routes look like https://www.firmenabc.at/<slug>_<id>.
  // ------------------------------------------------------------------
  const profileUrl = await page.evaluate(({ companyName, host }) => {
    const norm = (value) => String(value || '').normalize('NFKD')
      .replace(/[̀-ͯ]/g, '').toLowerCase().replace(/ß/g, 'ss')
      .replace(/[^a-z0-9]+/g, ' ').trim();
    const legal = new Set(['ag', 'co', 'gmbh', 'kg', 'mbh', 'og', 'se', 'und', 'company']);
    const tokens = norm(companyName).split(/\s+/).filter((t) => t.length >= 3 && !legal.has(t));
    const matches = (corpus) => {
      const hay = norm(corpus);
      if (tokens.length === 0 || !hay) return false;
      return tokens.filter((t) => hay.includes(t)).length >= Math.max(1, Math.ceil(tokens.length * 0.75));
    };
    const isProfile = (href) => {
      try {
        const url = new URL(href, location.href);
        if (url.protocol !== 'https:') return false;
        if (url.hostname.toLowerCase().replace(/^www\./, '') !== host) return false;
        if (url.hash) return false;
        const segments = url.pathname.split('/').filter(Boolean);
        // one segment, "<slug>_<base62 id>" — excludes /firmen/..., /suche/...
        return segments.length === 1 && /_[A-Za-z0-9]{2,}$/.test(segments[0]);
      } catch (err) {
        return false;
      }
    };
    const candidates = Array.from(document.querySelectorAll('a[href]'))
      .map((a) => ({ href: a.href, text: (a.textContent || '').replace(/\s+/g, ' ').trim() }))
      .filter((item) => isProfile(item.href) && item.text);
    // Prefer an exact normalized name match over a mere token match, so
    // "Red Bull GmbH" does not resolve to "Red Bull Media House GmbH".
    const exact = candidates.find((item) => norm(item.text) === norm(companyName));
    if (exact) return exact.href;
    const loose = candidates.find((item) => matches(item.text));
    return loose ? loose.href : null;
  }, { companyName: company, host: ALLOWED_HOST });

  if (!profileUrl) fail('no matching FirmenABC profile route found on the search results page');

  await sleep(2000); // politeness between navigations
  try {
    response = await page.goto(profileUrl, { waitUntil: 'domcontentloaded', timeout: 45000 });
  } catch (err) {
    fail('navigation to profile failed: ' + String(err.message || err).split('\n')[0]);
  }
  const profileStatus = (response && response.status()) || 0;
  if ([403, 429].includes(profileStatus) || profileStatus >= 500) {
    fail('profile returned HTTP ' + profileStatus);
  }
  await sleep(2500);

  // ------------------------------------------------------------------
  // 4. Extract from the profile body, using the same anchor/line layout
  //    that scripts/v1.js bodyProfile() relies on.
  // ------------------------------------------------------------------
  const profileText = await page.locator('body').innerText().catch(() => '');
  if (CHALLENGE_RE.test(profileText)) fail('blocked: anti-bot page detected on profile');

  const title = await page.title();
  const lines = profileText.split(/\r?\n/)
    .map((line) => line.replace(/\s+/g, ' ').trim())
    .filter(Boolean);
  const anchor = lines.findIndex((line) => normalize(line) === 'informationen zur firmenstruktur');
  const profileLines = anchor >= 0 ? lines.slice(anchor + 1, anchor + 16) : [];
  const postalIndex = profileLines.findIndex((line) => /^\d{4}\s+\S/.test(line));
  const postal = postalIndex >= 0 ? profileLines[postalIndex].match(/^(\d{4})\s+(.+)$/) : null;
  const contact = (prefix) => profileLines
    .find((line) => line.startsWith(prefix))?.slice(prefix.length).trim() || null;

  const name = String(title || '').replace(/\s+in\s+[^|]+(?:\|.*)?$/i, '').trim() || null;
  const street = postalIndex > 0 ? profileLines[postalIndex - 1] : null;

  const sourceUrl = page.url();
  const fields = {};
  const put = (key, value) => {
    const clean = String(value || '').replace(/\s+/g, ' ').trim();
    if (clean) fields[key] = { value: clean, source_url: sourceUrl };
  };
  put('firma_name', name);
  put('firma_anschrift', street);
  put('firma_plz', postal ? postal[1] : null);
  put('firma_ort', postal ? postal[2] : null);
  put('firma_telefon', contact('T:'));
  put('firma_email', contact('M:'));
  const website = contact('W:');
  if (website) {
    try {
      const absolute = /^https?:\/\//i.test(website) ? website : 'https://' + website;
      put('firma_domain', new URL(absolute).hostname.replace(/^www\./, ''));
    } catch (err) { /* unusable website line is simply dropped */ }
  }

  if (!fields.firma_name || !identityMatches(fields.firma_name.value)) {
    fail('profile identity mismatch (title: ' + JSON.stringify(title) + ')', fields);
  }
  const count = Object.keys(fields).length;
  if (count < 3) {
    fail('only ' + count + ' field(s) extracted (title: ' + JSON.stringify(title) + ')', fields);
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
