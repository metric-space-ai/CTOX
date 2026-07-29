// northdata.de — solo live probe (plain Playwright, no CTOX stack).
//
// Usage: node scrape-targets/northdata.de/solo/probe.mjs "<company name>"
//
// Drives the LIVE site headless, extracts the prospect.v1 field set for the
// company and prints ONE JSON object:
//   {target, input, fetched_at, fields: {<field_key>: {value, source_url}}}
// Exit 0 only when at least 3 non-empty fields were extracted; otherwise
// non-zero with a "reason" field.

import { chromium } from 'playwright';

const TARGET = 'northdata.de';
const HOME = 'https://www.northdata.de/';

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

const browser = await chromium.launch({ headless: true });
const context = await browser.newContext({ locale: 'de-DE' });
const page = await context.newPage();

try {
  // ------------------------------------------------------------------
  // 1. Portal search: GET /<company> performs the Northdata query.
  // ------------------------------------------------------------------
  const searchUrl = HOME + encodeURIComponent(company).replace(/%20/g, '+');
  let response;
  try {
    response = await page.goto(searchUrl, { waitUntil: 'domcontentloaded', timeout: 45000 });
  } catch (err) {
    fail('navigation to search failed: ' + String(err.message || err).split('\n')[0]);
  }
  const status = (response && response.status()) || 0;
  if (status === 403 || status === 429 || status >= 500) {
    fail('search returned HTTP ' + status);
  }
  await sleep(2000); // politeness between navigations

  // Ordinary consent dialog: click its visible button once.
  const consentPatterns = [/alle akzeptieren/i, /akzeptieren/i, /zustimmen/i, /einverstanden/i];
  for (const pattern of consentPatterns) {
    const button = page.getByRole('button', { name: pattern }).first();
    if (await button.count()) {
      await button.click({ timeout: 3000 }).catch(() => null);
      await sleep(2000);
      break;
    }
  }

  const bodyText = await page.locator('body').innerText().catch(() => '');
  if (/captcha|zugriff verweigert|access denied|gesperrt/i.test(bodyText)) {
    fail('blocked: anti-bot page detected');
  }

  // ------------------------------------------------------------------
  // 2. Resolve the canonical company profile route from the search page
  //    (or use the page itself if we already landed on a profile).
  // ------------------------------------------------------------------
  const profileUrl = await page.evaluate((companyName) => {
    const normalize = (value) => String(value || '').normalize('NFKD')
      .replace(/[̀-ͯ]/g, '').toLowerCase().replace(/ß/g, 'ss')
      .replace(/[^a-z0-9]+/g, ' ').trim();
    const legalTokens = new Set(['ag', 'gmbh', 'kg', 'mbh', 'se', 'und']);
    const identityTokens = (value) => normalize(value).split(/\s+/)
      .filter((token) => token.length >= 3 && !legalTokens.has(token));
    const identityMatches = (value) => {
      const tokens = identityTokens(companyName);
      const corpus = normalize(value);
      return tokens.length > 0 && corpus.length > 0
        && tokens.filter((token) => corpus.includes(token)).length >= Math.max(1, Math.ceil(tokens.length * 0.75));
    };
    const canonicalProfileRoute = (href) => {
      try {
        const candidate = new URL(href, location.href);
        const segments = candidate.pathname.split('/').filter(Boolean).map(decodeURIComponent);
        return candidate.protocol === 'https:'
          && candidate.hostname.toLowerCase().replace(/^www\./, '') === 'northdata.de'
          && segments.length >= 2
          && identityMatches(segments[0]);
      } catch (err) {
        return false;
      }
    };
    const declared = document.querySelector('link[rel~="canonical"]')
      ? document.querySelector('link[rel~="canonical"]').href : null;
    const ogUrl = document.querySelector('meta[property="og:url"]');
    const declaredUrl = declared || (ogUrl ? ogUrl.content : null);
    if (declaredUrl && canonicalProfileRoute(declaredUrl)) return declaredUrl;
    if (canonicalProfileRoute(location.href)) return location.href;
    const links = Array.from(document.querySelectorAll('a[href]')).map((a) => a.href);
    return links.find((href) => canonicalProfileRoute(href)) || null;
  }, company);

  if (!profileUrl) fail('no canonical Northdata profile route found on the search page');

  if (profileUrl !== page.url()) {
    await sleep(2000); // politeness between navigations
    try {
      response = await page.goto(profileUrl, { waitUntil: 'domcontentloaded', timeout: 45000 });
    } catch (err) {
      fail('navigation to profile failed: ' + String(err.message || err).split('\n')[0]);
    }
    const profileStatus = (response && response.status()) || 0;
    if (profileStatus === 403 || profileStatus === 429 || profileStatus >= 500) {
      fail('profile returned HTTP ' + profileStatus);
    }
  }

  // Wait for a profile marker (ribbon headings or qualified h1).
  await page.waitForFunction(() => Boolean(
    document.querySelector('h1.qualified')
      || Array.from(document.querySelectorAll("h3[class*='ribbon'], dt, [data-label]"))
        .some((node) => /^(name|adresse|anschrift)$/i.test((node.textContent || '').trim())),
  ), null, { timeout: 15000 }).catch(() => null);
  await sleep(2000);

  // ------------------------------------------------------------------
  // 3. Extract the prospect fields from the profile page DOM.
  // ------------------------------------------------------------------
  const extracted = await page.evaluate(() => {
    const clean = (value) => String(value || '').replace(/\s+/g, ' ').trim() || null;
    const normalizedLabel = (value) => (clean(value) || '').toLowerCase();
    const nextElement = (start) => {
      if (start && start.firstElementChild) return start.firstElementChild;
      let node = start;
      while (node) {
        if (node.nextElementSibling) return node.nextElementSibling;
        node = node.parentElement;
      }
      return null;
    };
    const ribbonValue = (label) => {
      const expectedLabel = normalizedLabel(label);
      const headings = Array.from(document.querySelectorAll("h3[class*='ribbon'], dt, [data-label]"));
      const heading = headings.find((node) => normalizedLabel(
        node.getAttribute('data-label') || node.textContent || '',
      ) === expectedLabel);
      if (!heading) return null;
      let node = nextElement(heading);
      for (let inspected = 0; node && inspected < 80; inspected += 1) {
        if (node.matches("h3[class*='ribbon'], dt, [data-label]")) break;
        let valueNode = null;
        if (node.matches('.content, dd, [data-value]')) valueNode = node;
        if (node.matches('.general-information, li')) {
          valueNode = node.querySelector('.content, dd, [data-value]') || node;
        }
        const value = (valueNode && valueNode.getAttribute && valueNode.getAttribute('data-value'))
          || (valueNode && valueNode.textContent);
        if (clean(value)) return clean(value);
        node = nextElement(node);
      }
      return null;
    };
    const heading = clean(document.querySelector('h1.qualified')
      ? document.querySelector('h1.qualified').textContent : null);
    const name = ribbonValue('Name') || (heading ? heading.split(',')[0].trim() : null);
    const address = ribbonValue('Adresse') || ribbonValue('Anschrift');
    let street = null;
    let plz = null;
    let ort = null;
    if (address) {
      const parts = address.split(',').map((part) => part.trim()).filter(Boolean);
      street = parts[0] || address;
      const last = parts[parts.length - 1] || '';
      const plzMatch = last.match(/\b(\d{4,5})\b\s*(.*)/);
      if (parts.length >= 2 && plzMatch) {
        plz = plzMatch[1];
        ort = clean(plzMatch[2]);
      } else if (parts.length >= 2) {
        ort = last;
      }
    }
    // Management persons rendered as <figure class="bizq" data-data='[...]'>.
    const persons = [];
    for (const figure of document.querySelectorAll('figure.bizq[data-data]')) {
      try {
        const data = JSON.parse(figure.getAttribute('data-data'));
        const items = Array.isArray(data) ? data : (data.items || []);
        for (const item of items) {
          if (item && !item.old && typeof item.text === 'string') persons.push(item.text);
        }
      } catch (err) { /* selector drift tolerated */ }
    }
    let person = null;
    for (const text of persons) {
      const match = text.trim().match(/^([A-Za-zÄÖÜäöü\-\s.]+?)\s+([A-ZÄÖÜ][A-Za-zÄÖÜäöü-]+)\s+([A-ZÄÖÜ][A-Za-zÄÖÜäöü-]+(?:\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöü-]+)*)$/);
      if (match) {
        person = { position: match[1].trim(), first: match[2].trim(), last: match[3].trim() };
        break;
      }
    }
    return { url: location.href, title: document.title, name, street, plz, ort, heading, person };
  });

  const sourceUrl = extracted.url || profileUrl;
  const fields = {};
  const put = (key, value) => {
    if (value) fields[key] = { value, source_url: sourceUrl };
  };
  put('firma_name', extracted.name);
  put('firma_anschrift', extracted.street);
  put('firma_plz', extracted.plz);
  put('firma_ort', extracted.ort);
  if (extracted.person) {
    put('person_position', extracted.person.position);
    put('person_vorname', extracted.person.first);
    put('person_nachname', extracted.person.last);
  }

  const count = Object.keys(fields).length;
  if (count < 3) {
    fail('only ' + count + ' field(s) extracted (title: ' + JSON.stringify(extracted.title) + ')', fields);
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
