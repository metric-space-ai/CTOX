#!/usr/bin/env node
// companyhouse.de — standalone live probe (plain Playwright, no CTOX stack).
//
// Usage: node scrape-targets/companyhouse.de/solo/probe.mjs "<company name>"
//
// Prints one JSON object:
//   {target, input, fetched_at, fields: {<field_key>: {value, source_url}}}
// Exit 0 only when at least one real prospect field was extracted from the
// LIVE site. Otherwise exit 1 with {reason: "..."}.
//
// Politeness: default UA, >=2 s between navigations, consent dialog handled
// by clicking its visible accept button. A passive (non-interactive)
// Cloudflare/Turnstile check is allowed to settle by waiting + reload; an
// interactive CAPTCHA is NEVER solved — the probe then reports blocked.

import { chromium } from 'playwright';

const TARGET = 'companyhouse.de';
const HOME = 'https://www.companyhouse.de/';
const NAV_GAP_MS = 2200;

const company = (process.argv[2] || '').trim();
if (!company) {
  console.log(JSON.stringify({ target: TARGET, input: '', reason: 'missing company CLI argument' }));
  process.exit(1);
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function normalizedIdentity(value) {
  return String(value || '')
    .normalize('NFKD')
    .replace(/\p{M}/gu, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ' ')
    .trim();
}

async function challengePresent(page) {
  const title = await page.title().catch(() => '');
  const text = await page.locator('body').innerText({ timeout: 3000 }).catch(() => '');
  const corpus = `${title} ${text}`.toLowerCase();
  return /cloudflare|cf-chl-|cf-mitigated|challenge-platform|turnstile|sicherheits(?:ü|u)berpr(?:ü|u)fung|noch einen schritt|nur einen moment|just a moment|captcha|verify (?:that )?you are human|nat(?:ü|u)rlichen zugriff|access denied|request blocked|wurden gesperrt|zugriff.{0,40}gesperrt/.test(corpus);
}

async function dismissConsent(page) {
  const button = page
    .getByRole('button', { name: /^(alle akzeptieren|akzeptieren|accept all|zustimmen|einverstanden)$/i })
    .first();
  if (await button.isVisible({ timeout: 1500 }).catch(() => false)) {
    await button.click({ timeout: 3000 }).catch(() => {});
    await page.waitForTimeout(600);
  }
}

// Let a PASSIVE interstitial clear on its own (JS runs by itself in a real
// browser). We never click an interactive "I am human" widget. Poll for up
// to ~30 s WITHOUT reloading (a reload restarts the challenge JS); only if
// it never settles do we try exactly one reload as a last resort.
async function settlePassiveChallenge(page) {
  const deadline = Date.now() + 30000;
  while (Date.now() < deadline) {
    if (!(await challengePresent(page))) return true;
    await page.waitForTimeout(2000);
  }
  await page.reload({ waitUntil: 'domcontentloaded', timeout: 15000 }).catch(() => null);
  const retryDeadline = Date.now() + 15000;
  while (Date.now() < retryDeadline) {
    if (!(await challengePresent(page))) return true;
    await page.waitForTimeout(2000);
  }
  return !(await challengePresent(page));
}

async function main() {
  const browser = await chromium.launch({ headless: true });
  // The site WAF hard-blocks (403) any UA announcing "HeadlessChrome". Use
  // the browser's own stable Chrome UA — same browser build, no stealth
  // plugin, mirrors the alignBrowserIdentity() approach in scripts/v1.js.
  const version = browser.version();
  const userAgent = version
    ? `Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/${version} Safari/537.36`
    : undefined;
  const context = await browser.newContext({ locale: 'de-DE', userAgent });
  const page = await context.newPage();

  const fields = {};
  const addField = (key, value, sourceUrl) => {
    const clean = String(value || '').replace(/\s+/g, ' ').trim();
    if (clean) fields[key] = { value: clean, source_url: sourceUrl };
  };

  let reason = null;
  try {
    // 1) Provider search page. Live finding (29.07.2026): the old `/s/`
    //    route is dead (404/403); search results live under
    //    `/Suche/<query>` where spaces MUST be `+` — a `%20`-encoded path
    //    trips the Cloudflare WAF into a hard 403 block page.
    const searchUrl = `${HOME}Suche/${encodeURIComponent(company).replace(/%20/g, '+')}`;
    await page.goto(searchUrl, { waitUntil: 'domcontentloaded', timeout: 30000 });
    await page.waitForTimeout(2500);
    await dismissConsent(page);
    if (await challengePresent(page)) {
      if (!(await settlePassiveChallenge(page))) {
        reason = 'blocked: cloudflare/captcha interstitial persisted without interaction';
      }
    }

    // 3) From the search results, follow the exact company profile link.
    let profileUrl = null;
    if (!reason) {
      const expected = normalizedIdentity(company);
      profileUrl = await page.locator('a[href]').evaluateAll((anchors, expectedValue) => {
        const normalize = (value) => String(value || '')
          .normalize('NFKD').replace(/\p{M}/gu, '')
          .toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
        const best = anchors.map((anchor) => {
          try {
            const url = new URL(anchor.href, document.baseURI);
            const host = url.hostname.toLowerCase().replace(/\.$/, '');
            const segments = url.pathname.split('/').filter(Boolean);
            const text = normalize(anchor.textContent);
            // Live finding (29.07.2026): company profiles moved from
            // `/<Firma>-<Ort>` to `/<Firma>-<Ort>/<Register>` (e.g.
            // `/BNT-Chemicals-GmbH-Bitterfeld-Wolfen/HRB-15222`).
            const first = (segments[0] || '').toLowerCase();
            const nonProfile = ['login', 'register', 'suche', 'search', 'impressum', 'agb',
              'datenschutz', 'faq', 'preise', 'kontakt', 's', 'l', 'person', 'personen'];
            const profilePath = segments.length >= 1 && segments.length <= 2
              && !nonProfile.includes(first);
            return {
              href: url.href,
              exact: text === expectedValue,
              matching: text.length > 0 && (text.includes(expectedValue) || expectedValue.includes(text)),
              providerOwned: url.protocol === 'https:'
                && (host === 'companyhouse.de' || host.endsWith('.companyhouse.de'))
                && profilePath,
            };
          } catch {
            return null;
          }
        }).filter((item) => item && item.providerOwned && item.matching)
          .sort((a, b) => Number(b.exact) - Number(a.exact))[0];
        return best ? best.href : null;
      }, expected).catch(() => null);
      if (!profileUrl) reason = 'no provider-owned profile link matched the company on the search page';
    }

    // 4) Profile page extraction.
    if (!reason) {
      await sleep(NAV_GAP_MS);
      await page.goto(profileUrl, { waitUntil: 'domcontentloaded', timeout: 30000 });
      await page.waitForTimeout(2500);
      await dismissConsent(page);
      if (await challengePresent(page)) {
        if (!(await settlePassiveChallenge(page))) {
          reason = 'blocked: cloudflare/captcha interstitial persisted on the profile page';
        }
      }
    }

    if (!reason) {
      const sourceUrl = page.url();
      const profile = await page.evaluate(() => {
        const clean = (value) => String(value || '').replace(/\s+/g, ' ').trim() || null;
        const heading = clean(document.querySelector('h1')?.textContent);
        const locationIcon = document.querySelector('[class*="ch-ico-location"]');
        const address = clean(
          locationIcon?.closest('div')?.querySelector('p')?.textContent
            || locationIcon?.parentElement?.nextElementSibling?.textContent,
        );
        const detail = (label) => {
          const headers = Array.from(document.querySelectorAll('[class*="tile-table-header"]'));
          const header = headers.find((node) => clean(node.textContent) === label);
          if (!header) return null;
          const row = header.parentElement;
          return clean(row?.querySelector('[class*="mb-3"]')?.textContent
            || header.nextElementSibling?.textContent);
        };
        return {
          heading,
          address,
          telephone: detail('Telefonnummer'),
          email: detail('E-Mail'),
          website: detail('Webseite'),
        };
      });

      addField('firma_name', profile.heading, sourceUrl);
      if (profile.address) {
        const match = profile.address.match(/^(.+?),\s*(\d{5})\s+(.+)$/u);
        if (match) {
          addField('firma_anschrift', match[1], sourceUrl);
          addField('firma_plz', match[2], sourceUrl);
          addField('firma_ort', match[3], sourceUrl);
        } else {
          addField('firma_anschrift', profile.address, sourceUrl);
        }
      }
      addField('firma_telefon', profile.telephone, sourceUrl);
      addField('firma_email', profile.email, sourceUrl);
      addField('firma_domain', profile.website, sourceUrl);

      // Identity guard: the extracted name must match the requested company.
      const expected = normalizedIdentity(company);
      const got = normalizedIdentity(fields.firma_name?.value);
      if (Object.keys(fields).length > 0 && (!got || !(got.includes(expected) || expected.includes(got)))) {
        reason = `identity mismatch: profile heading "${fields.firma_name?.value}" does not match "${company}"`;
        for (const key of Object.keys(fields)) delete fields[key];
      }
      if (!reason && Object.keys(fields).length === 0) {
        reason = 'portal drift: profile page loaded but no known selectors matched';
      }
    }
  } catch (error) {
    reason = `fatal: ${String(error?.message || error).slice(0, 300)}`;
  } finally {
    await browser.close().catch(() => {});
  }

  const output = {
    target: TARGET,
    input: company,
    fetched_at: new Date().toISOString(),
    fields,
  };
  if (reason) output.reason = reason;
  console.log(JSON.stringify(output, null, 2));
  process.exit(reason ? 1 : 0);
}

main();
