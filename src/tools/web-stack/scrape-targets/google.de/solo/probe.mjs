// google.de — standalone live probe (plain Playwright, no CTOX runtime).
//
// Usage: node scrape-targets/google.de/solo/probe.mjs "<company name>"
//
// Drives the LIVE google.de site (consent handling included), discovers the
// company's own website from the organic results, visits it (politely,
// >= 2 s between navigations, default user agent) and prints ONE JSON
// object:
//   { target, input, fetched_at, fields: { <field_key>: { value, source_url } } }
// Exit 0 only when real prospect fields were extracted; non-zero with a
// `reason` field otherwise. NEVER solves or bypasses a CAPTCHA: a hard bot
// challenge is reported as reason "blocked".

import { chromium } from "playwright";

const TARGET = "google.de";
const MIN_NAV_GAP_MS = 2000;

// Hosts that are directories/aggregators, never the company's own site.
const DIRECTORY_HOST_PARTS = [
  "google.", "linkedin.", "xing.", "facebook.", "instagram.", "youtube.",
  "wikipedia.", "northdata.", "companyhouse.", "moneyhouse.", "firmenabc.",
  "bundesanzeiger.", "handelsregister.", "firmenwissen.", "werkenntdenbesten.",
  "11880.", "golocal.", "kennstdueinen.", "cylex.", "dastelefonbuch.",
  "gelbeseiten.", "wlw.de", "europages.", "kompass.", "jobvector.",
  "indeed.", "stepstone.", "kununu.", "glassdoor.", "partcommunity.",
];

function fail(reason, extra = {}) {
  process.stdout.write(JSON.stringify({
    target: TARGET,
    input: process.argv[2] || null,
    fetched_at: new Date().toISOString(),
    fields: {},
    reason,
    ...extra,
  }, null, 2) + "\n");
  process.exit(1);
}

function succeed(fields) {
  process.stdout.write(JSON.stringify({
    target: TARGET,
    input: process.argv[2],
    fetched_at: new Date().toISOString(),
    fields,
  }, null, 2) + "\n");
  process.exit(0);
}

function normalizedTokens(value) {
  const legalForms = new Set(["ag", "gmbh", "mbh", "se", "kg", "kgaa", "ohg", "ug", "ltd", "inc"]);
  return String(value || "")
    .toLocaleLowerCase("de-DE")
    .normalize("NFKD")
    .replace(/[^a-z0-9äöüß]+/gi, " ")
    .split(/\s+/)
    .filter((token) => token.length >= 3 && !legalForms.has(token));
}

function identityMatches(company, corpus) {
  const tokens = normalizedTokens(company);
  if (tokens.length === 0) return false;
  const haystack = String(corpus || "").toLocaleLowerCase("de-DE").normalize("NFKD");
  return tokens.every((token) => haystack.includes(token));
}

function hostOf(url) {
  try {
    return new URL(url).hostname.replace(/^www\./, "").toLowerCase();
  } catch {
    return "";
  }
}

function isDirectoryHost(url) {
  const host = hostOf(url);
  if (!host) return true;
  return DIRECTORY_HOST_PARTS.some((part) => host.includes(part));
}

function looksBlocked(url, text) {
  return /\/sorry\//.test(url)
    || /unusual traffic|ungewöhnlicher Datenverkehr|not a robot|kein Roboter|automatisierte Anfragen/i.test(text || "");
}

async function politeGoto(page, url, state) {
  const elapsed = Date.now() - state.lastNav;
  if (elapsed < MIN_NAV_GAP_MS) {
    await page.waitForTimeout(MIN_NAV_GAP_MS - elapsed);
  }
  state.lastNav = Date.now();
  return page.goto(url, { waitUntil: "domcontentloaded", timeout: 30000 });
}

async function clickConsent(page) {
  const names = [
    /alle ablehnen/i, /alles ablehnen/i, /reject all/i,
    /alle akzeptieren/i, /alles akzeptieren/i, /accept all/i,
    /zustimmen/i, /akzeptieren/i, /i agree/i,
  ];
  for (const name of names) {
    const button = page.getByRole("button", { name }).first();
    if (await button.count() > 0 && await button.isVisible().catch(() => false)) {
      await button.click({ timeout: 3000 }).catch(() => null);
      await page.waitForLoadState("domcontentloaded", { timeout: 10000 }).catch(() => null);
      return true;
    }
  }
  return false;
}

async function extractResults(page) {
  return page.evaluate(() => {
    const out = [];
    for (const h3 of document.querySelectorAll("h3")) {
      const anchor = h3.closest("a[href]");
      if (!anchor) continue;
      const url = anchor.href;
      if (!/^https?:/.test(url)) continue;
      let container = anchor;
      for (let i = 0; i < 6 && container.parentElement; i += 1) {
        container = container.parentElement;
        if ((container.innerText || "").length > (anchor.innerText || "").length + 60) break;
      }
      out.push({
        url,
        title: (h3.innerText || "").replace(/\s+/g, " ").trim(),
        snippet: (container.innerText || "").replace(/\s+/g, " ").trim().slice(0, 600),
      });
    }
    return out;
  });
}

function extractContacts(text) {
  const emails = [...String(text).matchAll(/\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b/gi)]
    .map((m) => m[0].toLowerCase())
    .filter((email) => !/\.(?:png|jpe?g|gif|webp|svg)$/i.test(email))
    .filter((email) => !/(?:example|sentry|wixpress|googleapis|gstatic)/i.test(email));
  const phones = [...String(text).matchAll(/(?:\+|00)\d[\d\s()\/-]{7,}\d/g)]
    .map((m) => m[0].replace(/\s+/g, " ").trim());
  const postal = String(text).match(/\b(?:D-)?(\d{5})\s+([A-ZÄÖÜ][A-Za-zÄÖÜäöüß.'-]+)/);
  return { emails: [...new Set(emails)], phones: [...new Set(phones)], postal };
}

async function launchBrowser() {
  // Prefer the genuine installed Chrome (no stealth plugins, default UA);
  // fall back to the bundled Chromium when Chrome is not installed.
  // Pass --headed to open a visible window (default UA kept either way).
  const headless = !process.argv.includes("--headed");
  try {
    return await chromium.launch({ channel: "chrome", headless });
  } catch {
    return chromium.launch({ headless });
  }
}

async function main() {
  const company = (process.argv[2] || "").trim();
  if (!company) fail("missing_company_argument");

  const browser = await launchBrowser();
  try {
    const context = await browser.newContext({ locale: "de-DE" });
    const page = await context.newPage();
    const state = { lastNav: 0 };

    await politeGoto(
      page,
      "https://www.google.de/search?q=" + encodeURIComponent(company) + "&hl=de",
      state,
    );
    await clickConsent(page);
    await page.waitForSelector("h3", { timeout: 15000 }).catch(() => null);

    const serpText = await page.locator("body").innerText().catch(() => "");
    if (looksBlocked(page.url(), serpText)) {
      fail("blocked", { detail: "google presented a bot challenge on the results page", url: page.url() });
    }

    const results = await extractResults(page);
    if (results.length === 0) {
      fail("selector_drift", { detail: "no organic h3 results on loaded SERP", url: page.url() });
    }

    const hit = results.find((r) =>
      identityMatches(company, `${r.title} ${r.snippet}`) && !isDirectoryHost(r.url));
    if (!hit) {
      fail("no_company_site_result", {
        detail: "no organic result matched the company identity on a non-directory host",
        candidates: results.slice(0, 8).map((r) => ({ url: r.url, title: r.title })),
      });
    }

    const fields = {};
    const serpUrl = page.url();
    fields.firma_name = { value: company, source_url: hit.url };
    fields.firma_domain = { value: hostOf(hit.url), source_url: hit.url };

    // Visit the company's own site for contact details (polite gap enforced).
    await politeGoto(page, hit.url, state);
    await clickConsent(page);
    await page.waitForLoadState("networkidle", { timeout: 8000 }).catch(() => null);
    let text = await page.locator("body").innerText().catch(() => "");
    let contact = extractContacts(text);

    if (contact.emails.length === 0 || contact.phones.length === 0) {
      // Try the impressum page — German sites must publish one.
      const imprint = page.getByRole("link", { name: /impressum|legal notice|imprint/i }).first();
      if (await imprint.count() > 0) {
        const href = await imprint.getAttribute("href").catch(() => null);
        const target = href ? new URL(href, page.url()).href : null;
        if (target && hostOf(target) === hostOf(page.url())) {
          await politeGoto(page, target, state);
          text = await page.locator("body").innerText().catch(() => "");
          const deeper = extractContacts(text);
          contact = {
            emails: [...new Set([...contact.emails, ...deeper.emails])],
            phones: [...new Set([...contact.phones, ...deeper.phones])],
            postal: deeper.postal || contact.postal,
          };
        }
      }
    }

    const siteUrl = page.url();
    if (contact.emails[0]) fields.firma_email = { value: contact.emails[0], source_url: siteUrl };
    if (contact.phones[0]) fields.firma_telefon = { value: contact.phones[0], source_url: siteUrl };
    if (contact.postal) {
      fields.firma_plz = { value: contact.postal[1], source_url: siteUrl };
      fields.firma_ort = { value: contact.postal[2], source_url: siteUrl };
    }

    const nonEmpty = Object.values(fields).filter((f) => String(f.value || "").trim()).length;
    if (nonEmpty < 3) {
      fail("insufficient_extraction", { detail: `only ${nonEmpty} non-empty fields`, fields });
    }
    succeed(fields);
  } finally {
    await browser.close().catch(() => null);
  }
}

main().catch((error) => fail("probe_error", { detail: String(error?.message || error) }));
