// bundesanzeiger.de — SOLO probe (plain Playwright, no CTOX harness).
//
// Usage: node scrape-targets/bundesanzeiger.de/solo/probe.mjs "<company name>"
//
// Drives the LIVE site and prints ONE JSON object:
//   { target, input, fetched_at, fields: { <field_key>: { value, source_url } } }
// Exit 0 only on real extraction (>= 2 non-empty prospect fields from
// exact-match result rows); non-zero with a `reason` otherwise.
//
// Hard rule honoured here: the Jahresabschluss documents behind
// "Rechnungslegung/Finanzberichte" are gated by a per-document image
// CAPTCHA ("Sicherheitsabfrage"). This probe NEVER opens those documents
// and never attempts to solve or bypass that challenge. Free-text
// announcements ("Gesellschaftsbekanntmachungen" etc.) of the exact-match
// company are opened instead and mined for address evidence.

import { chromium } from "playwright";

const TARGET = "bundesanzeiger.de";
const SEARCH_URL = "https://www.bundesanzeiger.de/pub/de/suche?0";
const NAV_TIMEOUT_MS = 45_000;
const POLITENESS_MS = 2_200;

const BLOCKED_RE = /schutzma(?:ß|ss)nahme|sicherheitsabfrage|cf-chl-|turnstile|captcha|verify (?:that )?you are human|access denied|request blocked|zugriff verweigert|zu viele anfragen/i;

function fail(reason, extra = {}) {
  process.stdout.write(JSON.stringify({
    target: TARGET,
    input: process.argv[2] || null,
    fetched_at: new Date().toISOString(),
    fields: {},
    reason,
    ...extra,
  }, null, 2));
  process.exit(1);
}

function normalizeCompanyName(value) {
  return String(value || "")
    .toLocaleLowerCase("de-DE")
    .normalize("NFKD")
    .replace(/\p{M}/gu, "")
    .replace(/\s*\((?:vormals|ehemals|fruher|früher)\s*:[^)]+\)\s*$/i, "")
    .replace(/[^a-z0-9]+/g, " ")
    .trim()
    .replace(/\s+/g, " ");
}

async function main() {
  const company = String(process.argv[2] || "").trim();
  if (!company) fail("usage: probe.mjs <company name>");

  const browser = await chromium.launch({ headless: true });
  try {
    const page = await browser.newPage();
    await page.goto(SEARCH_URL, { waitUntil: "domcontentloaded", timeout: NAV_TIMEOUT_MS });
    await page.waitForTimeout(POLITENESS_MS);

    // --- challenge / block detection on first load -----------------------
    const initialText = await page.locator("body").innerText().catch(() => "");
    if (BLOCKED_RE.test(initialText) || /schutzma(?:ß|ss)nahme|to_nlp_start/i.test(await page.title().catch(() => ""))) {
      fail("blocked: visible access challenge on first load");
    }

    // --- ordinary consent dialog: click its visible button ---------------
    for (const pattern of [
      "Nur technisch notwendige Cookies akzeptieren",
      /alle akzeptieren|akzeptieren|zustimmen|verstanden/i,
    ]) {
      const button = page.getByRole("button", { name: pattern }).first();
      if (await button.count()) {
        await button.click({ timeout: 2_500 }).catch(() => null);
        break;
      }
    }
    await page.waitForTimeout(POLITENESS_MS);

    // --- search form ------------------------------------------------------
    const searchInput = page.locator('input[name="fulltext"]');
    if (await searchInput.count() !== 1) {
      fail("drift: fulltext search input not found on search page");
    }
    let onResults = false;
    for (let attempt = 0; attempt < 2 && !onResults; attempt += 1) {
      await searchInput.fill(company);
      const searchButton = page.locator('input[name="search-button"][value="Suchen"]');
      if (await searchButton.count() === 1) {
        await searchButton.click();
      } else {
        await searchInput.press("Enter");
      }
      await page.waitForLoadState("domcontentloaded", { timeout: 30_000 }).catch(() => null);
      onResults = await page
        .waitForSelector(".result_container", { timeout: 20_000 })
        .then(() => true)
        .catch(() => false);
      if (!onResults) {
        const text = await page.locator("body").innerText().catch(() => "");
        if (BLOCKED_RE.test(text)) fail("blocked: access challenge after search submit");
        if (/keine passenden Daten gefunden/i.test(text)) {
          fail("no_results: portal returned zero hits for the company");
        }
        // Wicket sometimes bounces back to the form; retry once before drift.
        if (attempt === 0 && (await searchInput.count()) === 1) {
          await page.waitForTimeout(POLITENESS_MS);
          continue;
        }
        fail("drift: result container selector did not appear after submit");
      }
    }

    // --- parse result rows ------------------------------------------------
    const result = await page.evaluate(() => {
      const container = document.querySelector(".result_container");
      const rows = container
        ? [...container.querySelectorAll(":scope > .row")].map((row) => {
            if (row.classList.contains("result_header")
                || row.classList.contains("concern_list")
                || row.classList.contains("subsidiary_list")) return null;
            const first = row.querySelector(":scope > .col-md-3 .first");
            if (!first) return null;
            const lines = first.innerText.split("\n").map((part) => part.trim()).filter(Boolean);
            const infoLink = row.querySelector(":scope > .col-md-5 .info > a");
            return {
              name: lines[0] || "",
              city: lines[1] || "",
              section: (row.querySelector(":scope > .col-md-2 .part")?.innerText || "").replace(/\s+/g, " ").trim(),
              information: (infoLink?.innerText || "").replace(/\s+/g, " ").trim(),
              href: infoLink?.href || "",
              date: (row.querySelector(":scope > .col-md-2 .date")?.innerText || "").replace(/\s+/g, " ").trim(),
            };
          }).filter((entry) => entry && entry.name)
        : [];
      return { url: location.href, title: document.title, rows };
    });

    const expected = normalizeCompanyName(company);
    const exact = result.rows.filter((row) => normalizeCompanyName(row.name) === expected);
    if (exact.length === 0) {
      fail("no_exact_match: result rows did not contain the exact company name", {
        observed_names: result.rows.slice(0, 5).map((row) => row.name),
      });
    }

    const fields = {
      firma_name: { value: exact[0].name, source_url: result.url },
    };
    if (exact[0].city) {
      fields.firma_ort = { value: exact[0].city, source_url: result.url };
    }

    // NOTE: no document drill-down. Every "Rechnungslegung/Finanzberichte"
    // document is gated by a per-document image CAPTCHA ("Sicherheitsabfrage")
    // which this probe never opens, and free-text announcements proved
    // unreliable for seat addresses (Hauptversammlung venues of third
    // parties co-occur with the company name). The results table is the
    // only precise public source, so the probe reports exactly what it says.

    const nonEmpty = Object.values(fields).filter((field) => String(field.value || "").trim()).length;
    if (nonEmpty < 2) fail("extraction_empty: parsed rows carried no usable fields");

    process.stdout.write(JSON.stringify({
      target: TARGET,
      input: company,
      fetched_at: new Date().toISOString(),
      fields,
    }, null, 2));
    process.exit(0);
  } finally {
    await browser.close();
  }
}

main().catch((error) => fail(`probe_error: ${error.message}`));
